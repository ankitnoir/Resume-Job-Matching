"""
test_wiring.py
==============
Checks that all modules fit together (imports, signatures, return formats, file paths)
WITHOUT downloading a model: embeddings come from a tiny deterministic stand-in with the same
interface as SemanticMatcher, and the training data is synthetic. This verifies the wiring only;
it says nothing about real-world accuracy. Run:  python test_wiring.py
"""
import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from job_classifier import JobClassifier, load_classifier, train_model
from job_recommendation import JobRecommender, results_to_dataframe
from scoring import compute_overall_score, evaluate_match, validate_weights
from semantic_matching import SemanticMatcher, split_into_chunks
from skill_extractor import SkillExtractor, extract_skills
from skill_gap import analyze_skill_gap
from text_preprocessing import clean_text, preprocess_text


class StubMatcher(SemanticMatcher):
    """Hashed bag-of-words embeddings: same interface as SemanticMatcher, no model download."""
    model_name = "stub-bow"

    def encode_documents(self, texts, batch_size=32):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in clean_text(text).lower().split():
                out[i, sum(map(ord, word)) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=out, where=norms > 0)

    def similarity(self, a, b):
        e = self.encode_documents([a, b])
        return float(e[0] @ e[1])


RESUME = "Python developer. Skills: Python, SQL, Docker, AWS, pandas, REST API. Built cloud services."
JOB = "We need Python, SQL, Docker, Kubernetes and AWS experience. Docker is essential."


class WiringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        vocab = {"Information-Technology": "python docker sql api cloud aws".split(),
                 "HR": "recruitment onboarding payroll employee hiring policy".split()}
        rows = [{"Resume_str": " ".join(rng.choice(w, 40)) + " team", "Category": c}
                for c, w in vocab.items() for _ in range(25)]
        pd.DataFrame(rows).to_csv(cls.tmp / "resumes.csv", index=False)
        cls.metrics = train_model(cls.tmp / "resumes.csv", "Resume_str", "Category", cls.tmp / "m.joblib")
        cls.clf = load_classifier(cls.tmp / "m.joblib")
        cls.matcher = StubMatcher()

    def test_training_outputs(self):
        self.assertTrue((self.tmp / "m.joblib").is_file())
        self.assertTrue((self.tmp / "m.metrics.json").is_file())
        self.assertEqual(set(self.clf.classes_), {"Information-Technology", "HR"})

    def test_predict_format(self):
        out = self.clf.predict(RESUME, top_k=2)
        self.assertEqual(set(out[0]), {"category", "probability"})
        self.assertAlmostEqual(sum(self.clf.predict_proba_dict(RESUME).values()), 1.0, places=5)

    def test_skills_and_gap(self):
        self.assertIn("Python", extract_skills(RESUME))
        self.assertNotIn("SQL", extract_skills("only mysql here"))
        self.assertNotIn("Java", extract_skills("javascript only"))
        job_counts = SkillExtractor().extract_with_counts(JOB)
        gap = analyze_skill_gap(extract_skills(RESUME), list(job_counts), job_counts)
        self.assertEqual([m["skill"] for m in gap["missing_skills"]], ["Kubernetes"])
        self.assertEqual(gap["method"], "rule_based_dictionary_matching")

    def test_evaluate_match(self):
        res = evaluate_match(RESUME, JOB, classifier=self.clf, matcher=self.matcher,
                             job_category="information-technology")
        self.assertTrue(0 <= res["overall_score"] <= 100)
        self.assertAlmostEqual(sum(res["weights_used"].values()), 1.0, places=3)
        contributions = sum(c["contribution"] for c in res["score_breakdown"].values())
        self.assertAlmostEqual(contributions, res["overall_score"], places=1)
        json.dumps(res)  # must be JSON-serialisable (st.json)
        no_clf = evaluate_match(RESUME, JOB, matcher=self.matcher)
        self.assertEqual(no_clf["weights_used"]["category_alignment"], 0.0)

    def test_weights_validation(self):
        with self.assertRaises(ValueError):
            validate_weights({"bogus": 1})
        with self.assertRaises(ValueError):
            compute_overall_score(0.5, None, None, {"semantic_similarity": 0, "skill_match": 1,
                                                    "category_alignment": 0})

    def test_recommender_ids_and_json(self):
        jobs = pd.DataFrame({"title": ["Py Dev", "HR Lead", "Cloud Eng"],
                             "description": [JOB, "Recruitment, onboarding, payroll.", "Docker, AWS, Kubernetes."],
                             "job_id": [np.int64(101), np.int64(102), np.int64(103)]}, index=[10, 20, 30])
        recs = JobRecommender(jobs, classifier=self.clf, matcher=self.matcher, job_id_col="job_id") \
            .recommend(RESUME, top_n=3)
        json.dumps(recs)  # numpy ints would raise here
        self.assertEqual([r["rank"] for r in recs], [1, 2, 3])
        self.assertTrue(all(isinstance(r["job_id"], int) for r in recs))
        # without job_id_col the caller's ORIGINAL index must be returned
        recs2 = JobRecommender(jobs, matcher=self.matcher).recommend(RESUME, top_n=3)
        self.assertEqual({r["job_id"] for r in recs2}, {10, 20, 30})
        self.assertEqual(len(results_to_dataframe(recs)), 3)

    def test_category_vocabulary_warning(self):
        jobs = pd.DataFrame({"title": ["A"], "description": [JOB], "category": ["Totally Unknown"]})
        with self.assertLogs("job_recommendation", level=logging.WARNING):
            JobRecommender(jobs, classifier=self.clf, matcher=self.matcher)

    def test_preprocessing_and_chunks(self):
        self.assertNotIn("@", clean_text("mail me a@b.com now"))
        self.assertIn("cplusplus", preprocess_text("Experienced in C++ and Node.js"))
        self.assertEqual([len(c.split()) for c in split_into_chunks("w " * 500, 180, 40)], [180, 180, 180, 80])


if __name__ == "__main__":
    unittest.main(verbosity=2)
