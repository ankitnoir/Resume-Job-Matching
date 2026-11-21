"""
job_recommendation.py
=====================
Rank a collection of job postings for one candidate resume.

Every job is scored with exactly the same components and weights as `scoring.evaluate_match`
(semantic similarity + rule-based skill match + ML category alignment), so the ranking is
explainable: each recommendation carries its score breakdown and skill gap.

Jobs DataFrame format
---------------------
required columns : title, description
optional columns : company, location, category, job_id
Column names are configurable in the constructor (e.g. company_col="company_name" for a
LinkedIn-style export). If a 'category' column exists, its labels should use the SAME names as
the classifier's classes (case-insensitive); otherwise the category component is unavailable
for those jobs (a warning is logged). Without a 'category' column, the classifier predicts a
category from the job text.

`job_id` in the results is the value of `job_id_col` if given, otherwise the ORIGINAL index of
the row in the DataFrame you passed in.

Performance: job embeddings are computed ONCE when the JobRecommender is built. In Streamlit,
create the recommender once and cache it (st.cache_resource), then call .recommend() per resume.
For very large files pass max_jobs=N (a reproducible random sample) to limit embedding time.

Public API
----------
load_jobs(csv_path)                                  -> pd.DataFrame
JobRecommender(jobs_df, classifier=None, matcher=None, extractor=None, weights=None, ...)
    .recommend(resume_text, top_n=5, min_score=0.0)  -> list[dict]
results_to_dataframe(results)                        -> pd.DataFrame
recommend_jobs(resume_text, jobs_df, classifier=None, top_n=5, ...) -> list[dict]  (convenience)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import pandas as pd

from job_classifier import JobClassifier
from scoring import DEFAULT_SEMANTIC_RANGE, category_alignment, compute_overall_score, validate_weights
from semantic_matching import SemanticMatcher, cosine_similarity_matrix, get_default_matcher
from skill_extractor import SkillExtractor, get_default_extractor
from skill_gap import analyze_skill_gap
from text_preprocessing import clean_text

logger = logging.getLogger(__name__)


def load_jobs(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Read a job-postings CSV (UTF-8, falling back to latin-1)."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Jobs file not found: {path}")
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _to_python(value: Any) -> Any:
    """Convert numpy scalars to plain Python types so results are JSON-serialisable."""
    return value.item() if hasattr(value, "item") else value


class JobRecommender:
    """Pre-indexes a set of jobs and ranks them for any resume."""

    def __init__(
        self,
        jobs_df: pd.DataFrame,
        classifier: Optional[JobClassifier] = None,
        matcher: Optional[SemanticMatcher] = None,
        extractor: Optional[SkillExtractor] = None,
        weights: Optional[Mapping[str, float]] = None,
        semantic_range=DEFAULT_SEMANTIC_RANGE,
        title_col: str = "title",
        description_col: str = "description",
        company_col: str = "company",
        location_col: str = "location",
        category_col: str = "category",
        job_id_col: Optional[str] = None,
        max_jobs: Optional[int] = None,
    ) -> None:
        if not isinstance(jobs_df, pd.DataFrame):
            raise TypeError("jobs_df must be a pandas DataFrame.")
        for col in (title_col, description_col):
            if col not in jobs_df.columns:
                raise ValueError(f"Jobs DataFrame needs a '{col}' column. Found: {list(jobs_df.columns)}")

        validate_weights(weights)  # fail early instead of inside the scoring loop

        self.classifier = classifier
        self.matcher = matcher or get_default_matcher()
        self.extractor = extractor or get_default_extractor()
        self.weights = weights
        self.semantic_range = semantic_range

        jobs = jobs_df.copy()
        jobs[description_col] = jobs[description_col].fillna("").astype(str)
        jobs[title_col] = jobs[title_col].fillna("").astype(str)
        jobs = jobs[jobs[description_col].str.strip() != ""]
        if jobs.empty:
            raise ValueError("No jobs with a non-empty description were provided.")
        if max_jobs is not None and len(jobs) > max_jobs:
            jobs = jobs.sample(n=max_jobs, random_state=42)  # reproducible sample

        self._orig_index = jobs.index.tolist()  # keep the caller's index for job_id
        self.jobs = jobs.reset_index(drop=True)
        self._cols = {"title": title_col, "company": company_col, "location": location_col,
                      "category": category_col}
        self._id_col = job_id_col if job_id_col and job_id_col in self.jobs.columns else None

        # Text used for embeddings / skills: title + description
        self._texts: List[str] = [
            clean_text(f"{t}. {d}") for t, d in zip(self.jobs[title_col], self.jobs[description_col])
        ]

        logger.info("Embedding %d job descriptions ...", len(self._texts))
        self._embeddings = self.matcher.encode_documents(self._texts)

        # Skills for every job (rule-based)
        self._job_skill_counts = [self.extractor.extract_with_counts(t) for t in self._texts]

        # Job categories: provided column, else classifier prediction, else unknown
        self.category_source: Optional[str] = None
        self._job_categories: List[Optional[str]] = [None] * len(self.jobs)
        if category_col in self.jobs.columns:
            self._job_categories = [
                str(c).strip() if pd.notna(c) and str(c).strip() else None
                for c in self.jobs[category_col]
            ]
            self.category_source = "provided_column"
            self._warn_if_categories_unknown()
        elif classifier is not None:
            self._job_categories = classifier.predict_labels(self._texts)
            self.category_source = "classifier_prediction_on_job_text"

    def _warn_if_categories_unknown(self) -> None:
        """Warn when the jobs' category labels do not match the classifier's classes."""
        if self.classifier is None:
            return
        known = {c.lower() for c in self.classifier.classes_}
        usable = sum(1 for c in self._job_categories if c and c.lower() in known)
        if usable < len(self._job_categories):
            logger.warning(
                "%d of %d jobs have a category that is not one of the classifier's classes %s; "
                "the category component is unavailable for them (weights are redistributed).",
                len(self._job_categories) - usable, len(self._job_categories),
                sorted(self.classifier.classes_))

    def _row_value(self, idx: int, key: str) -> Optional[str]:
        col = self._cols[key]
        if col in self.jobs.columns:
            value = self.jobs.at[idx, col]
            return None if pd.isna(value) else str(value)
        return None

    def recommend(self, resume_text: str, top_n: int = 5, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """
        Rank jobs for a resume.

        Returns a list (best first) of dicts:
        {
          "rank", "job_id", "title", "company", "location", "job_category",
          "overall_score", "match_label", "score_breakdown", "weights_used",
          "semantic_similarity" (raw cosine), "skill_gap" (see skill_gap.py),
          "resume_top_categories": [{"category", "probability"}, ...] ([] without classifier)
        }
        All values are plain Python types (JSON-serialisable).
        """
        resume_clean = clean_text(resume_text)
        if not resume_clean:
            raise ValueError("Resume text is empty.")

        # 1) Semantic similarity of the resume to all jobs (one matrix product)
        resume_embedding = self.matcher.encode_documents([resume_clean])
        cosines = cosine_similarity_matrix(resume_embedding, self._embeddings)[0]

        # 2) Resume skills (rule-based)
        resume_skills = list(self.extractor.extract_with_counts(resume_clean))

        # 3) Resume category probabilities (ML), if a classifier is available
        resume_probs: Optional[Dict[str, float]] = None
        top_categories: List[Dict[str, Any]] = []
        if self.classifier is not None:
            try:
                resume_probs = self.classifier.predict_proba_dict(resume_clean)
                ranked = sorted(resume_probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
                top_categories = [{"category": c, "probability": round(p, 4)} for c, p in ranked]
            except (ValueError, RuntimeError) as exc:
                logger.warning("Classifier skipped for this resume: %s", exc)

        # 4) Score every job with the shared scoring function
        results: List[Dict[str, Any]] = []
        for idx in range(len(self.jobs)):
            job_counts = self._job_skill_counts[idx]
            gap = analyze_skill_gap(resume_skills, list(job_counts), job_counts, self.extractor)
            cat_info = category_alignment(resume_probs, self._job_categories[idx]) if resume_probs else None
            score = compute_overall_score(float(cosines[idx]), gap, cat_info,
                                          self.weights, self.semantic_range)
            if score["overall_score"] < min_score:
                continue
            job_id = self.jobs.at[idx, self._id_col] if self._id_col else self._orig_index[idx]
            results.append({
                "job_id": _to_python(job_id),
                "title": self._row_value(idx, "title"),
                "company": self._row_value(idx, "company"),
                "location": self._row_value(idx, "location"),
                "job_category": self._job_categories[idx],
                "overall_score": score["overall_score"],
                "match_label": score["match_label"],
                "score_breakdown": score["score_breakdown"],
                "weights_used": score["weights_used"],
                "semantic_similarity": round(float(cosines[idx]), 4),
                "skill_gap": gap,
                "resume_top_categories": top_categories,
            })

        results.sort(key=lambda r: r["overall_score"], reverse=True)
        results = results[: max(1, top_n)]
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        return results


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Flatten recommendation results into a table for st.dataframe()."""
    rows = []
    for r in results:
        breakdown = r["score_breakdown"]
        rows.append({
            "Rank": r["rank"],
            "Job ID": r["job_id"],
            "Title": r["title"],
            "Company": r["company"],
            "Location": r["location"],
            "Overall score": r["overall_score"],
            "Match": r["match_label"],
            "Semantic similarity": breakdown["semantic_similarity"]["raw_value"],
            "Skill coverage": breakdown["skill_match"]["raw_value"],
            "Category alignment": breakdown["category_alignment"]["score"],
            "Matched skills": ", ".join(r["skill_gap"]["matched_skills"]),
            "Missing skills": ", ".join(m["skill"] for m in r["skill_gap"]["missing_skills"]),
        })
    return pd.DataFrame(rows)


def recommend_jobs(resume_text: str, jobs_df: pd.DataFrame, classifier: Optional[JobClassifier] = None,
                   top_n: int = 5, **recommender_kwargs: Any) -> List[Dict[str, Any]]:
    """
    One-shot helper. NOTE: it re-embeds all jobs on every call. For an app, build a
    JobRecommender once and reuse it.
    """
    return JobRecommender(jobs_df, classifier=classifier, **recommender_kwargs).recommend(
        resume_text, top_n=top_n)
