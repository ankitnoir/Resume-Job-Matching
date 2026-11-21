# AI/ML Backend: Resume–Job Matching and Skill Gap Analysis

Keep all `.py` files in **one folder** (they import each other as plain modules, e.g. `from scoring import ...`). Put `app.py` in the same folder, or add this folder to `PYTHONPATH` / `sys.path`.

```
resume_matcher/
  pdf_extractor.py  text_preprocessing.py  skill_extractor.py  job_classifier.py
  semantic_matching.py  skill_gap.py  job_recommendation.py  scoring.py
  demo_pipeline.py  test_wiring.py  requirements.txt
  models/            <- created by training (job_classifier.joblib + .metrics.json)
  app.py             <- the Streamlit app
```

## 1. Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_wiring.py        # checks module wiring, no model download needed
```

Install the package named **PyMuPDF** (`pip install PyMuPDF`), not a package called `fitz`.

The first call to `SemanticMatcher` downloads `all-MiniLM-L6-v2` from Hugging Face (internet needed once, then cached). No paid APIs are used.

## 2. Evidence types (kept separate on purpose)

| Evidence | Module | Method | Output |
|---|---|---|---|
| Category **classification** | `job_classifier.py` | TF-IDF + Logistic Regression, trained on your labelled data | probability per category |
| Semantic **similarity** | `semantic_matching.py` | pretrained Sentence-Transformer + cosine | cosine in [-1, 1] |
| **Rule-based** skill matching | `skill_extractor.py`, `skill_gap.py` | dictionary matching + set comparison | matched / missing / extra skills |

`scoring.py` combines them: `overall = 100 × Σ weight_i × score_i`, default weights semantic 0.40, skills 0.45, category 0.15 (design choices, tune and justify them in your report). Unavailable components have their weight redistributed; the weights actually used are returned.

## 3. Datasets

**Classifier training data (resume text + category).** Suggested public datasets on Kaggle (check the dataset page for the licence and the exact column names after download):

* *Resume Dataset* (livecareer resumes, ~2.4k resumes, 24 categories). Typical columns: `ID`, `Resume_str`, `Resume_html`, `Category`. Use `--text-col Resume_str --label-col Category`.
* *UpdatedResumeDataSet.csv* (~960 resumes, ~25 categories, mostly IT-oriented). Typical columns: `Category`, `Resume`. Use `--text-col Resume --label-col Category`.

Required format: a CSV (or DataFrame) with one column of resume text and one column of category labels (strings). Rows with empty text/label are dropped; classes with fewer than 2 samples are dropped (a warning is logged).

**Job postings (for recommendations).** Any CSV of postings, for example a public Kaggle job-postings dataset. Required columns: `title`, `description`. Optional: `company`, `location`, `category`, `job_id`. If you supply a `category` column, its labels should use the **same names as the classifier's classes** (case-insensitive), otherwise the category component is reported as unavailable for that job. If there is no `category` column, the classifier predicts a category from the job text.

**Skills dictionary.** Built in (`DEFAULT_SKILLS`). Extend it with a CSV: `skill,category,aliases` (aliases separated by `|`). Public sources to mine for more skills: ESCO and O*NET.

## 4. Train and save the classifier

```bash
python job_classifier.py --data data/Resume.csv --text-col Resume_str --label-col Category
```

The model is saved to `models/job_classifier.joblib` **next to the code** (absolute path `DEFAULT_MODEL_PATH`), so training and the app agree regardless of the working directory. Use `--output` to change it. This prints hold-out accuracy and macro-F1, and writes `models/job_classifier.metrics.json` (per-class report and confusion matrix). Report **your own** measured numbers. Options: `--model-type complement_nb`, `--test-size 0.25`, `--refit-on-full-data`.

From Python:

```python
from job_classifier import load_dataset, JobClassifier, DEFAULT_MODEL_PATH
df = load_dataset("data/Resume.csv", "Resume_str", "Category")
clf = JobClassifier(); metrics = clf.train(df); clf.save(DEFAULT_MODEL_PATH)   # import DEFAULT_MODEL_PATH too
```

## 5. Function signatures

```python
# pdf_extractor.py
extract_text_from_pdf(source) -> str                 # path | bytes | uploaded file; raises PDFExtractionError
extract_text_from_file(source, filename=None) -> str # PDF or TXT
# text_preprocessing.py
clean_text(text) -> str                              # light cleaning (skills, embeddings)
preprocess_text(text) -> str                         # heavy preprocessing (classifier)
# skill_extractor.py
extract_skills(text) -> list[str]
extract_skill_counts(text) -> dict[str, int]
extract_skills_by_category(text) -> dict[str, list[str]]
# job_classifier.py
load_classifier(path=DEFAULT_MODEL_PATH) -> JobClassifier ; .predict(text, top_k=3) -> [{"category","probability"}]
# semantic_matching.py
SemanticMatcher().similarity(a, b) -> float
# skill_gap.py
analyze_skill_gap(candidate_skills, job_skills, job_skill_counts=None) -> dict
# scoring.py
evaluate_match(resume_text, job_text, classifier=None, matcher=None, weights=None, job_category=None) -> dict
# job_recommendation.py
JobRecommender(jobs_df, classifier=None).recommend(resume_text, top_n=5) -> list[dict]
results_to_dataframe(results) -> pd.DataFrame
```

## 6. Example output formats

> The values below only illustrate the **structure**. They are not measured results.

```python
clf.predict(resume_text, top_k=2)
# [{"category": "<top category>", "probability": 0.0}, {"category": "<second>", "probability": 0.0}]

analyze_skill_gap(["Python", "SQL"], ["Python", "Docker", "AWS"], {"Python": 2, "Docker": 3, "AWS": 1})
# {"method": "rule_based_dictionary_matching",
#  "matched_skills": ["Python"],
#  "missing_skills": [{"skill": "Docker", "category": "Cloud & DevOps", "mentions": 3, "priority": "high"},
#                     {"skill": "AWS", "category": "Cloud & DevOps", "mentions": 1, "priority": "normal"}],
#  "extra_skills": ["SQL"], "coverage": 0.3333, "weighted_coverage": 0.3333, ...}

evaluate_match(resume_text, job_text, classifier=clf)
# {"overall_score": <0-100>, "match_label": "Good match",
#  "score_breakdown": {"semantic_similarity": {"raw_value", "score", "weight_used", "contribution", "explanation", ...},
#                      "skill_match": {...}, "category_alignment": {...}},
#  "weights_used": {...}, "semantic": {...}, "classification": {...},
#  "resume_skills": [...], "job_skills": [...], "skill_gap": {...}, "notes": []}
```

## 7. Integration with `app.py` (Streamlit)

```python
import streamlit as st
from pdf_extractor import extract_text_from_file, PDFExtractionError
from job_classifier import load_classifier
from semantic_matching import SemanticMatcher
from scoring import evaluate_match
from job_recommendation import JobRecommender, load_jobs, results_to_dataframe

@st.cache_resource
def get_models():
    clf = load_classifier()   # loads models/job_classifier.joblib (train first, section 4)
    return clf, SemanticMatcher()

@st.cache_resource
def get_recommender(jobs_csv: str):
    clf, matcher = get_models()
    return JobRecommender(load_jobs(jobs_csv), classifier=clf, matcher=matcher)

clf, matcher = get_models()
resume_file = st.file_uploader("Resume (PDF)", type=["pdf", "txt"])
job_text = st.text_area("Job description")

if resume_file and job_text and st.button("Analyze"):
    try:
        resume_text = extract_text_from_file(resume_file)
        result = evaluate_match(resume_text, job_text, classifier=clf, matcher=matcher)
    except (PDFExtractionError, ValueError, RuntimeError) as e:
        st.error(str(e))
    else:
        st.metric("Overall match", f"{result['overall_score']:.1f}/100", result["match_label"])
        st.json(result["score_breakdown"])
        st.write("Missing skills:", [m["skill"] for m in result["skill_gap"]["missing_skills"]])

# Job recommendations
# recs = get_recommender("data/jobs.csv").recommend(resume_text, top_n=5)
# st.dataframe(results_to_dataframe(recs))
```

## 8. Known limitations (mention in your report)

* Skill extraction only finds skills in the dictionary; a few words are ambiguous ("react", "spring", "C" in "Vitamin C").
* Scanned (image-only) PDFs are rejected; there is no OCR.
* The classifier is trained on resumes; applying it to job text is an approximation, and its category set is limited to your dataset's labels.
* Score weights and the cosine rescaling range (0.20–0.80) are documented design assumptions, not learned values. Validate them on labelled resume–job pairs if you have any.
* Only load `.joblib` model files you created yourself.
* `job_id` in recommendations is your `job_id_col` value, or else the row's original DataFrame index.
* Run `python demo_pipeline.py --resume cv.pdf --job-text "..."` for a command-line end-to-end run with the real model.
