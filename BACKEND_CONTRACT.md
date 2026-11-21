# Application ↔ Backend Integration

This file lists **exactly what `app.py` imports and calls** from the backend modules (verified against the
delivered backend by running the whole app end to end). If a backend signature changes, update this file and
`BACKEND_SPEC` / `run_analysis()` in `app.py` together.

All backend files sit in the project root next to `app.py`. Backend modules must not import Streamlit.

## Calls made by the application

| Step | Backend call | Input | Output used by the app |
|---|---|---|---|
| Extract text | `pdf_extractor.extract_text_from_pdf(bytes)` | uploaded PDF bytes | `str`; raises `PDFExtractionError` (scanned / encrypted / corrupt) |
| Clean | `text_preprocessing.clean_text(str)` | raw resume / job text | cleaned `str` (used for **all** later steps) |
| Skills | `skill_extractor.extract_skill_counts(cleaned)` | cleaned text | `{skill: mentions}` → skill list + chart counts |
| Skills by category | `skill_extractor.extract_skills_by_category(cleaned)` | cleaned resume | `{category: [skills]}` → "Resume information" table |
| Skill gap | `skill_gap.analyze_skill_gap(resume_skills, job_skills, job_counts)` | skill lists + job counts | uses `matched_skills`, `missing_skills` (`[{skill, category, mentions, priority}]`), `extra_skills`, `coverage` (may be `None`), `summary` |
| Classifier | `job_classifier.load_classifier(path)` → `.predict(text, top_k)` / `.predict_proba_dict(text)` / `.classes_` | cleaned resume | `[{"category", "probability"}]` |
| Semantic similarity | `semantic_matching.SemanticMatcher().similarity(a, b)` | cleaned texts | cosine in [-1, 1] (shown clipped to 0–1) |
| Category alignment | `scoring.category_alignment(probs, job_category)` | resume probabilities, job category | dict passed to `compute_overall_score` |
| Final score | `scoring.compute_overall_score(cosine, gap, category_info)` | the three components (any may be `None`) | `overall_score`, `match_label`, `score_breakdown`, `weights_used`, `notes` |
| Recommendations | `job_recommendation.JobRecommender(jobs_df, classifier=…, matcher=…, job_id_col="job_id").recommend(cleaned_resume, top_n)` | validated jobs DataFrame | list of dicts (`overall_score`, `match_label`, `score_breakdown`, `skill_gap`, `title`, `company`, `location`, `job_id`) |

### How the app uses the heavy objects
* The **classifier**, the **`SemanticMatcher`** and the **`JobRecommender`** (which embeds every job once) are created through
  `st.cache_resource`, so they are built once per session, not on every click.
* The classifier is loaded from `job_classifier.DEFAULT_MODEL_PATH` (`models/job_classifier.joblib`). If it is missing, the app
  still works: category prediction is skipped and `compute_overall_score` redistributes that weight.
* If the job dataset has a `category` column whose labels match none of the classifier's classes, the app drops that column
  before building the recommender so that the classifier predicts each job's category instead.

### Conversions done in `app.py`
| Backend value | App value |
|---|---|
| `probability` (classifier) | `confidence` |
| `missing_skills` list of dicts | list of names + a details table (category, mentions, priority) |
| `coverage = None` (no recognised job skills) | shown as "—" |
| `score_breakdown[*].score` (0–1) | component bars (0–100); full table in "How the score is calculated" |
| recommendation dicts | DataFrame: `rank, job_id, title, company, location, score, match_label, semantic_score, skill_score, matched_skills, missing_skills` (+ description looked up from the dataset by `job_id`) |

## Dataset formats

### Job dataset — `data/jobs.csv`
`title` (alias `job_title`, `position`, `role`) and `description` (alias `job_description`, `details`) are required;
`job_id`, `company`, `location`, `category` are optional. `data_loader.load_jobs_dataset()` normalises the names, drops empty and
duplicate rows and adds a display `label` column.

### Resume dataset (optional) — `data/resumes.csv`
`resume_text` (alias `Resume_str`, `resume`, `text`) is required; `resume_id` (`ID`) and `category` are optional. This file is also a
valid training file for the classifier (`--text-col Resume_str --label-col Category`).

## One-time backend setup
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python test_wiring.py                       # backend wiring check (no model download)
python job_classifier.py --data data/resumes.csv --text-col Resume_str --label-col Category   # trains models/job_classifier.joblib
streamlit run app.py                        # first analysis downloads the MiniLM embedding model (internet needed once)
```
See `BACKEND_README.md` for the backend developer's own documentation.

## Error-handling behaviour of the app

| Situation | What the user sees |
|---|---|
| Backend file missing / import error | ❌ in *Backend status*; only the dependent step is skipped |
| No trained classifier | Warning for "categories"; score still computed (weights redistributed) |
| Embedding model cannot be loaded (e.g. offline first run) | Warning for "similarity" and "recommendations"; score uses skills/category |
| Job text has no recognised skills | Skill coverage "—"; score then relies on the other components |
| PDF scanned / encrypted / corrupt | Analysis stops with the backend's error message |
| Wrong return type | Warning naming the step; other steps continue |
