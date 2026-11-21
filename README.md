# 🎯 AI-Based Resume–Job Matching and Skill Gap Analysis Using NLP and Machine Learning
Live Demo: https://resume-job-matcher-ankit.streamlit.app/

A Streamlit web application that reads a candidate's PDF resume, compares it with a job description,
and reports **how well the candidate fits**, **which skills are missing**, and **which other jobs suit the
resume best** — entirely locally, with no paid APIs.

> The repository has two layers: the **application** (`app.py`, `data_loader.py`, `dashboard.py`, `visualization.py`)
> and the **NLP/ML backend** (eight modules, documented in [`BACKEND_README.md`](BACKEND_README.md)).
> [`BACKEND_CONTRACT.md`](BACKEND_CONTRACT.md) lists exactly how the two connect.

---

## 1. Problem statement

Recruiters screen hundreds of resumes per opening, and candidates rarely know why they are rejected or what
to learn next. Keyword search misses synonyms ("ML" vs "machine learning") and gives no guidance.
This project combines **semantic text similarity** with **structured skill extraction** to
(1) score a resume against a job, (2) explain the score through matched and missing skills, and
(3) recommend the most suitable jobs from a dataset.

## 2. Features

| Area | What it does |
|---|---|
| **Resume upload** | PDF upload with format, size (≤ 5 MB) and header validation; text extracted by the backend (PyMuPDF) |
| **Job description input** | Paste manually, or select any posting from the job dataset (or upload your own CSV) |
| **Resume analysis** | Extracted skills, skill-mention distribution, predicted job categories with confidence, resume information |
| **Job matching** | Overall match score, semantic similarity, score components, matched / missing / extra skills |
| **Skill gap analysis** | Colour-coded skill chips, coverage donut, emphasis of each skill in the job description |
| **Job recommendations** | Ranked table with score bars, per-job matched/missing skills, CSV export |
| **Dashboard** | KPI cards, gauge, breakdown, ranked-jobs chart, most common missing skills across recommended jobs |
| **Robustness** | Missing backend module/model/dataset → clear warning instead of a crash; no hard-coded or fake scores |

## 3. System architecture

```
┌────────────────────────────── Streamlit application (this repo) ──────────────────────────────┐
│                                                                                                │
│  app.py  ── sidebar inputs, pipeline orchestration, output validation, session state           │
│    │                                                                                           │
│    ├── data_loader.py   upload checks · jobs/resumes CSV loading, column normalisation         │
│    ├── dashboard.py     tabs: Dashboard · Resume Analysis · Job Match · Recommendations        │
│    └── visualization.py Plotly figures (gauge, donut, bars, ranked jobs)                       │
│                                                                                                │
└───────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                            │  function calls defined in BACKEND_CONTRACT.md
┌───────────────────────────────────────────▼────────────────────────────────────────────────────┐
│ NLP / ML backend (separate developer)                                                          │
│ pdf_extractor · text_preprocessing · skill_extractor · job_classifier ·                        │
│ semantic_matching · skill_gap · job_recommendation · scoring                                   │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Processing pipeline** (`app.run_analysis`)

1. `extract_text_from_pdf` → raw resume text
2. `clean_text` → cleaned resume + job text
3. `extract_skill_counts` (resume and job) and `extract_skills_by_category`
4. `analyze_skill_gap` → matched / missing (with priority) / extra / coverage
5. `JobClassifier.predict` / `predict_proba_dict` → resume categories and category alignment
6. `SemanticMatcher.similarity` → cosine similarity of resume and job embeddings
7. `compute_overall_score` → overall 0–100 score, verdict and per-component explanation
8. `JobRecommender.recommend` → ranked jobs (job embeddings are computed once and cached)

Every step is isolated: if one fails, the others still run and the failure is listed in the UI.

## 4. Project structure

```
ai-resume-job-matcher/
├── app.py                  # Streamlit entry point
├── data_loader.py          # dataset + upload validation
├── dashboard.py            # dashboard / tab rendering
├── visualization.py        # Plotly chart builders
├── BACKEND_CONTRACT.md     # function signatures & data formats for the backend developer
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md           # accepted dataset columns
│   ├── sample_jobs.csv     # small sample (committed)
│   ├── jobs.csv            # full job dataset (you add this; git-ignored)
│   └── resumes.csv         # optional resume dataset (git-ignored)
│
│   ── NLP/ML backend ──
├── pdf_extractor.py   text_preprocessing.py   skill_extractor.py   job_classifier.py
├── semantic_matching.py   skill_gap.py   job_recommendation.py   scoring.py
├── test_wiring.py          # backend wiring tests (no model download)
├── demo_pipeline.py        # command-line demo of the backend
├── BACKEND_README.md       # backend documentation
└── models/                 # trained classifier (created by training; git-ignored)
```

## 5. Installation

Requires **Python 3.10+**.

```bash
git clone <your-repo-url>
cd ai-resume-job-matcher

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

```bash
python -m spacy download en_core_web_sm      # one-time; improves classifier preprocessing
python test_wiring.py                         # checks the backend wiring (no downloads)
```

`sentence-transformers` installs PyTorch (a few hundred MB); the CPU build is sufficient. The first analysis downloads the
`all-MiniLM-L6-v2` model once (internet needed); after that everything works offline.

**Train the classifier once** (needs a labelled resume CSV, e.g. a public Kaggle resume dataset):

```bash
python job_classifier.py --data data/resumes.csv --text-col Resume_str --label-col Category
```

This creates `models/job_classifier.joblib`. Without it the app still runs, but category prediction is skipped.

## 6. Running the application

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

1. **Backend:** already included; train the classifier once (see above).
2. **Data:** put your job dataset at `data/jobs.csv` (or upload a CSV in *⚙️ Data & settings*). Without it the app uses `data/sample_jobs.csv`.
3. In the sidebar, upload a PDF resume (or pick one from `data/resumes.csv`), paste or select a job description, and press **🚀 Analyze resume**.

The **🔌 Backend status** panel lists every backend module, the trained classifier and the embedding library with ✅ / ⚠️ / ❌. The first analysis is slower because the embedding model is downloaded and every job is embedded once; later runs reuse the cache. Use **🔄 Reload backend modules** after *editing* a backend file.

## 7. Expected UI

| Tab | Content |
|---|---|
| **📊 Dashboard** | Four KPI cards (overall match, semantic similarity, skill coverage, missing skills) with a verdict; gauge, score-component bars and matched/missing donut; ranked top-job bars; most common missing skills across recommended jobs; skill distribution and category predictions |
| **📄 Resume Analysis** | Green/blue skill chips and a mention-count chart, category table with confidence bars, word/character counts, resume information table, expandable raw text |
| **🎯 Job Match** | KPI row, gauge and score breakdown, ✅ matched / ❌ missing / ➕ extra skill chips, coverage donut, emphasis charts for matched and missing skills, skill comparison table |
| **💼 Job Recommendations** | Ranked table with progress-bar scores, bar chart, one expander per job (matched/missing skills, description), CSV download |

## 8. Methodology

The application layer performs no model inference itself. The backend (details in `BACKEND_README.md`) uses three separate kinds of evidence:

* **Text extraction & cleaning** — PyMuPDF text extraction (text-based PDFs only); light cleaning that removes e-mails, phone numbers and URLs.
* **Rule-based skill matching** — dictionary matching with aliases against a built-in skills vocabulary; the skill gap is a set comparison (matched / missing / extra) with a priority based on how often the job mentions each missing skill.
* **ML classification** — TF-IDF + Logistic Regression trained on labelled resumes; gives category probabilities for the resume and a *category alignment* score for the job.
* **Semantic similarity** — pretrained Sentence-Transformer (`all-MiniLM-L6-v2`) embeddings compared by cosine similarity, which captures meaning beyond exact keywords.
* **Match score** — `overall = 100 × Σ weight × component`, default weights: semantic 0.40, skills 0.45, category 0.15 (cosine rescaled from 0.20–0.80 to 0–1). Unavailable components have their weight redistributed; the weights used are shown in the dashboard. These are documented design assumptions, not learned values.
* **Recommendation** — the same scoring applied between the resume and every job, then ranked.

Application-side computations are limited to presentation: clipping values to their valid range, sorting recommendations by score and converting formats.

## 9. Limitations

* Only **text-based PDFs** are supported; scanned resumes need OCR, which is not included.
* Skill extraction only finds skills in the built-in dictionary (a few words such as "react" or "spring" are ambiguous).
* The classifier is trained on resumes and applied to job text as an approximation; its categories are limited to the training labels.
* When a job text contains no recognised skills, the score relies on the remaining components only.
* The first analysis needs internet once (embedding model download) and the first job ranking embeds every job, which can take minutes for thousands of jobs.
* Skill gap compares skill *names*; it does not judge proficiency or years of experience.
* The semantic model is generic and not fine-tuned for every industry; multi-column or graphic-heavy resume layouts can degrade text extraction.
* Recommendation speed depends on dataset size and on the backend caching job embeddings (the sidebar limits the rows scanned).
* Scores are decision-support, **not** hiring decisions; models can reflect biases present in training data.
* Resumes are handled in memory for the session only, but no authentication or multi-user isolation is implemented.

## 10. Future work

OCR for scanned resumes · multilingual support · fine-tuned embedding model · ATS-style resume improvement tips ·
course recommendations for missing skills · saved analysis history.

## 11. Team & integration

* **Application (UI, data loading, dashboard, visualisation):** this repository.
* **NLP/ML backend:** developed independently — see [`BACKEND_README.md`](BACKEND_README.md); the connection is described in [`BACKEND_CONTRACT.md`](BACKEND_CONTRACT.md).

## 12. License

Academic project — add your preferred license (e.g. MIT) before publishing.
