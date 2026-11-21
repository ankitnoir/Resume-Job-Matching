"""
app.py
======
Streamlit front end for
"AI-Based Resume–Job Matching and Skill Gap Analysis Using NLP and Machine Learning".

Run with:   streamlit run app.py

This file contains NO machine-learning logic. It
  1. collects the resume (PDF upload or dataset) and the job description,
  2. calls the backend modules (see BACKEND_CONTRACT.md for the exact functions used),
  3. validates / converts what the backend returns, and
  4. hands the result to dashboard.py for display.

If a backend module, trained model or dataset is missing, the app keeps running and
explains exactly what is missing instead of crashing or showing made-up numbers.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import dashboard as dash
import data_loader as dl

APP_TITLE = "AI Resume–Job Matcher"
DEFAULT_MODEL_FILE = dl.PROJECT_ROOT / "models" / "job_classifier.joblib"

# --------------------------------------------------------------------------- #
# Backend registry — names the app imports (keep in sync with BACKEND_CONTRACT.md)
# --------------------------------------------------------------------------- #
BACKEND_SPEC: dict[str, list[str]] = {
    "pdf_extractor": ["extract_text_from_pdf"],
    "text_preprocessing": ["clean_text"],
    "skill_extractor": ["extract_skill_counts", "extract_skills_by_category"],
    "job_classifier": ["load_classifier"],
    "semantic_matching": ["SemanticMatcher"],
    "skill_gap": ["analyze_skill_gap"],
    "scoring": ["compute_overall_score", "category_alignment"],
    "job_recommendation": ["JobRecommender"],
}


class BackendUnavailable(RuntimeError):
    """A backend module or function is missing or failed to import."""


class AnalysisError(Exception):
    """A fatal problem that stops the analysis (shown to the user)."""


# --------------------------------------------------------------------------- #
# Backend loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading backend modules…")
def load_backend() -> dict[str, dict[str, Any]]:
    """Import every backend module once (heavy models are loaded lazily, later)."""
    importlib.invalidate_caches()  # pick up files created after Streamlit started
    status: dict[str, dict[str, Any]] = {}
    for module_name, functions in BACKEND_SPEC.items():
        entry: dict[str, Any] = {"error": None, "functions": {}, "module": None}
        try:
            module = importlib.import_module(module_name)
            entry["module"] = module
            for fn_name in functions:
                fn = getattr(module, fn_name, None)
                entry["functions"][fn_name] = fn if callable(fn) else None
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                entry["error"] = f"{module_name}.py not found"
            else:
                entry["error"] = f"missing dependency '{exc.name}'"
        except Exception as exc:  # backend crashed while importing
            entry["error"] = f"{type(exc).__name__}: {exc}"
        status[module_name] = entry
    return status


def get_fn(backend: dict, module: str, name: str):
    entry = backend[module]
    if entry["error"]:
        raise BackendUnavailable(f"`{module}` is unavailable ({entry['error']}).")
    fn = entry["functions"].get(name)
    if fn is None:
        raise BackendUnavailable(f"`{module}.{name}` is not implemented yet.")
    return fn


def _model_path(backend: dict) -> Path:
    module = backend["job_classifier"]["module"]
    return Path(getattr(module, "DEFAULT_MODEL_PATH", DEFAULT_MODEL_FILE))


def _model_mtime(backend: dict) -> float:
    try:
        return _model_path(backend).stat().st_mtime
    except OSError:
        return 0.0


def backend_status_frame(backend: dict) -> pd.DataFrame:
    rows = []
    for module, functions in BACKEND_SPEC.items():
        entry = backend[module]
        if entry["error"]:
            status = f"❌ {entry['error']}"
        else:
            missing = [f for f in functions if entry["functions"].get(f) is None]
            status = "✅ Ready" if not missing else f"⚠️ missing {', '.join(missing)}"
        rows.append({"Module": f"{module}.py", "Used by the app": ", ".join(functions), "Status": status})
    if not backend["job_classifier"]["error"]:
        path = _model_path(backend)
        rows.append({"Module": "Trained classifier", "Used by the app": path.name,
                     "Status": "✅ Found" if path.is_file() else
                     "⚠️ Not trained yet (run: python job_classifier.py --data <csv> --text-col <col> --label-col <col>)"})
    has_st = importlib.util.find_spec("sentence_transformers") is not None
    rows.append({"Module": "sentence-transformers", "Used by the app": "semantic embeddings",
                 "Status": "✅ Installed" if has_st else "❌ Not installed (pip install sentence-transformers)"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cached heavy objects (classifier, embedding model, job index)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained classifier…")
def cached_classifier(_backend: dict, path: str, mtime: float):
    return get_fn(_backend, "job_classifier", "load_classifier")(path)


def get_classifier(backend: dict):
    """Return the trained classifier or raise with a helpful message."""
    if backend["job_classifier"]["error"]:
        raise BackendUnavailable(f"`job_classifier` is unavailable ({backend['job_classifier']['error']}).")
    path = _model_path(backend)
    if not path.is_file():
        raise FileNotFoundError(
            f"No trained classifier found at '{path}'. Train it once with: "
            "python job_classifier.py --data <resumes.csv> --text-col <text column> --label-col <category column>"
        )
    return cached_classifier(backend, str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner="Preparing the semantic model…")
def cached_matcher(_backend: dict):
    return get_fn(_backend, "semantic_matching", "SemanticMatcher")()


@st.cache_resource(show_spinner="Indexing job descriptions (first run only — every job is embedded once)…")
def cached_recommender(_backend: dict, _jobs_df: pd.DataFrame, jobs_key: str, model_mtime: float):
    classifier = None
    try:
        classifier = get_classifier(_backend)
    except Exception:
        pass  # recommender works without a classifier (category weight is redistributed)
    jobs = _jobs_df
    if classifier is not None and "category" in jobs.columns:
        known = {str(c).lower() for c in classifier.classes_}
        if not jobs["category"].astype(str).str.lower().isin(known).any():
            jobs = jobs.drop(columns=["category"])  # labels don't match -> let the classifier predict them
    return get_fn(_backend, "job_recommendation", "JobRecommender")(
        jobs, classifier=classifier, matcher=cached_matcher(_backend), job_id_col="job_id")


def _jobs_key(jobs_df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(jobs_df[["job_id", "title"]].astype(str), index=False).sum()
    return f"{len(jobs_df)}:{int(hashed)}"


# --------------------------------------------------------------------------- #
# Conversion / validation of backend outputs (guards the dashboard against wrong shapes)
# --------------------------------------------------------------------------- #
def _clean_str_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise TypeError(f"{label} must be a list of strings, got {type(value).__name__}.")
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            out.append(text)
    return out


def _number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a number, got {value!r}.") from exc
    if number != number:
        raise ValueError(f"{label} is NaN.")
    return number


def _unit(value: Any, label: str) -> float:
    return min(max(_number(value, label), 0.0), 1.0)


def _percent(value: Any, label: str) -> float:
    return min(max(_number(value, label), 0.0), 100.0)


def _counts(raw: Any, label: str) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise TypeError(f"{label} must return a dict of skill -> count.")
    return {str(k): int(v) for k, v in raw.items()}


def _validate_categories(raw: Any) -> list[dict[str, Any]]:
    """Backend format: [{"category": str, "probability": 0-1}] -> app format with 'confidence'."""
    if not isinstance(raw, (list, tuple)):
        raise TypeError("The classifier must return a list of dicts.")
    out = []
    for item in raw:
        if not isinstance(item, dict) or "category" not in item or "probability" not in item:
            raise TypeError("Each category must be a dict with 'category' and 'probability'.")
        out.append({"category": str(item["category"]), "confidence": _unit(item["probability"], "probability")})
    return sorted(out, key=lambda d: d["confidence"], reverse=True)


def _validate_gap(raw: Any) -> dict[str, Any]:
    """Convert skill_gap.analyze_skill_gap() output into the app's gap structure."""
    if not isinstance(raw, dict):
        raise TypeError("analyze_skill_gap() must return a dict.")
    matched = _clean_str_list(raw.get("matched_skills"), "matched_skills")
    extra = _clean_str_list(raw.get("extra_skills"), "extra_skills")
    details: list[dict[str, Any]] = []
    names: list[str] = []
    for item in raw.get("missing_skills") or []:
        if isinstance(item, dict) and item.get("skill"):
            try:
                mentions = int(item.get("mentions", 1))
            except (TypeError, ValueError):
                mentions = 1
            details.append({"skill": str(item["skill"]), "category": str(item.get("category", "")),
                            "mentions": mentions, "priority": str(item.get("priority", ""))})
            names.append(str(item["skill"]))
        elif isinstance(item, str) and item.strip():
            names.append(item.strip())
    coverage = raw.get("coverage")  # None when the job text contains no recognised skills
    return {"matched": matched, "missing": names, "extra": extra,
            "coverage": None if coverage is None else _unit(coverage, "coverage"),
            "missing_details": details, "summary": str(raw.get("summary") or "")}


def _validate_score(raw: Any) -> dict[str, Any]:
    """Convert scoring.compute_overall_score() output into the app's score structure."""
    if not isinstance(raw, dict) or raw.get("overall_score") is None:
        raise TypeError("compute_overall_score() must return a dict containing 'overall_score'.")
    breakdown = raw.get("score_breakdown") or {}
    components = {
        str(name): _percent(float(comp["score"]) * 100, f"components[{name}]")
        for name, comp in breakdown.items()
        if isinstance(comp, dict) and comp.get("available") and comp.get("score") is not None
    }
    return {"overall_score": _percent(raw["overall_score"], "overall_score"),
            "label": str(raw.get("match_label") or ""),
            "components": components, "details": breakdown,
            "notes": [str(n) for n in raw.get("notes", [])]}


def _to_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [p.strip() for p in value.replace("|", ",").split(",") if p.strip()]
    return []


def _recs_to_frame(results: Any) -> pd.DataFrame:
    """Convert JobRecommender.recommend() output (list of dicts) into the table the dashboard uses."""
    if not isinstance(results, list):
        raise TypeError("JobRecommender.recommend() must return a list of dicts.")
    rows = []
    for r in results:
        breakdown = r.get("score_breakdown", {})

        def pct(name: str):
            score = breakdown.get(name, {}).get("score")
            return None if score is None else float(score) * 100

        gap = r.get("skill_gap", {})
        rows.append({
            "job_id": r.get("job_id"), "title": r.get("title"), "company": r.get("company"),
            "location": r.get("location"), "score": r.get("overall_score"), "match_label": r.get("match_label"),
            "semantic_score": pct("semantic_similarity"), "skill_score": pct("skill_match"),
            "matched_skills": gap.get("matched_skills", []),
            "missing_skills": [m["skill"] for m in gap.get("missing_skills", []) if isinstance(m, dict)],
        })
    return pd.DataFrame(rows, columns=["job_id", "title", "company", "location", "score", "match_label",
                                       "semantic_score", "skill_score", "matched_skills", "missing_skills"])


def _validate_recommendations(raw: Any, jobs_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Recommendations must be a pandas DataFrame.")
    missing = [c for c in ("title", "score") if c not in raw.columns]
    if missing:
        raise ValueError(f"Recommendations are missing required column(s): {', '.join(missing)}.")
    df = raw.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce").clip(0, 100)
    for col in ("semantic_score", "skill_score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)
    df = df.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    # Fill company / location / description from the dataset when the backend does not return them.
    if "job_id" in df.columns and "job_id" in jobs_df.columns:
        lookup = jobs_df.assign(job_id=jobs_df["job_id"].astype(str)).drop_duplicates("job_id").set_index("job_id")
        for col in ("company", "location", "description"):
            if col in lookup.columns:
                filled = df["job_id"].astype(str).map(lookup[col])
                df[col] = filled if col not in df.columns else df[col].where(df[col].notna(), filled)
    for col in ("matched_skills", "missing_skills"):
        if col in df.columns:
            df[col] = df[col].apply(_to_list)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# --------------------------------------------------------------------------- #
# Analysis pipeline
# --------------------------------------------------------------------------- #
def run_analysis(
    backend: dict,
    *,
    resume_bytes: bytes | None,
    resume_text: str | None,
    resume_name: str,
    job_title: str,
    job_company: str,
    job_category: str,
    job_text: str,
    jobs_df: pd.DataFrame | None,
    top_n: int,
    status: Any,
) -> dict[str, Any]:
    """Run every backend step. Fatal problems raise AnalysisError; the rest are collected in ``errors``."""
    started = time.perf_counter()
    errors: dict[str, str] = {}

    def call(module: str, func: str, *args: Any, **kwargs: Any) -> Any:
        return get_fn(backend, module, func)(*args, **kwargs)

    def soft(key: str, producer, default: Any = None) -> Any:
        try:
            return producer()
        except Exception as exc:
            errors[key] = f"{type(exc).__name__}: {exc}"
            return default

    # 1. Text extraction ------------------------------------------------------
    if resume_bytes is not None:
        status.write("📄 Extracting text from the PDF…")
        try:
            resume_text = call("pdf_extractor", "extract_text_from_pdf", resume_bytes)
        except Exception as exc:
            raise AnalysisError(f"PDF text extraction failed — {exc}") from exc
    if not isinstance(resume_text, str) or len(resume_text.strip()) < dl.MIN_RESUME_CHARS:
        raise AnalysisError(
            "Too little text could be read from the resume. Scanned/image-only PDFs are not supported — "
            "please upload a text-based PDF."
        )

    # 2. Cleaning -------------------------------------------------------------
    status.write("🧹 Cleaning text…")
    try:
        resume_clean = call("text_preprocessing", "clean_text", resume_text)
        job_clean = call("text_preprocessing", "clean_text", job_text)
    except Exception as exc:
        raise AnalysisError(f"Text preprocessing failed — {type(exc).__name__}: {exc}") from exc
    if not (isinstance(resume_clean, str) and isinstance(job_clean, str) and resume_clean.strip() and job_clean.strip()):
        raise AnalysisError("clean_text() returned empty text for the resume or the job description.")

    # 3. Skills (rule-based dictionary matching) -------------------------------
    status.write("🧠 Extracting skills…")
    resume_counts = soft("resume_skills", lambda: _counts(
        call("skill_extractor", "extract_skill_counts", resume_clean), "extract_skill_counts(resume)"), {})
    job_counts = soft("job_skills", lambda: _counts(
        call("skill_extractor", "extract_skill_counts", job_clean), "extract_skill_counts(job)"), {})
    resume_info = soft("resume_info", lambda: {
        str(cat): list(skills) for cat, skills in
        call("skill_extractor", "extract_skills_by_category", resume_clean).items()}, None)

    # 4. Skill gap ------------------------------------------------------------
    status.write("🧩 Analysing skill gap…")
    gap = gap_raw = None
    if "resume_skills" in errors or "job_skills" in errors:
        errors["skill_gap"] = "Skipped because skill extraction failed."
    else:
        gap_raw = soft("skill_gap", lambda: call("skill_gap", "analyze_skill_gap",
                                                  list(resume_counts), list(job_counts), job_counts))
        gap = soft("skill_gap", lambda: _validate_gap(gap_raw)) if gap_raw is not None else None

    # 5. Classifier: resume categories + category alignment --------------------
    status.write("🏷️ Predicting job categories…")
    classifier, categories, cat_info = None, [], None
    try:
        classifier = get_classifier(backend)
    except Exception as exc:
        errors["categories"] = f"{type(exc).__name__}: {exc}"
    if classifier is not None:
        categories = soft("categories", lambda: _validate_categories(classifier.predict(resume_clean, top_k=5)), [])

        def alignment():
            probs = classifier.predict_proba_dict(resume_clean)
            known = {c.lower() for c in probs}
            target = job_category if job_category and job_category.strip().lower() in known else None
            if target is None:  # same approach as scoring.evaluate_match(): classify the job text
                target = classifier.predict(job_clean, top_k=1)[0]["category"]
            return call("scoring", "category_alignment", probs, target)

        cat_info = soft("category_alignment", alignment)

    # 6. Semantic similarity (Sentence-Transformer; first run downloads the model) --
    status.write("🔗 Computing semantic similarity (the first run downloads the embedding model)…")
    cosine = soft("similarity", lambda: _number(cached_matcher(backend).similarity(resume_clean, job_clean), "cosine"))
    similarity = None if cosine is None else max(0.0, min(1.0, cosine))

    # 7. Final score ----------------------------------------------------------
    status.write("🎯 Calculating match score…")
    score = None
    if cosine is None and gap_raw is None:
        errors["score"] = "Skipped because neither the semantic similarity nor the skill gap is available."
    else:
        score = soft("score", lambda: _validate_score(
            call("scoring", "compute_overall_score", cosine, gap_raw, cat_info)))

    # 8. Recommendations ------------------------------------------------------
    recommendations = None
    if jobs_df is None or jobs_df.empty:
        errors["recommendations"] = "No job dataset is loaded (add data/jobs.csv or upload a CSV in the sidebar)."
    else:
        status.write(f"💼 Ranking {len(jobs_df):,} jobs (the first run embeds all jobs and can take a while)…")

        def recommend():
            recommender = cached_recommender(backend, jobs_df, _jobs_key(jobs_df), _model_mtime(backend))
            return _validate_recommendations(_recs_to_frame(recommender.recommend(resume_clean, top_n=top_n)),
                                             jobs_df, top_n)
        recommendations = soft("recommendations", recommend)

    return {
        "meta": {"resume_name": resume_name, "job_title": job_title, "job_company": job_company,
                 "elapsed": time.perf_counter() - started},
        "resume": {"text": resume_text, "clean": resume_clean, "skills": list(resume_counts),
                   "skill_counts": resume_counts, "categories": categories, "info": resume_info},
        "job": {"title": job_title, "company": job_company, "text": job_text, "clean": job_clean,
                "skills": list(job_counts), "skill_counts": job_counts},
        "similarity": similarity,
        "cosine": cosine,
        "gap": gap,
        "score": score,
        "recommendations": recommendations,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Cached dataset loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading job dataset…")
def cached_jobs(path: str, mtime: float) -> pd.DataFrame:
    return dl.load_jobs_dataset(path)


@st.cache_data(show_spinner="Loading uploaded job dataset…")
def cached_jobs_upload(data: bytes) -> pd.DataFrame:
    return dl.load_jobs_dataset(io.BytesIO(data))


@st.cache_data(show_spinner="Loading resume dataset…")
def cached_resumes(path: str, mtime: float) -> pd.DataFrame:
    return dl.load_resume_dataset(path)


def resolve_jobs_dataset(uploaded_csv: Any) -> tuple[pd.DataFrame | None, str, str | None]:
    """Priority: uploaded CSV > data/jobs.csv > data/sample_jobs.csv. Returns (df, source_label, error)."""
    if uploaded_csv is not None:
        try:
            return cached_jobs_upload(uploaded_csv.getvalue()), f"uploaded file ({uploaded_csv.name})", None
        except dl.DatasetError as exc:
            return None, "", str(exc)
    for path in (dl.DEFAULT_JOBS_PATH, dl.SAMPLE_JOBS_PATH):
        if path.exists():
            try:
                return cached_jobs(str(path), path.stat().st_mtime), path.relative_to(dl.PROJECT_ROOT).as_posix(), None
            except dl.DatasetError as exc:
                return None, "", f"{path.name}: {exc}"
    return None, "", "No job dataset found. Add `data/jobs.csv` or upload a CSV in the sidebar."


def resolve_resume_dataset() -> tuple[pd.DataFrame | None, str | None]:
    path = dl.DEFAULT_RESUMES_PATH
    if not path.exists():
        return None, None
    try:
        return cached_resumes(str(path), path.stat().st_mtime), None
    except dl.DatasetError as exc:
        return None, f"{path.name}: {exc}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(backend: dict) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "resume_bytes": None, "resume_text": None, "resume_name": "", "resume_ready": False,
        "job_title": "", "job_company": "", "job_category": "", "job_text": "", "job_ready": False,
        "jobs_df": None, "top_n": 10, "analyze": False,
    }
    with st.sidebar:
        st.markdown("## 🎯 Resume–Job Matcher")

        # ---- Data & settings -------------------------------------------------
        with st.expander("⚙️ Data & settings"):
            uploaded_csv = st.file_uploader("Job dataset (CSV, optional)", type=["csv"], key="jobs_csv",
                                            help="Needs a title column and a description column.")
            top_n = st.slider("Recommendations to show", 3, 25, 10)
            max_jobs = st.number_input("Max jobs to scan", min_value=10, max_value=50000, value=2000, step=100,
                                       help="Limits how many dataset rows are embedded and ranked.")
            inputs["top_n"] = top_n

        jobs_df, jobs_source, jobs_error = resolve_jobs_dataset(uploaded_csv)
        resumes_df, resumes_error = resolve_resume_dataset()
        if jobs_df is not None:
            inputs["jobs_df"] = jobs_df.head(int(max_jobs)).reset_index(drop=True)
            st.caption(f"📚 {len(jobs_df):,} jobs loaded from {jobs_source}")
            for warning in jobs_df.attrs.get("warnings", []):
                st.caption(f"⚠️ {warning}")
        elif jobs_error:
            st.error(jobs_error)
        if resumes_error:
            st.error(resumes_error)

        # ---- 1. Resume ---------------------------------------------------------
        st.markdown("### 1 · Resume")
        modes = ["Upload PDF"] + (["Pick from resume dataset"] if resumes_df is not None else [])
        resume_mode = st.radio("Resume source", modes, label_visibility="collapsed") if len(modes) > 1 else modes[0]
        if resume_mode == "Upload PDF":
            pdf = st.file_uploader("Upload resume (PDF)", type=["pdf"], key="resume_pdf",
                                   help=f"Text-based PDF, up to {dl.MAX_PDF_SIZE_MB} MB.")
            if pdf is not None:
                ok, message = dl.validate_pdf_upload(pdf)
                if ok:
                    st.success(message)
                    inputs.update(resume_bytes=dl.read_upload_bytes(pdf), resume_name=pdf.name, resume_ready=True)
                else:
                    st.error(message)
        else:
            idx = st.selectbox("Resume", list(resumes_df.index), format_func=lambda i: resumes_df.loc[i, "label"])
            inputs.update(resume_text=resumes_df.loc[idx, "resume_text"], resume_name=resumes_df.loc[idx, "label"],
                          resume_ready=True)

        # ---- 2. Job description -------------------------------------------------
        st.markdown("### 2 · Job description")
        job_modes = ["Enter manually"] + (["Select from dataset"] if jobs_df is not None else [])
        job_mode = st.radio("Job source", job_modes, label_visibility="collapsed") if len(job_modes) > 1 else job_modes[0]
        if job_mode == "Enter manually":
            inputs["job_title"] = st.text_input("Job title", placeholder="e.g. Data Analyst").strip() or "Custom job description"
            inputs["job_company"] = st.text_input("Company (optional)").strip()
            inputs["job_text"] = st.text_area("Job description", height=220,
                                              placeholder="Paste the full job description here…")
        else:
            idx = st.selectbox("Job posting", list(jobs_df.index), format_func=lambda i: jobs_df.loc[i, "label"])
            row = jobs_df.loc[idx]
            category = "" if row["category"] == "Unspecified" else str(row["category"])
            inputs.update(job_title=row["title"], job_company=row["company"], job_category=category,
                          job_text=row["description"])
            with st.expander("Preview description"):
                st.text(row["description"])
        ok, message = dl.validate_job_description(inputs["job_text"])
        inputs["job_ready"] = ok
        if inputs["job_text"].strip() and not ok:
            st.warning(message)

        # ---- Run ------------------------------------------------------------------
        inputs["analyze"] = st.button("🚀 Analyze resume", type="primary",
                                      disabled=not (inputs["resume_ready"] and inputs["job_ready"]))
        if not (inputs["resume_ready"] and inputs["job_ready"]):
            st.caption("Provide a resume and a job description to enable analysis.")

        # ---- Backend status -----------------------------------------------------
        with st.expander("🔌 Backend status"):
            st.dataframe(backend_status_frame(backend)[["Module", "Status"]], hide_index=True)
            if st.button("🔄 Reload backend modules"):
                for module_name in BACKEND_SPEC:
                    sys.modules.pop(module_name, None)
                st.cache_resource.clear()
                st.rerun()
    return inputs


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
def render_welcome(backend: dict) -> None:
    st.info("👈 Upload a resume, choose or paste a job description, then click **Analyze resume**.")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("#### 1 · Resume\nUpload a text-based PDF (max 5 MB) or pick a resume from the dataset.")
    with c2, st.container(border=True):
        st.markdown("#### 2 · Job\nPaste a job description or select one from the job dataset.")
    with c3, st.container(border=True):
        st.markdown("#### 3 · Insights\nMatch score, skill gaps, predicted categories and ranked job recommendations.")
    frame = backend_status_frame(backend)
    all_ready = frame["Status"].str.startswith("✅").all()
    if not all_ready:
        st.warning("Some backend parts are not ready yet (see the table below). Missing parts only disable "
                   "the steps that depend on them.")
    with st.expander("Backend status", expanded=not all_ready):
        st.dataframe(frame, hide_index=True)


def run_and_store(backend: dict, inputs: dict[str, Any]) -> None:
    with st.status("Analyzing resume…", expanded=True) as status:
        try:
            result = run_analysis(
                backend,
                resume_bytes=inputs["resume_bytes"], resume_text=inputs["resume_text"],
                resume_name=inputs["resume_name"], job_title=inputs["job_title"],
                job_company=inputs["job_company"], job_category=inputs["job_category"],
                job_text=inputs["job_text"], jobs_df=inputs["jobs_df"], top_n=inputs["top_n"], status=status,
            )
            st.session_state["result"] = result
            status.update(label="Analysis complete", state="complete", expanded=False)
        except AnalysisError as exc:
            st.session_state.pop("result", None)
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:  # unexpected bug: show it instead of a blank page
            st.session_state.pop("result", None)
            status.update(label="Unexpected error", state="error", expanded=True)
            st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
            st.exception(exc)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🎯", layout="wide")
    dash.inject_css()
    backend = load_backend()
    if any((e["error"] or "").endswith(".py not found") for e in backend.values()):
        load_backend.clear()  # don't cache "file not found": retry on the next interaction
    inputs = render_sidebar(backend)

    st.title("🎯 AI Resume–Job Matching & Skill Gap Analysis")
    st.caption("NLP + machine learning powered resume analysis · runs fully locally")

    if inputs["analyze"]:
        run_and_store(backend, inputs)

    result = st.session_state.get("result")
    if result is None:
        render_welcome(backend)
        return

    shown = {k: v for k, v in result["errors"].items() if k != "recommendations" or inputs["jobs_df"] is not None}
    if shown:
        with st.expander(f"⚠️ {len(shown)} step(s) could not be completed", expanded=False):
            for key, message in shown.items():
                st.markdown(f"- **{key.replace('_', ' ')}** — {message}")

    tab_dash, tab_resume, tab_match, tab_recs = st.tabs(
        ["📊 Dashboard", "📄 Resume Analysis", "🎯 Job Match", "💼 Job Recommendations"]
    )
    with tab_dash:
        dash.render_dashboard(result)
    with tab_resume:
        dash.render_resume_analysis(result)
    with tab_match:
        dash.render_job_match(result)
    with tab_recs:
        dash.render_recommendations(result)

    st.divider()
    st.caption("Resumes are processed in memory for the current session only and are not stored by this app.")


main()
