"""
data_loader.py
==============
Loading and validation helpers for the Resume-Job Matching application.

Responsibilities
----------------
* Validate uploaded resume PDFs (extension, size, PDF header).
* Load and validate the job-description dataset (CSV).
* Load and validate the (optional) resume dataset (CSV).
* Normalise common column names (e.g. Kaggle's ``Job Title`` / ``Resume_str``)
  to the canonical names used by the whole project.

Nothing in this module performs NLP/ML work.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths and limits
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_JOBS_PATH = DATA_DIR / "jobs.csv"
SAMPLE_JOBS_PATH = DATA_DIR / "sample_jobs.csv"
DEFAULT_RESUMES_PATH = DATA_DIR / "resumes.csv"

MAX_PDF_SIZE_MB = 5
MIN_RESUME_CHARS = 200          # fewer extracted characters => probably a scanned PDF
MIN_JOB_DESCRIPTION_CHARS = 50

# Canonical schemas (the first columns of every DataFrame returned below)
JOB_COLUMNS = ["job_id", "title", "company", "location", "category", "description"]
RESUME_COLUMNS = ["resume_id", "category", "resume_text"]

# Accepted header names, in priority order (compared after lower-casing and replacing
# non-alphanumerics by "_"). If a file has several matching columns, the first alias wins.
_JOB_ALIASES: dict[str, tuple[str, ...]] = {
    "job_id": ("job_id", "jobid", "posting_id", "id"),
    "title": ("title", "job_title", "jobtitle", "job_position", "position", "role"),
    "company": ("company", "company_name", "employer", "organization"),
    "location": ("location", "job_location", "city"),
    "category": ("category", "job_category", "industry", "function"),
    "description": ("description", "job_description", "jobdescription", "job_desc", "details", "text"),
}
_RESUME_ALIASES: dict[str, tuple[str, ...]] = {
    "resume_id": ("resume_id", "resumeid", "id"),
    "category": ("category", "job_category", "label"),
    "resume_text": ("resume_text", "resume_str", "resume", "content", "cv", "text"),
}


class DatasetError(Exception):
    """Raised when a dataset is missing, unreadable or does not follow the expected schema."""


# --------------------------------------------------------------------------- #
# Resume upload validation
# --------------------------------------------------------------------------- #
def read_upload_bytes(uploaded_file: Any) -> bytes:
    """Return the raw bytes of a Streamlit ``UploadedFile`` (or any file-like object)."""
    if hasattr(uploaded_file, "getvalue"):
        return uploaded_file.getvalue()
    uploaded_file.seek(0)
    return uploaded_file.read()


def validate_pdf_upload(uploaded_file: Any, max_mb: float = MAX_PDF_SIZE_MB) -> tuple[bool, str]:
    """Check an uploaded resume. Returns ``(is_valid, message)``."""
    if uploaded_file is None:
        return False, "No file uploaded."
    name = getattr(uploaded_file, "name", "") or ""
    if not name.lower().endswith(".pdf"):
        return False, "Only PDF files (.pdf) are supported."
    data = read_upload_bytes(uploaded_file)
    size = len(data)
    if size == 0:
        return False, "The uploaded file is empty."
    if size > max_mb * 1024 * 1024:
        return False, f"File is {size / 1024 / 1024:.1f} MB; the limit is {max_mb:g} MB."
    if b"%PDF" not in data[:1024]:
        return False, "This does not look like a valid PDF (missing PDF header)."
    return True, f"{name} · {size / 1024:.0f} KB"


def validate_job_description(text: str) -> tuple[bool, str]:
    """Check a manually entered / selected job description."""
    text = (text or "").strip()
    if not text:
        return False, "Please enter a job description."
    if len(text) < MIN_JOB_DESCRIPTION_CHARS:
        return False, f"The job description is too short (minimum {MIN_JOB_DESCRIPTION_CHARS} characters)."
    return True, "OK"


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def _normalise_name(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _rename_columns(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    """Rename matching columns to their canonical names (one column per canonical name)."""
    normalised = {col: _normalise_name(col) for col in df.columns}
    rename: dict[Any, str] = {}
    for canonical, names in aliases.items():
        for alias in names:  # priority order
            match = next((col for col, norm in normalised.items() if norm == alias and col not in rename), None)
            if match is not None:
                rename[match] = canonical
                break
    return df.rename(columns=rename)


def _read_csv(source: Any) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, trying common encodings."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DatasetError(f"Dataset file not found: {path}")
        source = path
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DatasetError("The CSV file is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DatasetError(f"The CSV file could not be parsed: {exc}") from exc
    raise DatasetError("The CSV file could not be decoded (tried UTF-8 and Latin-1).")


def _text_column(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _fill_default(df: pd.DataFrame, column: str, default: str) -> None:
    if column not in df.columns:
        df[column] = default
    else:
        df[column] = _text_column(df[column]).replace("", default)


def _reorder(df: pd.DataFrame, canonical: list[str]) -> pd.DataFrame:
    front = canonical + ["label"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


# --------------------------------------------------------------------------- #
# Job-description dataset
# --------------------------------------------------------------------------- #
def load_jobs_dataset(source: Any) -> pd.DataFrame:
    """
    Load and validate a job-description dataset.

    ``source`` may be a path or a file-like object (e.g. a Streamlit upload).
    Required columns: a title and a description (aliases accepted).
    Returns a DataFrame with the canonical columns
    ``job_id, title, company, location, category, description, label``
    followed by any extra columns of the original file.
    ``df.attrs["warnings"]`` holds a list of human-readable data-quality notes.
    """
    raw = _read_csv(source)
    df = _rename_columns(raw, _JOB_ALIASES)

    missing = [c for c in ("title", "description") if c not in df.columns]
    if missing:
        raise DatasetError(
            f"Job dataset is missing required column(s): {', '.join(missing)}. "
            f"Columns found: {list(raw.columns)}. Accepted names include "
            "'title'/'job_title' and 'description'/'job_description'."
        )

    warnings: list[str] = []
    n_start = len(df)

    df["title"] = _text_column(df["title"])
    df["description"] = _text_column(df["description"])
    valid = (df["title"] != "") & (df["description"].str.len() >= MIN_JOB_DESCRIPTION_CHARS)
    if (~valid).sum():
        warnings.append(f"Dropped {int((~valid).sum())} row(s) with an empty title or a very short description.")
    df = df[valid]

    n_before = len(df)
    df = df.drop_duplicates(subset=["title", "description"])
    if n_before - len(df):
        warnings.append(f"Dropped {n_before - len(df)} duplicate job posting(s).")

    if df.empty:
        raise DatasetError(f"No valid job rows remain after validation ({n_start} row(s) read).")

    df = df.reset_index(drop=True)
    _fill_default(df, "company", "Not specified")
    _fill_default(df, "location", "Not specified")
    _fill_default(df, "category", "Unspecified")
    if "job_id" in df.columns:
        df["job_id"] = _text_column(df["job_id"]).replace("", pd.NA)
        df["job_id"] = df["job_id"].fillna(pd.Series(df.index.astype(str), index=df.index))
    else:
        df["job_id"] = df.index.astype(str)

    df["label"] = df["title"] + " — " + df["company"] + " (#" + df["job_id"].astype(str) + ")"
    df = _reorder(df, JOB_COLUMNS)
    df.attrs["warnings"] = warnings
    return df


# --------------------------------------------------------------------------- #
# Resume dataset (optional)
# --------------------------------------------------------------------------- #
def load_resume_dataset(source: Any) -> pd.DataFrame:
    """
    Load and validate a resume dataset (plain-text resumes, one per row).

    Required column: resume text (``resume_text`` / ``Resume_str`` / ``resume`` ...).
    Returns canonical columns ``resume_id, category, resume_text, label``.
    """
    raw = _read_csv(source)
    df = _rename_columns(raw, _RESUME_ALIASES)
    if "resume_text" not in df.columns:
        raise DatasetError(
            "Resume dataset needs a text column named 'resume_text' (or 'Resume_str', 'resume', 'text'). "
            f"Columns found: {list(raw.columns)}."
        )

    warnings: list[str] = []
    df["resume_text"] = _text_column(df["resume_text"])
    valid = df["resume_text"].str.len() >= MIN_RESUME_CHARS
    if (~valid).sum():
        warnings.append(f"Dropped {int((~valid).sum())} resume(s) with fewer than {MIN_RESUME_CHARS} characters.")
    df = df[valid].drop_duplicates(subset=["resume_text"])
    if df.empty:
        raise DatasetError("No valid resumes remain after validation.")

    df = df.reset_index(drop=True)
    _fill_default(df, "category", "Unspecified")
    if "resume_id" in df.columns:
        df["resume_id"] = _text_column(df["resume_id"]).replace("", pd.NA)
        df["resume_id"] = df["resume_id"].fillna(pd.Series(df.index.astype(str), index=df.index))
    else:
        df["resume_id"] = df.index.astype(str)

    df["label"] = df["category"] + " — resume #" + df["resume_id"].astype(str)
    df = _reorder(df, RESUME_COLUMNS)
    df.attrs["warnings"] = warnings
    return df


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def dataset_summary(df: pd.DataFrame, text_column: str = "description") -> dict[str, Any]:
    """Small summary dictionary used by the sidebar."""
    words = df[text_column].str.split().str.len() if text_column in df.columns else pd.Series(dtype=float)
    return {
        "rows": int(len(df)),
        "unique_titles": int(df["title"].nunique()) if "title" in df.columns else None,
        "avg_words": float(words.mean()) if len(words) else 0.0,
    }
