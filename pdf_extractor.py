"""
pdf_extractor.py
================
Extract plain text from PDF resumes / job descriptions using PyMuPDF.

Public API
----------
extract_text_from_pdf(source)  -> str
extract_pdf_details(source)    -> dict   (text + page count + warnings)
extract_text_from_file(source, filename=None) -> str   (PDF or TXT)

`source` may be:
  * a file path (str or pathlib.Path)
  * raw bytes
  * a file-like object, e.g. the object returned by Streamlit's st.file_uploader
    (anything with .getvalue() or .read())

All failures raise `PDFExtractionError` with a user-friendly message, so the
Streamlit app can simply do:  try: ... except PDFExtractionError as e: st.error(str(e))
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # PyMuPDF >= 1.24.3 exposes 'pymupdf'; older versions only expose 'fitz'
    import pymupdf as fitz
except ImportError:
    import fitz  # install with: pip install PyMuPDF   (NOT 'pip install fitz')

logger = logging.getLogger(__name__)

# If fewer characters than this are extracted, the PDF is probably a scanned image.
MIN_TEXT_CHARS = 50


class PDFExtractionError(Exception):
    """Raised when a document cannot be opened or contains no extractable text."""


def _read_bytes(source: Any) -> bytes:
    """Read the content of a path, bytes object or file-like object as bytes."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise PDFExtractionError(f"File not found: {path}")
        return path.read_bytes()

    if isinstance(source, (bytes, bytearray)):
        return bytes(source)

    # Streamlit UploadedFile and BytesIO both provide getvalue(); plain file objects provide read().
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)  # make sure we read from the beginning
        return source.read()

    raise PDFExtractionError(
        "Unsupported input type. Provide a file path, bytes, or an uploaded file object."
    )


def extract_pdf_details(source: Any, min_chars: int = MIN_TEXT_CHARS) -> Dict[str, Any]:
    """
    Extract text and basic metadata from a PDF.

    Returns
    -------
    {
        "text": str,            # full text, pages joined with newlines
        "num_pages": int,
        "char_count": int,
        "warnings": list[str],  # e.g. pages that could not be read
    }
    """
    data = _read_bytes(source)
    if not data:
        raise PDFExtractionError("The file is empty.")

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several different exception types
        raise PDFExtractionError(
            f"Could not open the file as a PDF (it may be corrupted or not a PDF): {exc}"
        ) from exc

    warnings: List[str] = []
    pages: List[str] = []
    try:
        if doc.needs_pass:
            raise PDFExtractionError("The PDF is password-protected and cannot be read.")

        num_pages = doc.page_count
        for page_number, page in enumerate(doc, start=1):
            try:
                # sort=True orders text blocks top-to-bottom, left-to-right,
                # which gives a more natural reading order for multi-column resumes.
                pages.append(page.get_text("text", sort=True))
            except Exception as exc:  # keep going if a single page fails
                warnings.append(f"Page {page_number} could not be read: {exc}")
                pages.append("")
    finally:
        doc.close()

    text = "\n".join(pages).strip()
    if len(text) < min_chars:
        raise PDFExtractionError(
            "No readable text was found. The PDF may be a scanned image; "
            "this project does not include OCR. Please upload a text-based PDF."
        )

    return {
        "text": text,
        "num_pages": num_pages,
        "char_count": len(text),
        "warnings": warnings,
    }


def extract_text_from_pdf(source: Any) -> str:
    """Return the extracted text of a PDF (see `extract_pdf_details` for metadata)."""
    return extract_pdf_details(source)["text"]


def extract_text_from_file(source: Any, filename: Optional[str] = None) -> str:
    """
    Convenience wrapper that accepts PDF or plain-text (.txt) input.

    The type is decided by `filename`, or by source.name / the path suffix.
    Anything that is not .txt is treated as a PDF.
    """
    if filename is None:
        if isinstance(source, (str, Path)):
            filename = str(source)
        else:
            filename = getattr(source, "name", "") or ""

    if Path(filename).suffix.lower() == ".txt":
        raw = _read_bytes(source)
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            raise PDFExtractionError("The text file is empty.")
        return text

    return extract_text_from_pdf(source)
