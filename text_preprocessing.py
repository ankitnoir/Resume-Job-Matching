"""
text_preprocessing.py
=====================
Clean and preprocess resume / job-description text.

Two levels of processing are provided because different components need different input:

* clean_text(text)       -> LIGHT cleaning. Keeps wording, case and punctuation such as
                            "C++", "Node.js", "CI/CD". Used by the skill extractor and by
                            the Sentence-Transformer (which works best on natural text).
* preprocess_text(text)  -> HEAVY preprocessing (lowercase, tokenise, remove stop-words,
                            lemmatise). Used by the TF-IDF based job classifier.

Public API
----------
clean_text(text)                          -> str
preprocess_text(text, remove_stopwords=True, lemmatize=True) -> str
preprocess_batch(texts, remove_stopwords=True, lemmatize=True) -> list[str]
get_nlp()                                 -> spaCy pipeline or None

spaCy model `en_core_web_sm` is used for lemmatisation. If spaCy or the model is not
installed, the code logs a warning and falls back to simple tokenisation without
lemmatisation, so the pipeline keeps working.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Regular expressions used for light cleaning
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Candidate phone numbers; digits are counted afterwards so date ranges such as
# "2018 - 2022" are NOT removed.
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BULLET_RE = re.compile(r"[•●▪■◦◆►▶✓✔★☆➢➤]")

# Technology names whose punctuation would be lost by a plain word tokeniser.
# They are replaced by plain-letter tokens ONLY in the heavy preprocessing path.
_TECH_TOKEN_MAP = {
    "asp.net": "aspnet",
    "c++": "cplusplus",
    "c#": "csharp",
    ".net": "dotnet",
    "node.js": "nodejs",
    "react.js": "reactjs",
    "vue.js": "vuejs",
    "next.js": "nextjs",
    "express.js": "expressjs",
}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")  # tokens of 2+ characters starting with a letter

# --------------------------------------------------------------------------- #
# Lazy spaCy loading
# --------------------------------------------------------------------------- #
_NLP = None
_NLP_LOAD_ATTEMPTED = False


def get_nlp():
    """Load spaCy `en_core_web_sm` once (parser and NER disabled for speed). Returns None on failure."""
    global _NLP, _NLP_LOAD_ATTEMPTED
    if _NLP_LOAD_ATTEMPTED:
        return _NLP
    _NLP_LOAD_ATTEMPTED = True
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        _NLP.max_length = max(_NLP.max_length, 2_000_000)
    except (ImportError, OSError) as exc:
        logger.warning(
            "spaCy model 'en_core_web_sm' unavailable (%s). Falling back to simple "
            "tokenisation without lemmatisation. Install with: "
            "pip install spacy && python -m spacy download en_core_web_sm",
            exc,
        )
        _NLP = None
    return _NLP


@lru_cache(maxsize=1)
def _get_stopwords() -> frozenset:
    """Stop-word list from spaCy if available, otherwise from scikit-learn (built once)."""
    nlp = get_nlp()
    if nlp is not None:
        return frozenset(nlp.Defaults.stop_words)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    return frozenset(ENGLISH_STOP_WORDS)


# --------------------------------------------------------------------------- #
# Light cleaning
# --------------------------------------------------------------------------- #
def _strip_phone(match: re.Match) -> str:
    digits = sum(ch.isdigit() for ch in match.group(0))
    return " " if 10 <= digits <= 13 else match.group(0)


def clean_text(text: Optional[str]) -> str:
    """
    Light cleaning that keeps the wording of the document.

    Steps: Unicode normalisation, fix words broken by line-end hyphenation, remove control
    characters, URLs, e-mail addresses and phone numbers (not useful for matching and
    privacy-sensitive), remove bullet symbols, and normalise whitespace.
    Paragraph breaks (newlines) are preserved.
    """
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN from pandas
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)  # also expands ligatures such as "ﬁ"
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # "develop-\nment" -> "development"
    text = _CONTROL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_CANDIDATE_RE.sub(_strip_phone, text)
    text = _BULLET_RE.sub(" ", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Heavy preprocessing (for the ML classifier)
# --------------------------------------------------------------------------- #
def _basic_tokens(text: Optional[str]) -> List[str]:
    """Lower-case the cleaned text, protect tech names, and split into word tokens."""
    lowered = clean_text(text).lower()
    for original, replacement in _TECH_TOKEN_MAP.items():
        lowered = lowered.replace(original, replacement)
    return _TOKEN_RE.findall(lowered)


def preprocess_batch(
    texts: Iterable[Optional[str]],
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    batch_size: int = 16,
) -> List[str]:
    """
    Preprocess many documents efficiently (uses spaCy's nlp.pipe for lemmatisation).

    Note: lemmatisation maps e.g. "pandas" -> "panda". That is harmless for the classifier
    because the SAME transformation is applied at training and prediction time. It is NOT
    used by the skill extractor, which works on `clean_text` output.
    """
    token_lists = [_basic_tokens(t) for t in texts]
    stop = _get_stopwords() if remove_stopwords else set()
    nlp = get_nlp() if lemmatize else None

    # Fallback: no lemmatisation
    if nlp is None:
        return [" ".join(tok for tok in toks if tok not in stop) for toks in token_lists]

    joined = [" ".join(toks) for toks in token_lists]
    processed: List[str] = []
    # Tagging the full token sequence first gives better lemmas; stop-words are removed afterwards.
    for doc in nlp.pipe(joined, batch_size=batch_size):
        kept = []
        for token in doc:
            lemma = token.lemma_.lower().strip()
            if len(lemma) < 2:
                continue
            if token.text in stop or lemma in stop:
                continue
            kept.append(lemma)
        processed.append(" ".join(kept))
    return processed


def preprocess_text(
    text: Optional[str], remove_stopwords: bool = True, lemmatize: bool = True
) -> str:
    """Preprocess a single document (see `preprocess_batch`)."""
    return preprocess_batch([text], remove_stopwords=remove_stopwords, lemmatize=lemmatize)[0]
