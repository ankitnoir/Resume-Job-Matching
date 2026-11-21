"""
semantic_matching.py
====================
Semantic similarity between resumes and job descriptions using a PRETRAINED
Sentence-Transformer model and cosine similarity. No training is needed here and the
result is a SIMILARITY score, not a classification and not a skill match.

Default model: sentence-transformers/all-MiniLM-L6-v2 (free, small, runs on CPU).
It is downloaded from the Hugging Face hub on first use (internet needed once) and then cached.

Important detail - long documents
---------------------------------
MiniLM only reads about 256 word-pieces, so a long resume would be silently truncated.
This module therefore splits each document into overlapping word chunks, embeds every chunk,
averages the chunk embeddings and re-normalises the result, so the WHOLE document contributes.

Public API
----------
SemanticMatcher(model_name=..., chunk_words=180, chunk_overlap=40)
    .encode_documents(texts)                      -> np.ndarray (n_docs, dim), unit vectors
    .similarity(text_a, text_b)                   -> float cosine in [-1, 1]
    .similarity_to_embeddings(text, embeddings)   -> np.ndarray (n_docs,)
    .rank_texts(query_text, texts, top_n=None)    -> [{"index": int, "score": float}, ...]
get_default_matcher()                             -> shared SemanticMatcher
compute_semantic_similarity(resume_text, job_text, matcher=None) -> float
cosine_similarity_matrix(a, b)                    -> np.ndarray
split_into_chunks(text, chunk_words, overlap)     -> list[str]
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from text_preprocessing import clean_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loaded models are cached so several SemanticMatcher objects share one model in memory.
_MODEL_CACHE: Dict[Tuple[str, Optional[str]], Any] = {}


def chunk_overlap_invalid(chunk_words: int, overlap: int) -> bool:
    return chunk_words <= 0 or overlap < 0 or overlap >= chunk_words


def split_into_chunks(text: str, chunk_words: int = 180, overlap: int = 40) -> List[str]:
    """Split text into overlapping chunks of `chunk_words` words. Returns [] for empty text."""
    if chunk_overlap_invalid(chunk_words, overlap):
        raise ValueError("overlap must be smaller than chunk_words")
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = chunk_words - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):  # last chunk reached the end
            break
    return chunks


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between every row of `a` and every row of `b` -> shape (len(a), len(b)).
    Works for non-normalised vectors; all-zero vectors give similarity 0.
    """
    a = np.atleast_2d(np.asarray(a, dtype=np.float32))
    b = np.atleast_2d(np.asarray(b, dtype=np.float32))
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


class SemanticMatcher:
    """Wraps a Sentence-Transformer model for document-level embeddings and cosine similarity."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, chunk_words: int = 180,
                 chunk_overlap: int = 40, device: Optional[str] = None) -> None:
        if chunk_overlap_invalid(chunk_words, chunk_overlap):
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_words")
        self.model_name = model_name
        self.chunk_words = chunk_words
        self.chunk_overlap = chunk_overlap
        self.device = device

    def _get_model(self):
        """Load the model lazily (so importing this module stays fast)."""
        key = (self.model_name, self.device)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. Run: pip install sentence-transformers"
            ) from exc
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the Sentence-Transformer model '{self.model_name}'. "
                f"The first run needs an internet connection to download it. Details: {exc}"
            ) from exc
        _MODEL_CACHE[key] = model
        return model

    def encode_documents(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed whole documents (chunked + mean-pooled). Returns unit-length vectors of shape
        (len(texts), dim). Empty documents get an all-zero vector (cosine similarity 0).
        """
        model = self._get_model()
        # method name differs between sentence-transformers versions
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dim = get_dim()
        out = np.zeros((len(texts), dim), dtype=np.float32)

        chunks: List[str] = []
        owners: List[int] = []
        for doc_index, text in enumerate(texts):
            for chunk in split_into_chunks(clean_text(text), self.chunk_words, self.chunk_overlap):
                chunks.append(chunk)
                owners.append(doc_index)
        if not chunks:
            return out

        embeddings = model.encode(
            chunks, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        np.add.at(out, np.array(owners), embeddings)       # sum chunk vectors per document
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)     # re-normalise to unit length
        return out

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity (in [-1, 1]) between two documents."""
        if not clean_text(text_a) or not clean_text(text_b):
            raise ValueError("Both texts must be non-empty to compute semantic similarity.")
        emb = self.encode_documents([text_a, text_b])
        return float(cosine_similarity_matrix(emb[0], emb[1])[0, 0])

    def similarity_to_embeddings(self, text: str, embeddings: np.ndarray) -> np.ndarray:
        """Cosine similarity between one text and pre-computed document embeddings."""
        if not clean_text(text):
            raise ValueError("Text is empty.")
        query = self.encode_documents([text])
        return cosine_similarity_matrix(query, embeddings)[0]

    def rank_texts(self, query_text: str, texts: Sequence[str],
                   top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rank `texts` by semantic similarity to `query_text` (highest first)."""
        if not texts:
            return []
        scores = self.similarity_to_embeddings(query_text, self.encode_documents(texts))
        order = np.argsort(-scores)
        if top_n is not None:
            order = order[:top_n]
        return [{"index": int(i), "score": float(scores[i])} for i in order]


_DEFAULT_MATCHER: Optional[SemanticMatcher] = None


def get_default_matcher() -> SemanticMatcher:
    """Shared matcher using the default model (created on first use)."""
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = SemanticMatcher()
    return _DEFAULT_MATCHER


def compute_semantic_similarity(resume_text: str, job_text: str,
                                matcher: Optional[SemanticMatcher] = None) -> float:
    """Convenience wrapper: cosine similarity between a resume and a job description."""
    return (matcher or get_default_matcher()).similarity(resume_text, job_text)
