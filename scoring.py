"""
scoring.py
==========
Explainable overall resume-job matching score (0-100) built from three clearly separated
components. Each component comes from a different kind of evidence:

  1. semantic_similarity  - Sentence-Transformer cosine similarity   (pretrained model, no training)
  2. skill_match          - share of job skills found on the resume  (rule-based dictionary matching)
  3. category_alignment   - agreement between the job's category and the resume's predicted
                            category                                  (supervised ML classifier)

Overall score = sum( weight_i * component_score_i ) * 100, using weights that sum to 1.

Default weights (editable, see DEFAULT_WEIGHTS): semantic 0.40, skills 0.45, category 0.15.
These are design choices, not learned values. Tune them (e.g. on labelled resume-job pairs)
and report your choice.

If a component cannot be computed (no classifier loaded, or no skills found in the job text),
its weight is redistributed proportionally over the remaining components. The weights that
were actually used are returned in `weights_used`.

Rescaling of cosine similarity
------------------------------
Raw cosine similarity between a resume and a job description with MiniLM models typically
falls in a middle band (roughly 0.2-0.7) even for good matches, so it is linearly rescaled to
0-1 using DEFAULT_SEMANTIC_RANGE = (0.20, 0.80). The raw cosine is always returned too.
The range is a documented, adjustable assumption.

Public API
----------
DEFAULT_WEIGHTS, DEFAULT_SEMANTIC_RANGE
rescale_similarity(cosine, low, high)                          -> float in [0, 1]
validate_weights(weights)                                      -> dict (raises ValueError if invalid)
category_alignment(resume_probs, job_category)                 -> dict
compute_overall_score(semantic_cosine, skill_gap, category_info, weights=None, semantic_range=...) -> dict
evaluate_match(resume_text, job_text, classifier=None, matcher=None, extractor=None,
               weights=None, job_category=None)                -> dict
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from job_classifier import JobClassifier
from semantic_matching import SemanticMatcher, get_default_matcher
from skill_extractor import SkillExtractor, get_default_extractor
from skill_gap import analyze_skill_gap
from text_preprocessing import clean_text

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic_similarity": 0.40,
    "skill_match": 0.45,
    "category_alignment": 0.15,
}
DEFAULT_SEMANTIC_RANGE: Tuple[float, float] = (0.20, 0.80)

# Interpretation bands for the 0-100 score (descriptive labels, not predictions).
MATCH_BANDS: List[Tuple[float, str]] = [
    (80.0, "Excellent match"),
    (65.0, "Good match"),
    (50.0, "Moderate match"),
    (35.0, "Weak match"),
    (0.0, "Poor match"),
]


def _label_for(score: float) -> str:
    for threshold, label in MATCH_BANDS:
        if score >= threshold:
            return label
    return MATCH_BANDS[-1][1]


def rescale_similarity(cosine: float, low: float = DEFAULT_SEMANTIC_RANGE[0],
                       high: float = DEFAULT_SEMANTIC_RANGE[1]) -> float:
    """Linearly map cosine similarity from [low, high] to [0, 1] (clipped)."""
    if high <= low:
        raise ValueError("high must be greater than low")
    return float(min(1.0, max(0.0, (cosine - low) / (high - low))))


def validate_weights(weights: Optional[Mapping[str, float]]) -> Dict[str, float]:
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"Unknown weight keys {sorted(unknown)}. Valid keys: {list(DEFAULT_WEIGHTS)}")
        merged.update({k: float(v) for k, v in weights.items()})
    if any(v < 0 for v in merged.values()):
        raise ValueError("Weights must be non-negative.")
    if sum(merged.values()) <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    return merged


def category_alignment(resume_probs: Optional[Mapping[str, float]],
                       job_category: Optional[str]) -> Dict[str, Any]:
    """
    How well does the job's category agree with the classifier's prediction for the resume?

    score = P(resume in job_category) / P(resume in its top predicted category)
    -> 1.0 when the job's category is the resume's top prediction, smaller otherwise.

    `resume_probs` comes from JobClassifier.predict_proba_dict(). Category names are
    compared case-insensitively. If the job category is unknown to the classifier the
    component is reported as unavailable (it is never guessed).
    """
    info: Dict[str, Any] = {
        "available": False, "score": None, "job_category": job_category,
        "job_category_probability": None, "resume_top_category": None,
        "resume_top_probability": None,
    }
    if not resume_probs or not job_category:
        return info

    top_category = max(resume_probs, key=resume_probs.get)
    info["resume_top_category"] = top_category
    info["resume_top_probability"] = float(resume_probs[top_category])

    lookup = {c.lower(): c for c in resume_probs}
    matched = lookup.get(str(job_category).strip().lower())
    if matched is None:
        return info  # job category not among the classifier's classes
    top_prob = resume_probs[top_category]
    job_prob = float(resume_probs[matched])
    info.update({
        "available": True,
        "job_category": matched,
        "job_category_probability": job_prob,
        "score": float(job_prob / top_prob) if top_prob > 0 else 0.0,
    })
    return info


def compute_overall_score(
    semantic_cosine: Optional[float],
    skill_gap: Optional[Mapping[str, Any]],
    category_info: Optional[Mapping[str, Any]] = None,
    weights: Optional[Mapping[str, float]] = None,
    semantic_range: Tuple[float, float] = DEFAULT_SEMANTIC_RANGE,
) -> Dict[str, Any]:
    """
    Combine the components into an explainable overall score.

    Parameters
    ----------
    semantic_cosine : raw cosine similarity from semantic_matching (or None).
    skill_gap       : dict returned by skill_gap.analyze_skill_gap (uses 'coverage').
    category_info   : dict returned by category_alignment() (or None).

    Returns
    -------
    {
      "overall_score": float (0-100),
      "match_label": str,
      "score_breakdown": {
          "<component>": {"raw_value", "score" (0-1 or None), "weight_requested",
                          "weight_used", "contribution" (points out of 100),
                          "available": bool, "method": str, "explanation": str}, ...},
      "weights_requested": {...}, "weights_used": {...}, "notes": [str, ...]
    }
    """
    requested = validate_weights(weights)
    notes: List[str] = []

    components: Dict[str, Dict[str, Any]] = {}

    # 1) Semantic similarity (pretrained embeddings + cosine)
    if semantic_cosine is None:
        components["semantic_similarity"] = {
            "raw_value": None, "score": None, "method": "sentence_transformer_cosine",
            "explanation": "Semantic similarity was not computed."}
    else:
        sem_score = rescale_similarity(semantic_cosine, *semantic_range)
        components["semantic_similarity"] = {
            "raw_value": round(float(semantic_cosine), 4), "score": sem_score,
            "method": "sentence_transformer_cosine",
            "explanation": (f"Cosine similarity {semantic_cosine:.3f} between resume and job "
                            f"embeddings, rescaled from [{semantic_range[0]}, {semantic_range[1]}] "
                            f"to 0-1 = {sem_score:.2f}.")}

    # 2) Skill match (rule-based coverage of the job's skills)
    coverage = skill_gap.get("coverage") if skill_gap else None
    if coverage is None:
        components["skill_match"] = {
            "raw_value": None, "score": None, "method": "rule_based_dictionary_matching",
            "explanation": "No recognised skills in the job description, so skill match is unavailable."}
    else:
        components["skill_match"] = {
            "raw_value": round(float(coverage), 4), "score": float(coverage),
            "method": "rule_based_dictionary_matching",
            "explanation": (f"{skill_gap['matched_skill_count']} of {skill_gap['required_skill_count']} "
                            f"job skills were found on the resume ({coverage:.0%}).")}

    # 3) Category alignment (supervised classifier)
    if category_info and category_info.get("available"):
        components["category_alignment"] = {
            "raw_value": round(float(category_info["job_category_probability"]), 4),
            "score": float(category_info["score"]), "method": "ml_classification",
            "explanation": (f"The classifier ranks the resume as '{category_info['resume_top_category']}' "
                            f"({category_info['resume_top_probability']:.0%}); the job's category "
                            f"'{category_info['job_category']}' gets "
                            f"{category_info['job_category_probability']:.0%} -> alignment "
                            f"{category_info['score']:.2f}.")}
    else:
        reason = "No classifier was used." if not category_info else \
            "The job category is unknown or not one of the classifier's classes."
        components["category_alignment"] = {
            "raw_value": None, "score": None, "method": "ml_classification",
            "explanation": f"Category alignment unavailable. {reason}"}

    # Redistribute weights of unavailable components.
    available = [k for k, v in components.items() if v["score"] is not None and requested[k] > 0]
    total_weight = sum(requested[k] for k in available)
    if total_weight <= 0:
        raise ValueError("No scoring component with a positive weight is available.")

    overall = 0.0
    for name, comp in components.items():
        comp["available"] = name in available
        comp["weight_requested"] = round(requested[name], 4)
        used = requested[name] / total_weight if name in available else 0.0
        comp["weight_used"] = round(used, 4)
        comp["contribution"] = round(used * comp["score"] * 100, 2) if name in available else 0.0
        overall += used * comp["score"] * 100 if name in available else 0.0
        if name not in available and requested[name] > 0:
            notes.append(f"'{name}' was unavailable; its weight was redistributed to other components.")

    overall = round(float(overall), 2)
    return {
        "overall_score": overall,
        "match_label": _label_for(overall),
        "score_breakdown": components,
        "weights_requested": {k: round(v, 4) for k, v in requested.items()},
        "weights_used": {k: components[k]["weight_used"] for k in components},
        "notes": notes,
    }


def evaluate_match(
    resume_text: str,
    job_text: str,
    classifier: Optional[JobClassifier] = None,
    matcher: Optional[SemanticMatcher] = None,
    extractor: Optional[SkillExtractor] = None,
    weights: Optional[Mapping[str, float]] = None,
    job_category: Optional[str] = None,
    semantic_range: Tuple[float, float] = DEFAULT_SEMANTIC_RANGE,
) -> Dict[str, Any]:
    """
    Full analysis of ONE resume against ONE job description (main entry point for the app).

    The classifier is optional. If given, the resume's category probabilities are computed;
    the job's category is `job_category` if provided, otherwise the classifier's top prediction
    for the job text (the classifier was trained on resumes, so this is an approximation).

    Returns everything from `compute_overall_score` plus:
      "semantic":       {"method", "model", "cosine_similarity"}
      "classification": {"method", "available", "resume_top_categories", "job_category", "job_category_source"}
      "resume_skills", "job_skills", "skill_gap"
    """
    resume_clean, job_clean = clean_text(resume_text), clean_text(job_text)
    if not resume_clean:
        raise ValueError("Resume text is empty.")
    if not job_clean:
        raise ValueError("Job description text is empty.")

    matcher = matcher or get_default_matcher()
    extractor = extractor or get_default_extractor()
    notes: List[str] = []

    # Semantic similarity (pretrained model)
    cosine = matcher.similarity(resume_clean, job_clean)

    # Rule-based skill matching
    resume_counts = extractor.extract_with_counts(resume_clean)
    job_counts = extractor.extract_with_counts(job_clean)
    gap = analyze_skill_gap(list(resume_counts), list(job_counts), job_counts, extractor)

    # ML classification (optional)
    classification: Dict[str, Any] = {
        "method": "ml_classification", "available": False, "resume_top_categories": [],
        "job_category": job_category, "job_category_source": "provided" if job_category else None,
    }
    category_info = None
    if classifier is not None:
        try:
            probs = classifier.predict_proba_dict(resume_clean)
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
            classification["resume_top_categories"] = [
                {"category": c, "probability": round(p, 4)} for c, p in ranked]
            if not job_category:
                job_category = classifier.predict(job_clean, top_k=1)[0]["category"]
                classification["job_category_source"] = "classifier_prediction_on_job_text"
            classification["job_category"] = job_category
            classification["available"] = True
            category_info = category_alignment(probs, job_category)
        except (ValueError, RuntimeError) as exc:
            notes.append(f"Category classification skipped: {exc}")

    result = compute_overall_score(cosine, gap, category_info, weights, semantic_range)
    result["notes"] = notes + result["notes"]
    result["semantic"] = {"method": "sentence_transformer_cosine", "model": matcher.model_name,
                          "cosine_similarity": round(cosine, 4)}
    result["classification"] = classification
    result["resume_skills"] = list(resume_counts)
    result["job_skills"] = list(job_counts)
    result["skill_gap"] = gap
    return result
