"""
skill_gap.py
============
Compare candidate skills with the skills mentioned in a job description and identify gaps.

This is RULE-BASED set comparison on skills found by `skill_extractor` (dictionary matching).
It is not machine learning and not semantic similarity.

Note on terminology: a job description does not say which skills are mandatory, so every
skill the extractor finds in the job text is treated as a "job skill". Skills mentioned
more often are ranked as higher priority when they are missing.

Public API
----------
analyze_skill_gap(candidate_skills, job_skills, job_skill_counts=None, extractor=None) -> dict
analyze_resume_vs_job(resume_text, job_text, extractor=None) -> dict

Return format of analyze_skill_gap
----------------------------------
{
  "method": "rule_based_dictionary_matching",
  "job_skills": [str, ...],
  "matched_skills": [str, ...],
  "missing_skills": [{"skill": str, "category": str, "mentions": int, "priority": "high|medium|normal"}, ...],
  "extra_skills": [str, ...],              # candidate skills not mentioned in the job
  "required_skill_count": int,             # number of job skills
  "matched_skill_count": int,
  "coverage": float | None,                # matched / job skills (None if the job has no skills)
  "weighted_coverage": float | None,       # same, but weighted by how often the job mentions each skill
  "missing_by_category": {category: [skill, ...]},
  "summary": str
}
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

from skill_extractor import SkillExtractor, get_default_extractor


def _key(skill: str) -> str:
    return skill.strip().lower()


def _priority(mentions: int) -> str:
    """Simple, transparent rule: skills repeated in the job text are more important."""
    if mentions >= 3:
        return "high"
    if mentions == 2:
        return "medium"
    return "normal"


def analyze_skill_gap(
    candidate_skills: Iterable[str],
    job_skills: Iterable[str],
    job_skill_counts: Optional[Mapping[str, int]] = None,
    extractor: Optional[SkillExtractor] = None,
) -> Dict[str, Any]:
    """Compare two lists of canonical skill names (as produced by SkillExtractor)."""
    extractor = extractor or get_default_extractor()

    candidate = {_key(s): s for s in candidate_skills if isinstance(s, str) and s.strip()}
    job_list = list(dict.fromkeys(s for s in job_skills if isinstance(s, str) and s.strip()))
    counts = {_key(k): int(v) for k, v in (job_skill_counts or {}).items()}

    matched: List[str] = []
    missing: List[Dict[str, Any]] = []
    matched_mentions = 0
    total_mentions = 0

    for skill in job_list:
        mentions = max(1, counts.get(_key(skill), 1))
        total_mentions += mentions
        if _key(skill) in candidate:
            matched.append(skill)
            matched_mentions += mentions
        else:
            missing.append({
                "skill": skill,
                "category": extractor.get_category(skill),
                "mentions": mentions,
                "priority": _priority(mentions),
            })

    # Most important gaps first
    missing.sort(key=lambda m: (-m["mentions"], m["skill"].lower()))

    job_keys = {_key(s) for s in job_list}
    extra = [orig for k, orig in candidate.items() if k not in job_keys]

    missing_by_category: Dict[str, List[str]] = defaultdict(list)
    for item in missing:
        missing_by_category[item["category"]].append(item["skill"])

    n_job = len(job_list)
    coverage = len(matched) / n_job if n_job else None
    weighted = matched_mentions / total_mentions if total_mentions else None

    if n_job == 0:
        summary = ("No recognised skills were found in the job description, "
                   "so a skill-gap analysis is not possible.")
    else:
        summary = (f"The candidate covers {len(matched)} of {n_job} skills mentioned in the job "
                   f"description ({coverage:.0%}); {len(missing)} skill(s) are missing.")

    return {
        "method": "rule_based_dictionary_matching",
        "job_skills": job_list,
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "required_skill_count": n_job,
        "matched_skill_count": len(matched),
        "coverage": coverage,
        "weighted_coverage": weighted,
        "missing_by_category": dict(missing_by_category),
        "summary": summary,
    }


def analyze_resume_vs_job(resume_text: str, job_text: str,
                          extractor: Optional[SkillExtractor] = None) -> Dict[str, Any]:
    """
    Convenience: extract skills from both texts, then run `analyze_skill_gap`.
    Returns the gap dictionary plus 'resume_skills' and 'job_skill_counts'.
    """
    extractor = extractor or get_default_extractor()
    resume_counts = extractor.extract_with_counts(resume_text)
    job_counts = extractor.extract_with_counts(job_text)
    gap = analyze_skill_gap(list(resume_counts), list(job_counts), job_counts, extractor)
    gap["resume_skills"] = list(resume_counts)
    gap["job_skill_counts"] = job_counts
    return gap
