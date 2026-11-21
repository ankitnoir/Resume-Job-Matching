"""
visualization.py
================
Plotly figure builders used by the dashboard.

Every function returns a ``plotly.graph_objects.Figure`` (or ``None`` when
there is nothing to draw). No Streamlit calls are made here, so the functions
can be reused in notebooks or reports.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "matched": "#2ea043",
    "missing": "#e5534b",
    "extra": "#8b949e",
    "primary": "#4f8bf9",
    "warning": "#f5a524",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def score_color(score: float) -> str:
    """Traffic-light colour for a 0-100 score (bands follow scoring.MATCH_BANDS: 65 = Good, 35 = Weak)."""
    if score >= 65:
        return COLORS["matched"]
    if score >= 35:
        return COLORS["warning"]
    return COLORS["missing"]


def score_band(score: float) -> tuple[str, str]:
    """Return ``(verdict, streamlit_colour_name)`` for a 0-100 score."""
    if score >= 65:
        return "Good match", "green"
    if score >= 35:
        return "Moderate match", "orange"
    return "Weak match", "red"


def count_mentions(text: str, skill: str) -> int:
    """Case-insensitive, whole-token count of ``skill`` inside ``text``."""
    if not text or not skill or not skill.strip():
        return 0
    pattern = r"(?<![A-Za-z0-9])" + re.escape(skill.strip()) + r"(?![A-Za-z0-9])"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=20, t=48 if title else 16, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        showlegend=False,
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)))
    return fig


def _short(text: str, limit: int = 42) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --------------------------------------------------------------------------- #
# Match score charts
# --------------------------------------------------------------------------- #
def match_gauge(score: float, title: str = "Overall match score") -> go.Figure:
    """Semi-circular gauge for a 0-100 match score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(score),
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": title, "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": score_color(score)},
                "steps": [
                    {"range": [0, 35], "color": "rgba(229,83,75,0.15)"},
                    {"range": [35, 65], "color": "rgba(245,165,36,0.15)"},
                    {"range": [65, 100], "color": "rgba(46,160,67,0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def score_breakdown(components: Mapping[str, float], title: str = "Score components") -> go.Figure | None:
    """Horizontal bars for the score components returned by ``scoring.py`` (0-100)."""
    if not components:
        return None
    items = sorted(components.items(), key=lambda kv: kv[1])
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [float(v) for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[score_color(v) for v in values],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, 115], showgrid=False, visible=False)
    return _layout(fig, max(200, 60 * len(values) + 70), title)


def skill_overlap_donut(matched: Sequence[str], missing: Sequence[str], title: str = "Required skills coverage") -> go.Figure | None:
    """Donut chart: matched vs missing job skills."""
    if not matched and not missing:
        return None
    fig = go.Figure(
        go.Pie(
            labels=["Matched", "Missing"],
            values=[len(matched), len(missing)],
            hole=0.58,
            marker=dict(colors=[COLORS["matched"], COLORS["missing"]]),
            textinfo="label+value",
            sort=False,
        )
    )
    return _layout(fig, 300, title)


# --------------------------------------------------------------------------- #
# Skill charts
# --------------------------------------------------------------------------- #
def skill_distribution(
    skills: Sequence[str],
    text: str,
    title: str,
    color: str = COLORS["primary"],
    top_n: int = 15,
    counts: Mapping[str, int] | None = None,
) -> go.Figure | None:
    """
    Bar chart of how often each skill is mentioned.

    ``counts`` (skill -> mentions, e.g. from ``skill_extractor.extract_skill_counts``) is used when
    given, so aliases are counted too; otherwise the skill names are searched for in ``text``.

    Used for: skills in the resume (resume text), matched skills / missing
    skills (job-description text, i.e. how strongly the job asks for them).
    """
    if not skills:
        return None
    lookup = {k.lower(): int(v) for k, v in (counts or {}).items()}
    rows = sorted(
        ((s, lookup.get(s.lower(), 0) if counts is not None else count_mentions(text, s)) for s in skills),
        key=lambda r: (-r[1], r[0].lower()),
    )[:top_n]
    labels = [r[0] for r in rows]
    counts = [r[1] for r in rows]
    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker_color=color,
            text=counts,
            textposition="outside",
            hovertemplate="%{y}: %{x} mention(s)<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(rangemode="tozero", title="Mentions", showgrid=False)
    return _layout(fig, max(240, 28 * len(rows) + 100), title)


def category_predictions(categories: Sequence[Mapping[str, float]], title: str = "Predicted job categories") -> go.Figure | None:
    """Bar chart of ``[{"category": str, "confidence": 0-1}, ...]``."""
    if not categories:
        return None
    labels = [str(c["category"]) for c in categories]
    values = [float(c["confidence"]) * 100 for c in categories]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=COLORS["primary"],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 115], visible=False)
    return _layout(fig, max(220, 52 * len(labels) + 80), title)


# --------------------------------------------------------------------------- #
# Recommendation charts
# --------------------------------------------------------------------------- #
def job_matches_bar(recs: pd.DataFrame, top_n: int = 10, title: str = "Top job matches") -> go.Figure | None:
    """Ranked horizontal bars of recommended jobs (expects ``rank``, ``title``, ``score``)."""
    if recs is None or recs.empty:
        return None
    df = recs.head(top_n)
    labels = [f"{int(r.rank)}. {_short(r.title)}" for r in df.itertuples()]
    values = [float(s) for s in df["score"]]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=[score_color(v) for v in values],
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, 118], visible=False)
    return _layout(fig, max(260, 34 * len(labels) + 90), title)


def missing_skills_across_jobs(recs: pd.DataFrame, top_n: int = 12, title: str = "Most common missing skills across recommended jobs") -> go.Figure | None:
    """Count the skills that recur in the ``missing_skills`` column of the recommendations."""
    if recs is None or recs.empty or "missing_skills" not in recs.columns:
        return None
    counter: Counter[str] = Counter()
    for skills in recs["missing_skills"]:
        counter.update(skills if isinstance(skills, list) else [])
    if not counter:
        return None
    rows = counter.most_common(top_n)
    fig = go.Figure(
        go.Bar(
            x=[c for _, c in rows],
            y=[s for s, _ in rows],
            orientation="h",
            marker_color=COLORS["missing"],
            text=[c for _, c in rows],
            textposition="outside",
            hovertemplate="%{y}: missing in %{x} job(s)<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(rangemode="tozero", title="Number of jobs", showgrid=False)
    return _layout(fig, max(240, 28 * len(rows) + 100), title)
