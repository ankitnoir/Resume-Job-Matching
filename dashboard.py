"""
dashboard.py
============
Streamlit rendering functions for the analysis results.

``app.py`` builds a *result dictionary* (see ``app.run_analysis``) and passes it
to the ``render_*`` functions below. This module only displays data; it never
calls the NLP/ML backend.

Result dictionary layout
------------------------
{
  "meta":   {"resume_name", "job_title", "job_company", "elapsed"},
  "resume": {"text", "clean", "skills", "categories", "info"},
  "job":    {"title", "company", "text", "clean", "skills"},
  "similarity": float | None,            # 0-1
  "gap":    {"matched", "missing", "extra", "coverage"} | None,
  "score":  {"overall_score", "label", "components"} | None,
  "recommendations": DataFrame | None,
  "errors": {step_name: message},
}
"""
from __future__ import annotations

import html
import inspect
from typing import Any, Sequence

import pandas as pd
import streamlit as st

import visualization as viz

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.chip-wrap {line-height: 2.1; margin-bottom: .4rem;}
.chip {display:inline-block; padding:2px 11px; margin:0 6px 6px 0; border-radius:999px;
       font-size:.82rem; font-weight:500; border:1px solid transparent;}
.chip-matched {background:rgba(46,160,67,.15); color:#2ea043; border-color:rgba(46,160,67,.45);}
.chip-missing {background:rgba(229,83,75,.13); color:#e5534b; border-color:rgba(229,83,75,.45);}
.chip-skill   {background:rgba(79,139,249,.14); color:#4f8bf9; border-color:rgba(79,139,249,.45);}
.chip-extra   {background:rgba(139,148,158,.15); color:#8b949e; border-color:rgba(139,148,158,.45);}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show_plot(fig: Any, key: str) -> None:
    """Render a Plotly figure full-width (``key`` must be unique on the page)."""
    if fig is None:
        return
    if _PLOTLY_HAS_WIDTH:  # newer Streamlit
        st.plotly_chart(fig, width="stretch", key=key)
    else:  # older Streamlit (``width`` would be silently ignored there)
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_table(df: pd.DataFrame, **kwargs: Any) -> None:
    """Render a DataFrame full-width without the index."""
    try:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    except Exception:  # older Streamlit versions
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


def chips_html(items: Sequence[str], kind: str, limit: int = 80) -> str:
    shown = list(items)[:limit]
    out = "".join(f'<span class="chip chip-{kind}">{html.escape(str(i))}</span>' for i in shown)
    if len(items) > limit:
        out += f'<span class="chip chip-extra">+{len(items) - limit} more</span>'
    return f'<div class="chip-wrap">{out}</div>'


def render_chips(title: str, items: Sequence[str], kind: str, empty: str = "None found.") -> None:
    st.markdown(f"**{title}** ({len(items)})")
    if items:
        st.markdown(chips_html(items, kind), unsafe_allow_html=True)
    else:
        st.caption(empty)


def _md_escape(text: Any) -> str:
    """Escape characters that Streamlit's markdown would treat as formatting/LaTeX."""
    return str(text).replace("$", "\\$").replace("[", "\\[").replace("]", "\\]")


def _text_or_empty(value: Any) -> str:
    """Return ``value`` as text, treating None/NaN as an empty string."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _warn_step(result: dict, *keys: str) -> bool:
    """Show a warning for the first failed step among ``keys``. Returns True if shown."""
    for key in keys:
        if key in result["errors"]:
            st.warning(f"Not available — {result['errors'][key]}")
            return True
    return False


def _pct(value: float | None, scale: float = 1.0) -> str:
    return "—" if value is None else f"{value * scale:.1f}%"


# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
def render_kpi_row(result: dict) -> None:
    score, sim, gap = result.get("score"), result.get("similarity"), result.get("gap")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall match", _pct(score["overall_score"]) if score else "—",
              help="Final score from scoring.compute_match_score().")
    c2.metric("Semantic similarity", _pct(sim, 100),
              help="Cosine similarity of resume and job-description embeddings.")
    c3.metric("Skill coverage", _pct(gap["coverage"], 100) if gap else "—",
              help="Share of the job's required skills found in the resume.")
    c4.metric("Missing skills", str(len(gap["missing"])) if gap else "—",
              help="Job skills not found in the resume.")
    if score:
        verdict, colour = viz.score_band(score["overall_score"])
        label = score.get("label") or verdict
        st.markdown(f"Verdict: :{colour}[**{label}**]")


# --------------------------------------------------------------------------- #
# Resume analysis tab
# --------------------------------------------------------------------------- #
def render_resume_analysis(result: dict) -> None:
    res = result["resume"]
    st.subheader("📄 Resume Analysis")

    left, right = st.columns([3, 2])
    with left, st.container(border=True):
        st.markdown("##### Extracted skills")
        if not _warn_step(result, "resume_skills"):
            render_chips("Skills detected", res["skills"], "skill", "No skills were detected in this resume.")
            show_plot(
                viz.skill_distribution(res["skills"], res["clean"], "Skill distribution (mentions in resume)",
                                       counts=res.get("skill_counts")),
                key="resume_skill_dist",
            )
    with right, st.container(border=True):
        st.markdown("##### Predicted job categories")
        if not _warn_step(result, "categories"):
            cats = res["categories"]
            if cats:
                table = pd.DataFrame(
                    {"Rank": range(1, len(cats) + 1),
                     "Category": [c["category"] for c in cats],
                     "Confidence": [c["confidence"] * 100 for c in cats]}
                )
                show_table(table, column_config={
                    "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%")
                })
                show_plot(viz.category_predictions(cats), key="resume_categories")
            else:
                st.caption("No category predictions returned.")

    with st.container(border=True):
        st.markdown("##### Resume information · skills by category")
        text = res["text"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Words", f"{len(text.split()):,}")
        m2.metric("Characters", f"{len(text):,}")
        m3.metric("Skills detected", len(res["skills"]))
        info = res.get("info")
        if _warn_step(result, "resume_info"):
            pass
        elif info:
            rows = [
                {"Field": str(k) if any(c.isupper() for c in str(k)) else str(k).replace("_", " ").title(),
                 "Value": ", ".join(map(str, v)) if isinstance(v, (list, tuple, set)) else str(v)}
                for k, v in info.items()
            ]
            show_table(pd.DataFrame(rows))
        else:
            st.caption("Skills grouped by category appear here when the skill extractor is available.")

    with st.expander("View extracted resume text"):
        st.text_area("Extracted text", value=text, height=320, disabled=True,
                     key="resume_text_view", label_visibility="collapsed")


# --------------------------------------------------------------------------- #
# Job match tab
# --------------------------------------------------------------------------- #
def render_job_match(result: dict) -> None:
    job = result["job"]
    gap, score = result.get("gap"), result.get("score")
    st.subheader("🎯 Resume–Job Match")
    st.caption(f"Target role: **{_md_escape(job['title'])}**" + (f" · {_md_escape(job['company'])}" if job.get("company") else ""))

    render_kpi_row(result)
    for key in ("similarity", "skill_gap", "score"):
        _warn_step(result, key)
    if score and score.get("notes"):
        for note in score["notes"]:
            st.info(note)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        if score:
            show_plot(viz.match_gauge(score["overall_score"]), key="match_gauge")
        else:
            st.info("The overall score is not available (see warnings above).")
    with c2, st.container(border=True):
        if score and score["components"]:
            show_plot(viz.score_breakdown(score["components"]), key="match_breakdown")
        elif result.get("similarity") is not None:
            st.metric("Semantic similarity", _pct(result["similarity"], 100))
        else:
            st.info("No score components available.")

    st.markdown("### Skill gap analysis")
    if gap is None:
        # The reason (failed step) is already listed in the warnings above.
        st.info("Skill gap analysis is not available.")
        return

    if gap.get("summary"):
        st.caption(gap["summary"])
    a, b = st.columns([3, 2])
    with a, st.container(border=True):
        render_chips("✅ Matched skills", gap["matched"], "matched", "No required skills were found in the resume.")
        render_chips("❌ Missing skills (skill gap)", gap["missing"], "missing", "No skill gap detected — all required skills are present.")
        if gap["extra"]:
            render_chips("➕ Additional resume skills (not required by this job)", gap["extra"], "extra")
    with b, st.container(border=True):
        show_plot(viz.skill_overlap_donut(gap["matched"], gap["missing"]), key="match_donut")

    d1, d2 = st.columns(2)
    with d1, st.container(border=True):
        show_plot(viz.skill_distribution(gap["matched"], job["clean"], "Matched skills — emphasis in job description",
                                         viz.COLORS["matched"], counts=job.get("skill_counts")), key="match_matched_dist")
    with d2, st.container(border=True):
        show_plot(viz.skill_distribution(gap["missing"], job["clean"], "Missing skills — emphasis in job description",
                                         viz.COLORS["missing"], counts=job.get("skill_counts")), key="match_missing_dist")

    if gap.get("missing_details"):
        with st.container(border=True):
            st.markdown("##### Skills to learn first (ordered by emphasis in the job description)")
            table = pd.DataFrame(gap["missing_details"]).rename(columns={
                "skill": "Skill", "category": "Category", "mentions": "Mentions", "priority": "Priority"})
            show_table(table)

    if score and score.get("details"):
        with st.expander("How the score is calculated"):
            rows = []
            for name, comp in score["details"].items():
                rows.append({
                    "Component": name.replace("_", " ").title(),
                    "Raw value": comp.get("raw_value"),
                    "Score (0–100)": None if comp.get("score") is None else round(comp["score"] * 100, 1),
                    "Weight used": comp.get("weight_used"),
                    "Contribution": comp.get("contribution"),
                    "Explanation": comp.get("explanation", ""),
                })
            show_table(pd.DataFrame(rows))
            st.caption("Weights and rescaling ranges are design choices defined in scoring.py; unavailable "
                       "components have their weight redistributed to the others.")

    status_rows = (
        [{"Skill": s, "Status": "Matched"} for s in gap["matched"]]
        + [{"Skill": s, "Status": "Missing"} for s in gap["missing"]]
        + [{"Skill": s, "Status": "Extra (resume only)"} for s in gap["extra"]]
    )
    if status_rows:
        with st.expander("Skill comparison table"):
            show_table(pd.DataFrame(status_rows))
    with st.expander("View job description"):
        st.text(job["text"])


# --------------------------------------------------------------------------- #
# Recommendations tab
# --------------------------------------------------------------------------- #
def _export_frame(recs: pd.DataFrame) -> pd.DataFrame:
    out = recs.copy()
    for col in ("matched_skills", "missing_skills"):
        if col in out.columns:
            out[col] = out[col].apply(lambda v: ", ".join(v) if isinstance(v, list) else v)
    return out


def render_recommendations(result: dict) -> None:
    st.subheader("💼 Job Recommendations")
    if _warn_step(result, "recommendations"):
        return
    recs = result.get("recommendations")
    if recs is None or recs.empty:
        st.info("No recommendations. Add a job dataset (`data/jobs.csv`) so the backend has jobs to rank.")
        return

    st.caption(f"{len(recs)} best-matching jobs, ranked by match score.")
    columns = [c for c in ("rank", "title", "company", "location", "score", "match_label", "semantic_score", "skill_score")
               if c in recs.columns]
    config: dict[str, Any] = {
        "rank": st.column_config.NumberColumn("#", width="small"),
        "title": st.column_config.TextColumn("Job title"),
        "company": st.column_config.TextColumn("Company"),
        "location": st.column_config.TextColumn("Location"),
        "match_label": st.column_config.TextColumn("Verdict"),
    }
    for col, label in (("score", "Match score"), ("semantic_score", "Semantic"), ("skill_score", "Skills")):
        config[col] = st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.1f")
    show_table(recs[columns], column_config={k: v for k, v in config.items() if k in columns})

    with st.container(border=True):
        show_plot(viz.job_matches_bar(recs), key="recs_bar")

    st.markdown("##### Job details")
    for row in recs.head(10).to_dict("records"):
        company = _text_or_empty(row.get("company"))
        header = f"#{int(row['rank'])} · {_md_escape(row['title'])}" + (f" — {_md_escape(company)}" if company else "") + f" · {row['score']:.1f}%"
        with st.expander(header):
            matched, missing = row.get("matched_skills"), row.get("missing_skills")
            if isinstance(matched, list) and matched:
                render_chips("Matched skills", matched, "matched")
            if isinstance(missing, list) and missing:
                render_chips("Missing skills", missing, "missing")
            desc = _text_or_empty(row.get("description"))
            if desc:
                st.text(desc[:1500] + ("…" if len(desc) > 1500 else ""))

    st.download_button(
        "⬇️ Download recommendations (CSV)",
        data=_export_frame(recs).to_csv(index=False).encode("utf-8"),
        file_name="job_recommendations.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Dashboard overview tab
# --------------------------------------------------------------------------- #
def render_dashboard(result: dict) -> None:
    st.subheader("📊 Dashboard")
    meta = result["meta"]
    st.caption(f"Resume: **{meta['resume_name']}** · Target job: **{meta['job_title']}** · analysed in {meta['elapsed']:.1f}s")
    render_kpi_row(result)

    score, gap, res, job = result.get("score"), result.get("gap"), result["resume"], result["job"]

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        if score:
            show_plot(viz.match_gauge(score["overall_score"]), key="dash_gauge")
        else:
            st.info("Overall score unavailable.")
    with c2, st.container(border=True):
        if score and score["components"]:
            show_plot(viz.score_breakdown(score["components"]), key="dash_breakdown")
        else:
            st.info("No score components available.")
    with c3, st.container(border=True):
        fig = viz.skill_overlap_donut(gap["matched"], gap["missing"]) if gap else None
        if fig is not None:
            show_plot(fig, key="dash_donut")
        else:
            st.info("Skill coverage unavailable.")

    recs = result.get("recommendations")
    c4, c5 = st.columns(2)
    with c4, st.container(border=True):
        fig = viz.job_matches_bar(recs) if recs is not None else None
        if fig is not None:
            show_plot(fig, key="dash_recs")
        else:
            st.info("No job recommendations available.")
    with c5, st.container(border=True):
        fig = viz.missing_skills_across_jobs(recs) if recs is not None else None
        if fig is not None:
            show_plot(fig, key="dash_missing_across")
        elif gap and gap["missing"]:
            show_plot(viz.skill_distribution(gap["missing"], job["clean"], "Missing skills — emphasis in job description",
                                             viz.COLORS["missing"], counts=job.get("skill_counts")), key="dash_missing_dist")
        else:
            st.info("No skill gap data available.")

    c6, c7 = st.columns(2)
    with c6, st.container(border=True):
        fig = viz.skill_distribution(res["skills"], res["clean"], "Skill distribution (mentions in resume)",
                                     counts=res.get("skill_counts"))
        if fig is not None:
            show_plot(fig, key="dash_skill_dist")
        else:
            st.info("No resume skills available.")
    with c7, st.container(border=True):
        fig = viz.category_predictions(res["categories"])
        if fig is not None:
            show_plot(fig, key="dash_categories")
        else:
            st.info("No category predictions available.")

    if gap:
        with st.container(border=True):
            st.markdown("##### Skills at a glance")
            render_chips("✅ Matched", gap["matched"], "matched")
            render_chips("❌ Missing", gap["missing"], "missing")
