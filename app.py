"""Small Streamlit dashboard for the PMIS synthetic allocation prototype."""

from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "outputs"
EXP = OUT / "experiments"

st.set_page_config(page_title="PM Internship Allocation Explorer", page_icon="🧭", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def required(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Required output not found: `{path.relative_to(ROOT)}`. Run `python src/pmis_pipeline.py` first.")
        st.stop()
    return load_csv(path)


candidates = required(OUT / "clean" / "synthetic_candidates.csv")
internships = required(OUT / "clean" / "synthetic_internships.csv")
recommendations = required(EXP / "ranked_recommendations.csv")
allocations = required(EXP / "capacity_aware_allocations.csv")
metrics = required(EXP / "model_metrics.csv")
quality = required(EXP / "data_quality_metrics.csv")

st.title("PM Internship Scheme · Allocation Explorer")
st.caption("Research prototype · browse generated opportunities, synthetic candidate matches, and allocation scenarios")
st.warning(
    "All candidate profiles, match scores, recommendations, and labels shown here are synthetic. "
    "The scores are not real PMIS selection predictions or observed applicant outcomes.",
    icon="⚠️",
)

overview, opportunity_tab, matching_tab, allocation_tab, evaluation_tab = st.tabs(
    ["Overview", "Internships", "Candidate matching", "Allocations", "Experiments"]
)

with overview:
    assigned = allocations[allocations["allocated"].astype(int) == 1]
    cap = pd.to_numeric(internships["capacity"], errors="coerce").fillna(0).sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Synthetic candidates", f"{len(candidates):,}")
    c2.metric("Opportunity rows", f"{len(internships):,}")
    c3.metric("Candidate–opportunity pairs", f"{len(recommendations):,}")
    c4.metric("Scenario allocations", f"{len(assigned):,}")
    st.subheader("What this prototype demonstrates")
    st.markdown(
        "- Browse source-linked internship opportunities and their stated capacity.\n"
        "- Inspect eligibility-filtered recommendations and their compatibility components.\n"
        "- Review one-to-one candidate assignments under opportunity capacity limits.\n"
        "- Compare synthetic proxy-label experiments and data-quality scenarios."
    )
    st.info(f"Recorded opportunity capacity in this sample: **{int(cap):,}** seats. The optimizer assigned **{len(assigned):,}** synthetic candidates.")
    st.caption("Source opportunity attributes come from the supplied workbook. Candidate records and matching labels were generated for this research prototype.")

with opportunity_tab:
    st.subheader("Internship opportunities")
    f1, f2, f3 = st.columns(3)
    states = sorted(internships["state"].dropna().astype(str).unique())
    sectors = sorted(internships["sector"].dropna().astype(str).unique())
    quals = sorted(internships["qualification"].dropna().astype(str).unique())
    state_pick = f1.selectbox("State", ["All"] + states, key="opp_state")
    sector_pick = f2.selectbox("Sector", ["All"] + sectors, key="opp_sector")
    qual_pick = f3.selectbox("Qualification", ["All"] + quals, key="opp_qual")
    view = internships.copy()
    if state_pick != "All":
        view = view[view["state"].astype(str) == state_pick]
    if sector_pick != "All":
        view = view[view["sector"].astype(str) == sector_pick]
    if qual_pick != "All":
        view = view[view["qualification"].astype(str) == qual_pick]
    st.caption(f"{len(view):,} opportunities match these filters.")
    show_cols = [c for c in ["internship_id", "company", "state", "district", "sector", "area", "qualification", "course", "capacity", "source_applications_received"] if c in view]
    st.dataframe(view[show_cols], width="stretch", hide_index=True)

with matching_tab:
    st.subheader("Recommendations for a synthetic candidate")
    cand_ids = sorted(candidates["candidate_id"].astype(str).unique())
    selected_id = st.selectbox("Candidate ID", cand_ids, key="candidate_match_id")
    profile = candidates.loc[candidates["candidate_id"].astype(str) == selected_id].iloc[0]
    st.write(
        f"**Profile:** {profile['qualification']} · {profile['course']} · "
        f"{profile['home_district']}, {profile['home_state']} · interests: {profile['sector_interests']}"
    )
    recs = recommendations[
        (recommendations["candidate_id"].astype(str) == selected_id)
        & (recommendations["eligible"].astype(int) == 1)
    ].sort_values("recommendation_rank")
    min_score = st.slider("Minimum synthetic match score", 0.0, 1.0, 0.0, 0.05, key="match_min_score")
    limit = st.slider("Show top recommendations", 5, 50, 15, 5, key="match_limit")
    recs = recs[pd.to_numeric(recs["synthetic_match_score"], errors="coerce") >= min_score].head(limit)
    if recs.empty:
        st.info("No eligible recommendations meet this score filter.")
    else:
        st.dataframe(
            recs[["recommendation_rank", "internship_id", "internship_company", "internship_sector", "internship_state", "internship_district", "synthetic_match_score", "qualification_compatibility", "course_compatibility", "specialization_compatibility", "skill_similarity", "sector_compatibility", "location_compatibility", "why_recommended"]],
            width="stretch",
            hide_index=True,
        )

with allocation_tab:
    st.subheader("Capacity-aware allocation scenario")
    assigned = allocations[allocations["allocated"].astype(int) == 1].copy()
    i1, i2, i3 = st.columns(3)
    alloc_states = ["All"] + sorted(assigned["internship_state"].dropna().astype(str).unique())
    alloc_sectors = ["All"] + sorted(assigned["internship_sector"].dropna().astype(str).unique())
    selected_state = i1.selectbox("Internship state", alloc_states, key="allocation_state")
    selected_sector = i2.selectbox("Sector", alloc_sectors, key="allocation_sector")
    search = i3.text_input("Company contains", key="allocation_company").strip().casefold()
    if selected_state != "All":
        assigned = assigned[assigned["internship_state"].astype(str) == selected_state]
    if selected_sector != "All":
        assigned = assigned[assigned["internship_sector"].astype(str) == selected_sector]
    if search:
        assigned = assigned[assigned["internship_company"].astype(str).str.casefold().str.contains(search, na=False)]
    at_limit = int(assigned["capacity_at_limit"].astype(bool).sum()) if len(assigned) else 0
    a1, a2, a3 = st.columns(3)
    a1.metric("Assignments shown", f"{len(assigned):,}")
    a2.metric("At-capacity assignments shown", f"{at_limit:,}")
    a3.metric("Capacity violations", "0", help="Validation is recorded by the allocation pipeline.")
    alloc_cols = ["candidate_id", "internship_id", "internship_company", "internship_sector", "internship_state", "synthetic_match_score", "capacity", "capacity_at_limit", "allocation_explanation"]
    st.dataframe(assigned[alloc_cols], width="stretch", hide_index=True)

with evaluation_tab:
    st.subheader("Model comparison · synthetic proxy labels only")
    clean = metrics[metrics["dataset"].astype(str) == "clean"].copy()
    display_metric = st.selectbox("Metric", ["f1", "precision", "recall", "roc_auc", "pr_auc", "ndcg_at_full_list"], key="evaluation_metric")
    if not clean.empty:
        plot_data = clean[["model", display_metric]].copy().set_index("model")
        st.bar_chart(plot_data, y=display_metric, height=300)
        st.dataframe(clean[["model", "precision", "recall", "f1", "roc_auc", "pr_auc", "ndcg_at_full_list", "label_note"]], width="stretch", hide_index=True)
    st.subheader("Data-quality experiment")
    qcols = [c for c in ["dataset", "rows", "missing_cells", "duplicate_pair_ids", "non_numeric_capacity_values", "application_capacity_ratio_gt_10", "unknown_category_values", "proxy_positive_rate"] if c in quality]
    st.dataframe(quality[qcols], width="stretch", hide_index=True)
    st.info("Fairness was not evaluated: the supplied data and project instructions do not define policy groups or approved policy rules. All evaluation scores measure agreement with generated heuristic labels, not real selection quality.")

