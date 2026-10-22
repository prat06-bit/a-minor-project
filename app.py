"""Streamlit research dashboard for the PM internship allocation prototype."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.pmis_pipeline import (
    MATCH_WEIGHTS,
    allocate,
    allocation_fairness_summary,
    load_opportunity_catalog,
    match_candidates,
    validate_candidate_roster,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "outputs"
EXP = OUT / "experiments"
FEATURE_COLUMNS = {
    "candidate_id", "qualification", "course", "specialization", "skills",
    "sector_interests", "primary_sector", "home_state", "home_district",
    "preferred_state", "preferred_district", "past_participation",
}
NON_GROUP_COLUMNS = {
    "candidate_id", "record_origin", "source_profile_row", "generation_seed",
    "generation_rule", "provenance_json", "label", "selection_outcome",
}

st.set_page_config(
    page_title="PMIS · Smart Allocation Engine",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    :root { --ink: #13233a; --muted: #50627a; --line: #d9e2ee; --accent: #126b5b; }
    .stApp, [data-testid="stAppViewContainer"] { background: #f5f8fc; color: var(--ink); }
    .stApp p, .stApp label, [data-testid="stMarkdownContainer"] { color: var(--ink); }
    [data-testid="stCaptionContainer"] p { color: #50627a !important; }
    [data-testid="stSidebar"] { background: #10253c; }
    [data-testid="stSidebar"] * { color: #f2f6fb !important; }
    h1, h2, h3 { color: #13233a; letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid var(--line); border-radius: 12px;
        padding: 14px 16px; box-shadow: 0 2px 8px rgba(22, 45, 75, .04);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: #10253c; }
    [data-testid="stDataFrame"] { color: #13233a !important; }
    div[data-testid="stTabs"] button { color: #34465c !important; font-weight: 650; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0b5d4d !important;
        border-bottom-color: #0b8068 !important;
    }
    [data-testid="stAlert"] {
        color: #13233a !important;
        background-color: #fff4cc !important;
        border: 1px solid #e2c45a !important;
        border-radius: 10px;
    }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    [data-testid="stAlert"] div {
        color: #13233a !important;
    }
    .stButton > button {
        color: #123247 !important;
        background: #ffffff !important;
        border: 1px solid #b9c8d8 !important;
    }
    .stButton > button[kind="primary"] {
        color: #ffffff !important;
        background: #126b5b !important;
        border-color: #126b5b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text, low_memory=False)


@st.cache_data(show_spinner=False)
def load_source_catalog(path_text: str) -> pd.DataFrame:
    return load_opportunity_catalog(Path(path_text))


def load_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(
            f"Required data file not found: `{path.relative_to(ROOT)}`. "
            "Run `python src/pmis_pipeline.py` to generate the research dataset."
        )
        st.stop()
    try:
        return load_csv(str(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read `{path.relative_to(ROOT)}`: {error}")
        st.stop()


def upload_candidate_roster(upload) -> pd.DataFrame:
    if upload.size > 5 * 1024 * 1024:
        raise ValueError("Candidate CSV upload is limited to 5 MB")
    try:
        roster = pd.read_csv(upload, low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"Candidate CSV could not be parsed: {error}") from error
    return validate_candidate_roster(roster)


def pretty_provenance(frame: pd.DataFrame, column: str) -> str:
    if column not in frame or frame.empty:
        return "Not recorded"
    values = frame[column].dropna().astype(str)
    return ", ".join(sorted(values.unique())) if len(values) else "Not recorded"


candidates = load_required(OUT / "clean" / "synthetic_candidates.csv")
internships = load_required(OUT / "clean" / "synthetic_internships.csv")
pairs = load_required(OUT / "clean" / "synthetic_clean_pairs.csv")
try:
    source_path = ROOT / "data" / "source" / "PMIS_MyBharat_Qualification_District_Mapping.xlsx"
    source_opportunities = load_source_catalog(str(source_path))
except (FileNotFoundError, ImportError, OSError, ValueError) as error:
    st.error(f"Could not load the observed source internship catalog: {error}")
    st.stop()
recommendations_path = EXP / "ranked_recommendations.csv"
allocation_path = EXP / "capacity_aware_allocations.csv"
metrics_path = EXP / "model_metrics.csv"
quality_path = EXP / "data_quality_metrics.csv"
cv_path = EXP / "grouped_cross_validation.csv"

recommendations = load_csv(str(recommendations_path)) if recommendations_path.exists() else pd.DataFrame()
saved_allocations = load_csv(str(allocation_path)) if allocation_path.exists() else pd.DataFrame()
metrics = load_csv(str(metrics_path)) if metrics_path.exists() else pd.DataFrame()
quality = load_csv(str(quality_path)) if quality_path.exists() else pd.DataFrame()
grouped_cv = load_csv(str(cv_path)) if cv_path.exists() else pd.DataFrame()

if "candidate_roster" not in st.session_state:
    st.session_state["candidate_roster"] = candidates.copy()
    st.session_state["roster_uploaded"] = False
if "match_results" not in st.session_state:
    st.session_state["match_results"] = pd.DataFrame()
if "allocation_results" not in st.session_state:
    st.session_state["allocation_results"] = saved_allocations.copy()
if "allocation_summary" not in st.session_state:
    st.session_state["allocation_summary"] = {}
if "equity_policy" not in st.session_state:
    st.session_state["equity_policy"] = {}
if "ignore_candidate_upload" not in st.session_state:
    st.session_state["ignore_candidate_upload"] = False

if "allocation_summary" not in st.session_state or not st.session_state["allocation_summary"]:
    validation_path = EXP / "allocation_validation.json"
    saved_summary = {}
    if validation_path.exists():
        try:
            saved_summary = json.loads(validation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            saved_summary = {}
    allocated_ids = (
        set(saved_allocations["candidate_id"].astype(str))
        if not saved_allocations.empty and "candidate_id" in saved_allocations
        else set()
    )
    scenario_candidates = set(pairs["candidate_id"].astype(str)) if "candidate_id" in pairs else set()
    scenario_internships = pairs.drop_duplicates("internship_id").set_index("internship_id")["capacity"]
    saved_count = len(allocated_ids)
    saved_summary.update({
        "allocations": saved_count,
        "unique_candidates_allocated": saved_count,
        "unallocated_candidates": len(scenario_candidates - allocated_ids),
        "total_seats": int(pd.to_numeric(scenario_internships, errors="coerce").fillna(0).sum()),
    })
    saved_summary["unused_seats"] = max(0, saved_summary["total_seats"] - saved_count)
    saved_summary["seat_utilization"] = (
        saved_count / saved_summary["total_seats"] if saved_summary["total_seats"] else 0.0
    )
    saved_summary["allocation_rate"] = (
        saved_count / len(scenario_candidates) if scenario_candidates else 0.0
    )
    if not saved_allocations.empty and {"internship_id", "capacity"}.issubset(saved_allocations):
        counts = saved_allocations.groupby("internship_id").size()
        capacities = saved_allocations.groupby("internship_id")["capacity"].first()
        saved_summary["capacity_violations"] = int((counts > capacities).sum())
    if not saved_allocations.empty and {
        "eligible", "qualification_compatibility", "course_compatibility"
    }.issubset(saved_allocations.columns):
        valid = (
            pd.to_numeric(saved_allocations["eligible"], errors="coerce").eq(1)
            & pd.to_numeric(saved_allocations["qualification_compatibility"], errors="coerce").eq(1)
            & pd.to_numeric(saved_allocations["course_compatibility"], errors="coerce").eq(1)
        )
        saved_summary["eligibility_violations"] = int((~valid).sum())
    else:
        saved_summary["eligibility_violations"] = 0
    st.session_state["allocation_summary"] = saved_summary

roster = st.session_state["candidate_roster"]
current_allocations = st.session_state["allocation_results"]

st.title("PM Internship Scheme")
st.caption("AI-Based Smart Allocation Engine · research prototype")
st.warning(
    "This prototype uses observed internship listings plus synthetic candidate profiles, "
    "matches, and proxy labels. Its recommendations are not real applicant predictions or "
    "government selection decisions. Do not use it to select or reject real applicants.",
    icon="⚠️",
)

overview_tab, opportunities_tab, matching_tab, allocation_tab, fairness_tab, explain_tab, evaluation_tab = st.tabs(
    [
        "Overview",
        "Internship Opportunities",
        "Candidate Matching",
        "Smart Allocation",
        "Fairness & Equity",
        "Explainability",
        "Model Evaluation",
    ]
)

with overview_tab:
    st.subheader("Prototype at a glance")
    assigned_count = int(current_allocations["candidate_id"].nunique()) if not current_allocations.empty and "candidate_id" in current_allocations else 0
    available_seats = int(pd.to_numeric(internships["capacity"], errors="coerce").fillna(0).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Candidate profiles", f"{len(roster):,}")
    c2.metric("Source internship listings", f"{len(source_opportunities):,}")
    c3.metric("Candidate–internship pairs", f"{len(pairs):,}")
    c4.metric("Current scenario assignments", f"{assigned_count:,}")
    st.markdown("#### What is connected today")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            "- Browse source-derived internship attributes and recorded capacities.\n"
            "- Score a candidate against the source internship catalog using transparent compatibility rules.\n"
            "- Re-run a capacity-constrained assignment with optional explicit group minimums.\n"
            "- Inspect synthetic proxy-label evaluation and auditable data-quality experiments."
        )
        st.info(
            f"The generated allocation scenario contains **{available_seats:,}** recorded seats. "
            "This is the sampled research scenario, not a count of current PMIS vacancies."
        )
    with right:
        st.markdown("#### Data provenance")
        st.write(
            "**Opportunity listings:** " + pretty_provenance(source_opportunities, "record_origin")
        )
        st.write("**Candidate profiles:** " + pretty_provenance(roster, "record_origin"))
        st.write(
            "**Match and label records:** generated research data. "
            "Evaluation labels are synthetic heuristic proxy labels."
        )
        st.caption(
            "The supplied workbook contains internship opportunities and aggregate counts, "
            "not candidate-level applications or observed selection outcomes."
        )
    st.markdown("#### Methodology")
    st.write(
        "Eligibility is a separate qualification/course gate. Eligible pairs receive a "
        "weighted rule-based compatibility score; allocation maximizes total score subject "
        "to one assignment per candidate and each opportunity's capacity."
    )

with opportunities_tab:
    st.subheader("Internship opportunities")
    filters = st.columns(4)
    state_values = sorted(source_opportunities["state"].dropna().astype(str).unique())
    sector_values = sorted(source_opportunities["sector"].dropna().astype(str).unique())
    qualification_values = sorted(source_opportunities["qualification"].dropna().astype(str).unique())
    state_filter = filters[0].selectbox("State / UT", ["All"] + state_values, key="opp_state")
    sector_filter = filters[1].selectbox("Sector", ["All"] + sector_values, key="opp_sector")
    qualification_filter = filters[2].selectbox(
        "Qualification", ["All"] + qualification_values, key="opp_qualification"
    )
    company_search = filters[3].text_input("Company contains", key="opp_company").strip().casefold()
    opportunity_view = source_opportunities.copy()
    if state_filter != "All":
        opportunity_view = opportunity_view[opportunity_view["state"].astype(str) == state_filter]
    if sector_filter != "All":
        opportunity_view = opportunity_view[opportunity_view["sector"].astype(str) == sector_filter]
    if qualification_filter != "All":
        opportunity_view = opportunity_view[
            opportunity_view["qualification"].astype(str) == qualification_filter
        ]
    if company_search:
        opportunity_view = opportunity_view[
            opportunity_view["company"].astype(str).str.casefold().str.contains(company_search, na=False)
        ]
    st.caption(f"{len(opportunity_view):,} records match the selected filters.")
    opportunity_columns = [
        column for column in [
            "internship_id", "source_internship_id", "company", "state", "district",
            "sector", "area", "qualification", "course", "specialization", "capacity",
        ]
        if column in opportunity_view
    ]
    if opportunity_view.empty:
        st.info("No opportunities match these filters. Clear or adjust a filter to continue.")
    else:
        st.dataframe(opportunity_view[opportunity_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered opportunities",
            opportunity_view[opportunity_columns].to_csv(index=False).encode("utf-8"),
            file_name="pmis_filtered_opportunities.csv",
            mime="text/csv",
        )
    st.caption(
        "These rows retain observed source internship IDs and attributes. The workbook's "
        "aggregate application counts are not individual applicant records."
    )

with matching_tab:
    st.subheader("Candidate-to-internship matching")
    st.caption("Scoring engine: documented rule-based baseline · no fitted model is loaded for live matching.")
    st.warning(
        "Optional uploads are session-only in this dashboard. Use synthetic or properly de-identified "
        "research examples; this prototype is not approved for real applicant decisions."
    )
    upload = st.file_uploader(
        "Upload candidate CSV (optional; max 250 candidates)",
        type=["csv"],
        help="Required columns: candidate_id, qualification, course. Optional: specialization, skills, sector_interests, preferred_state, preferred_district.",
        key="candidate_csv_upload",
    )
    if upload is None:
        st.session_state["ignore_candidate_upload"] = False
    ignore_upload = st.session_state["ignore_candidate_upload"]
    if upload is not None and not ignore_upload:
        try:
            uploaded_roster = upload_candidate_roster(upload)
            st.session_state["candidate_roster"] = uploaded_roster
            st.session_state["roster_uploaded"] = True
            roster = uploaded_roster
            st.success(f"Validated {len(uploaded_roster):,} uploaded candidate rows for this session.")
        except ValueError as error:
            st.error(str(error))
    if st.button("Use generated demonstration candidates", key="restore_demo_roster"):
        st.session_state["candidate_roster"] = candidates.copy()
        st.session_state["roster_uploaded"] = False
        st.session_state["match_results"] = pd.DataFrame()
        st.session_state["ignore_candidate_upload"] = True
        roster = candidates
        st.rerun()
    if roster.empty:
        st.info("No candidate profiles are available. Upload a valid candidate CSV to begin.")
    else:
        candidate_ids = sorted(roster["candidate_id"].astype(str).unique())
        selected_candidate = st.selectbox("Candidate ID", candidate_ids, key="selected_match_candidate")
        profile = roster.loc[roster["candidate_id"].astype(str) == selected_candidate].iloc[0]
        st.write(
            f"**Profile:** {profile['qualification']} · {profile['course']} · "
            f"{profile.get('home_district', 'location not supplied')}, "
            f"{profile.get('home_state', 'location not supplied')} · "
            f"Interests: {profile.get('sector_interests', 'not supplied')}"
        )
        if st.button("Generate recommendations", type="primary", key="generate_matches"):
            with st.spinner("Scoring this candidate against the internship sample…"):
                try:
                    st.session_state["match_results"] = match_candidates(
                        profile.to_frame().T, source_opportunities
                    )
                    st.session_state["matched_candidate_id"] = selected_candidate
                except (ValueError, TypeError) as error:
                    st.error(f"Could not generate recommendations: {error}")
        results = st.session_state["match_results"]
        if results.empty or st.session_state.get("matched_candidate_id") != selected_candidate:
            st.info("Generate recommendations to view eligibility and ranked alternatives.")
        else:
            minimum_score = st.slider(
                "Minimum compatibility score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="match_minimum_score",
            )
            show_ineligible = st.checkbox("Include ineligible alternatives", value=False)
            shown = results[results["synthetic_match_score"] >= minimum_score].copy()
            if not show_ineligible:
                shown = shown[shown["eligible"] == 1]
            st.caption(
                f"{int(results['eligible'].sum()):,} eligible of {len(results):,} scored opportunities."
            )
            if shown.empty:
                st.info("No alternatives meet the selected eligibility and score filters.")
            else:
                display_columns = [
                    column for column in [
                        "recommendation_rank", "internship_id", "internship_company",
                        "internship_sector", "internship_state", "internship_district",
                        "eligible", "eligibility_explanation", "synthetic_match_score",
                        *MATCH_WEIGHTS.keys(), "why_recommended",
                    ]
                    if column in shown.columns
                ]
                st.dataframe(shown[display_columns], use_container_width=True, hide_index=True)
                st.download_button(
                    "Download recommendations",
                    shown[display_columns].to_csv(index=False).encode("utf-8"),
                    file_name=f"recommendations_{selected_candidate}.csv",
                    mime="text/csv",
                )

with allocation_tab:
    st.subheader("Capacity-constrained smart allocation")
    st.write(
        "The optimizer maximizes summed compatibility scores. Each candidate can receive "
        "at most one placement; each internship is capped at its recorded capacity."
    )
    use_uploaded = bool(st.session_state.get("roster_uploaded", False))
    source_label = (
        "Uploaded roster × internship sample"
        if use_uploaded
        else "Generated 12,000-pair research scenario"
    )
    st.caption(f"Allocation input: **{source_label}**")
    minimum_allocation_score = st.slider(
        "Minimum compatibility score to consider",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="An explicit score cutoff; pairs below it are excluded from assignment consideration.",
        key="allocation_minimum_score",
    )
    policy = st.session_state.get("equity_policy", {})
    if policy:
        st.info(
            f"Configured equity constraint: at least {policy['minimum']} allocation(s) "
            f"from `{policy['group_column']} = {policy['group_value']}`. "
            "This explicit constraint does not alter compatibility scores."
        )
    if st.button("Run allocation", type="primary", key="run_allocation"):
        with st.spinner("Solving the capacity-constrained allocation…"):
            try:
                if use_uploaded:
                    group_column = policy.get("group_column")
                    scenario_pairs = match_candidates(
                        st.session_state["candidate_roster"],
                        internships,
                        policy_group_column=group_column,
                    )
                else:
                    scenario_pairs = pairs.copy()
                scenario_pairs.loc[
                    scenario_pairs["synthetic_match_score"] < minimum_allocation_score,
                    "eligible",
                ] = 0
                minimums = {}
                if use_uploaded and policy:
                    minimums = {str(policy["group_value"]): int(policy["minimum"])}
                with tempfile.TemporaryDirectory(prefix="pmis-allocation-") as temp_dir:
                    output = Path(temp_dir) / "allocations.csv"
                    summary = allocate(
                        scenario_pairs,
                        output,
                        time_limit=30.0,
                        policy_group_column=policy.get("group_column") if minimums else None,
                        policy_minimums=minimums,
                    )
                    result = pd.read_csv(output)
                st.session_state["allocation_results"] = result
                st.session_state["allocation_summary"] = summary
                st.session_state["allocation_input"] = source_label
                st.success(
                    f"Allocation completed with {summary['solver']} "
                    f"({summary['solver_status']})."
                )
            except (ValueError, RuntimeError, OSError) as error:
                st.error(f"Allocation could not be completed: {error}")
    allocation_results = st.session_state["allocation_results"]
    summary = st.session_state.get("allocation_summary", {})
    if not allocation_results.empty:
        metric_columns = st.columns(5)
        metric_columns[0].metric("Assignments", f"{len(allocation_results):,}")
        metric_columns[1].metric("Unallocated candidates", f"{summary.get('unallocated_candidates', '—')}")
        metric_columns[2].metric("Unused seats", f"{summary.get('unused_seats', '—')}")
        metric_columns[3].metric("Capacity violations", f"{summary.get('capacity_violations', '—')}")
        metric_columns[4].metric("Eligibility violations", f"{summary.get('eligibility_violations', '—')}")
        st.caption(
            f"Seat utilization: {summary.get('seat_utilization', 0):.1%} · "
            f"Candidate allocation rate: {summary.get('allocation_rate', 0):.1%} · "
            f"Objective score sum: {summary.get('total_match_score', 0):.3f}"
        )
        state_values = ["All"] + sorted(
            allocation_results.get("internship_state", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        sector_values = ["All"] + sorted(
            allocation_results.get("internship_sector", pd.Series(dtype=str)).dropna().astype(str).unique()
        )
        state_choice, sector_choice = st.columns(2)
        selected_state = state_choice.selectbox("Allocated internship state", state_values, key="alloc_state_filter")
        selected_sector = sector_choice.selectbox("Allocated sector", sector_values, key="alloc_sector_filter")
        allocation_view = allocation_results.copy()
        if selected_state != "All" and "internship_state" in allocation_view:
            allocation_view = allocation_view[allocation_view["internship_state"].astype(str) == selected_state]
        if selected_sector != "All" and "internship_sector" in allocation_view:
            allocation_view = allocation_view[allocation_view["internship_sector"].astype(str) == selected_sector]
        allocation_columns = [
            column for column in [
                "candidate_id", "internship_id", "internship_company", "internship_sector",
                "internship_state", "synthetic_match_score", "capacity",
                "capacity_at_limit", "allocation_explanation",
            ]
            if column in allocation_view
        ]
        st.dataframe(allocation_view[allocation_columns], use_container_width=True, hide_index=True)
        st.download_button(
            "Download allocation result",
            allocation_results.to_csv(index=False).encode("utf-8"),
            file_name="pmis_allocation_scenario.csv",
            mime="text/csv",
        )
    else:
        st.info(
            "No current allocation result. Run the optimizer to create one. "
            "If there are no eligible pairs, all candidates remain unallocated."
        )

with fairness_tab:
    st.subheader("Fairness & equity")
    st.info(
        "No policy group or authoritative equity rule is present in the checked-in candidate data. "
        "Group fairness is therefore not evaluated for the generated baseline scenario."
    )
    allowed_group_columns = [
        column for column in roster.columns
        if column not in NON_GROUP_COLUMNS and column not in FEATURE_COLUMNS
        and not column.lower().endswith(("id", "_id"))
    ]
    if not st.session_state.get("roster_uploaded", False) or not allowed_group_columns:
        st.write(
            "To evaluate a real, authorized policy scenario, upload candidate records containing "
            "an explicitly approved group field. The field will be carried only into policy "
            "constraints and descriptive reporting; it is not used in match scoring."
        )
        if not st.session_state.get("roster_uploaded", False):
            st.caption("The generated profiles contain no demographic or policy-group attributes.")
    else:
        st.markdown("#### Configure one auditable minimum-representation constraint")
        selected_group_column = st.selectbox(
            "Approved group field",
            allowed_group_columns,
            key="equity_group_column",
        )
        group_values = sorted(roster[selected_group_column].dropna().astype(str).unique())
        if not group_values:
            st.warning("The selected group field has no non-missing values.")
        else:
            selected_group_value = st.selectbox(
                "Group receiving the minimum", group_values, key="equity_group_value"
            )
            group_count = int(
                roster[selected_group_column].astype(str).eq(selected_group_value).sum()
            )
            minimum = st.number_input(
                "Minimum assignments from this group",
                min_value=0,
                max_value=group_count,
                value=0,
                step=1,
                key="equity_minimum",
            )
            if st.button("Save explicit policy constraint", key="save_equity_policy"):
                st.session_state["equity_policy"] = {
                    "group_column": selected_group_column,
                    "group_value": selected_group_value,
                    "minimum": int(minimum),
                    "available_group_candidates": group_count,
                } if minimum else {}
                st.success(
                    "Policy constraint saved for the next uploaded-roster allocation."
                    if minimum
                    else "No minimum constraint saved; matching scores remain unchanged."
                )
        policy = st.session_state.get("equity_policy", {})
        if policy:
            st.write(
                f"**Active policy:** minimum {policy['minimum']} allocations for "
                f"`{policy['group_column']} = {policy['group_value']}` "
                f"({policy['available_group_candidates']} candidate(s) available)."
            )
            if st.button("Clear saved policy", key="clear_equity_policy"):
                st.session_state["equity_policy"] = {}
                st.rerun()
        allocations_now = st.session_state["allocation_results"]
        if (
            selected_group_column in allocations_now.columns
            and not allocations_now.empty
            and "candidate_id" in allocations_now
        ):
            try:
                fairness = allocation_fairness_summary(
                    roster, allocations_now, selected_group_column
                )
                st.markdown("#### Descriptive allocation rates")
                st.dataframe(fairness, use_container_width=True, hide_index=True)
                st.caption(
                    "Group rate differences are descriptive diagnostics, not proof of fairness, "
                    "legal compliance, causality, or improved outcomes."
                )
            except ValueError as error:
                st.warning(f"Group metrics are unavailable: {error}")

with explain_tab:
    st.subheader("How each recommendation is produced")
    st.write(
        "The live matcher is a rule-based baseline, not a trained predictive model. "
        "Eligibility is a hard filter; the weighted compatibility score is calculated only "
        "after that gate and does not use group attributes."
    )
    weights = pd.DataFrame(
        [{"component": key.replace("_compatibility", "").replace("_", " "),
          "weight": value, "meaning": {
              "qualification_compatibility": "Qualification level meets or exceeds the stated requirement.",
              "course_compatibility": "Course matches the requirement or the posting accepts ANY / All Subjects.",
              "specialization_compatibility": "Exact specialization match; missing/wildcard requirements are compatible.",
              "skill_similarity": "Jaccard token overlap between candidate skills and posting course/area/sector terms.",
              "sector_compatibility": "Posting sector appears in the candidate's stated sector interests.",
              "location_compatibility": "0.55 same preferred state plus 0.45 same preferred district.",
          }[key]}
         for key, value in MATCH_WEIGHTS.items()]
    )
    st.dataframe(weights, use_container_width=True, hide_index=True)
    st.caption("Weights total 1.00. Missing preferences are not inferred as a positive location/sector match.")
    st.markdown("#### Hard eligibility rules")
    st.markdown(
        "- Candidate qualification must meet the internship qualification level.\n"
        "- Candidate course must match the required course unless the posting lists `ANY` or `All Subjects`.\n"
        "- An ineligible pair is excluded from allocation even if its incoming score is high.\n"
        "- Allocation adds one-placement-per-candidate and per-internship capacity constraints."
    )
    match_results = st.session_state["match_results"]
    if not match_results.empty:
        st.markdown("#### Feature-level explanation for the current candidate")
        display_columns = [
            column for column in [
                "internship_id", "eligible", "eligibility_explanation",
                *MATCH_WEIGHTS.keys(), "synthetic_match_score", "why_recommended",
            ]
            if column in match_results
        ]
        st.dataframe(match_results[display_columns].head(20), use_container_width=True, hide_index=True)
    elif not recommendations.empty:
        st.markdown("#### Example from the saved synthetic recommendation run")
        if "why_recommended" in recommendations:
            st.dataframe(
                recommendations[
                    [column for column in [
                        "candidate_id", "internship_id", "eligible", "synthetic_match_score",
                        *MATCH_WEIGHTS.keys(), "why_recommended", "why_alternative_lower",
                    ]
                     if column in recommendations]
                ].head(10),
                use_container_width=True,
                hide_index=True,
            )
    st.warning(
        "A score is not a probability of selection. The heuristic and generated proxy labels "
        "are not evidence of candidate merit or real PMIS outcomes."
    )

with evaluation_tab:
    st.subheader("Model comparison · synthetic proxy-label agreement")
    st.warning(
        "All checked-in evaluation targets are generated heuristic proxy labels. They are not "
        "observed PMIS decisions, applicant outcomes, or measured selection accuracy."
    )
    clean_metrics = (
        metrics[metrics["dataset"].astype(str) == "clean"].copy()
        if not metrics.empty and {"dataset", "model"}.issubset(metrics.columns)
        else pd.DataFrame()
    )
    if clean_metrics.empty:
        st.info("No evaluation metrics found. Run `python src/pmis_pipeline.py` to reproduce them.")
    else:
        selected_metric = st.selectbox(
            "Compare metric",
            [column for column in [
                "precision", "recall", "f1", "pr_auc", "roc_auc",
                "ndcg_at_5", "ndcg_at_10", "ndcg_at_full_list",
                "train_f1", "validation_f1",
                "fit_seconds", "inference_ms_per_row",
            ] if column in clean_metrics],
            key="evaluation_metric",
        )
        chart = clean_metrics[["model", selected_metric]].set_index("model")
        st.bar_chart(chart, y=selected_metric, height=320)
        metric_columns = [
            column for column in [
                "model", "precision", "recall", "f1", "pr_auc", "roc_auc",
                "ndcg_at_full_list", "train_f1", "validation_f1",
                "fit_seconds", "inference_ms_per_row", "label_note",
            ]
            if column in clean_metrics
        ]
        st.dataframe(clean_metrics[metric_columns], use_container_width=True, hide_index=True)
        if not {"ndcg_at_5", "ndcg_at_10"}.issubset(clean_metrics.columns):
            st.info(
                "The checked-in evaluation report predates NDCG@5/@10 output. "
                "Rerun `python src/pmis_pipeline.py` to reproduce metrics with those ranking cutoffs."
            )
    if not grouped_cv.empty:
        st.markdown("#### Candidate-grouped cross-validation")
        st.dataframe(grouped_cv, use_container_width=True, hide_index=True)
        st.caption(
            "Candidate IDs are kept together across each grouped fold. Preprocessing is fitted "
            "inside each model pipeline on training-fold data."
        )
    if not quality.empty:
        st.markdown("#### Data-quality experiments")
        quality_columns = [
            column for column in [
                "dataset", "rows", "missing_cells", "duplicate_pair_ids",
                "non_numeric_capacity_values", "application_capacity_ratio_gt_10",
                "unknown_category_values", "out_of_range_skill_similarity",
                "proxy_positive_rate",
            ]
            if column in quality
        ]
        st.dataframe(quality[quality_columns], use_container_width=True, hide_index=True)
        st.caption(
            "Application/capacity ratios above 10 are a diagnostic from the experiment, not an "
            "official validity threshold; source aggregate semantics are unconfirmed."
        )
    validation_path = EXP / "allocation_validation.json"
    if validation_path.exists():
        try:
            allocation_validation = json.loads(validation_path.read_text(encoding="utf-8"))
            st.markdown("#### Saved allocation validation")
            st.json(allocation_validation)
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not read allocation validation summary: {error}")
    saved_allocation_summary = st.session_state.get("allocation_summary", {})
    if saved_allocation_summary:
        st.markdown("#### Allocation constraint checks")
        st.dataframe(
            pd.DataFrame([
                {"constraint": "Capacity violations", "count": saved_allocation_summary.get("capacity_violations", 0)},
                {"constraint": "Eligibility violations", "count": saved_allocation_summary.get("eligibility_violations", 0)},
            ]),
            use_container_width=True,
            hide_index=True,
        )
    st.caption(
        "The repository also includes grouped cross-validation and unseen-internship diagnostics. "
        "No saved fitted model is used for live matching; no observed outcome benchmark is available."
    )
