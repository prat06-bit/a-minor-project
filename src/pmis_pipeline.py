"""Reproducible, synthetic-only PMIS matching research pipeline.

The supplied workbook is used as a reference catalog. It has no candidate-level
outcomes, so this module never treats aggregate application counts as labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source" / "PMIS_MyBharat_Qualification_District_Mapping.xlsx"
DEFAULT_CONFIG = ROOT / "config.json"
NUMERIC_FEATURES = [
    "qualification_compatibility", "course_compatibility",
    "specialization_compatibility", "skill_similarity",
    "sector_compatibility", "location_compatibility", "capacity_log",
    "past_participation",
]
CATEGORICAL_FEATURES = [
    "candidate_qualification", "internship_qualification", "candidate_course",
    "internship_course", "candidate_primary_sector", "internship_sector",
    "candidate_preferred_state", "internship_state", "candidate_preferred_district",
    "internship_district",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MATCH_WEIGHTS = {
    "qualification_compatibility": .25,
    "course_compatibility": .17,
    "specialization_compatibility": .08,
    "skill_similarity": .18,
    "sector_compatibility": .14,
    "location_compatibility": .18,
}


def read_config(path: Path = DEFAULT_CONFIG) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_sources(source: Path = SOURCE):
    if not source.exists():
        raise FileNotFoundError(f"Source workbook not found: {source}. Copy the supplied workbook to data/source.")
    try:
        book = pd.ExcelFile(source)
        main = pd.read_excel(source, sheet_name="Main")
        states = pd.read_excel(source, sheet_name="States")
        districts = pd.read_excel(source, sheet_name="Districts")
        return book.sheet_names, main, states, districts
    except ImportError:
        # CSV extracts are deterministic read-only projections created during
        # source audit. This lets the pipeline run where an XLSX reader is absent.
        extract_dir = source.parent / "sheets_csv"
        needed = [extract_dir / f"{name}.csv" for name in ("Main", "States", "Districts")]
        if not all(path.exists() for path in needed):
            raise
        main, states, districts = [pd.read_csv(path) for path in needed]
        return ["Main", "States", "Districts"], main, states, districts


def load_opportunity_catalog(source: Path = SOURCE) -> pd.DataFrame:
    """Return validated source internship rows without assigning synthetic IDs."""
    _, main, _, _ = load_sources(source)
    required = {
        "Internship ID", "Company/Network Entity Name", "State / UT", "District",
        "Sector", "Area/Field", "Qualification", "Course", "No Of Opportunities",
    }
    missing = required - set(main.columns)
    if missing:
        raise ValueError(f"Source internship catalog is missing columns: {sorted(missing)}")
    identifiers = main["Internship ID"]
    if identifiers.isna().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError("Source internship IDs must be present and non-empty")
    if identifiers.astype(str).duplicated().any():
        raise ValueError("Source internship IDs must be unique")
    capacity = pd.to_numeric(main["No Of Opportunities"], errors="coerce")
    if capacity.isna().any() or not np.isfinite(capacity).all():
        raise ValueError("Source internship capacities must be finite numbers")
    if (capacity < 1).any() or not np.equal(capacity, np.floor(capacity)).all():
        raise ValueError("Source internship capacities must be positive whole numbers")
    for column in ("Company/Network Entity Name", "State / UT", "District", "Sector", "Qualification", "Course"):
        if main[column].isna().any() or main[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Source internship `{column}` values must be present")
    applications = np.nan
    if "Applications Received" in main:
        applications = pd.to_numeric(main["Applications Received"], errors="coerce")
        if applications.isna().any() or not np.isfinite(applications).all():
            raise ValueError("Source application totals must be finite numbers")
        if (applications < 0).any() or not np.equal(applications, np.floor(applications)).all():
            raise ValueError("Source application totals must be non-negative whole numbers")
    catalog = pd.DataFrame({
        "internship_id": identifiers.astype(str),
        "source_internship_id": identifiers.astype(str),
        "company": main["Company/Network Entity Name"],
        "state": main["State / UT"],
        "district": main["District"],
        "sector": main["Sector"],
        "area": main["Area/Field"],
        "qualification": main["Qualification"],
        "course": main["Course"],
        "specialization": main["Specialization"] if "Specialization" in main else np.nan,
        "capacity": capacity.astype(int),
        "source_applications_received": applications,
        "record_origin": "observed_source_opportunity",
        "source_row_number": np.arange(2, len(main) + 2),
    })
    catalog["provenance_json"] = catalog["source_row_number"].map(
        lambda row: json.dumps(
            {
                "record_origin": "observed_source_opportunity",
                "source_workbook": source.name,
                "source_sheet": "Main",
                "source_row_number": int(row),
                "capacity": "source No Of Opportunities field",
                "applications_received": "aggregate source count; not a candidate record",
            },
            sort_keys=True,
        )
    )
    return catalog


def validate_candidate_roster(candidates: pd.DataFrame, max_candidates: int = 250) -> pd.DataFrame:
    """Validate an uploaded roster without persisting or mixing it with generated data."""
    required = {"candidate_id", "qualification", "course"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate roster is missing required columns: {sorted(missing)}")
    if candidates.empty:
        raise ValueError("Candidate roster contains no rows")
    if len(candidates) > max_candidates:
        raise ValueError(f"Candidate roster exceeds the {max_candidates}-candidate upload limit")
    if candidates["candidate_id"].isna().any() or candidates["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate IDs must be present and non-empty")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise ValueError("Candidate IDs must be unique")
    for column in ("qualification", "course"):
        if candidates[column].isna().any() or candidates[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Candidate `{column}` values must be present and non-empty")
    roster = candidates.copy()
    roster["candidate_id"] = roster["candidate_id"].astype(str)
    if "past_participation" in roster:
        participation = pd.to_numeric(roster["past_participation"], errors="coerce")
        invalid = roster["past_participation"].notna() & participation.isna()
        if invalid.any() or not participation.dropna().isin([0, 1]).all():
            raise ValueError("past_participation must contain only 0 or 1 when provided")
        roster["past_participation"] = participation.fillna(0).astype(int)
    roster["record_origin"] = "user_uploaded_candidate"
    return roster


def _words(value) -> set[str]:
    if pd.isna(value):
        return set()
    return set(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _compat(candidate, required, wildcard=()) -> float:
    if pd.isna(required) or str(required).strip().casefold() in {"", "nan"}:
        return 1.0
    rv = str(required).strip().casefold()
    if rv in {str(x).casefold() for x in wildcard}:
        return 1.0
    return float(not pd.isna(candidate) and str(candidate).strip().casefold() == rv)


def _qualification_level(q) -> int:
    levels = {"10th": 1, "12th": 2, "iti": 3, "diploma": 4, "graduation": 5, "post graduation": 6}
    return levels.get(str(q).strip().casefold(), 0)


def _qualification_eligible(candidate, required) -> bool:
    candidate_level = _qualification_level(candidate)
    required_text = str(required).strip().casefold()
    required_level = _qualification_level(required)
    if candidate_level == 0:
        return False
    if required_text in {"any", "any qualification", "all"}:
        return True
    return required_level > 0 and candidate_level >= required_level


def match_candidates(candidates: pd.DataFrame, internships: pd.DataFrame,
                     policy_group_column: str | None = None) -> pd.DataFrame:
    """Score candidates against opportunities using the documented rule baseline."""
    candidate_required = {"candidate_id", "qualification", "course"}
    internship_required = {
        "internship_id", "qualification", "course", "capacity", "state",
        "district", "sector", "area",
    }
    missing_candidates = candidate_required - set(candidates.columns)
    missing_internships = internship_required - set(internships.columns)
    if missing_candidates or missing_internships:
        raise ValueError(
            f"Missing matching columns; candidates: {sorted(missing_candidates)}, "
            f"internships: {sorted(missing_internships)}"
        )
    if candidates["candidate_id"].isna().any() or candidates["candidate_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Candidate IDs must be present and non-empty")
    if candidates["candidate_id"].astype(str).duplicated().any():
        raise ValueError("Candidate IDs must be unique")
    candidates = candidates.copy()
    if "past_participation" not in candidates:
        candidates["past_participation"] = 0
    else:
        participation = pd.to_numeric(candidates["past_participation"], errors="coerce")
        invalid_participation = candidates["past_participation"].notna() & participation.isna()
        if invalid_participation.any() or not participation.dropna().isin([0, 1]).all():
            raise ValueError("past_participation must contain only 0 or 1 when provided")
        candidates["past_participation"] = participation.fillna(0).astype(int)
    if internships["internship_id"].isna().any() or internships["internship_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Internship IDs must be present and non-empty")
    if internships["internship_id"].astype(str).duplicated().any():
        raise ValueError("Internship IDs must be unique")
    if policy_group_column:
        if policy_group_column not in candidates.columns:
            raise ValueError(f"Policy group column not found: {policy_group_column}")
        if candidates[policy_group_column].isna().any():
            raise ValueError("Policy group values must be present for every candidate")

    rows = []
    for _, candidate in candidates.iterrows():
        interests = {
            item.strip().casefold()
            for item in str(candidate.get("sector_interests", "")).split(";")
            if item.strip()
        }
        for _, internship in internships.iterrows():
            qualification = float(
                _qualification_eligible(candidate["qualification"], internship["qualification"])
            )
            course = _compat(candidate["course"], internship["course"], ("ANY", "All Subjects"))
            specialization = _compat(
                candidate.get("specialization", np.nan),
                internship.get("specialization", np.nan),
            )
            candidate_skills = _words(candidate.get("skills", ""))
            internship_skills = _words(
                f"{internship.get('course', '')} {internship.get('specialization', '')} "
                f"{internship.get('area', '')} {internship.get('sector', '')}"
            )
            skill = len(candidate_skills & internship_skills) / max(
                1, len(candidate_skills | internship_skills)
            )
            sector = float(str(internship["sector"]).casefold() in interests)
            preferred_state = candidate.get("preferred_state", "")
            preferred_district = candidate.get("preferred_district", "")
            location = (
                .55 * float(str(internship["state"]).casefold() == str(preferred_state).casefold())
                + .45 * float(
                    str(internship["district"]).casefold() == str(preferred_district).casefold()
                )
            )
            eligible = bool(qualification and course)
            components = {
                "qualification_compatibility": qualification,
                "course_compatibility": course,
                "specialization_compatibility": specialization,
                "skill_similarity": skill,
                "sector_compatibility": sector,
                "location_compatibility": location,
            }
            score = sum(MATCH_WEIGHTS[name] * value for name, value in components.items())
            failed = []
            if not qualification:
                failed.append("qualification requirement not met")
            if not course:
                failed.append("course requirement not met")
            strong = [
                name.replace("_compatibility", "").replace("_", " ")
                for name, value in components.items()
                if value >= .75
            ]
            explanation = (
                "Eligible. Strongest compatibility: "
                + (", ".join(strong) if strong else "combined feature score")
                + f". Rule-based compatibility score {score:.3f}."
                if eligible
                else "Ineligible: " + "; ".join(failed) + "."
            )
            row = {
                "candidate_id": str(candidate["candidate_id"]),
                "internship_id": str(internship["internship_id"]),
                "source_internship_id": internship.get("source_internship_id", ""),
                "internship_company": internship.get("company", ""),
                "internship_sector": internship["sector"],
                "internship_state": internship["state"],
                "internship_district": internship["district"],
                "eligible": int(eligible),
                "eligibility_explanation": "; ".join(failed) if failed else "All hard requirements passed",
                "synthetic_match_score": round(score, 6),
                **components,
                "capacity_log": float(np.log1p(float(internship["capacity"]))),
                "past_participation": int(candidate["past_participation"]),
                "candidate_qualification": candidate["qualification"],
                "internship_qualification": internship["qualification"],
                "candidate_course": candidate["course"],
                "internship_course": internship["course"],
                "candidate_primary_sector": candidate.get("primary_sector", ""),
                "internship_sector": internship["sector"],
                "candidate_preferred_state": preferred_state,
                "internship_state": internship["state"],
                "candidate_preferred_district": preferred_district,
                "internship_district": internship["district"],
                "capacity": int(internship["capacity"]),
                "why_recommended": explanation,
                "record_origin": "rule_based_match_from_supplied_profiles",
                "provenance_json": json.dumps(
                    {
                        "score": "documented_weighted_rule_baseline",
                        "candidate": candidate.get("record_origin", "user_supplied"),
                        "internship": internship.get("record_origin", "source_derived_sample"),
                        "eligibility": "qualification_and_course_hard_filters",
                    },
                    sort_keys=True,
                ),
            }
            if policy_group_column:
                row[policy_group_column] = str(candidate[policy_group_column])
            rows.append(row)
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked.sort_values(
        ["candidate_id", "eligible", "synthetic_match_score", "internship_id"],
        ascending=[True, False, False, True],
    )
    ranked["recommendation_rank"] = (
        ranked.groupby("candidate_id").cumcount() + 1
    )
    return ranked.reset_index(drop=True)


def allocation_fairness_summary(candidates: pd.DataFrame, allocations: pd.DataFrame,
                                group_column: str) -> pd.DataFrame:
    """Measure descriptive group allocation rates without using group in merit scores."""
    if group_column not in candidates.columns:
        raise ValueError(f"Policy group column not found: {group_column}")
    if candidates["candidate_id"].duplicated().any():
        raise ValueError("Candidate IDs must be unique for group allocation metrics")
    groups = candidates[["candidate_id", group_column]].copy()
    if groups[group_column].isna().any():
        raise ValueError("Policy group values must be present to calculate group rates")
    assigned_ids = set(allocations.get("candidate_id", pd.Series(dtype=str)).astype(str))
    groups["allocated"] = groups["candidate_id"].astype(str).isin(assigned_ids)
    overall_rate = float(groups["allocated"].mean()) if len(groups) else 0.0
    counts = groups.groupby(group_column, dropna=False).agg(
        candidate_count=("candidate_id", "size"),
        allocated_count=("allocated", "sum"),
        allocation_rate=("allocated", "mean"),
    ).reset_index()
    counts["allocation_rate_difference_vs_overall"] = (
        counts["allocation_rate"] - overall_rate
    )
    counts["representation_share_of_allocations"] = (
        counts["allocated_count"] / max(1, int(groups["allocated"].sum()))
    )
    return counts


def generate(config: dict, source: Path = SOURCE, output_dir: Path | None = None) -> dict[str, Path]:
    """Generate catalog-linked synthetic candidates, internships, and pair labels."""
    _, main, _, districts = load_sources(source)
    out = output_dir or ROOT / config.get("output_dir", "data/outputs")
    clean_dir = out / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    n_c = int(config["n_candidates"]); n_i = int(config["n_internships"]); n_p = int(config["n_pairs"])
    if min(n_c, n_i, n_p) <= 0:
        raise ValueError("n_candidates, n_internships, and n_pairs must be positive")
    if n_p > n_c * n_i:
        raise ValueError("n_pairs cannot exceed n_candidates * n_internships")

    # The source tuple is sampled as a unit so course/qualification/specialization
    # combinations remain supported by the observed internship catalog.
    tuples = main[["Qualification", "Course", "Specialization"]].copy()
    tuple_idx = rng.choice(len(tuples), size=n_c, replace=True)
    combos = tuples.iloc[tuple_idx].reset_index(drop=True)
    district_rows = districts.dropna(subset=["District"]).reset_index(drop=True)
    weights = district_rows["Internships"].astype(float).to_numpy()
    weights = weights / weights.sum()
    loc_idx = rng.choice(len(district_rows), size=n_c, replace=True, p=weights)
    locations = district_rows.iloc[loc_idx].reset_index(drop=True)
    sectors = main["Sector"].dropna().value_counts(normalize=True)
    sector_names = sectors.index.to_numpy(); sector_probs = sectors.to_numpy()
    cands = []
    for j in range(n_c):
        row = combos.iloc[j]; loc = locations.iloc[j]
        interests = rng.choice(sector_names, size=int(rng.integers(1, 4)), replace=False, p=sector_probs)
        skill_text = " ".join(dict.fromkeys([str(row["Course"]), str(row["Specialization"] or "") if pd.notna(row["Specialization"]) else "", *list(interests)]))
        cands.append({
            "candidate_id": f"SYN-CAN-{j+1:07d}", "qualification": row["Qualification"],
            "course": row["Course"], "specialization": row["Specialization"],
            "skills": skill_text, "sector_interests": ";".join(interests),
            "primary_sector": interests[0], "home_state": loc["State / UT"],
            "home_district": loc["District"], "preferred_state": loc["State / UT"],
            "preferred_district": loc["District"],
            "past_participation": int(rng.random() < 0.08),
            "record_origin": "synthetic", "source_profile_row": int(tuple_idx[j] + 2),
            "generation_seed": seed, "generation_rule": "observed_tuple_weighted_profile_v1",
            "provenance_json": json.dumps({"qualification":"sampled_observed_distribution",
                "course":"sampled_observed_tuple","specialization":"sampled_observed_tuple",
                "home_location":"sampled_observed_district_distribution","skills":"generated_from_profile_tokens",
                "sector_interests":"generated_preferences","preferred_location":"generated_from_sampled_home_location",
                "past_participation":"synthetic_bernoulli_proxy"},sort_keys=True),
        })
    candidates = pd.DataFrame(cands)

    int_idx = rng.choice(len(main), size=n_i, replace=n_i>len(main))
    selected = main.iloc[int_idx].reset_index(drop=True)
    internships = pd.DataFrame({
        "internship_id": [f"SYN-INT-{j+1:07d}" for j in range(n_i)],
        "source_internship_id": selected["Internship ID"].astype(str),
        "company": selected["Company/Network Entity Name"],
        "state": selected["State / UT"], "district": selected["District"],
        "sector": selected["Sector"], "area": selected["Area/Field"],
        "qualification": selected["Qualification"], "course": selected["Course"],
        "specialization": selected["Specialization"],
        "capacity": selected["No Of Opportunities"].astype(int),
        # This is a source aggregate retained for data-quality stress scenarios,
        # not an applicant-level target or a generated outcome.
        "source_applications_received": selected["Applications Received"].astype(int),
        "record_origin": "synthetic_scenario_from_observed_opportunity",
        "source_row_number": int_idx + 2, "generation_seed": seed,
    })
    internships["provenance_json"] = json.dumps({"opportunity_attributes":"copied_from_observed_source_row",
        "internship_id":"synthetic_identifier","capacity":"copied_observed_opportunity_capacity",
        "source_applications_received":"observed_aggregate_only"},sort_keys=True)
    pair_indices = rng.choice(n_c * n_i, size=n_p, replace=False)
    c_index = pair_indices // n_i
    i_index = pair_indices % n_i
    cand_lookup = candidates.iloc[c_index].reset_index(drop=True)
    int_lookup = internships.iloc[i_index].reset_index(drop=True)
    semantic_scores=None
    if config.get("skill_similarity_backend","token_overlap")=="sentence_transformers":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install requirements-optional.txt to use Sentence-Transformers") from exc
        model=SentenceTransformer(config.get("sentence_transformer_model","sentence-transformers/all-MiniLM-L6-v2"))
        cand_embeddings=model.encode(candidates.skills.fillna("").tolist(),normalize_embeddings=True,show_progress_bar=False)
        internship_text=(internships.course.fillna("")+" "+internships.specialization.fillna("")+" "+internships.area.fillna("")+" "+internships.sector.fillna(""))
        internship_embeddings=model.encode(internship_text.tolist(),normalize_embeddings=True,show_progress_bar=False)
        semantic_scores=np.einsum("ij,ij->i",cand_embeddings[c_index],internship_embeddings[i_index])
    pairs = []
    label_threshold = float(config.get("label_threshold", .56))
    for k in range(n_p):
        c = cand_lookup.iloc[k]; it = int_lookup.iloc[k]
        q = float(_qualification_eligible(c.qualification, it.qualification))
        course = _compat(c.course, it.course, ("ANY", "All Subjects"))
        spec = _compat(c.specialization, it.specialization)
        skill = (float((semantic_scores[k]+1)/2) if semantic_scores is not None else
                 len(_words(c.skills) & _words(f"{it.course} {it.specialization} {it.area} {it.sector}")) / max(1, len(_words(c.skills) | _words(f"{it.course} {it.specialization} {it.area} {it.sector}"))))
        sector = float(it.sector in str(c.sector_interests).split(";"))
        loc = float(it.state == c.preferred_state) * .55 + float(it.district == c.preferred_district) * .45
        eligible = bool(q and course)
        ratio = int(it.source_applications_received) / max(1, int(it.capacity))
        features = {"qualification_compatibility": q, "course_compatibility": course,
                    "specialization_compatibility": spec, "skill_similarity": skill,
                    "sector_compatibility": sector, "location_compatibility": loc,
                    "capacity_log": float(np.log1p(it.capacity)), "past_participation": int(c.past_participation)}
        score = (.25*q + .17*course + .08*spec + .18*skill + .14*sector + .18*loc)
        noisy_score = score + float(rng.normal(0, .07))
        label = int(eligible and noisy_score >= label_threshold)
        pairs.append({
            "pair_id": f"SYN-PAIR-{k+1:09d}", "candidate_id": c.candidate_id,
            "internship_id": it.internship_id, "source_internship_id": it.source_internship_id,
            "internship_company": it.company,
            "eligible": int(eligible), "synthetic_match_score": round(score, 6),
            **features,
            "candidate_qualification": c.qualification, "internship_qualification": it.qualification,
            "candidate_course": c.course, "internship_course": it.course,
            "candidate_primary_sector": c.primary_sector, "internship_sector": it.sector,
            "candidate_preferred_state": c.preferred_state, "internship_state": it.state,
            "candidate_preferred_district": c.preferred_district, "internship_district": it.district,
            "capacity": int(it.capacity), "applications_received": int(it.source_applications_received),
            "record_origin": "synthetic_candidate_internship_pair",
            "generation_seed": seed, "label_rule": "eligible_and_noisy_weighted_match_score_v1",
            "synthetic_proxy_label": label,
            "provenance_json": json.dumps({"compatibility_features":"computed_from_synthetic_profile_and_observed_opportunity_attributes",
                "eligible":"inferred_rule","synthetic_match_score":"weighted_heuristic",
                "skill_similarity_backend":config.get("skill_similarity_backend","token_overlap"),
                "synthetic_proxy_label":"synthetic_heuristic_with_seeded_noise"},sort_keys=True),
        })
    pair_df = pd.DataFrame(pairs)
    files = {
        "candidates": clean_dir / "synthetic_candidates.csv",
        "internships": clean_dir / "synthetic_internships.csv",
        "pairs": clean_dir / "synthetic_clean_pairs.csv",
    }
    candidates.to_csv(files["candidates"], index=False)
    internships.to_csv(files["internships"], index=False)
    pair_df.to_csv(files["pairs"], index=False)
    manifest = {"seed": seed, "config": config, "source_workbook": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_sheets": ["Main", "States", "Districts"],
                "source_main_rows": int(len(main)), "candidate_rows": int(len(candidates)),
                "internship_rows": int(len(internships)), "pair_rows": int(len(pair_df)),
                "positive_proxy_labels": int(pair_df.synthetic_proxy_label.sum()),
                "negative_proxy_labels": int((1-pair_df.synthetic_proxy_label).sum()),
                "target_definition": "synthetic heuristic; not observed PMIS selections",
                "files": {
                    key: str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
                    for key, path in files.items()
                }}
    (out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return files


def corrupt_dataset(input_csv: Path, output_csv: Path, rate: float, seed: int) -> tuple[Path, Path]:
    """Make an auditable corruption copy. Synthetic labels remain unchanged."""
    if not 0 <= rate <= 1:
        raise ValueError("corruption rate must be between 0 and 1")
    df = pd.read_csv(input_csv, keep_default_na=True)
    original = df.copy(deep=True); rng = np.random.default_rng(seed)
    logs = []
    eligible_cols = [c for c in CATEGORICAL_FEATURES if c in df]
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df]
    def mutate(row_idx, col, value, operation):
        if col == "capacity" and not pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(object)
        old = df.at[row_idx, col]
        df.at[row_idx, col] = value
        logs.append({"pair_id": original.at[row_idx, "pair_id"], "row_index": int(row_idx), "column": col,
                     "original_value": "<NA>" if pd.isna(old) else str(old),
                     "corrupted_value": "<NA>" if pd.isna(value) else str(value),
                     "transformation": operation, "seed": seed, "corruption_rate": rate})
    n = int(round(len(df) * rate))
    for col in eligible_cols:
        for idx in rng.choice(len(df), size=n, replace=False) if n else []:
            val = df.at[idx, col]
            if pd.notna(val): mutate(idx, col, f"  {str(val).swapcase()}  ", "case_whitespace_drift")
    for col in numeric_cols:
        if col in ("capacity_log", "skill_similarity", "sector_compatibility"):
            for idx in rng.choice(len(df), size=n, replace=False) if n else []:
                val = df.at[idx, col]
                if pd.notna(val): mutate(idx, col, float(val) + float(rng.normal(.15, .08)), "feature_noise_stress")
    # Explicit range and demand-ratio stress cases, separate from modest noise.
    if n:
        for idx in rng.choice(len(df), size=n, replace=False):
            mutate(idx, "skill_similarity", 5.0, "out_of_range_numeric_stress")
            mutate(idx, "applications_received", int(max(1, df.at[idx,"capacity"]) * 100), "implausible_application_capacity_ratio")
    for col in ["candidate_course", "internship_sector"]:
        if col in df and n:
            for idx in rng.choice(len(df), size=n, replace=False):
                val = df.at[idx, col]
                mutate(idx, col, str(val) + "_unknown", "unknown_category_stress")
    if "internship_sector" in df and n:
        for idx in rng.choice(len(df), size=n, replace=False):
            val=str(df.at[idx,"internship_sector"])
            mutate(idx,"internship_sector",val[:-1]+"x" if val else "unknownx","spelling_variation_stress")
    if n:
        states=original.internship_state.dropna().unique()
        for idx in rng.choice(len(df), size=n, replace=False):
            mutate(idx,"internship_state",str(rng.choice(states)),"conflicting_location_stress")
    # Missingness (rate of rows, spread across a subset of feature columns).
    miss_cols = ["candidate_course", "candidate_primary_sector", "skill_similarity", "location_compatibility"]
    for col in miss_cols:
        if col in df and n:
            for idx in rng.choice(len(df), size=n, replace=False): mutate(idx, col, np.nan, "controlled_missingness")
    # Numeric string ingestion errors. Keep labels and group identifiers intact.
    if n:
        for idx in rng.choice(len(df), size=n, replace=False):
            val = df.at[idx, "capacity"]
            mutate(idx, "capacity", f"{val} seats", "numeric_as_text")
    # Change class prevalence by sampling rows, leaving each retained label intact.
    imbalance_n=int(round(len(df)*rate))
    if imbalance_n and df.synthetic_proxy_label.nunique()>1:
        majority=int(df.synthetic_proxy_label.value_counts().idxmax())
        candidates=df.index[df.synthetic_proxy_label==majority].to_numpy()
        drop=rng.choice(candidates,size=min(imbalance_n,max(0,len(candidates)-1)),replace=False)
        for idx in drop:
            logs.append({"pair_id":df.at[idx,"pair_id"],"row_index":int(idx),"column":"row",
                         "original_value":"retained","corrupted_value":"removed",
                         "transformation":"class_imbalance_sampling_stress","seed":seed,"corruption_rate":rate})
        if len(drop): df=df.drop(index=drop).reset_index(drop=True)
    # Duplicate delivery rows and duplicate identifiers are deliberate stress tests.
    dup_n = int(round(len(df) * rate * .25))
    if dup_n:
        dup_rows = df.sample(dup_n, random_state=seed).copy()
        dup_rows["record_origin"] = "synthetic_corruption_duplicate"
        df = pd.concat([df, dup_rows], ignore_index=True)
        # Exact duplicate pair IDs are deliberate delivery/redelivery errors.
        for _, dup in dup_rows.iterrows():
            logs.append({"pair_id": dup["pair_id"], "row_index": -1, "column": "pair_id",
                         "original_value": dup["pair_id"], "corrupted_value": dup["pair_id"],
                         "transformation": "duplicate_identifier_stress", "seed": seed,
                         "corruption_rate": rate})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    log_path = output_csv.with_name(output_csv.stem + "_change_log.csv")
    log_columns=["pair_id","row_index","column","original_value","corrupted_value","transformation","seed","corruption_rate"]
    pd.DataFrame(logs,columns=log_columns).to_csv(log_path,index=False)
    return output_csv, log_path


def clean_dataset(corrupted_csv: Path, candidates_csv: Path, internships_csv: Path,
                  output_csv: Path, rejected_csv: Path) -> tuple[Path, Path]:
    """Normalize fields, recover ID-backed attributes, and audit rejected rows."""
    pairs = pd.read_csv(corrupted_csv)
    cands = pd.read_csv(candidates_csv).set_index("candidate_id")
    ints = pd.read_csv(internships_csv).set_index("internship_id")
    reject_reasons = []; cleaning_log=[]
    duplicate_mask=pairs.pair_id.duplicated(keep="first")
    for idx,row in pairs.loc[duplicate_mask].iterrows():
        cleaning_log.append({"pair_id":row.pair_id,"column":"pair_id","original_value":row.pair_id,
            "cleaned_value":"<dropped>","transformation":"drop_duplicate_pair_id_keep_first"})
    pairs = pairs.drop_duplicates(subset=["pair_id"], keep="first").copy()
    def repair(idx,col,value,action):
        old=pairs.at[idx,col]
        old_txt="<NA>" if pd.isna(old) else str(old)
        new_txt="<NA>" if pd.isna(value) else str(value)
        if old_txt!=new_txt:
            cleaning_log.append({"pair_id":pairs.at[idx,"pair_id"],"column":col,"original_value":old_txt,
                "cleaned_value":new_txt,"transformation":action})
        pairs.at[idx,col]=value
    for idx, row in pairs.iterrows():
        reasons=[]
        if pd.isna(row.get("candidate_id")) or row.get("candidate_id") not in cands.index: reasons.append("unknown_candidate_id")
        if pd.isna(row.get("internship_id")) or row.get("internship_id") not in ints.index: reasons.append("unknown_internship_id")
        if pd.isna(row.get("synthetic_proxy_label")): reasons.append("missing_proxy_label")
        if reasons: reject_reasons.append((idx, ";".join(reasons)))
        else:
            c=cands.loc[row.candidate_id]; it=ints.loc[row.internship_id]
            # Stable IDs are authoritative for these relationship fields.
            for col, value in [("candidate_qualification",c.qualification),("candidate_course",c.course),
                               ("candidate_primary_sector",c.primary_sector),
                               ("candidate_preferred_state",c.preferred_state),
                               ("candidate_preferred_district",c.preferred_district),
                               ("internship_qualification",it.qualification),("internship_course",it.course),
                               ("internship_sector",it.sector),("internship_state",it.state),
                               ("internship_district",it.district)]: repair(idx,col,value,"restore_from_stable_entity_id")
            for col, value in [("capacity",it.capacity),("applications_received",it.source_applications_received)]: repair(idx,col,value,"restore_from_stable_entity_id")
    rejected_idxs={x[0] for x in reject_reasons}
    rejected=pairs.loc[list(rejected_idxs)].copy() if rejected_idxs else pd.DataFrame(columns=list(pairs.columns)+["rejection_reason"])
    if reject_reasons:
        rejected["rejection_reason"]=[dict(reject_reasons)[i] for i in rejected.index]
    accepted=pairs.drop(index=list(rejected_idxs)).copy() if rejected_idxs else pairs.copy()
    for col in CATEGORICAL_FEATURES:
        if col in accepted:
            accepted[col]=accepted[col].map(lambda x: re.sub(r"\s+"," ",str(x)).strip().casefold() if pd.notna(x) else np.nan)
    for col in NUMERIC_FEATURES+["capacity","applications_received","eligible","synthetic_proxy_label"]:
        if col in accepted: accepted[col]=pd.to_numeric(accepted[col],errors="coerce")
    for col in ["qualification_compatibility","course_compatibility","specialization_compatibility",
                "skill_similarity","sector_compatibility","location_compatibility"]:
        invalid_range=accepted[col].notna() & ~accepted[col].between(0,1)
        for idx in accepted.index[invalid_range]:
            cleaning_log.append({"pair_id":accepted.at[idx,"pair_id"],"column":col,
                "original_value":str(accepted.at[idx,col]),"cleaned_value":"<NA>",
                "transformation":"out_of_range_to_missing_train_only_imputation"})
        accepted.loc[invalid_range,col]=np.nan
    max_capacity_log=float(np.log1p(150))
    bad_capacity_log=accepted.capacity_log.notna() & ~accepted.capacity_log.between(0,max_capacity_log)
    for idx in accepted.index[bad_capacity_log]:
        cleaning_log.append({"pair_id":accepted.at[idx,"pair_id"],"column":"capacity_log",
            "original_value":str(accepted.at[idx,"capacity_log"]),"cleaned_value":"<NA>",
            "transformation":"out_of_range_to_missing_train_only_imputation"})
    accepted.loc[bad_capacity_log,"capacity_log"]=np.nan
    accepted["synthetic_proxy_label"]=accepted["synthetic_proxy_label"].astype(int)
    # Reject impossible structural rows; do not silently discard them.
    invalid=(accepted["capacity"].isna() | (accepted["capacity"]<1) |
             accepted["eligible"].isna() | accepted["synthetic_proxy_label"].isna())
    if invalid.any():
        bad=accepted.loc[invalid].copy(); bad["rejection_reason"]="invalid_capacity_eligibility_or_label"
        rejected=pd.concat([rejected,bad],ignore_index=True); accepted=accepted.loc[~invalid]
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    accepted.to_csv(output_csv,index=False); rejected.to_csv(rejected_csv,index=False)
    pd.DataFrame(cleaning_log,columns=["pair_id","column","original_value","cleaned_value","transformation"]).to_csv(
        output_csv.with_name(output_csv.stem+"_cleaning_audit_log.csv"),index=False)
    return output_csv,rejected_csv


def _make_model(kind: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    prep=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median", add_indicator=True)),("scale",StandardScaler())]),NUMERIC_FEATURES),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),CATEGORICAL_FEATURES),
    ],sparse_threshold=0)
    clf={"logistic_regression":LogisticRegression(max_iter=500,class_weight="balanced",C=.5),
         "random_forest":RandomForestClassifier(n_estimators=100,min_samples_leaf=3,max_features="sqrt",class_weight="balanced_subsample",n_jobs=1,random_state=42),
         "gradient_boosting":HistGradientBoostingClassifier(max_iter=60,learning_rate=.08,max_leaf_nodes=15,l2_regularization=1.0)}[kind]
    return Pipeline([("preprocess",prep),("model",clf)])


def _ndcg_at_k(relevance, scores, k: int | None = None) -> float | None:
    relevance = np.asarray(relevance, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(relevance) != len(scores) or len(relevance) == 0:
        return None
    limit = len(relevance) if k is None else min(int(k), len(relevance))
    if limit <= 0:
        return None
    order = np.argsort(-scores)[:limit]
    ideal = np.argsort(-relevance)[:limit]
    discounts = np.log2(np.arange(2, limit + 2))
    dcg = float(np.sum((2 ** relevance[order] - 1) / discounts))
    ideal_dcg = float(np.sum((2 ** relevance[ideal] - 1) / discounts))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def candidate_grouped_split(df: pd.DataFrame, seed: int = 42):
    """Return train/validation/test row indices with candidates isolated per split."""
    from sklearn.model_selection import GroupShuffleSplit

    if "candidate_id" not in df or df["candidate_id"].isna().any():
        raise ValueError("Evaluation data must contain non-missing candidate_id values")
    groups = df["candidate_id"].astype(str).to_numpy()
    all_rows = np.arange(len(df))
    development, test = next(
        GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
            all_rows, groups=groups
        )
    )
    train_relative, validation_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed + 1).split(
            development, groups=groups[development]
        )
    )
    train = development[train_relative]
    validation = development[validation_relative]
    split_groups = [
        set(groups[indexes])
        for indexes in (train, validation, test)
    ]
    if any(
        split_groups[left] & split_groups[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("Candidate leakage detected between evaluation partitions")
    return train, validation, test


def evaluate_dataset(name: str, csv_path: Path, output_dir: Path, seed: int=42) -> list[dict]:
    from sklearn.metrics import (precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score,
                                 average_precision_score)
    df=pd.read_csv(csv_path)
    for col in NUMERIC_FEATURES: df[col]=pd.to_numeric(df[col],errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col]=df[col].where(df[col].notna(),np.nan).astype(object)
    y=df["synthetic_proxy_label"].astype(int).to_numpy()
    tr,val,te=candidate_grouped_split(df,seed)
    groups=df["candidate_id"].astype(str).to_numpy()
    x_train=df.iloc[tr][FEATURES]; x_val=df.iloc[val][FEATURES]; x_test=df.iloc[te][FEATURES]
    y_train=y[tr]; y_val=y[val]; y_test=y[te]
    reports=[]
    # Transparent rule baseline uses only documented eligibility and score.
    rule=(df.iloc[te]["eligible"].fillna(0).to_numpy().astype(float)>0) & (df.iloc[te]["synthetic_match_score"].fillna(0).to_numpy()>=.56)
    for model_name in ["rule_baseline","logistic_regression","random_forest","gradient_boosting"]:
        if model_name=="rule_baseline":
            fit_seconds=0.0; infer_start=time.perf_counter()
            pred=rule.astype(int); proba=df.iloc[te]["synthetic_match_score"].fillna(0).to_numpy()
            val_pred=((df.iloc[val]["eligible"].fillna(0).to_numpy()>0)&(df.iloc[val]["synthetic_match_score"].fillna(0).to_numpy()>=.56)).astype(int)
            train_pred=((df.iloc[tr]["eligible"].fillna(0).to_numpy()>0)&(df.iloc[tr]["synthetic_match_score"].fillna(0).to_numpy()>=.56)).astype(int)
            infer_seconds=time.perf_counter()-infer_start
        else:
            model=_make_model(model_name); fit_start=time.perf_counter(); model.fit(x_train,y_train)
            fit_seconds=time.perf_counter()-fit_start; infer_start=time.perf_counter()
            pred=model.predict(x_test); proba=model.predict_proba(x_test)[:,1]
            val_pred=model.predict(x_val); train_pred=model.predict(x_train)
            infer_seconds=time.perf_counter()-infer_start
        tn,fp,fn,tp=confusion_matrix(y_test,pred,labels=[0,1]).ravel()
        try: auc=roc_auc_score(y_test,proba) if len(np.unique(y_test))>1 else None
        except ValueError: auc=None
        try: ap=average_precision_score(y_test,proba) if len(np.unique(y_test))>1 else None
        except ValueError: ap=None
        # Per-candidate ranking quality: NDCG over held-out pairs.
        ranked=df.iloc[te][["candidate_id","synthetic_proxy_label"]].copy(); ranked["score"]=proba
        ndcg_at_5=[]; ndcg_at_10=[]; ndcg_full=[]
        for _,g in ranked.groupby("candidate_id"):
            if len(g)>1 and g.synthetic_proxy_label.sum()>0:
                relevance=g.synthetic_proxy_label.to_numpy()
                scores=g.score.to_numpy()
                ndcg_at_5.append(_ndcg_at_k(relevance,scores,5))
                ndcg_at_10.append(_ndcg_at_k(relevance,scores,10))
                ndcg_full.append(_ndcg_at_k(relevance,scores))
        reports.append({"dataset":name,"model":model_name,"train_rows":len(tr),"validation_rows":len(val),"test_rows":len(te),
            "train_candidates":int(pd.Series(groups[tr]).nunique()),"validation_candidates":int(pd.Series(groups[val]).nunique()),
            "test_candidates":int(pd.Series(groups[te]).nunique()),"train_f1":f1_score(y_train,train_pred,zero_division=0),
            "validation_f1":f1_score(y_val,val_pred,zero_division=0),"precision":precision_score(y_test,pred,zero_division=0),
            "recall":recall_score(y_test,pred,zero_division=0),"f1":f1_score(y_test,pred,zero_division=0),
            "roc_auc":auc,"pr_auc":ap,
            "ndcg_at_5":float(np.mean(ndcg_at_5)) if ndcg_at_5 else None,
            "ndcg_at_10":float(np.mean(ndcg_at_10)) if ndcg_at_10 else None,
            "ndcg_at_full_list":float(np.mean(ndcg_full)) if ndcg_full else None,
            "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp),"fit_seconds":fit_seconds,
            "inference_seconds":infer_seconds,"inference_ms_per_row":infer_seconds/max(1,len(te))*1000,
            "label_note":"agreement with synthetic heuristic labels only; not observed PMIS outcome accuracy"})
    return reports


def allocate(pairs_csv: Path | pd.DataFrame, output_csv: Path, time_limit: float = 15.0,
             policy_group_column: str | None = None,
             policy_minimums: dict[str, int] | None = None) -> dict:
    """Maximize compatibility under eligibility, candidate, capacity, and explicit policy rules."""
    df = pairs_csv.copy() if isinstance(pairs_csv, pd.DataFrame) else pd.read_csv(pairs_csv)
    required = {
        "candidate_id", "internship_id", "eligible", "synthetic_match_score", "capacity",
        "candidate_qualification", "internship_qualification", "candidate_course",
        "internship_course",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Allocation input is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Allocation input contains no candidate–internship pairs")
    for column in ("candidate_id", "internship_id"):
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} values must be present and non-empty")
    if df.duplicated(["candidate_id", "internship_id"]).any():
        raise ValueError("Allocation input contains duplicate candidate–internship pairs")

    eligible = pd.to_numeric(df["eligible"], errors="coerce")
    scores = pd.to_numeric(df["synthetic_match_score"], errors="coerce")
    capacities = pd.to_numeric(df["capacity"], errors="coerce")
    if eligible.isna().any() or not eligible.isin([0, 1]).all():
        raise ValueError("Eligibility values must be binary (0 or 1)")
    if scores.isna().any() or not np.isfinite(scores).all() or not scores.between(0, 1).all():
        raise ValueError("Compatibility scores must be finite values between 0 and 1")
    if capacities.isna().any() or not np.isfinite(capacities).all() or (capacities < 0).any():
        raise ValueError("Internship capacities must be finite, non-negative numbers")
    if not np.equal(capacities, np.floor(capacities)).all():
        raise ValueError("Internship capacities must be whole numbers")
    df["eligible"] = eligible.astype(int)
    df["synthetic_match_score"] = scores.astype(float)
    df["capacity"] = capacities.astype(int)
    if df.groupby("internship_id")["capacity"].nunique().gt(1).any():
        raise ValueError("Each internship must have one consistent capacity value")

    policy_minimums = policy_minimums or {}
    if policy_minimums:
        if not policy_group_column or policy_group_column not in df.columns:
            raise ValueError("Configured group minimums require a policy group column in the input")
        if df[policy_group_column].isna().any():
            raise ValueError("Policy group values must be present for every candidate")
        group_counts = df.groupby("candidate_id")[policy_group_column].nunique()
        if group_counts.gt(1).any():
            raise ValueError("Each candidate must have exactly one policy group")
        for group, minimum in policy_minimums.items():
            if not isinstance(minimum, int) or minimum < 0:
                raise ValueError("Policy minimums must be non-negative whole numbers")
            if minimum > int(df.loc[df[policy_group_column] == group, "candidate_id"].nunique()):
                raise ValueError(f"Policy minimum for {group!r} exceeds the available group size")

    # Recompute hard requirements rather than trusting an incoming eligibility flag.
    hard_eligible = (
        df.apply(
            lambda row: (
                _qualification_eligible(
                    row["candidate_qualification"], row["internship_qualification"]
                )
                and _compat(row["candidate_course"], row["internship_course"], ("ANY", "All Subjects"))
            ),
            axis=1,
        )
    )
    all_candidates = df["candidate_id"].astype(str).unique()
    internships = df.drop_duplicates("internship_id").set_index("internship_id")["capacity"]
    edges = df.loc[(df["eligible"] == 1) & hard_eligible].copy().reset_index(drop=True)
    edges["utility"] = edges["synthetic_match_score"]
    if policy_minimums:
        eligible_group_counts = edges.groupby(policy_group_column)["candidate_id"].nunique()
        for group, minimum in policy_minimums.items():
            available = int(eligible_group_counts.get(group, 0))
            if minimum > available:
                raise RuntimeError(
                    f"Policy minimum for {group!r} is infeasible: "
                    f"{available} eligible candidate(s), minimum {minimum}"
                )
    solver_name = "not_run"
    solver_status = "no_hard_eligible_pairs"
    if not edges.empty:
        candidates = edges["candidate_id"].unique()
        edge_internships = edges["internship_id"].unique()
        try:
            from ortools.sat.python import cp_model
            model = cp_model.CpModel()
            variables = [model.NewBoolVar(f"x_{j}") for j in range(len(edges))]
            by_candidate = edges.groupby("candidate_id").indices
            by_internship = edges.groupby("internship_id").indices
            for indexes in by_candidate.values():
                model.Add(sum(variables[int(j)] for j in indexes) <= 1)
            for internship, indexes in by_internship.items():
                model.Add(
                    sum(variables[int(j)] for j in indexes)
                    <= int(internships.loc[internship])
                )
            if policy_minimums:
                for group, minimum in policy_minimums.items():
                    group_indexes = edges.index[edges[policy_group_column] == group].tolist()
                    model.Add(sum(variables[j] for j in group_indexes) >= minimum)
            model.Maximize(
                sum(
                    int(round(float(edges.loc[j, "utility"]) * 1_000_000)) * variables[j]
                    for j in range(len(edges))
                )
            )
            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = float(time_limit)
            solver.parameters.num_search_workers = 8
            status = solver.Solve(model)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                raise RuntimeError(f"Capacity allocation failed: {solver.StatusName(status)}")
            edges["allocated"] = [solver.Value(variable) for variable in variables]
            solver_name = "ortools_cp_sat"
            solver_status = solver.StatusName(status)
        except ImportError:
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
            policy_groups = list(policy_minimums)
            rows = len(candidates) + len(edge_internships) + len(policy_groups)
            matrix = lil_matrix((rows, len(edges)), dtype=float)
            candidate_map = {value: index for index, value in enumerate(candidates)}
            internship_map = {
                value: index + len(candidates)
                for index, value in enumerate(edge_internships)
            }
            policy_map = {
                value: index + len(candidates) + len(edge_internships)
                for index, value in enumerate(policy_groups)
            }
            for index, edge in edges.iterrows():
                matrix[candidate_map[edge.candidate_id], index] = 1
                matrix[internship_map[edge.internship_id], index] = 1
                if policy_minimums and edge[policy_group_column] in policy_map:
                    matrix[policy_map[edge[policy_group_column]], index] = 1
            lower = np.r_[
                np.zeros(len(candidates) + len(edge_internships)),
                [policy_minimums[group] for group in policy_groups],
            ]
            upper = np.r_[
                np.ones(len(candidates)),
                internships.reindex(edge_internships).to_numpy(dtype=float),
                np.full(len(policy_groups), np.inf),
            ]
            result = milp(
                c=-edges["utility"].to_numpy(),
                integrality=np.ones(len(edges)),
                bounds=Bounds(0, 1),
                constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                options={"time_limit": time_limit},
            )
            if result.x is None:
                raise RuntimeError(f"Capacity allocation failed: {result.message}")
            edges["allocated"] = (result.x > .5).astype(int)
            solver_name = "scipy.optimize.milp"
            solver_status = str(result.message)

    chosen = edges[edges["allocated"] == 1].copy() if "allocated" in edges else edges.copy()
    if not chosen.empty:
        alternatives = edges[
            ~edges.index.isin(chosen.index)
        ].sort_values("synthetic_match_score", ascending=False).drop_duplicates("candidate_id")
        alternative_scores = alternatives.set_index("candidate_id")["synthetic_match_score"]
        chosen["best_unselected_alternative_score"] = chosen["candidate_id"].map(alternative_scores)
        chosen["capacity_at_limit"] = (
            chosen.groupby("internship_id")["internship_id"].transform("size")
            >= chosen.groupby("internship_id")["capacity"].transform("first")
        )

        def explain(row):
            components = [
                ("qualification_compatibility", "qualification"),
                ("course_compatibility", "course"),
                ("specialization_compatibility", "specialization"),
                ("skill_similarity", "skill overlap"),
                ("sector_compatibility", "sector preference"),
                ("location_compatibility", "location preference"),
            ]
            details = [
                f"{label}={float(row[column]):.2f}"
                for column, label in components
                if column in row.index and pd.notna(row[column])
            ]
            capacity_note = (
                "assigned internship reached its capacity"
                if row["capacity_at_limit"]
                else "assigned internship retained capacity"
            )
            alternative = row["best_unselected_alternative_score"]
            if pd.isna(alternative):
                alternative_note = "No other eligible option was present."
            elif float(alternative) > float(row["synthetic_match_score"]) + 1e-6:
                alternative_note = (
                    f"A higher-scoring unassigned alternative scored {float(alternative):.3f}; "
                    "capacity competition affected the assignment."
                )
            else:
                alternative_note = (
                    f"Best unassigned alternative scored {float(alternative):.3f}, "
                    "below the assigned score."
                )
            return (
                f"Eligible; score {float(row['synthetic_match_score']):.3f} "
                f"({', '.join(details)}); {capacity_note}. {alternative_note}"
            )

        chosen["allocation_explanation"] = chosen.apply(explain, axis=1)
        chosen["decision_layer_note"] = (
            "Eligibility and capacity optimization are separate. "
            + (
                f"Explicit group minimums applied: {policy_minimums}."
                if policy_minimums
                else "No equity policy constraint configured."
            )
        )
    else:
        chosen = edges.copy()
        for column, dtype in (
            ("best_unselected_alternative_score", float),
            ("capacity_at_limit", bool),
            ("allocation_explanation", object),
            ("decision_layer_note", object),
        ):
            chosen[column] = pd.Series(dtype=dtype)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(output_csv, index=False)
    assigned_counts = chosen.groupby("internship_id").size()
    violations = int(
        sum(
            max(0, int(count) - int(internships.loc[internship]))
            for internship, count in assigned_counts.items()
        )
    )
    total_seats = int(internships.sum())
    assigned_count = int(len(chosen))
    allocated_candidates = int(chosen["candidate_id"].nunique())
    eligibility_violations = int(
        sum(
            not (
                _qualification_eligible(row["candidate_qualification"], row["internship_qualification"])
                and _compat(row["candidate_course"], row["internship_course"], ("ANY", "All Subjects"))
            )
            for _, row in chosen.iterrows()
        )
    )
    return {
        "solver": solver_name,
        "solver_status": solver_status,
        "eligible_pairs": int(len(edges)),
        "eligible_candidates": int(edges["candidate_id"].nunique()),
        "allocations": assigned_count,
        "unique_candidates_allocated": allocated_candidates,
        "unallocated_candidates": int(len(all_candidates) - allocated_candidates),
        "total_seats": total_seats,
        "unused_seats": max(0, total_seats - assigned_count),
        "seat_utilization": assigned_count / total_seats if total_seats else 0.0,
        "allocation_rate": allocated_candidates / len(all_candidates) if len(all_candidates) else 0.0,
        "capacity_violations": violations,
        "eligibility_violations": eligibility_violations,
        "total_match_score": float(chosen["synthetic_match_score"].sum()),
        "policy_group_column": policy_group_column if policy_minimums else None,
        "policy_minimums": policy_minimums,
        "output": str(output_csv),
    }


def learning_curve(clean_pairs: Path, output_csv: Path, seed: int) -> pd.DataFrame:
    """Candidate-grouped LR learning curve on a fixed held-out candidate set."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score
    df=pd.read_csv(clean_pairs); y=df.synthetic_proxy_label.astype(int).to_numpy()
    groups=df.candidate_id.astype(str).to_numpy()
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(df,y,groups))
    rng=np.random.default_rng(seed); unique=pd.unique(groups[dev]); rows=[]
    for fraction in [.25,.5,.75,1.0]:
        chosen=rng.permutation(unique)[:max(1,int(round(len(unique)*fraction)))]
        tr=dev[np.isin(groups[dev],chosen)]
        model=_make_model("logistic_regression"); model.fit(df.iloc[tr][FEATURES],y[tr])
        rows.append({"training_fraction_of_development_candidates":fraction,"training_rows":len(tr),
                     "training_candidates":len(chosen),"heldout_test_rows":len(test),
                     "train_f1":f1_score(y[tr],model.predict(df.iloc[tr][FEATURES]),zero_division=0),
                     "test_f1":f1_score(y[test],model.predict(df.iloc[test][FEATURES]),zero_division=0),
                     "split_unit":"candidate_id","label_note":"synthetic proxy-label agreement only"})
    result=pd.DataFrame(rows); result.to_csv(output_csv,index=False); return result


def grouped_cross_validation(clean_pairs: Path, output_csv: Path, seed: int) -> pd.DataFrame:
    """Three-fold GroupKFold diagnostics; each fold fits preprocessing on training candidates."""
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit
    from sklearn.metrics import f1_score
    df=pd.read_csv(clean_pairs); y=df.synthetic_proxy_label.astype(int).to_numpy()
    groups=df.candidate_id.astype(str).to_numpy(); rows=[]
    development,_=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(df,y,groups))
    dev_candidates=pd.unique(groups[development]); rng=np.random.default_rng(seed)
    cv_candidates=rng.permutation(dev_candidates)[:max(3,int(round(len(dev_candidates)*.35)))]
    cv_rows=development[np.isin(groups[development],cv_candidates)]
    for model_name in ["logistic_regression","random_forest","gradient_boosting"]:
        for fold,(tr_rel,te_rel) in enumerate(GroupKFold(n_splits=3).split(df.iloc[cv_rows][FEATURES],y[cv_rows],groups[cv_rows]),1):
            tr=cv_rows[tr_rel]; te=cv_rows[te_rel]
            model=_make_model(model_name); model.fit(df.iloc[tr][FEATURES],y[tr])
            rows.append({"model":model_name,"fold":fold,"train_rows":len(tr),"validation_rows":len(te),
                         "f1":f1_score(y[te],model.predict(df.iloc[te][FEATURES]),zero_division=0),
                         "split_unit":"candidate_id","seed":seed,"development_candidate_fraction_for_cv":.35,
                         "label_note":"synthetic proxy-label agreement only"})
    result=pd.DataFrame(rows); result.to_csv(output_csv,index=False); return result


def unseen_internship_validation(clean_pairs: Path, output_csv: Path, seed: int) -> dict:
    """Held-out source internship IDs, with candidate overlap removed from training."""
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
    df=pd.read_csv(clean_pairs); y=df.synthetic_proxy_label.astype(int).to_numpy()
    groups=df.source_internship_id.astype(str).to_numpy()
    dev,test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(df,y,groups))
    test_candidates=set(df.iloc[test].candidate_id.astype(str))
    train=dev[~df.iloc[dev].candidate_id.astype(str).isin(test_candidates).to_numpy()]
    model=_make_model("logistic_regression"); model.fit(df.iloc[train][FEATURES],y[train])
    probs=model.predict_proba(df.iloc[test][FEATURES])[:,1]; pred=model.predict(df.iloc[test][FEATURES])
    result={"split":"unseen_source_internship_ids_and_unseen_candidates","model":"logistic_regression",
        "train_rows":int(len(train)),"test_rows":int(len(test)),
        "train_candidates":int(df.iloc[train].candidate_id.nunique()),
        "test_candidates":int(df.iloc[test].candidate_id.nunique()),
        "heldout_source_internships":int(df.iloc[test].source_internship_id.nunique()),
        "precision":float(__import__("sklearn.metrics",fromlist=["precision_score"]).precision_score(y[test],pred,zero_division=0)),
        "recall":float(__import__("sklearn.metrics",fromlist=["recall_score"]).recall_score(y[test],pred,zero_division=0)),
        "f1":float(f1_score(y[test],pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y[test],probs)) if len(np.unique(y[test]))>1 else None,
        "pr_auc":float(average_precision_score(y[test],probs)) if len(np.unique(y[test]))>1 else None,
        "label_note":"synthetic proxy-label agreement only"}
    output_csv.write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def rank_internships(clean_pairs: Path, output_csv: Path) -> pd.DataFrame:
    """Emit per-candidate eligible internship rankings with component reasons."""
    df=pd.read_csv(clean_pairs)
    ranked=df[df.eligible==1].copy()
    ranked=ranked.sort_values(["candidate_id","synthetic_match_score","internship_id"],ascending=[True,False,True])
    ranked["recommendation_rank"]=ranked.groupby("candidate_id").cumcount()+1
    component_cols=[("qualification_compatibility","qualification"),("course_compatibility","course"),
        ("specialization_compatibility","specialization"),("skill_similarity","skill overlap"),
        ("sector_compatibility","sector preference"),("location_compatibility","location preference")]
    def reason(r):
        strengths=[label for col,label in component_cols if pd.notna(r[col]) and float(r[col])>=.75]
        return "Eligible; strongest compatibility: "+(", ".join(strengths) if strengths else "combined score across the listed features")+f" (score {float(r.synthetic_match_score):.3f})."
    ranked["why_recommended"]=ranked.apply(reason,axis=1)
    next_scores=ranked.groupby("candidate_id").synthetic_match_score.shift(-1)
    ranked["next_eligible_score"]=next_scores
    ranked["alternative_score_below"]=(ranked.synthetic_match_score-next_scores).round(6)
    ranked["why_alternative_lower"]=[("Next option scores lower by {:.3f}.".format(delta) if pd.notna(delta) and delta>1e-6
        else "Next option ties on score and follows in the deterministic internship-ID tie-break." if pd.notna(delta)
        else "No lower-ranked eligible option in this generated pair sample.") for delta in ranked.alternative_score_below]
    output_csv.parent.mkdir(parents=True,exist_ok=True); ranked.to_csv(output_csv,index=False)
    return ranked


def profile_quality(path: Path, name: str) -> dict:
    df=pd.read_csv(path)
    cap=pd.to_numeric(df.capacity,errors="coerce")
    apps=pd.to_numeric(df.applications_received,errors="coerce")
    ratio=apps/cap.replace(0,np.nan)
    return {"dataset":name,"rows":int(len(df)),"columns":int(len(df.columns)),
            "duplicate_pair_ids":int(df.pair_id.duplicated().sum()),
            "missing_cells":int(df.isna().sum().sum()),
            "non_numeric_capacity_values":int(cap.isna().sum()),
            "application_capacity_ratio_gt_10":int((ratio>10).sum()),
            "unknown_category_values":int(sum(df[c].astype(str).str.endswith("_unknown").sum() for c in ["candidate_course","internship_sector"] if c in df)),
            "out_of_range_skill_similarity":int(((pd.to_numeric(df.skill_similarity,errors="coerce")<0)|(pd.to_numeric(df.skill_similarity,errors="coerce")>1)).sum()),
            "proxy_positive_rate":float(pd.to_numeric(df.synthetic_proxy_label,errors="coerce").mean())}


def run_all(config: dict, source: Path=SOURCE, output_dir: Path|None=None) -> None:
    out=output_dir or ROOT/config.get("output_dir","data/outputs")
    generated=generate(config,source,out)
    print(f"Generated {config['n_candidates']} candidates, {config['n_internships']} internships, and {config['n_pairs']} pairs.",flush=True)
    clean_pairs=generated["pairs"]
    expdir=out/"experiments"; expdir.mkdir(parents=True,exist_ok=True)
    level_reports=[]; clean_report=evaluate_dataset("clean",clean_pairs,expdir,int(config["seed"]))
    print("Evaluated clean dataset.",flush=True)
    level_reports.extend(clean_report)
    experiment_manifest=[]
    quality_rows=[profile_quality(clean_pairs,"clean")]
    default_tag=f"{int(round(float(config.get('default_corruption_rate',.1))*100)):02d}pct"
    for rate in config.get("corruption_levels",[config.get("default_corruption_rate",.1)]):
        tag=f"{int(round(rate*100)):02d}pct"
        print(f"Evaluating {tag} corruption and cleaned copy.",flush=True)
        corrupt_path=expdir/f"synthetic_corrupted_{tag}.csv"
        corrupt_dataset(clean_pairs,corrupt_path,float(rate),int(config["seed"])+int(rate*1000))
        candidates=generated["candidates"]; internships=generated["internships"]
        cleaned_path=expdir/f"synthetic_cleaned_{tag}.csv"; rejected_path=expdir/f"rejected_{tag}.csv"
        clean_dataset(corrupt_path,candidates,internships,cleaned_path,rejected_path)
        level_reports += evaluate_dataset(f"corrupted_{tag}",corrupt_path,expdir,int(config["seed"]))
        level_reports += evaluate_dataset(f"cleaned_{tag}",cleaned_path,expdir,int(config["seed"]))
        quality_rows += [profile_quality(corrupt_path,f"corrupted_{tag}"),profile_quality(cleaned_path,f"cleaned_{tag}")]
        log=pd.read_csv(corrupt_path.with_name(corrupt_path.stem+"_change_log.csv"))
        experiment_manifest.append({"seed":int(config["seed"])+int(rate*1000),"corruption_rate":rate,
            "clean_rows":len(pd.read_csv(clean_pairs)),"corrupted_rows":len(pd.read_csv(corrupt_path)),
            "cleaned_rows":len(pd.read_csv(cleaned_path)),"affected_columns":";".join(sorted(log.column.dropna().astype(str).unique())) if len(log) else "",
            "transformations":";".join(sorted(log.transformation.dropna().astype(str).unique())) if len(log) else "none",
            "change_log":str(corrupt_path.with_name(corrupt_path.stem+"_change_log.csv"))})
        if tag==default_tag:
            shutil.copy2(corrupt_path,out/"synthetic_corrupted_experiment.csv")
            shutil.copy2(cleaned_path,out/"synthetic_cleaned_experiment.csv")
    shutil.copy2(clean_pairs,out/"synthetic_clean.csv")
    pd.DataFrame(experiment_manifest).to_csv(expdir/"experiment_manifest.csv",index=False)
    pd.DataFrame(quality_rows).to_csv(expdir/"data_quality_metrics.csv",index=False)
    metrics=pd.DataFrame(level_reports)
    metrics_path=expdir/"model_metrics.csv"; metrics.to_csv(metrics_path,index=False)
    curve=learning_curve(clean_pairs,expdir/"learning_curve.csv",int(config["seed"]))
    print("Learning curve complete; running grouped cross-validation.",flush=True)
    grouped_cross_validation(clean_pairs,expdir/"grouped_cross_validation.csv",int(config["seed"]))
    print("Cross-validation complete; running unseen-internship experiment.",flush=True)
    unseen=unseen_internship_validation(clean_pairs,expdir/"unseen_internship_validation.json",int(config["seed"]))
    print("Unseen-internship experiment complete.",flush=True)
    rank_internships(clean_pairs,expdir/"ranked_recommendations.csv")
    # Capacity-aware allocation on the clean synthetic pair set.
    allocation_path=expdir/"capacity_aware_allocations.csv"
    allocation_summary=allocate(clean_pairs,allocation_path)
    (expdir/"allocation_validation.json").write_text(json.dumps(allocation_summary,indent=2),encoding="utf-8")
    (expdir/"fairness_validation.json").write_text(json.dumps({
        "status":"not_evaluated","reason":"No legitimate candidate demographic/policy data or authoritative affirmative-action rules were supplied.",
        "sensitive_attributes_used_as_merit_features":False,
        "allocation_capacity_violations":allocation_summary["capacity_violations"],
        "note":"Capacity compliance is validated separately. Do not infer fairness from synthetic proxy-label metrics."},indent=2),encoding="utf-8")
    # Compact visual report; scores are proxy-label agreement only.
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5.5))
    model_order=["rule_baseline","logistic_regression","random_forest","gradient_boosting"]
    grouped=metrics.pivot(index="dataset",columns="model",values="f1").reindex(columns=model_order)
    order=["clean"]+[f"{kind}_{tag}" for tag in ["00pct","05pct","10pct","20pct"] for kind in ["corrupted","cleaned"]]
    grouped=grouped.reindex(order)
    x=np.arange(len(grouped)); width=.19
    colors=["#667985","#315a78","#9d6b3f","#52745b"]
    for i,model_name in enumerate(model_order):
        ax.bar(x+(i-1.5)*width,grouped[model_name],width,label=model_name.replace("_"," "),color=colors[i])
    labels=["Clean"]+[f"{kind.title()} {tag[:2]}%" for tag in ["00pct","05pct","10pct","20pct"] for kind in ["corrupted","cleaned"]]
    ax.set_xticks(x,labels,rotation=0,ha="center",fontsize=8)
    ax.set_ylabel("F1 score (synthetic proxy-label agreement)"); ax.set_xlabel("Dataset condition")
    ax.set_ylim(0,1); ax.legend(ncols=4,frameon=False,loc="upper center",bbox_to_anchor=(.5,1.12))
    ax.set_title("Model comparison across corruption levels",pad=32)
    ax.text(.01,-.22,"Source: generated experimental pairs · target: synthetic_proxy_label · candidate-grouped holdout",transform=ax.transAxes,fontsize=8)
    ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True); fig.tight_layout(rect=(0,.08,1,.90))
    fig.savefig(expdir/"model_comparison.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    ax.plot(curve.training_rows,curve.train_f1,marker="o",label="Train F1")
    ax.plot(curve.training_rows,curve.test_f1,marker="o",label="Held-out candidate F1")
    ax.set_xlabel("Training rows (candidate-grouped)" ); ax.set_ylabel("F1 (synthetic proxy-label agreement)")
    ax.set_ylim(0,1); ax.set_title("Learning curve: Logistic Regression on synthetic pairs")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(expdir/"learning_curve.png",dpi=160); plt.close(fig)
    report={"synthetic_only":True,"metrics_csv":str(metrics_path),"allocation":allocation_summary,
            "data_quality":{"clean_pair_rows":len(pd.read_csv(clean_pairs)),
                "clean_pair_duplicate_ids":int(pd.read_csv(clean_pairs).pair_id.duplicated().sum()),
                "corruption_levels":config.get("corruption_levels",[]),
                "corrupted_and_cleaned_files":True},
            "caveat":"All model targets are synthetic heuristic labels. No actual PMIS selection outcome was available."}
    (expdir/"evaluation_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    clean_metrics=metrics[metrics.dataset=="clean"].set_index("model")
    report_lines=["# Synthetic experiment results","",
        "All metrics below measure agreement with `synthetic_proxy_label`, a generated heuristic target. They are not real PMIS selection accuracy.","",
        "## Clean candidate-grouped holdout","",
        "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | NDCG@5 | NDCG@10 | NDCG full | Train F1 | Validation F1 | Fit seconds | Inference ms/row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name,row in clean_metrics.iterrows():
        report_lines.append(
            f"| {model_name} | {row.precision:.3f} | {row.recall:.3f} | {row.f1:.3f} | "
            f"{row.roc_auc:.3f} | {row.pr_auc:.3f} | {row.ndcg_at_5:.3f} | "
            f"{row.ndcg_at_10:.3f} | {row.ndcg_at_full_list:.3f} | {row.train_f1:.3f} | "
            f"{row.validation_f1:.3f} | {row.fit_seconds:.3f} | {row.inference_ms_per_row:.3f} |"
        )
    report_lines += ["", "## Data quality by condition", "",
        "| Dataset | Rows | Duplicate IDs | Missing cells | Non-numeric capacity | App/capacity >10 | Unknown categories | Out-of-range skill values | Positive proxy rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for q in quality_rows:
        report_lines.append(f"| {q['dataset']} | {q['rows']} | {q['duplicate_pair_ids']} | {q['missing_cells']} | {q['non_numeric_capacity_values']} | {q['application_capacity_ratio_gt_10']} | {q['unknown_category_values']} | {q['out_of_range_skill_similarity']} | {q['proxy_positive_rate']:.3f} |")
    report_lines += ["", "## Validation notes", "",
        f"- Capacity allocation used `{allocation_summary['solver']}` with status `{allocation_summary['solver_status']}`; {allocation_summary['allocations']} assignments were made across {allocation_summary['unique_candidates_allocated']} candidates, with {allocation_summary['capacity_violations']} capacity violations.",
        f"- Unseen-internship experiment: {unseen['heldout_source_internships']} source internships and {unseen['test_candidates']} held-out candidates; F1 {unseen['f1']:.3f}, ROC-AUC {unseen['roc_auc'] if unseen['roc_auc'] is not None else 'not defined'}.",
        "- Corruption experiments include nulls, case/whitespace drift, category noise/unknowns, numeric text, out-of-range values, impossible demand ratios, conflicting locations, feature noise, duplicate deliveries/IDs, and valid-row sampling to alter class prevalence. Labels on retained rows were not changed.",
        "- Cleaning restores fields from stable synthetic entity IDs, removes duplicate pair IDs with a log, normalizes categories, and converts out-of-range match features to missing values. Remaining missing feature values are imputed within each training fold by the sklearn pipeline.",
        "- `App/capacity >10` is a diagnostic threshold for the injected demand-ratio stress test, not an official validity rule. The workbook's aggregate application counts already include high ratios; their semantics must be confirmed before operational use.",
        "- Fairness metrics are not reported: no legitimate policy group data or authoritative policy rules were supplied. The fairness validation JSON records this limitation.",
        "- No company-level real-world generalization or true applicant outcome validation is possible from the supplied source workbook.",""]
    (expdir/"evaluation_report.md").write_text("\n".join(report_lines),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    ap.add_argument("--source",type=Path,default=SOURCE)
    ap.add_argument("--output",type=Path,default=None)
    args=ap.parse_args(); config=read_config(args.config)
    run_all(config,args.source,args.output)
    print(f"Pipeline complete. Outputs: {args.output or (ROOT/config.get('output_dir','data/outputs'))}")


if __name__=="__main__": main()
