import json
from pathlib import Path

import pandas as pd
import pytest

from src.pmis_pipeline import (
    FEATURES,
    _ndcg_at_k,
    allocate,
    allocation_fairness_summary,
    candidate_grouped_split,
    clean_dataset,
    corrupt_dataset,
    generate,
    load_opportunity_catalog,
    match_candidates,
    validate_candidate_roster,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source/PMIS_MyBharat_Qualification_District_Mapping.xlsx"


def small_config(seed=51):
    return {"seed": seed, "n_candidates": 30, "n_internships": 20, "n_pairs": 120,
            "label_threshold": .56, "output_dir": "tmp-test"}


def test_seeded_generation_is_reproducible(tmp_path):
    a = generate(small_config(), SOURCE, tmp_path / "a")
    b = generate(small_config(), SOURCE, tmp_path / "b")
    assert pd.read_csv(a["candidates"]).equals(pd.read_csv(b["candidates"]))
    pa, pb = pd.read_csv(a["pairs"]), pd.read_csv(b["pairs"])
    assert pa.equals(pb)
    assert pa.pair_id.is_unique
    assert set(pa.synthetic_proxy_label.unique()) <= {0, 1}
    assert pa.record_origin.eq("synthetic_candidate_internship_pair").all()


def test_source_opportunity_catalog_preserves_observed_ids_and_capacity():
    catalog = load_opportunity_catalog(SOURCE)
    assert len(catalog) == 1062
    assert catalog.internship_id.is_unique
    assert catalog.internship_id.eq(catalog.source_internship_id).all()
    assert catalog.record_origin.eq("observed_source_opportunity").all()
    assert catalog.capacity.ge(1).all()
    assert catalog.provenance_json.str.contains('"source_sheet": "Main"').all()


def test_uploaded_candidate_roster_validation_preserves_separate_provenance():
    roster = validate_candidate_roster(pd.DataFrame([
        {"candidate_id": "U1", "qualification": "Graduation", "course": "Mechanical",
         "past_participation": 0, "policy_group": "A"},
    ]))
    assert roster.record_origin.tolist() == ["user_uploaded_candidate"]
    assert roster.policy_group.tolist() == ["A"]
    with pytest.raises(ValueError, match="must be unique"):
        validate_candidate_roster(pd.DataFrame([
            {"candidate_id": "U1", "qualification": "Graduation", "course": "Mechanical"},
            {"candidate_id": "U1", "qualification": "Diploma", "course": "Civil"},
        ]))
    with pytest.raises(ValueError, match="past_participation"):
        validate_candidate_roster(pd.DataFrame([
            {"candidate_id": "U1", "qualification": "Graduation", "course": "Mechanical",
             "past_participation": 2},
        ]))


def test_corruption_keeps_proxy_labels_and_logs_changes(tmp_path):
    paths = generate(small_config(), SOURCE, tmp_path / "data")
    base = pd.read_csv(paths["pairs"])
    corrupted, log = corrupt_dataset(paths["pairs"], tmp_path / "corrupted.csv", .2, 99)
    dirty = pd.read_csv(corrupted)
    changes = pd.read_csv(log)
    assert dirty.synthetic_proxy_label.notna().all()
    assert "duplicate_identifier_stress" in set(changes.transformation)
    assert dirty.pair_id.duplicated().any()
    assert "class_imbalance_sampling_stress" in set(changes.transformation)


def test_cleaning_restores_id_backed_relations_and_audits_rejections(tmp_path):
    paths = generate(small_config(), SOURCE, tmp_path / "data")
    corrupted, _ = corrupt_dataset(paths["pairs"], tmp_path / "corrupted.csv", .2, 12)
    cleaned, rejected = clean_dataset(corrupted, paths["candidates"], paths["internships"],
                                      tmp_path / "cleaned.csv", tmp_path / "rejected.csv")
    result = pd.read_csv(cleaned)
    assert result.pair_id.is_unique
    assert result.synthetic_proxy_label.notna().all()
    assert Path(rejected).exists()
    assert not result.skill_similarity.dropna().gt(1).any()
    assert Path(cleaned).with_name(Path(cleaned).stem + "_cleaning_audit_log.csv").exists()


def test_allocation_respects_capacity(tmp_path):
    paths = generate(small_config(), SOURCE, tmp_path / "data")
    out = tmp_path / "allocations.csv"
    summary = allocate(paths["pairs"], out)
    allocations = pd.read_csv(out)
    assert summary["capacity_violations"] == 0
    assert summary["eligibility_violations"] == 0
    assert allocations.candidate_id.is_unique
    assert (allocations.groupby("internship_id").size() <=
            allocations.groupby("internship_id").capacity.first()).all()
    assert allocations.allocation_explanation.notna().all()
    assert summary["unallocated_candidates"] == (
        pd.read_csv(paths["pairs"]).candidate_id.nunique() - allocations.candidate_id.nunique()
    )
    assert summary["unused_seats"] == summary["total_seats"] - len(allocations)
    assert summary["allocation_rate"] == summary["unique_candidates_allocated"] / 30


def test_matching_separates_hard_eligibility_from_compatibility_scores():
    candidates = pd.DataFrame([
        {"candidate_id": "C1", "qualification": "Diploma", "course": "Civil Engineering",
         "skills": "Civil Engineering", "sector_interests": "Construction",
         "preferred_state": "State A", "preferred_district": "District A", "policy_group": 1},
        {"candidate_id": "C2", "qualification": "Graduation", "course": "Mechanical Engineering",
         "skills": "Mechanical Engineering", "sector_interests": "Automotive",
         "preferred_state": "State A", "preferred_district": "District A", "policy_group": 2},
    ])
    internships = pd.DataFrame([{
        "internship_id": "I1", "qualification": "Graduation", "course": "Mechanical Engineering",
        "specialization": "Design", "capacity": 1, "state": "State A", "district": "District A",
        "sector": "Automotive", "area": "Manufacturing",
    }])
    matches = match_candidates(candidates, internships, policy_group_column="policy_group")
    by_candidate = matches.set_index("candidate_id")
    assert by_candidate.loc["C1", "eligible"] == 0
    assert "qualification requirement not met" in by_candidate.loc["C1", "eligibility_explanation"]
    assert by_candidate.loc["C2", "eligible"] == 1
    assert by_candidate.loc["C2", "qualification_compatibility"] == 1
    assert by_candidate.loc["C2", "course_compatibility"] == 1
    assert by_candidate.loc["C2", "policy_group"] == "2"
    assert by_candidate.loc["C2", "synthetic_match_score"] <= 1


def test_allocation_applies_explicit_group_minimum_without_scoring_group(tmp_path):
    pairs = pd.DataFrame([
        {"candidate_id": "A", "internship_id": "I", "eligible": 1, "synthetic_match_score": .6,
         "capacity": 1, "candidate_qualification": "Diploma", "internship_qualification": "10th",
         "candidate_course": "Mechanical", "internship_course": "ANY", "policy_group": "A"},
        {"candidate_id": "B", "internship_id": "I", "eligible": 1, "synthetic_match_score": .9,
         "capacity": 1, "candidate_qualification": "Diploma", "internship_qualification": "10th",
         "candidate_course": "Mechanical", "internship_course": "ANY", "policy_group": "B"},
        {"candidate_id": "C", "internship_id": "I", "eligible": 0, "synthetic_match_score": 1.0,
         "capacity": 1, "candidate_qualification": "12th", "internship_qualification": "Graduation",
         "candidate_course": "Mechanical", "internship_course": "ANY", "policy_group": "A"},
    ])
    output = tmp_path / "policy_allocations.csv"
    summary = allocate(
        pairs, output, policy_group_column="policy_group", policy_minimums={"A": 1}
    )
    allocations = pd.read_csv(output)
    assert allocations.candidate_id.tolist() == ["A"]
    assert summary["capacity_violations"] == 0
    assert summary["unallocated_candidates"] == 2
    assert summary["policy_minimums"] == {"A": 1}
    with pytest.raises(ValueError, match="available group size"):
        allocate(
            pairs, tmp_path / "infeasible.csv", policy_group_column="policy_group",
            policy_minimums={"missing": 1},
        )
    with pytest.raises(RuntimeError, match="infeasible"):
        allocate(
            pairs, tmp_path / "ineligible_group.csv", policy_group_column="policy_group",
            policy_minimums={"A": 2},
        )

    profiles = pd.DataFrame([
        {"candidate_id": "A", "policy_group": "A"},
        {"candidate_id": "B", "policy_group": "B"},
        {"candidate_id": "C", "policy_group": "A"},
    ])
    fairness = allocation_fairness_summary(profiles, allocations, "policy_group").set_index("policy_group")
    assert fairness.loc["A", "candidate_count"] == 2
    assert fairness.loc["A", "allocated_count"] == 1
    assert fairness.loc["B", "allocation_rate"] == 0


def test_allocation_rejects_invalid_capacity_and_duplicate_pairs(tmp_path):
    row = {"candidate_id": "A", "internship_id": "I", "eligible": 1, "synthetic_match_score": .5,
           "capacity": 1, "candidate_qualification": "Diploma", "internship_qualification": "10th",
           "candidate_course": "Mechanical", "internship_course": "ANY"}
    duplicate = pd.DataFrame([row, row])
    with pytest.raises(ValueError, match="duplicate candidate"):
        allocate(duplicate, tmp_path / "duplicate.csv")
    invalid_capacity = pd.DataFrame([{**row, "capacity": -1}])
    with pytest.raises(ValueError, match="non-negative"):
        allocate(invalid_capacity, tmp_path / "invalid.csv")


def test_allocation_reports_zero_feasible_assignments_and_unused_capacity(tmp_path):
    pairs = pd.DataFrame([{
        "candidate_id": "A", "internship_id": "I", "eligible": 0,
        "synthetic_match_score": .9, "capacity": 3,
        "candidate_qualification": "12th", "internship_qualification": "Graduation",
        "candidate_course": "Mechanical", "internship_course": "ANY",
    }])
    output = tmp_path / "empty_allocations.csv"
    summary = allocate(pairs, output)
    assert pd.read_csv(output).empty
    assert summary["allocations"] == 0
    assert summary["unallocated_candidates"] == 1
    assert summary["unused_seats"] == 3
    assert summary["capacity_violations"] == 0


def test_evaluation_feature_set_excludes_targets_identifiers_and_provenance():
    forbidden = {
        "candidate_id", "internship_id", "pair_id", "source_internship_id",
        "synthetic_proxy_label", "record_origin", "provenance_json", "label_rule",
    }
    assert forbidden.isdisjoint(FEATURES)


def test_evaluation_splits_keep_each_candidate_in_one_partition():
    frame = pd.DataFrame({
        "candidate_id": [f"C{i}" for i in range(40)],
        "synthetic_proxy_label": [i % 2 for i in range(40)],
    })
    train, validation, test = candidate_grouped_split(frame, seed=13)
    groups = frame.candidate_id.to_numpy()
    train_groups = set(groups[train])
    validation_groups = set(groups[validation])
    test_groups = set(groups[test])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)


def test_ndcg_at_k_rewards_correct_top_rank_order():
    relevance = [1, 0, 1]
    assert _ndcg_at_k(relevance, [3, 1, 2], 1) == 1
    assert _ndcg_at_k(relevance, [3, 1, 2], 3) == 1
    assert _ndcg_at_k(relevance, [1, 2, 3], 3) < 1
