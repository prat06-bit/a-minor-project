import json
from pathlib import Path

import pandas as pd

from src.pmis_pipeline import generate, corrupt_dataset, clean_dataset, allocate


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
    assert allocations.candidate_id.is_unique
    assert (allocations.groupby("internship_id").size() <=
            allocations.groupby("internship_id").capacity.first()).all()
    assert allocations.allocation_explanation.notna().all()
