from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from gnn_siamese.builders import build_dataset_bundle, build_split_bundle
from gnn_siamese.config import load_config
from gnn_siamese.data import load_leave_position_out_split


REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_SPLIT_PATH = REPO_ROOT / "splits" / "leave_position_out_seed_42.json"
NATIVE_WT_KEY = "residue-srv:A:419:Leucine->Leucine:PKP2_WT"


def test_full_split_artifact_is_complete_leakage_free_and_excludes_native_wt() -> None:
    payload = json.loads(FULL_SPLIT_PATH.read_text(encoding="utf-8"))
    split = load_leave_position_out_split(FULL_SPLIT_PATH)
    variant_ids = split.variant_ids_by_partition()
    positions = split.positions_by_partition()
    variant_sets = {name: set(values) for name, values in variant_ids.items()}
    position_sets = {name: set(values) for name, values in positions.items()}

    assert split.split_type == "leave_position_out"
    assert split.config.seed == 42
    assert sum(len(values) for values in variant_sets.values()) == 483
    assert len(set.union(*variant_sets.values())) == 483
    assert sum(len(partition) for partition in split.assignments_by_partition().values()) == 483
    assert variant_sets["train"].isdisjoint(variant_sets["validation"])
    assert variant_sets["train"].isdisjoint(variant_sets["test"])
    assert variant_sets["validation"].isdisjoint(variant_sets["test"])
    assert position_sets["train"].isdisjoint(position_sets["validation"])
    assert position_sets["train"].isdisjoint(position_sets["test"])
    assert position_sets["validation"].isdisjoint(position_sets["test"])
    assert all(NATIVE_WT_KEY not in values for values in variant_sets.values())

    metadata = payload["audit_metadata"]
    assert metadata["created_from_variant_count"] == 483
    assert metadata["native_wt_control_count"] == 1
    assert metadata["evaluation_controls"] == [NATIVE_WT_KEY]
    assert metadata["a1_inventory"]["exact_match"] is True


def test_model_a_pilot_is_frozen_to_the_full_split_not_smoke() -> None:
    config = load_config(REPO_ROOT / "configs" / "model_a_pilot.yaml")
    split_path = config["split"]["persist_path"]
    split = load_leave_position_out_split(REPO_ROOT / split_path)

    assert split_path == "splits/leave_position_out_seed_42.json"
    assert config["split"]["type"] == "leave_position_out"
    assert config["split"]["seed"] == 42
    assert config["split"]["allow_create"] is False
    assert len(split.assignments) == 483
    assert len(split.assignments) != 8


@pytest.mark.skipif(
    not os.environ.get("MODEL_A_MUT_HDF5") or not os.environ.get("MODEL_A_WT_HDF5"),
    reason="Set canonical HDF5 paths to verify the shared full split through A and B builders.",
)
def test_model_a_and_model_b_load_the_exact_same_full_split_without_creation() -> None:
    assignments = []
    for config_name in ("model_a_pilot.yaml", "model_b_baseline.yaml"):
        config_path = REPO_ROOT / "configs" / config_name
        config = load_config(config_path)
        config["__config_path__"] = str(config_path)
        config["paths"]["mutants_hdf5"] = os.environ["MODEL_A_MUT_HDF5"]
        config["paths"]["wt_companion_hdf5"] = os.environ["MODEL_A_WT_HDF5"]
        config["split"].update(
            {
                "type": "leave_position_out",
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "group_key": "position",
                "shuffle_groups": True,
                "seed": 42,
                "persist_path": str(FULL_SPLIT_PATH),
                "allow_create": False,
            }
        )
        dataset = build_dataset_bundle(config).dataset
        bundle = build_split_bundle(config, dataset)
        assert bundle.created is False
        assert len(dataset.pairs) == 483
        assert len(dataset.native_wt_controls) == 1
        assignments.append(bundle.split.assignments)

    assert assignments[0] == assignments[1]
