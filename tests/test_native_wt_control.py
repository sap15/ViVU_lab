from __future__ import annotations

import os
from pathlib import Path

import pytest

from gnn_siamese.data import (
    BIOLOGICAL_VARIANT,
    NATIVE_WT_CONTROL,
    MutWtPairDataset,
    build_leave_position_out_split,
    classify_variant_record,
)
from tests.test_mut_wt_dataset import _build_config, _build_schema, _create_graph


NATIVE_WT_KEY = "residue-srv:A:419:Leucine->Leucine:PKP2_WT"


def _dataset_with_control(tmp_path: Path) -> MutWtPairDataset:
    mutant_path = tmp_path / "mutants.h5"
    wt_path = tmp_path / "wt.h5"
    variants = (
        "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
        # STOP-like generation may use WT->WT query amino acids; its explicit
        # pos_*_STOP identity must keep it in the biological inventory.
        "residue-srv:A:200:Alanine->Alanine:pos_200_A_STOP",
    )
    for key, diff in zip(variants, ([0.0, 1.0, 0.0], [0.0, 0.0, 0.0])):
        _create_graph(mutant_path, key, diff_mass=diff)
    _create_graph(mutant_path, NATIVE_WT_KEY, diff_mass=[0.0, 0.0, 0.0])
    _create_graph(
        wt_path,
        "residue-srv:A:100:Glycine->Glycine:PKP2_WT",
        diff_mass=[0.0, 0.0, 0.0],
    )
    _create_graph(
        wt_path,
        "residue-srv:A:200:Alanine->Alanine:PKP2_WT",
        diff_mass=[0.0, 0.0, 0.0],
    )
    _create_graph(wt_path, NATIVE_WT_KEY, diff_mass=[0.0, 0.0, 0.0])
    return MutWtPairDataset(
        mutant_h5_path=mutant_path,
        wt_h5_path=wt_path,
        config=_build_config(),
        schema=_build_schema(),
    )


def _split_config() -> dict:
    return {
        "split": {
            "type": "leave_position_out",
            "validation_fraction": 0.0,
            "test_fraction": 0.0,
            "group_key": "position",
            "shuffle_groups": True,
            "seed": 42,
            "enforce_no_position_overlap": True,
            "enforce_no_variant_overlap": True,
            "leave_neighborhood_out": {"enabled": False},
        }
    }


def test_native_wt_control_is_classified_safely_and_reproducibly() -> None:
    assert classify_variant_record(NATIVE_WT_KEY) == NATIVE_WT_CONTROL
    assert classify_variant_record(NATIVE_WT_KEY) == NATIVE_WT_CONTROL
    assert (
        classify_variant_record(
            "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"
        )
        == BIOLOGICAL_VARIANT
    )
    assert (
        classify_variant_record(
            "residue-srv:A:200:Alanine->Alanine:pos_200_A_STOP"
        )
        == BIOLOGICAL_VARIANT
    )


def test_native_wt_control_is_outside_splits_but_available_for_inference(
    tmp_path: Path,
) -> None:
    dataset = _dataset_with_control(tmp_path)

    assert [record.variant_id for record in dataset.pairs] == [
        "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
        "residue-srv:A:200:Alanine->Alanine:pos_200_A_STOP",
    ]
    assert len(dataset.native_wt_controls) == 1
    control_record = dataset.native_wt_controls[0]
    assert control_record.variant_id == NATIVE_WT_KEY
    assert control_record.record_type == NATIVE_WT_CONTROL
    assert control_record.role == "evaluation_control"
    assert control_record.trainable is False
    assert control_record.available_for_inference is True

    split = build_leave_position_out_split(dataset, _split_config())
    assigned = {assignment.variant_id for assignment in split.assignments}
    assert NATIVE_WT_KEY not in assigned

    control = dataset.get_native_wt_control()
    assert control.variant_id == NATIVE_WT_KEY
    assert control.metadata["record_type"] == NATIVE_WT_CONTROL
    assert control.metadata["role"] == "evaluation_control"
    assert control.metadata["trainable"] is False
    assert control.metadata["used_for_split"] is False
    assert control.metadata["used_for_training"] is False
    assert control.metadata["used_for_validation"] is False
    assert control.metadata["used_for_test_loss"] is False
    assert control.metadata["available_for_inference"] is True


@pytest.mark.skipif(
    not os.environ.get("MODEL_A_MUT_HDF5") or not os.environ.get("MODEL_A_WT_HDF5"),
    reason="Set MODEL_A_MUT_HDF5 and MODEL_A_WT_HDF5 for the controlled inventory integration.",
)
def test_real_inventory_has_483_variants_and_one_native_wt_control() -> None:
    dataset = MutWtPairDataset(
        mutant_h5_path=os.environ["MODEL_A_MUT_HDF5"],
        wt_h5_path=os.environ["MODEL_A_WT_HDF5"],
        config=_build_config(),
        schema=_build_schema(),
    )

    assert len(dataset.pairs) == 483
    assert len(dataset.native_wt_controls) == 1
    assert dataset.native_wt_controls[0].variant_id == NATIVE_WT_KEY
    assert NATIVE_WT_KEY not in {record.variant_id for record in dataset.pairs}
