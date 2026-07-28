from __future__ import annotations

import dataclasses
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from gnn_siamese.data.node_pair_alignment import (
    DuplicateResidueKeyError,
    InvalidCoordinateError,
    InvalidResidueKeyError,
    align_node_pair,
)


def _align(
    *,
    mut_res=(10, 11, 13),
    wt_res=(10, 11, 14),
    mut_xyz=((0, 0, 0), (4, 0, 0), (3, 0, 0)),
    wt_xyz=((0, 0, 0), (12, 0, 0), (3, 0, 0)),
    anchor=("A", 10),
    radii=(8.0,),
):
    return align_node_pair(
        [b"A"] * len(mut_res),
        mut_res,
        np.asarray(mut_xyz),
        ["A"] * len(wt_res),
        wt_res,
        np.asarray(wt_xyz),
        anchor_key=anchor,
        radii=radii,
    )


def test_deterministic_union_presence_supports_and_global_indices():
    result = _align(mut_res=(13, 10, 11), wt_res=(11, 14, 10))
    assert result.union_keys == (("A", 10), ("A", 11), ("A", 13), ("A", 14))
    assert result.aligned_keys == (("A", 10), ("A", 11))
    assert result.graph_mut_only_keys == (("A", 13),)
    assert result.graph_wt_only_keys == (("A", 14),)
    assert result.exists_MUT == (True, True, True, False)
    assert result.exists_WT == (True, True, False, True)
    assert tuple(item.support for item in result.presence) == (
        "aligned",
        "aligned",
        "graph_mut_only",
        "graph_wt_only",
    )
    mut_keys = (("A", 13), ("A", 10), ("A", 11))
    wt_keys = (("A", 11), ("A", 14), ("A", 10))
    for position, key in enumerate(result.aligned_keys):
        assert mut_keys[result.mut_aligned_index[position]] == key
        assert wt_keys[result.wt_aligned_index[position]] == key
    assert result.mut_only_index == (0,)
    assert result.wt_only_index == (1,)
    assert all(index >= 0 for index in (*result.mut_aligned_index, *result.wt_aligned_index))


def test_input_order_does_not_change_semantic_contract():
    first = _align()
    second = _align(
        mut_res=(13, 11, 10),
        wt_res=(14, 10, 11),
        mut_xyz=((3, 0, 0), (4, 0, 0), (0, 0, 0)),
        wt_xyz=((3, 0, 0), (0, 0, 0), (12, 0, 0)),
    )
    assert first.union_keys == second.union_keys
    assert first.aligned_keys == second.aligned_keys
    assert first.presence == second.presence
    assert first.metrics == second.metrics
    assert first.local_view(8).K_local_union == second.local_view(8).K_local_union


@pytest.mark.parametrize(
    ("residues", "code"),
    [
        ((10, 10), "duplicate_key"),
        ((10, 10.5), "nonintegral_res_id"),
        ((10, np.nan), "nonfinite_res_id"),
        ((10, np.inf), "nonfinite_res_id"),
    ],
)
def test_duplicate_and_invalid_residue_keys_fail_structurally(residues, code):
    error = DuplicateResidueKeyError if code == "duplicate_key" else InvalidResidueKeyError
    with pytest.raises(error) as caught:
        _align(
            mut_res=residues,
            mut_xyz=((0, 0, 0), (1, 0, 0)),
        )
    assert caught.value.code == code


def test_anchor_aligned_and_missing_anchor_is_described_not_filtered():
    valid = _align()
    assert valid.anchor_aligned
    missing = _align(anchor=("A", 99))
    assert not missing.anchor_aligned
    assert missing.alignment_quality_group == "invalid_identity"
    assert missing.quality_reason_codes == ("anchor_missing_mut", "anchor_missing_wt")
    assert missing.include_in_full_dataset
    assert missing.training_eligibility == "pending"
    assert missing.local_views == ()


def test_empty_alignment_and_zero_denominators_are_explicit():
    empty_alignment = _align(
        mut_res=(10,),
        wt_res=(20,),
        mut_xyz=((0, 0, 0),),
        wt_xyz=((0, 0, 0),),
    )
    assert empty_alignment.alignment_quality_group == "invalid_identity"
    assert "empty_alignment" in empty_alignment.quality_reason_codes
    assert empty_alignment.metrics.coverage_union == 0.0

    empty_graphs = _align(
        mut_res=(),
        wt_res=(),
        mut_xyz=np.empty((0, 3)),
        wt_xyz=np.empty((0, 3)),
    )
    assert empty_graphs.metrics.coverage_union is None
    assert empty_graphs.metrics.coverage_mut is None
    assert empty_graphs.metrics.coverage_wt is None


def test_global_metrics_and_coverages():
    metrics = _align().metrics
    assert dataclasses.asdict(metrics) == {
        "n_union": 4,
        "n_aligned": 2,
        "n_graph_mut_only": 1,
        "n_graph_wt_only": 1,
        "coverage_union": 0.5,
        "coverage_mut": 2 / 3,
        "coverage_wt": 2 / 3,
        "graph_mut_only_fraction_union": 0.25,
        "graph_wt_only_fraction_union": 0.25,
    }


def test_local_sets_indices_global_exclusivity_and_radial_exclusivity():
    result = _align()
    view = result.local_view(8)
    assert view.K_MUT == (("A", 10), ("A", 11), ("A", 13))
    assert view.K_WT == (("A", 10), ("A", 14))
    assert view.K_local_union == (("A", 10), ("A", 11), ("A", 13), ("A", 14))
    assert view.K_local_aligned == (("A", 10), ("A", 11))
    assert view.local_mut_aligned_index == (0, 1)
    assert view.local_wt_aligned_index == (0, 1)
    assert view.local_graph_mut_only_keys == (("A", 13),)
    assert view.local_graph_wt_only_keys == (("A", 14),)
    assert view.radial_mut_only == (("A", 11),)
    assert view.radial_wt_only == ()
    states = {state.key: state for state in view.aligned_radial_states}
    assert states[("A", 11)].inside_radius_MUT
    assert not states[("A", 11)].inside_radius_WT
    assert states[("A", 11)].delta_distance == -8.0
    assert set(view.radial_mut_only).isdisjoint(result.graph_mut_only_keys)
    assert set(view.radial_wt_only).isdisjoint(result.graph_wt_only_keys)


def test_radial_difference_alone_does_not_degrade_comparability():
    result = _align(
        mut_res=(10, 11),
        wt_res=(10, 11),
        mut_xyz=((0, 0, 0), (4, 0, 0)),
        wt_xyz=((0, 0, 0), (12, 0, 0)),
    )
    assert result.local_view(8).radial_mut_only == (("A", 11),)
    assert result.alignment_quality_group == "high_comparability"
    assert result.baseline_clean_candidate

    reverse = _align(
        mut_res=(10, 11),
        wt_res=(10, 11),
        mut_xyz=((0, 0, 0), (12, 0, 0)),
        wt_xyz=((0, 0, 0), (4, 0, 0)),
    )
    assert reverse.local_view(8).radial_wt_only == (("A", 11),)
    assert reverse.alignment_quality_group == "high_comparability"


def test_global_exclusive_inside_and_outside_technical_radius():
    inside = _align(
        mut_res=(10, 11, 13),
        wt_res=(10, 11),
        wt_xyz=((0, 0, 0), (12, 0, 0)),
    )
    assert inside.alignment_quality_group == "limited_local_comparability"
    assert not inside.baseline_clean_candidate
    assert inside.include_in_full_dataset

    outside = _align(
        mut_res=(10, 11, 13),
        wt_res=(10, 11),
        mut_xyz=((0, 0, 0), (4, 0, 0), (9, 0, 0)),
        wt_xyz=((0, 0, 0), (12, 0, 0)),
    )
    assert outside.alignment_quality_group == "high_comparability"
    assert outside.baseline_clean_candidate


@pytest.mark.parametrize(
    ("mut_xyz", "code"),
    [
        (((0, 0, 0), (np.nan, 0, 0), (3, 0, 0)), "nonfinite_coordinate"),
        (((0, 0, 0), (np.inf, 0, 0), (3, 0, 0)), "nonfinite_coordinate"),
        (((0, 0), (1, 0), (3, 0)), "invalid_coordinate_shape"),
    ],
)
def test_invalid_coordinates_fail_structurally(mut_xyz, code):
    with pytest.raises(InvalidCoordinateError) as caught:
        _align(mut_xyz=mut_xyz)
    assert caught.value.code == code


def test_result_is_deeply_immutable_and_contains_no_padding_contract():
    result = _align()
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.alignment_quality_group = "changed"
    with pytest.raises(TypeError):
        result.union_keys[0] = ("B", 1)
    assert -1 not in (
        *result.mut_aligned_index,
        *result.wt_aligned_index,
        *result.mut_only_index,
        *result.wt_only_index,
    )
    assert not hasattr(result, "H_MUT_union")
    assert not hasattr(result, "H_WT_union")


def test_inputs_are_not_modified_and_calls_are_deterministic():
    mut_chain = np.asarray([b"A", b"A", b"A"])
    mut_res = np.asarray([10.0, 11.0, 13.0])
    mut_xyz = np.asarray(((0, 0, 0), (4, 0, 0), (3, 0, 0)), dtype=float)
    snapshots = tuple(value.copy() for value in (mut_chain, mut_res, mut_xyz))
    kwargs = dict(
        mut_chain_ids=mut_chain,
        mut_res_ids=mut_res,
        mut_positions=mut_xyz,
        wt_chain_ids=["A", "A"],
        wt_res_ids=[10, 11],
        wt_positions=np.asarray(((0, 0, 0), (12, 0, 0))),
        anchor_key=("A", 10),
        radii=(4, 8, 12),
    )
    first = align_node_pair(**kwargs)
    second = align_node_pair(**kwargs)
    assert first == second
    for value, snapshot in zip((mut_chain, mut_res, mut_xyz), snapshots):
        np.testing.assert_array_equal(value, snapshot)


@pytest.mark.integration
@pytest.mark.hdf5
def test_real_483_alignment_if_configured():
    """Run with MODEL_A_MUT_HDF5 and MODEL_A_WT_HDF5; otherwise skip."""

    mut_value = os.environ.get("MODEL_A_MUT_HDF5")
    wt_value = os.environ.get("MODEL_A_WT_HDF5")
    if not mut_value or not wt_value:
        pytest.skip("Set MODEL_A_MUT_HDF5 and MODEL_A_WT_HDF5 for the controlled integration.")
    mut_path, wt_path = Path(mut_value), Path(wt_value)
    if not mut_path.is_file() or not wt_path.is_file():
        pytest.skip("Configured Model A integration HDF5 paths are unavailable.")

    import h5py

    from gnn_siamese.data.pairing import pair_mutants_with_wt, parse_variant_signature

    with h5py.File(mut_path, "r") as mut_h5, h5py.File(wt_path, "r") as wt_h5:
        mutants = [
            {"variant_id": key, "graph_id": key}
            for key in sorted(mut_h5)
            if parse_variant_signature(key).mut_aa != parse_variant_signature(key).wt_aa
        ]
        companions = [{"variant_id": key, "graph_id": key} for key in sorted(wt_h5)]
        pairs = pair_mutants_with_wt(mutants, companions)
        results = []
        for pair in pairs:
            mut_group = mut_h5[pair["graph_id"]]["node_features"]
            wt_group = wt_h5[pair["wt_companion_id"]]["node_features"]
            signature = parse_variant_signature(pair["variant_id"])
            results.append(
                (
                    signature,
                    align_node_pair(
                        mut_group["_chain_id"][()],
                        mut_group["res_id"][()],
                        mut_group["_position"][()],
                        wt_group["_chain_id"][()],
                        wt_group["res_id"][()],
                        wt_group["_position"][()],
                        anchor_key=(signature.chain_id, signature.position),
                        radii=(8.0,),
                    ),
                )
            )

    counts = Counter(result.alignment_quality_group for _, result in results)
    assert len(results) == 483
    assert counts == {
        "high_comparability": 464,
        "limited_local_comparability": 19,
    }
    r101h = [
        result
        for signature, result in results
        if signature.position == 101 and signature.mut_aa == "H"
    ]
    assert len(r101h) == 1
    assert r101h[0].anchor_aligned
    assert r101h[0].alignment_quality_group == "limited_local_comparability"
    assert r101h[0].include_in_full_dataset
    assert not r101h[0].baseline_clean_candidate
