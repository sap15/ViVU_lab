from __future__ import annotations

import dataclasses

import pytest
import torch

from gnn_siamese.models import (
    BranchMultiscalePooling,
    ModelAMultiscalePooling,
    ModelAMultiscalePoolingOutput,
    ScalePoolResult,
    aligned_selection_mask,
    indices_to_mask,
    segmented_pool,
)
from gnn_siamese.models.delta_block import NodeDeltaBlock


def _inputs(
    *,
    requires_grad: bool = False,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor | int]:
    H_MUT = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
        device=device,
        requires_grad=requires_grad,
    )
    H_WT = torch.tensor(
        [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [20.0, 10.0], [40.0, 30.0], [60.0, 50.0]],
        device=device,
        requires_grad=requires_grad,
    )
    H_delta = torch.tensor(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [7.0, 70.0, 700.0]],
        device=device,
        requires_grad=requires_grad,
    )
    return {
        "H_MUT": H_MUT,
        "H_WT": H_WT,
        "H_delta": H_delta,
        "batch_MUT": torch.tensor([0, 0, 1, 2, 2], device=device),
        "batch_WT": torch.tensor([0, 0, 0, 1, 2, 2], device=device),
        "aligned_pair_batch": torch.tensor([0, 0, 2], device=device),
        "alignment_ptr": torch.tensor([0, 2, 2, 3], device=device),
        "num_pairs": 3,
    }


def _mask(values: list[bool], device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bool, device=device)


def _all_scale_masks(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    return {
        "mutation_mask_MUT": _mask([False, True, True, False, True], device),
        "mutation_mask_WT": _mask([False, True, False, True, False, True], device),
        # Pair 1 mutation is not aligned and is therefore explicitly invalid in delta.
        "mutation_mask_delta": _mask([False, True, False], device),
        "local_mask_MUT": _mask([True, True, True, False, True], device),
        "local_mask_WT": _mask([True, True, False, True, False, True], device),
        "local_mask_delta": _mask([True, False, True], device),
        "domain_mask_MUT": _mask([True, True, False, True, True], device),
        "domain_mask_WT": _mask([True, False, True, False, True, True], device),
        "domain_mask_delta": _mask([False, True, True], device),
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("sum", [[4.0, 6.0], [0.0, 0.0], [50.0, 60.0]]),
        ("mean", [[2.0, 3.0], [0.0, 0.0], [50.0, 60.0]]),
    ],
)
def test_segmented_pool_mean_sum_and_empty_segment(mode: str, expected: list[list[float]]) -> None:
    embeddings = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [50.0, 60.0]],
        dtype=torch.float64,
    )
    result = segmented_pool(
        embeddings,
        torch.tensor([0, 0, 2, 2]),
        _mask([True, True, False, True]),
        num_pairs=3,
        mode=mode,
    )

    assert isinstance(result, ScalePoolResult)
    torch.testing.assert_close(result.values, torch.tensor(expected, dtype=torch.float64))
    assert result.counts.tolist() == [2, 0, 1]
    assert result.valid_mask.tolist() == [True, False, True]
    assert result.values.dtype == embeddings.dtype
    assert result.counts.dtype == torch.long
    assert result.valid_mask.dtype == torch.bool


def test_heterogeneous_global_pooling_keeps_all_three_branches_separate() -> None:
    output = ModelAMultiscalePooling(enabled_scales=("global",))(**_inputs())

    assert isinstance(output, ModelAMultiscalePoolingOutput)
    assert isinstance(output.MUT, BranchMultiscalePooling)
    torch.testing.assert_close(
        output.MUT.global_.values,
        torch.tensor([[2.0, 3.0], [10.0, 20.0], [40.0, 50.0]]),
    )
    torch.testing.assert_close(
        output.WT.global_.values,
        torch.tensor([[4.0, 3.0], [20.0, 10.0], [50.0, 40.0]]),
    )
    torch.testing.assert_close(
        output.delta.global_.values,
        torch.tensor([[1.5, 15.0, 150.0], [0.0, 0.0, 0.0], [7.0, 70.0, 700.0]]),
    )
    assert output.delta.global_.counts.tolist() == [2, 0, 1]
    assert output.delta.global_.valid_mask.tolist() == [True, False, True]
    assert output.MUT.mutation is None
    assert output.WT.local is None
    assert output.delta.domain is None


def test_mutation_pooling_requires_one_mut_and_wt_but_allows_unaligned_delta() -> None:
    output = ModelAMultiscalePooling(enabled_scales=("mutation",))(
        **_inputs(),
        **{
            key: value
            for key, value in _all_scale_masks().items()
            if key.startswith("mutation_")
        },
    )

    torch.testing.assert_close(
        output.MUT.mutation.values,
        torch.tensor([[3.0, 4.0], [10.0, 20.0], [50.0, 60.0]]),
    )
    torch.testing.assert_close(
        output.WT.mutation.values,
        torch.tensor([[4.0, 3.0], [20.0, 10.0], [60.0, 50.0]]),
    )
    assert output.delta.mutation.counts.tolist() == [1, 0, 0]
    assert output.delta.mutation.valid_mask.tolist() == [True, False, False]


def test_mutation_pooling_rejects_multiple_or_missing_required_nodes() -> None:
    inputs = _inputs()
    mutation_masks = {
        key: value
        for key, value in _all_scale_masks().items()
        if key.startswith("mutation_")
    }
    mutation_masks["mutation_mask_MUT"] = _mask([True, True, True, False, True])
    with pytest.raises(ValueError, match="mutation_mask_MUT must select at most one"):
        ModelAMultiscalePooling(enabled_scales=("mutation",))(
            **inputs, **mutation_masks
        )

    mutation_masks["mutation_mask_MUT"] = _mask([False, True, False, False, True])
    with pytest.raises(ValueError, match="mutation_mask_MUT must select exactly one"):
        ModelAMultiscalePooling(enabled_scales=("mutation",))(
            **inputs, **mutation_masks
        )

    output = ModelAMultiscalePooling(
        enabled_scales=("mutation",), allow_missing_mutation=True
    )(**inputs, **mutation_masks)
    assert output.MUT.mutation.valid_mask.tolist() == [True, False, True]


def test_local_a1_indices_map_exactly_to_noncontiguous_delta_rows() -> None:
    # Global aligned MUT indices by pair: [0, 1] | [] | [4].
    # Local A1 MUT indices:             [1]    | [] | [4].
    local_delta_mask = aligned_selection_mask(
        torch.tensor([0, 1, 4]),
        torch.tensor([0, 2, 2, 3]),
        torch.tensor([1, 4]),
        torch.tensor([0, 1, 1, 2]),
        num_pairs=3,
    )
    assert local_delta_mask.tolist() == [False, True, True]

    output = ModelAMultiscalePooling(enabled_scales=("local",))(
        **_inputs(),
        local_mask_MUT=indices_to_mask(torch.tensor([1, 4]), row_count=5),
        local_mask_WT=indices_to_mask(torch.tensor([1, 5]), row_count=6),
        local_mask_delta=local_delta_mask,
    )
    torch.testing.assert_close(
        output.delta.local.values,
        torch.tensor([[2.0, 20.0, 200.0], [0.0, 0.0, 0.0], [7.0, 70.0, 700.0]]),
    )
    assert output.delta.local.counts.tolist() == [1, 0, 1]


def test_aligned_selection_rejects_missing_or_ambiguous_correspondence() -> None:
    with pytest.raises(ValueError, match="must occur exactly once.*found 0"):
        aligned_selection_mask(
            torch.tensor([0, 1]),
            torch.tensor([0, 2]),
            torch.tensor([2]),
            torch.tensor([0, 1]),
            num_pairs=1,
        )
    with pytest.raises(ValueError, match="must occur exactly once.*found 2"):
        aligned_selection_mask(
            torch.tensor([1, 1]),
            torch.tensor([0, 2]),
            torch.tensor([1]),
            torch.tensor([0, 1]),
            num_pairs=1,
        )


def test_aligned_selection_handles_unsorted_segments_without_pair_mixing() -> None:
    mask = aligned_selection_mask(
        torch.tensor([4, 1, 3, 8, 6]),
        torch.tensor([0, 3, 3, 5]),
        torch.tensor([1, 3, 6]),
        torch.tensor([0, 2, 2, 3]),
        num_pairs=3,
    )
    assert mask.tolist() == [False, True, True, False, True]


def test_domain_external_masks_are_required_only_when_enabled() -> None:
    without_domain = ModelAMultiscalePooling(enabled_scales=("global",))(**_inputs())
    assert without_domain.MUT.domain is None

    with pytest.raises(ValueError, match="domain_mask_MUT is required"):
        ModelAMultiscalePooling(enabled_scales=("domain",))(**_inputs())

    masks = {
        key: value
        for key, value in _all_scale_masks().items()
        if key.startswith("domain_")
    }
    output = ModelAMultiscalePooling(enabled_scales=("domain",))(
        **_inputs(), **masks
    )
    assert output.MUT.domain.counts.tolist() == [2, 0, 2]
    assert output.WT.domain.counts.tolist() == [2, 0, 2]
    assert output.delta.domain.counts.tolist() == [1, 0, 1]


@pytest.mark.parametrize(
    "enabled",
    [
        ("mutation", "global"),
        ("local", "global"),
        ("mutation", "local", "global"),
        ("mutation", "local", "domain", "global"),
    ],
)
def test_scale_ablations_return_none_for_disabled_scales(enabled: tuple[str, ...]) -> None:
    masks = {
        key: value
        for key, value in _all_scale_masks().items()
        if key.split("_mask_")[0] in enabled
    }
    output = ModelAMultiscalePooling(enabled_scales=enabled)(**_inputs(), **masks)
    attribute = {"global": "global_"}
    for branch in (output.MUT, output.WT, output.delta):
        for scale in ("mutation", "local", "domain", "global"):
            value = getattr(branch, attribute.get(scale, scale))
            assert (value is not None) == (scale in enabled)


def test_shapes_dtypes_devices_determinism_and_immutable_output() -> None:
    pooling = ModelAMultiscalePooling(
        enabled_scales=("mutation", "local", "domain", "global")
    )
    inputs = _inputs()
    masks = _all_scale_masks()
    output_a = pooling(**inputs, **masks)
    output_b = pooling(**inputs, **masks)

    for branch_name, dimension in (("MUT", 2), ("WT", 2), ("delta", 3)):
        branch_a = getattr(output_a, branch_name)
        branch_b = getattr(output_b, branch_name)
        for scale_name in ("mutation", "local", "domain", "global_"):
            result_a = getattr(branch_a, scale_name)
            result_b = getattr(branch_b, scale_name)
            assert result_a.values.shape == (3, dimension)
            assert result_a.values.device.type == "cpu"
            assert torch.equal(result_a.values, result_b.values)
    with pytest.raises(dataclasses.FrozenInstanceError):
        output_a.MUT = output_b.MUT


def test_backward_reaches_selected_rows_in_all_branches_without_detach() -> None:
    inputs = _inputs(requires_grad=True)
    pooling = ModelAMultiscalePooling(enabled_scales=("local",))
    masks = {
        key: value
        for key, value in _all_scale_masks().items()
        if key.startswith("local_")
    }
    output = pooling(**inputs, **masks)
    loss = sum(
        branch.local.values.square().sum()
        for branch in (output.MUT, output.WT, output.delta)
    )
    loss.backward()

    for name in ("H_MUT", "H_WT", "H_delta"):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert inputs["H_MUT"].grad[0].abs().sum() > 0
    assert inputs["H_MUT"].grad[3].abs().sum() == 0
    assert inputs["H_WT"].grad[0].abs().sum() > 0
    assert inputs["H_WT"].grad[2].abs().sum() == 0
    assert inputs["H_delta"].grad[0].abs().sum() > 0
    assert inputs["H_delta"].grad[1].abs().sum() == 0


@pytest.mark.parametrize("scale", ["mutation", "domain", "global"])
def test_each_remaining_scale_propagates_gradients_to_all_three_branches(
    scale: str,
) -> None:
    inputs = _inputs(requires_grad=True)
    masks = {
        key: value
        for key, value in _all_scale_masks().items()
        if key.startswith(f"{scale}_")
    }
    output = ModelAMultiscalePooling(enabled_scales=(scale,))(**inputs, **masks)
    attribute = "global_" if scale == "global" else scale
    results = [getattr(branch, attribute) for branch in (output.MUT, output.WT, output.delta)]
    loss = sum(result.values.square().sum() for result in results)
    loss.backward()

    for name in ("H_MUT", "H_WT", "H_delta"):
        gradient = inputs[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_validation_messages_cover_embeddings_batches_masks_and_ptr() -> None:
    with pytest.raises(ValueError, match="H_MUT contains NaN or Inf"):
        ModelAMultiscalePooling(enabled_scales=("global",))(
            **{**_inputs(), "H_MUT": torch.tensor([[float("nan"), 0.0]])}
        )
    with pytest.raises(TypeError, match="batch_MUT must use torch.long"):
        ModelAMultiscalePooling(enabled_scales=("global",))(
            **{**_inputs(), "batch_MUT": torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0])}
        )
    with pytest.raises(TypeError, match="local_mask_MUT must use torch.bool"):
        ModelAMultiscalePooling(enabled_scales=("local",))(
            **_inputs(),
            local_mask_MUT=torch.ones(5),
            local_mask_WT=_mask([True] * 6),
            local_mask_delta=_mask([True] * 3),
        )
    with pytest.raises(ValueError, match="alignment_ptr must end at 3"):
        ModelAMultiscalePooling(enabled_scales=("global",))(
            **{**_inputs(), "alignment_ptr": torch.tensor([0, 1, 1, 2])}
        )
    with pytest.raises(ValueError, match="incompatible with alignment_ptr"):
        ModelAMultiscalePooling(enabled_scales=("global",))(
            **{**_inputs(), "aligned_pair_batch": torch.tensor([0, 2, 2])}
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"enabled_scales": ()}, ValueError, "at least one"),
        ({"enabled_scales": ("global", "global")}, ValueError, "duplicates"),
        ({"enabled_scales": ("unknown",)}, ValueError, "unknown scales"),
        ({"mode": "max"}, ValueError, "mode must be one of"),
    ],
)
def test_configuration_validation(kwargs: dict[str, object], error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        ModelAMultiscalePooling(**kwargs)


def test_node_delta_output_is_directly_compatible_with_global_pooling() -> None:
    block = NodeDeltaBlock(
        input_dim=2, hidden_dim=5, output_dim=4, dropout=0.0
    )
    H_MUT = torch.randn(5, 2)
    H_WT = torch.randn(6, 2)
    mut_index = torch.tensor([0, 1, 4])
    wt_index = torch.tensor([0, 1, 5])
    pair_batch = torch.tensor([0, 0, 2])
    ptr = torch.tensor([0, 2, 2, 3])
    delta_output = block(
        H_MUT, H_WT, mut_index, wt_index, pair_batch, ptr
    )

    output = ModelAMultiscalePooling(enabled_scales=("global",))(
        H_MUT=H_MUT,
        H_WT=H_WT,
        H_delta=delta_output.H_delta,
        batch_MUT=torch.tensor([0, 0, 1, 2, 2]),
        batch_WT=torch.tensor([0, 0, 0, 1, 2, 2]),
        aligned_pair_batch=delta_output.aligned_pair_batch,
        alignment_ptr=delta_output.alignment_ptr,
        num_pairs=3,
    )
    assert output.delta.global_.values.shape == (3, block.output_dim)
    assert output.delta.global_.valid_mask.tolist() == [True, False, True]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_preserves_device_and_backward() -> None:
    device = torch.device("cuda")
    with pytest.raises(ValueError, match="must be on the same device"):
        ModelAMultiscalePooling(enabled_scales=("global",))(
            **{**_inputs(), "H_delta": _inputs(device=device)["H_delta"]}
        )

    inputs = _inputs(requires_grad=True, device=device)
    masks = {
        key: value
        for key, value in _all_scale_masks(device).items()
        if key.startswith("local_")
    }
    output = ModelAMultiscalePooling(enabled_scales=("local",)).cuda()(
        **inputs, **masks
    )
    assert output.delta.local.values.device == device
    output.delta.local.values.sum().backward()
    assert inputs["H_delta"].grad is not None
