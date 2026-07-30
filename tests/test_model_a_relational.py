from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from gnn_siamese.models.multiscale_pooling_a import (
    BranchMultiscalePooling,
    ModelAMultiscalePoolingOutput,
    ScalePoolResult,
)
from gnn_siamese.models.multiscale_relational_a import (
    ModelAMultiscaleRelational,
    build_scale_relational,
)


def _result(values: torch.Tensor, counts: list[int] | None = None) -> ScalePoolResult:
    actual = [1] * values.shape[0] if counts is None else counts
    count_tensor = torch.tensor(actual, dtype=torch.long, device=values.device)
    return ScalePoolResult(values, count_tensor > 0, count_tensor)


def _pooling(
    *,
    dimension: int = 8,
    domain: bool = False,
    empty_local: bool = False,
) -> ModelAMultiscalePoolingOutput:
    branches: dict[str, BranchMultiscalePooling] = {}
    for branch_index, branch in enumerate(("MUT", "WT", "delta"), start=1):
        results = {}
        for scale_index, scale in enumerate(("mutation", "local", "domain", "global"), start=1):
            if scale == "domain" and not domain:
                results[scale] = None
                continue
            values = torch.full((2, dimension), float(10 * branch_index + scale_index))
            counts = [1, 0] if empty_local and scale == "local" else [1, 1]
            if counts[1] == 0:
                values[1].zero_()
            results[scale] = _result(values, counts)
        branches[branch] = BranchMultiscalePooling(
            mutation=results["mutation"],
            local=results["local"],
            domain=results["domain"],
            global_=results["global"],
        )
    return ModelAMultiscalePoolingOutput(
        MUT=branches["MUT"], WT=branches["WT"], delta=branches["delta"]
    )


def _replace_scale_result(
    pooling: ModelAMultiscalePoolingOutput,
    *,
    branch: str,
    scale: str,
    result: ScalePoolResult,
) -> ModelAMultiscalePoolingOutput:
    branch_output = getattr(pooling, branch)
    branch_output = replace(
        branch_output,
        **{("global_" if scale == "global" else scale): result},
    )
    return replace(pooling, **{branch: branch_output})


def test_scale_relational_has_exact_five_block_composition() -> None:
    mut = torch.tensor([[1.0, -2.0]])
    wt = torch.tensor([[3.0, 4.0]])
    delta = torch.tensor([[7.0, 8.0]])
    output = build_scale_relational(mut, wt, delta, scale="local")
    expected = torch.cat((mut, wt, mut - wt, (mut - wt).abs(), delta), dim=-1)
    assert torch.equal(output, expected)
    assert output.shape == (1, 10)


def test_shapes_canonical_order_and_domain_ablation() -> None:
    no_domain = ModelAMultiscaleRelational(
        embedding_dim=8,
        active_scales=("global", "mutation", "local"),
        pair_fusion_enabled=False,
        pair_fusion_output_dim=120,
    )(_pooling())
    assert no_domain.active_scales == ("mutation", "local", "global")
    assert no_domain.scale_order == ("mutation", "local", "domain", "global")
    assert no_domain.z_delta_mutation.shape == (2, 40)
    assert no_domain.h_pair_delta.shape == (2, 120)
    assert no_domain.z_delta_domain is None
    assert no_domain.pair_fusion_mode == "identity"
    assert torch.equal(no_domain.z_delta_pair, no_domain.h_pair_delta)
    assert torch.equal(
        no_domain.h_pair_delta,
        torch.cat(
            (
                no_domain.z_delta_mutation,
                no_domain.z_delta_local,
                no_domain.z_delta_global,
            ),
            dim=-1,
        ),
    )

    with_domain = ModelAMultiscaleRelational(
        embedding_dim=8,
        active_scales=("global", "domain", "local", "mutation"),
        pair_fusion_enabled=False,
        pair_fusion_output_dim=160,
    )(_pooling(domain=True))
    assert with_domain.active_scales == ("mutation", "local", "domain", "global")
    assert with_domain.h_pair_delta.shape == (2, 160)


def test_empty_scale_metadata_is_preserved_without_nan() -> None:
    output = ModelAMultiscaleRelational(
        embedding_dim=8,
        pair_fusion_enabled=False,
        pair_fusion_output_dim=120,
    )(_pooling(empty_local=True))
    assert output.scale_counts["local"]["MUT"].tolist() == [1, 0]
    assert output.scale_valid_masks["local"].tolist() == [True, False]
    assert output.pair_valid_mask.tolist() == [True, False]
    assert torch.isfinite(output.h_pair_delta).all()
    assert torch.count_nonzero(output.z_delta_local[1]) == 0


def test_direction_changes_when_mut_and_wt_are_swapped() -> None:
    mut = torch.tensor([[2.0, 5.0]])
    wt = torch.tensor([[1.0, 3.0]])
    delta = torch.tensor([[0.5, -0.5]])
    forward = build_scale_relational(mut, wt, delta, scale="global")
    reverse = build_scale_relational(wt, mut, delta, scale="global")
    assert torch.equal(forward[:, 4:6], -reverse[:, 4:6])
    assert not torch.equal(forward, reverse)


def test_swapping_mut_and_wt_changes_pair_and_active_fusion_output() -> None:
    torch.manual_seed(9)
    pooling = _pooling()
    swapped = ModelAMultiscalePoolingOutput(
        MUT=pooling.WT,
        WT=pooling.MUT,
        delta=pooling.delta,
    )
    module = ModelAMultiscaleRelational(
        embedding_dim=8,
        pair_fusion_hidden_dim=13,
        pair_fusion_output_dim=7,
        pair_fusion_dropout=0.0,
    )
    forward = module(pooling)
    reverse = module(swapped)
    assert not torch.equal(forward.h_pair_delta, reverse.h_pair_delta)
    assert not torch.equal(forward.z_delta_pair, reverse.z_delta_pair)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((torch.ones(2, 3), torch.ones(3, 3), torch.ones(2, 3)), "batch sizes"),
        ((torch.ones(2, 3), torch.ones(2, 4), torch.ones(2, 3)), "embedding dimensions"),
        ((torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 4)), "embedding dimensions"),
    ],
)
def test_scale_tensor_validation(values, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_scale_relational(*values, scale="local")
    with pytest.raises(ValueError, match="unknown scale"):
        build_scale_relational(torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2), scale="shell")


def test_scale_relational_rejects_incompatible_or_non_floating_dtypes() -> None:
    with pytest.raises(TypeError, match="must share one dtype"):
        build_scale_relational(
            torch.ones(2, 3, dtype=torch.float32),
            torch.ones(2, 3, dtype=torch.float64),
            torch.ones(2, 3, dtype=torch.float32),
            scale="mutation",
        )

    for field_index, field_name in enumerate(
        ("h_local_MUT", "h_local_WT", "h_local_delta")
    ):
        tensors = [torch.ones(2, 3) for _ in range(3)]
        tensors[field_index] = torch.ones(2, 3, dtype=torch.long)
        with pytest.raises(TypeError, match=rf"{field_name} must use a floating-point dtype"):
            build_scale_relational(*tensors, scale="local")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_scale_relational_rejects_mixed_cpu_and_cuda_devices() -> None:
    with pytest.raises(ValueError, match="must share one device"):
        build_scale_relational(
            torch.ones(2, 3, device="cpu"),
            torch.ones(2, 3, device="cuda"),
            torch.ones(2, 3, device="cpu"),
            scale="global",
        )


@pytest.mark.parametrize(
    ("valid_mask", "message"),
    [
        (torch.tensor([[True, True]]), "local MUT valid_mask must have shape"),
        (torch.tensor([True]), "local MUT valid_mask must have shape"),
        (torch.tensor([1, 1]), "local MUT valid_mask must use torch.bool"),
    ],
)
def test_scale_metadata_rejects_invalid_valid_mask(
    valid_mask: torch.Tensor, message: str
) -> None:
    pooling = _pooling()
    original = pooling.MUT.local
    corrupted = ScalePoolResult(original.values, valid_mask, original.counts)
    pooling = _replace_scale_result(
        pooling, branch="MUT", scale="local", result=corrupted
    )
    with pytest.raises((TypeError, ValueError), match=message):
        ModelAMultiscaleRelational(embedding_dim=8)(pooling)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (torch.tensor([[1, 1]]), "global WT counts must have shape"),
        (torch.tensor([1]), "global WT counts must have shape"),
        (torch.tensor([1.0, 1.0]), "global WT counts must use torch.long"),
        (torch.tensor([1, -1]), "global WT counts must be non-negative"),
    ],
)
def test_scale_metadata_rejects_invalid_counts(
    counts: torch.Tensor, message: str
) -> None:
    pooling = _pooling()
    original = pooling.WT.global_
    corrupted = ScalePoolResult(original.values, original.valid_mask, counts)
    pooling = _replace_scale_result(
        pooling, branch="WT", scale="global", result=corrupted
    )
    with pytest.raises((TypeError, ValueError), match=message):
        ModelAMultiscaleRelational(embedding_dim=8)(pooling)


@pytest.mark.parametrize(
    ("branch", "counts", "valid_mask"),
    [
        ("MUT", torch.tensor([0, 1]), torch.tensor([True, True])),
        ("WT", torch.tensor([1, 1]), torch.tensor([False, True])),
        ("delta", torch.tensor([1, 0]), torch.tensor([True, True])),
    ],
)
def test_scale_metadata_rejects_mask_count_inconsistency_per_branch(
    branch: str, counts: torch.Tensor, valid_mask: torch.Tensor
) -> None:
    pooling = _pooling()
    original = getattr(pooling, branch).mutation
    corrupted = ScalePoolResult(original.values, valid_mask, counts)
    pooling = _replace_scale_result(
        pooling, branch=branch, scale="mutation", result=corrupted
    )
    with pytest.raises(
        ValueError, match=rf"mutation {branch} valid_mask must equal counts > 0"
    ):
        ModelAMultiscaleRelational(embedding_dim=8)(pooling)


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_scale_relational_rejects_non_finite_values(non_finite: float) -> None:
    for field_index, field_name in enumerate(
        ("h_domain_MUT", "h_domain_WT", "h_domain_delta")
    ):
        tensors = [torch.ones(2, 3) for _ in range(3)]
        tensors[field_index][0, 0] = non_finite
        with pytest.raises(ValueError, match=rf"{field_name} contains NaN or Inf"):
            build_scale_relational(*tensors, scale="domain")


def test_configuration_and_a3_availability_validation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ModelAMultiscaleRelational(embedding_dim=8, active_scales=())
    with pytest.raises(ValueError, match="unknown scales"):
        ModelAMultiscaleRelational(embedding_dim=8, active_scales=("shell",))
    with pytest.raises(ValueError, match="derived 120"):
        ModelAMultiscaleRelational(
            embedding_dim=8, pair_fusion_input_dim=121
        )
    module = ModelAMultiscaleRelational(
        embedding_dim=8, active_scales=("domain",)
    )
    with pytest.raises(ValueError, match="active scale 'domain' is unavailable"):
        module(_pooling(domain=False))
