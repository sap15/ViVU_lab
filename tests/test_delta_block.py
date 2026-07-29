from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from gnn_siamese.models.delta_block import (
    NodeDeltaBlock,
    NodeDeltaOutput,
    build_node_delta_features,
)


def _block(*, dropout: float = 0.0) -> NodeDeltaBlock:
    return NodeDeltaBlock(
        input_dim=2,
        hidden_dim=7,
        output_dim=3,
        activation="relu",
        dropout=dropout,
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    H_MUT = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]]
    )
    H_WT = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
    )
    return H_MUT, H_WT, torch.tensor([2, 0, 3]), torch.tensor([4, 1, 0])


def test_basic_shape_parameters_and_structured_immutable_output() -> None:
    block = _block()
    H_MUT, H_WT, mut_idx, wt_idx = _inputs()
    pair_batch = torch.tensor([0, 0, 1])
    ptr = torch.tensor([0, 2, 3])

    output = block(H_MUT, H_WT, mut_idx, wt_idx, pair_batch, ptr)

    assert isinstance(output, NodeDeltaOutput)
    assert output.H_delta.shape == (3, 3)
    assert block.network[0].in_features == 4 * block.input_dim
    assert block.network[0].out_features == block.hidden_dim
    assert block.network[-1].out_features == block.output_dim
    assert sum(parameter.numel() for parameter in block.parameters()) > 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        output.H_delta = torch.empty(0)


def test_gathering_uses_exact_indices_independent_of_node_order() -> None:
    H_MUT, H_WT, mut_idx, wt_idx = _inputs()

    features = build_node_delta_features(
        H_MUT.index_select(0, mut_idx),
        H_WT.index_select(0, wt_idx),
    )

    assert torch.equal(features[:, :2], H_MUT[[2, 0, 3]])
    assert torch.equal(features[:, 2:4], H_WT[[4, 1, 0]])


def test_delta_feature_blocks_have_exact_semantic_order() -> None:
    H_MUT_aligned = torch.tensor([[3.0, -2.0], [1.0, 7.0]])
    H_WT_aligned = torch.tensor([[1.0, 4.0], [5.0, 2.0]])

    features = build_node_delta_features(H_MUT_aligned, H_WT_aligned)

    expected = torch.cat(
        (
            H_MUT_aligned,
            H_WT_aligned,
            H_MUT_aligned - H_WT_aligned,
            (H_MUT_aligned - H_WT_aligned).abs(),
        ),
        dim=-1,
    )
    assert torch.equal(features, expected)
    assert torch.equal(features[:, 0:2], H_MUT_aligned)
    assert torch.equal(features[:, 2:4], H_WT_aligned)
    assert torch.equal(features[:, 4:6], H_MUT_aligned - H_WT_aligned)
    assert torch.equal(features[:, 6:8], (H_MUT_aligned - H_WT_aligned).abs())


def test_operation_is_directed_when_mut_and_wt_are_swapped() -> None:
    block = NodeDeltaBlock(
        input_dim=1,
        hidden_dim=1,
        output_dim=1,
        activation="relu",
        dropout=0.0,
    )
    with torch.no_grad():
        block.network[0].weight.zero_()
        block.network[0].weight[0, 2] = 1.0
        block.network[0].bias.fill_(2.0)
        block.network[-1].weight.fill_(1.0)
        block.network[-1].bias.zero_()
    H_MUT = torch.tensor([[3.0]])
    H_WT = torch.tensor([[1.0]])
    index = torch.tensor([0])

    mut_wt = block(H_MUT, H_WT, index, index).H_delta
    wt_mut = block(H_WT, H_MUT, index, index).H_delta

    assert torch.equal(mut_wt, torch.tensor([[4.0]]))
    assert torch.equal(wt_mut, torch.tensor([[0.0]]))
    assert not torch.equal(mut_wt, wt_mut)


def test_heterogeneous_batch_preserves_pair_segments_and_row_order() -> None:
    H_MUT, H_WT, mut_idx, wt_idx = _inputs()
    pair_batch = torch.tensor([0, 0, 2])
    ptr = torch.tensor([0, 2, 2, 3])
    block = _block()

    output = block(H_MUT, H_WT, mut_idx, wt_idx, pair_batch, ptr)
    expected = block.network(
        build_node_delta_features(H_MUT[mut_idx], H_WT[wt_idx])
    )

    assert output.aligned_pair_batch is pair_batch
    assert output.alignment_ptr is ptr
    assert torch.equal(output.H_delta, expected)
    assert torch.equal(output.H_delta[ptr[0] : ptr[1]], expected[:2])
    assert output.H_delta[ptr[1] : ptr[2]].shape == (0, 3)
    assert torch.equal(output.H_delta[ptr[2] : ptr[3]], expected[2:])


def test_completely_empty_alignment_returns_no_fictitious_rows() -> None:
    block = _block()
    H_MUT = torch.randn(2, 2)
    H_WT = torch.randn(3, 2)
    pair_batch = torch.empty(0, dtype=torch.long)
    ptr = torch.tensor([0, 0, 0], dtype=torch.long)

    output = block(
        H_MUT,
        H_WT,
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        pair_batch,
        ptr,
    )

    assert output.H_delta.shape == (0, block.output_dim)
    assert output.H_delta.dtype == H_MUT.dtype
    assert output.H_delta.device == H_MUT.device
    assert output.aligned_pair_batch is pair_batch
    assert output.alignment_ptr is ptr


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("H_MUT", torch.ones(3), ValueError, "H_MUT must be two-dimensional"),
        ("H_WT", torch.ones(3), ValueError, "H_WT must be two-dimensional"),
        ("H_WT", torch.ones(3, 3), ValueError, "latent dimensions must match"),
        ("mut_idx", torch.tensor([0.0]), TypeError, "mut_aligned_index must use torch.long"),
        ("wt_idx", torch.tensor([0.0]), TypeError, "wt_aligned_index must use torch.long"),
        ("mut_idx", torch.tensor([-1]), IndexError, "mut_aligned_index contains a negative"),
        ("wt_idx", torch.tensor([-1]), IndexError, "wt_aligned_index contains a negative"),
        ("mut_idx", torch.tensor([4]), IndexError, "mut_aligned_index contains index 4"),
        ("wt_idx", torch.tensor([5]), IndexError, "wt_aligned_index contains index 5"),
        ("H_MUT", torch.tensor([[float("nan"), 0.0]]), ValueError, "H_MUT contains NaN or Inf"),
        ("H_WT", torch.tensor([[float("inf"), 0.0]]), ValueError, "H_WT contains NaN or Inf"),
    ],
)
def test_required_input_validation(field, value, error, message) -> None:
    H_MUT, H_WT, _, _ = _inputs()
    arguments = {
        "H_MUT": H_MUT,
        "H_WT": H_WT,
        "mut_idx": torch.tensor([0]),
        "wt_idx": torch.tensor([0]),
    }
    arguments[field] = value

    with pytest.raises(error, match=message):
        _block()(
            arguments["H_MUT"],
            arguments["H_WT"],
            arguments["mut_idx"],
            arguments["wt_idx"],
        )


def test_index_lengths_must_match() -> None:
    H_MUT, H_WT, _, _ = _inputs()
    with pytest.raises(ValueError, match="must have the same length"):
        _block()(H_MUT, H_WT, torch.tensor([0, 1]), torch.tensor([0]))


def test_matching_latent_dimensions_must_equal_configured_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim=2"):
        _block()(
            torch.ones(3, 4),
            torch.ones(5, 4),
            torch.tensor([0]),
            torch.tensor([0]),
        )


@pytest.mark.parametrize(
    ("pair_batch", "ptr", "error", "message"),
    [
        (torch.tensor([0]), None, ValueError, "length A=2"),
        (torch.tensor([[0, 0]]), None, ValueError, "one-dimensional"),
        (torch.tensor([0.0, 0.0]), None, TypeError, "torch.long"),
        (None, torch.tensor([[0, 2]]), ValueError, "one-dimensional"),
        (None, torch.tensor([0.0, 2.0]), TypeError, "torch.long"),
        (None, torch.tensor([1, 2]), ValueError, "start at zero"),
        (None, torch.tensor([0, 2, 1, 2]), ValueError, "monotonically"),
        (None, torch.tensor([0, 1]), ValueError, "end at A=2"),
        (torch.tensor([0, 2]), torch.tensor([0, 1, 2]), ValueError, "incompatible"),
        (torch.tensor([0, 0]), torch.tensor([0, 1, 2]), ValueError, "segment 1"),
    ],
)
def test_batch_metadata_validation(pair_batch, ptr, error, message) -> None:
    H_MUT, H_WT, _, _ = _inputs()
    with pytest.raises(error, match=message):
        _block()(
            H_MUT,
            H_WT,
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            pair_batch,
            ptr,
        )


def test_eval_is_deterministic_with_dropout_configured() -> None:
    torch.manual_seed(7)
    block = _block(dropout=0.5).eval()
    H_MUT, H_WT, mut_idx, wt_idx = _inputs()

    first = block(H_MUT, H_WT, mut_idx, wt_idx).H_delta
    second = block(H_MUT, H_WT, mut_idx, wt_idx).H_delta

    assert torch.equal(first, second)


def test_backward_reaches_embeddings_parameters_and_optimizer_changes_weights() -> None:
    torch.manual_seed(11)
    block = NodeDeltaBlock(
        input_dim=2,
        hidden_dim=5,
        output_dim=3,
        activation="silu",
        dropout=0.0,
    )
    H_MUT = torch.randn(4, 2, requires_grad=True)
    H_WT = torch.randn(5, 2, requires_grad=True)
    mut_idx = torch.tensor([0, 2, 3])
    wt_idx = torch.tensor([4, 1, 0])
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    before = [parameter.detach().clone() for parameter in block.parameters()]

    optimizer.zero_grad()
    output = block(H_MUT, H_WT, mut_idx, wt_idx)
    loss = output.H_delta.square().mean()
    loss.backward()

    assert H_MUT.grad is not None
    assert H_WT.grad is not None
    assert torch.isfinite(H_MUT.grad).all()
    assert torch.isfinite(H_WT.grad).all()
    assert H_MUT.grad[mut_idx].abs().sum() > 0
    assert H_WT.grad[wt_idx].abs().sum() > 0
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    optimizer.step()
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, block.parameters())
    )


def test_cpu_execution() -> None:
    H_MUT, H_WT, mut_idx, wt_idx = _inputs()
    output = _block().cpu()(H_MUT.cpu(), H_WT.cpu(), mut_idx.cpu(), wt_idx.cpu())
    assert output.H_delta.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable.")
def test_cuda_execution_and_metadata_device() -> None:
    device = torch.device("cuda")
    H_MUT, H_WT, mut_idx, wt_idx = _inputs()
    pair_batch = torch.tensor([0, 0, 1], device=device)
    ptr = torch.tensor([0, 2, 3], device=device)

    output = _block().to(device)(
        H_MUT.to(device),
        H_WT.to(device),
        mut_idx.to(device),
        wt_idx.to(device),
        pair_batch,
        ptr,
    )

    assert output.H_delta.device.type == "cuda"
    assert output.aligned_pair_batch is pair_batch
    assert output.alignment_ptr is ptr


def test_supported_activations_and_constructor_validation() -> None:
    assert isinstance(
        NodeDeltaBlock(input_dim=2, hidden_dim=3, output_dim=4, activation="gelu").network[1],
        nn.GELU,
    )
    assert isinstance(
        NodeDeltaBlock(input_dim=2, hidden_dim=3, output_dim=4, activation="silu").network[1],
        nn.SiLU,
    )
    with pytest.raises(ValueError, match="activation must be one of"):
        NodeDeltaBlock(input_dim=2, hidden_dim=3, output_dim=4, activation="tanh")
    with pytest.raises(ValueError, match="dropout must be in"):
        NodeDeltaBlock(input_dim=2, hidden_dim=3, output_dim=4, dropout=1.0)
