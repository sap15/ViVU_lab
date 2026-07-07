from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from gnn_siamese.models.encoder import EdgeAwareGraphEncoder
from gnn_siamese.models.model import SharedSiameseEncoderModel


def _make_graph(
    *,
    x: list[list[float]],
    edge_index: list[list[int]],
    edge_attr: list[list[float]],
    is_mutation: list[float],
    availability_mask: list[float],
) -> Data:
    graph = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        is_mutation=torch.tensor(is_mutation, dtype=torch.float32),
    )
    graph.availability_mask_diff_mass = torch.tensor(availability_mask, dtype=torch.float32)
    return graph


def _attach_masks(batch: Batch) -> Batch:
    batch.node_availability_masks = {
        "diff_mass": batch.availability_mask_diff_mass,
    }
    return batch


def _make_branch_batch(graphs: list[Data]) -> Batch:
    return _attach_masks(Batch.from_data_list(graphs))


def _build_siamese_batches() -> tuple[Batch, Batch]:
    graph_mut_a = _make_graph(
        x=[[1.0, 0.0, 0.0], [0.5, 1.0, 1.0], [0.2, 0.5, 0.0]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
        edge_attr=[[1.0, 0.1], [1.0, 0.1], [0.5, 0.2], [0.5, 0.2]],
        is_mutation=[0.0, 1.0, 0.0],
        availability_mask=[1.0, 1.0, 0.0],
    )
    graph_mut_b = _make_graph(
        x=[[0.3, 0.0, 1.0], [0.6, 0.2, 0.0], [0.9, 0.1, 0.0], [0.4, 0.7, 0.0]],
        edge_index=[[0, 1, 2, 2, 3], [1, 0, 3, 1, 2]],
        edge_attr=[[0.9, 0.4], [0.9, 0.4], [1.2, 0.6], [0.8, 0.5], [1.2, 0.6]],
        is_mutation=[0.0, 0.0, 1.0, 0.0],
        availability_mask=[1.0, 1.0, 1.0, 1.0],
    )
    graph_wt_a = _make_graph(
        x=[[0.8, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.4, 0.0]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
        edge_attr=[[1.0, 0.1], [1.0, 0.1], [0.5, 0.2], [0.5, 0.2]],
        is_mutation=[0.0, 0.0, 0.0],
        availability_mask=[1.0, 1.0, 1.0],
    )
    graph_wt_b = _make_graph(
        x=[[0.2, 0.0, 0.0], [0.5, 0.2, 0.0], [0.7, 0.1, 0.0], [0.1, 0.3, 0.0], [0.4, 0.6, 0.0]],
        edge_index=[[0, 1, 2, 3, 4], [1, 0, 3, 4, 3]],
        edge_attr=[[0.7, 0.3], [0.7, 0.3], [1.1, 0.2], [0.6, 0.5], [0.6, 0.5]],
        is_mutation=[0.0, 0.0, 0.0, 0.0, 0.0],
        availability_mask=[1.0, 0.0, 1.0, 1.0, 1.0],
    )
    return _make_branch_batch([graph_mut_a, graph_mut_b]), _make_branch_batch([graph_wt_a, graph_wt_b])


def test_shared_encoder_outputs_shapes_and_aliases_on_cpu() -> None:
    graph_mut, graph_wt = _build_siamese_batches()
    encoder = EdgeAwareGraphEncoder(
        node_input_dim=3,
        edge_input_dim=2,
        hidden_dim=8,
        graph_output_dim=10,
        fusion_hidden_dim=12,
    )
    model = SharedSiameseEncoderModel(encoder).cpu()

    output = model(graph_mut=graph_mut.cpu(), graph_wt=graph_wt.cpu())

    assert encoder.fusion_input_dim == 5 * encoder.hidden_dim
    assert encoder.mlp_fusion[0].in_features == 5 * encoder.hidden_dim
    assert output.H_mut.device.type == "cpu"
    assert output.H_WT.device.type == "cpu"
    assert output.h_encoder_mut.device.type == "cpu"
    assert output.h_encoder_wt.device.type == "cpu"
    assert output.H_mut.shape == (7, 8)
    assert output.H_WT.shape == (8, 8)
    assert output.h_encoder_mut.shape == (2, 10)
    assert output.h_encoder_wt.shape == (2, 10)
    assert output.h_mut is output.h_encoder_mut
    assert output.h_wt is output.h_encoder_wt
    assert model.shared_encoder is encoder
    assert not hasattr(model, "encoder_mut")
    assert not hasattr(model, "encoder_wt")


def test_shared_encoder_backward_produces_nonzero_gradients() -> None:
    graph_mut, graph_wt = _build_siamese_batches()
    model = SharedSiameseEncoderModel(
        EdgeAwareGraphEncoder(
            node_input_dim=3,
            edge_input_dim=2,
            hidden_dim=8,
            graph_output_dim=10,
            fusion_hidden_dim=12,
        )
    )

    output = model(graph_mut=graph_mut, graph_wt=graph_wt)
    loss = (
        output.h_encoder_mut.square().mean()
        + output.h_encoder_wt.square().mean()
        + output.H_mut.square().mean()
        + output.H_WT.square().mean()
    )
    loss.backward()

    gradient_norms = [
        parameter.grad.norm().item()
        for parameter in model.shared_encoder.parameters()
        if parameter.grad is not None
    ]
    assert gradient_norms
    assert any(norm > 0.0 for norm in gradient_norms)


def test_edge_attr_changes_shared_encoder_output() -> None:
    base_graph = _make_graph(
        x=[[1.0, 0.0, 0.0], [0.2, 1.0, 1.0], [0.4, 0.5, 0.0]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
        edge_attr=[[1.0, 0.1], [1.0, 0.1], [0.5, 0.2], [0.5, 0.2]],
        is_mutation=[0.0, 1.0, 0.0],
        availability_mask=[1.0, 1.0, 1.0],
    )
    changed_graph = _make_graph(
        x=[[1.0, 0.0, 0.0], [0.2, 1.0, 1.0], [0.4, 0.5, 0.0]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
        edge_attr=[[2.0, 0.9], [2.0, 0.9], [1.5, 0.8], [1.5, 0.8]],
        is_mutation=[0.0, 1.0, 0.0],
        availability_mask=[1.0, 1.0, 1.0],
    )
    graph_wt = _make_graph(
        x=[[0.8, 0.0, 0.0], [0.3, 0.0, 0.0], [0.1, 0.4, 0.0]],
        edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
        edge_attr=[[1.0, 0.1], [1.0, 0.1], [0.5, 0.2], [0.5, 0.2]],
        is_mutation=[0.0, 0.0, 0.0],
        availability_mask=[1.0, 1.0, 1.0],
    )

    model = SharedSiameseEncoderModel(
        EdgeAwareGraphEncoder(
            node_input_dim=3,
            edge_input_dim=2,
            hidden_dim=8,
            graph_output_dim=10,
            fusion_hidden_dim=12,
            dropout=0.0,
        )
    ).eval()

    with torch.no_grad():
        output_a = model(
            graph_mut=_make_branch_batch([base_graph]),
            graph_wt=_make_branch_batch([graph_wt]),
        )
        output_b = model(
            graph_mut=_make_branch_batch([changed_graph]),
            graph_wt=_make_branch_batch([graph_wt]),
        )

    assert not torch.allclose(output_a.H_mut, output_b.H_mut)
    assert not torch.allclose(output_a.h_encoder_mut, output_b.h_encoder_mut)
