from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from torch_geometric.data import Batch, Data

from gnn_siamese.models.delta_block import NodeDeltaBlock
from gnn_siamese.models.encoder import EdgeAwareGraphEncoder
from gnn_siamese.models.model_a import ModelAOneView
from gnn_siamese.models.multiscale_pooling_a import ModelAMultiscalePooling
from gnn_siamese.models.multiscale_relational_a import ModelAMultiscaleRelational


class TinySharedEncoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dimension, dimension)

    def forward(self, graph: object) -> object:
        return SimpleNamespace(H=self.linear(graph.x))


def _model(*, fusion_enabled: bool = True) -> ModelAOneView:
    dimension = 8
    return ModelAOneView(
        shared_encoder=TinySharedEncoder(dimension),
        node_delta_block=NodeDeltaBlock(
            input_dim=dimension,
            hidden_dim=12,
            output_dim=dimension,
            dropout=0.0,
        ),
        multiscale_pooling=ModelAMultiscalePooling(
            enabled_scales=("mutation", "local", "global")
        ),
        multiscale_relational=ModelAMultiscaleRelational(
            embedding_dim=dimension,
            active_scales=("global", "local", "mutation"),
            pair_fusion_enabled=fusion_enabled,
            pair_fusion_hidden_dim=20,
            pair_fusion_output_dim=16 if fusion_enabled else 120,
            pair_fusion_dropout=0.0,
        ),
    )


def _inputs() -> dict[str, object]:
    graph_mut = SimpleNamespace(
        x=torch.randn(5, 8), batch=torch.tensor([0, 0, 1, 1, 1])
    )
    graph_wt = SimpleNamespace(
        x=torch.randn(6, 8), batch=torch.tensor([0, 0, 0, 1, 1, 1])
    )
    return {
        "graph_mut": graph_mut,
        "graph_wt": graph_wt,
        "mut_aligned_index": torch.tensor([0, 1, 2, 3, 4]),
        "wt_aligned_index": torch.tensor([0, 1, 3, 4, 5]),
        "aligned_pair_batch": torch.tensor([0, 0, 1, 1, 1]),
        "alignment_ptr": torch.tensor([0, 2, 5]),
        "num_pairs": 2,
        "mutation_mask_MUT": torch.tensor([True, False, True, False, False]),
        "mutation_mask_WT": torch.tensor([True, False, False, True, False, False]),
        "mutation_mask_delta": torch.tensor([True, False, True, False, False]),
        "local_mask_MUT": torch.tensor([True, True, True, False, False]),
        "local_mask_WT": torch.tensor([True, True, False, True, True, False]),
        "local_mask_delta": torch.tensor([True, True, True, True, False]),
        "variant_id": ("v1", "v2"),
    }


def test_one_view_forward_shapes_metadata_direction_and_no_domain() -> None:
    model = _model()
    output = model(**_inputs())
    assert output.H_MUT.shape == (5, 8)
    assert output.H_WT.shape == (6, 8)
    assert output.H_delta.shape == (5, 8)
    assert output.z_delta_mutation.shape == (2, 40)
    assert output.h_pair_delta.shape == (2, 120)
    assert output.z_delta_pair.shape == (2, 16)
    assert output.h_domain_MUT is None
    assert output.z_delta_domain is None
    assert output.active_scales == ("mutation", "local", "global")
    assert output.variant_id == ("v1", "v2")
    assert output.alignment_metadata["alignment_ptr"].tolist() == [0, 2, 5]
    assert output.scale_counts["local"]["delta"].tolist() == [2, 2]


def test_backward_optimizer_membership_and_pair_fusion_weight_change() -> None:
    torch.manual_seed(4)
    model = _model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    pair_parameters = list(model.multiscale_relational.pair_fusion.parameters())
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(parameter) in optimizer_ids for parameter in pair_parameters)
    before = [parameter.detach().clone() for parameter in pair_parameters]

    output = model(**_inputs())
    output.z_delta_pair.square().mean().backward()
    for module in (
        model.shared_encoder,
        model.node_delta_block,
        model.multiscale_relational.pair_fusion,
    ):
        gradients = [p.grad for p in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)
    assert list(model.multiscale_pooling.parameters()) == []

    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, pair_parameters)
    )


def test_identity_forward_is_unambiguous() -> None:
    output = _model(fusion_enabled=False)(**_inputs())
    assert output.pair_fusion_mode == "identity"
    assert torch.equal(output.z_delta_pair, output.h_pair_delta)


def _pyg_graph(
    *,
    x: list[list[float]],
    edge_index: list[list[int]],
    edge_attr: list[list[float]],
    is_mutation: list[float],
) -> Data:
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        is_mutation=torch.tensor(is_mutation, dtype=torch.float32),
    )


def test_one_view_integrates_real_pyg_batch_and_edge_aware_encoder_on_cpu() -> None:
    torch.manual_seed(17)
    graph_mut = Batch.from_data_list(
        [
            _pyg_graph(
                x=[[1.0, 0.0, 0.0], [0.5, 1.0, 1.0], [0.2, 0.5, 0.0]],
                edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
                edge_attr=[[1.0, 0.1], [1.0, 0.1], [0.5, 0.2], [0.5, 0.2]],
                is_mutation=[0.0, 1.0, 0.0],
            ),
            _pyg_graph(
                x=[[0.3, 0.0, 1.0], [0.6, 0.2, 0.0], [0.9, 0.1, 0.0], [0.4, 0.7, 0.0]],
                edge_index=[[0, 1, 2, 2, 3], [1, 0, 3, 1, 2]],
                edge_attr=[[0.9, 0.4], [0.9, 0.4], [1.2, 0.6], [0.8, 0.5], [1.2, 0.6]],
                is_mutation=[0.0, 0.0, 1.0, 0.0],
            ),
        ]
    ).cpu()
    graph_wt = Batch.from_data_list(
        [
            _pyg_graph(
                x=[[0.8, 0.0, 0.0], [0.4, 0.0, 0.0], [0.1, 0.4, 0.0]],
                edge_index=[[0, 1, 1, 2], [1, 0, 2, 1]],
                edge_attr=[[1.0, 0.1], [1.0, 0.1], [0.5, 0.2], [0.5, 0.2]],
                is_mutation=[0.0, 0.0, 0.0],
            ),
            _pyg_graph(
                x=[[0.2, 0.0, 0.0], [0.5, 0.2, 0.0], [0.7, 0.1, 0.0], [0.1, 0.3, 0.0], [0.4, 0.6, 0.0]],
                edge_index=[[0, 1, 2, 3, 4], [1, 0, 3, 4, 3]],
                edge_attr=[[0.7, 0.3], [0.7, 0.3], [1.1, 0.2], [0.6, 0.5], [0.6, 0.5]],
                is_mutation=[0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        ]
    ).cpu()
    encoder = EdgeAwareGraphEncoder(
        node_input_dim=3,
        edge_input_dim=2,
        hidden_dim=8,
        num_layers=1,
        edge_mlp_hidden_dim=6,
        fusion_hidden_dim=10,
        graph_output_dim=6,
        dropout=0.0,
    )
    model = ModelAOneView(
        shared_encoder=encoder,
        node_delta_block=NodeDeltaBlock(
            input_dim=8, hidden_dim=12, output_dim=8, dropout=0.0
        ),
        multiscale_pooling=ModelAMultiscalePooling(
            enabled_scales=("mutation", "local", "global")
        ),
        multiscale_relational=ModelAMultiscaleRelational(
            embedding_dim=8,
            active_scales=("global", "mutation", "local"),
            pair_fusion_hidden_dim=20,
            pair_fusion_output_dim=16,
            pair_fusion_dropout=0.0,
        ),
    ).cpu()
    output = model(
        graph_mut=graph_mut,
        graph_wt=graph_wt,
        mut_aligned_index=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        wt_aligned_index=torch.tensor([0, 1, 2, 3, 4, 5, 6]),
        aligned_pair_batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),
        alignment_ptr=torch.tensor([0, 3, 7]),
        num_pairs=2,
        mutation_mask_MUT=torch.tensor([False, True, False, False, False, True, False]),
        mutation_mask_WT=torch.tensor([False, True, False, False, False, True, False, False]),
        mutation_mask_delta=torch.tensor([False, True, False, False, False, True, False]),
        local_mask_MUT=torch.tensor([True, True, False, False, True, True, False]),
        local_mask_WT=torch.tensor([True, True, False, False, True, True, False, False]),
        local_mask_delta=torch.tensor([True, True, False, False, True, True, False]),
    )

    assert output.H_MUT.shape == (7, 8)
    assert output.H_WT.shape == (8, 8)
    assert output.H_delta.shape == (7, 8)
    assert output.z_delta_mutation.shape == (2, 40)
    assert output.h_pair_delta.shape == (2, 120)
    assert output.z_delta_pair.shape == (2, 16)
    assert torch.isfinite(output.z_delta_pair).all()
    assert output.scale_counts["global"]["MUT"].tolist() == [3, 4]
    assert output.scale_counts["global"]["WT"].tolist() == [3, 5]
    torch.testing.assert_close(
        output.h_global_MUT[0], output.H_MUT[:3].mean(dim=0)
    )
    torch.testing.assert_close(
        output.h_global_MUT[1], output.H_MUT[3:].mean(dim=0)
    )
    torch.testing.assert_close(
        output.h_global_WT[0], output.H_WT[:3].mean(dim=0)
    )
    torch.testing.assert_close(
        output.h_global_WT[1], output.H_WT[3:].mean(dim=0)
    )

    output.z_delta_pair.square().mean().backward()
    for module in (model.node_delta_block, model.multiscale_relational.pair_fusion):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients
        assert all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)

    encoder_gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ]
    assert encoder_gradients
    assert all(torch.isfinite(gradient).all() for gradient in encoder_gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in encoder_gradients)
    assert encoder.input_projection.weight.grad is not None
    assert encoder.convs[0].lin.weight.grad is not None
