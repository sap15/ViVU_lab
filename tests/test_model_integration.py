from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torch_geometric = pytest.importorskip("torch_geometric")

from torch_geometric.data import Batch, Data

from gnn_siamese.models.encoder import EdgeAwareGraphEncoder
from gnn_siamese.models.model import SharedSiameseEncoderModel
from gnn_siamese.models.projection import (
    InstanceProjectionHead,
    PairProjectionHead,
    ProjectionHeadConfig,
)
from gnn_siamese.models.relational import RelationalRepresentation


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


def _make_encoder(*, graph_output_dim: int = 10) -> EdgeAwareGraphEncoder:
    return EdgeAwareGraphEncoder(
        node_input_dim=3,
        edge_input_dim=2,
        hidden_dim=8,
        graph_output_dim=graph_output_dim,
        fusion_hidden_dim=12,
        dropout=0.0,
    )


def test_shared_model_integrates_relational_and_both_projection_heads_on_cpu() -> None:
    graph_mut, graph_wt = _build_siamese_batches()
    graph_dim = 10
    model = SharedSiameseEncoderModel(
        shared_encoder=_make_encoder(graph_output_dim=graph_dim),
        relational_module=RelationalRepresentation(embedding_dim=graph_dim, mlp_delta_enabled=False),
        projection_instance=InstanceProjectionHead(
            config=ProjectionHeadConfig(
                input_dim=graph_dim,
                hidden_dim=12,
                output_dim=6,
                dropout=0.0,
            )
        ),
        projection_pair=PairProjectionHead(
            config=ProjectionHeadConfig(
                input_dim=5 * graph_dim,
                hidden_dim=14,
                output_dim=7,
                dropout=0.0,
            )
        ),
        pair_projection_source="r_delta",
    ).cpu()

    output = model(graph_mut=graph_mut.cpu(), graph_wt=graph_wt.cpu())

    assert output.H_mut.device.type == "cpu"
    assert output.H_WT.device.type == "cpu"
    assert output.h_encoder_mut.shape == (2, graph_dim)
    assert output.h_encoder_wt.shape == (2, graph_dim)
    assert output.r_delta is not None
    assert output.r_delta.shape == (2, 5 * graph_dim)
    assert output.r_delta.shape[-1] == 5 * graph_dim
    assert output.z_instance is not None
    assert output.z_instance.shape == (2, 6)
    assert output.z_instance_pair is not None
    assert output.z_instance_pair.shape == (2, 7)
    assert output.severity is not None
    assert output.severity.shape == (2,)
    assert output.mechanism_direction is not None
    assert output.mechanism_direction.shape == (2, graph_dim)


def test_projection_pair_can_be_disabled_without_breaking_instance_projection() -> None:
    graph_mut, graph_wt = _build_siamese_batches()
    graph_dim = 10
    model = SharedSiameseEncoderModel(
        shared_encoder=_make_encoder(graph_output_dim=graph_dim),
        relational_module=RelationalRepresentation(embedding_dim=graph_dim, mlp_delta_enabled=False),
        projection_instance=InstanceProjectionHead(
            config=ProjectionHeadConfig(input_dim=graph_dim, hidden_dim=12, output_dim=6, dropout=0.0)
        ),
        projection_pair=None,
    ).cpu()

    output = model(graph_mut=graph_mut.cpu(), graph_wt=graph_wt.cpu())

    assert output.z_instance is not None
    assert output.z_instance.shape == (2, 6)
    assert output.z_instance_pair is None
    assert output.r_delta is not None
    assert output.severity is not None
    assert output.mechanism_direction is not None


def test_pair_projection_source_z_delta_fails_when_not_validated() -> None:
    graph_mut, graph_wt = _build_siamese_batches()
    graph_dim = 10
    model = SharedSiameseEncoderModel(
        shared_encoder=_make_encoder(graph_output_dim=graph_dim),
        relational_module=RelationalRepresentation(
            embedding_dim=graph_dim,
            mlp_delta_enabled=True,
            mlp_delta_hidden_dim=12,
            mlp_delta_output_dim=9,
        ),
        projection_instance=InstanceProjectionHead(
            config=ProjectionHeadConfig(input_dim=graph_dim, hidden_dim=12, output_dim=6, dropout=0.0)
        ),
        projection_pair=PairProjectionHead(
            config=ProjectionHeadConfig(input_dim=9, hidden_dim=11, output_dim=5, dropout=0.0)
        ),
        pair_projection_source="z_delta",
    ).cpu()

    with pytest.raises(ValueError, match="z_delta is not validated"):
        model(graph_mut=graph_mut.cpu(), graph_wt=graph_wt.cpu())


def test_pair_projection_source_z_delta_works_when_validated(tmp_path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(
        (
            '{"modules": {"mlp_delta": {"status": "trained", "optimizer_group": "g1", '
            '"connected_losses": ["L_delta"], "mean_gradient_norm": 0.25, '
            '"max_gradient_norm": 0.5, "relative_weight_change": 0.01, '
            '"has_nan_or_inf": false}}}'
        ),
        encoding="utf-8",
    )
    graph_mut, graph_wt = _build_siamese_batches()
    graph_dim = 10
    z_delta_dim = 9
    model = SharedSiameseEncoderModel(
        shared_encoder=_make_encoder(graph_output_dim=graph_dim),
        relational_module=RelationalRepresentation(
            embedding_dim=graph_dim,
            mlp_delta_enabled=True,
            mlp_delta_hidden_dim=12,
            mlp_delta_output_dim=z_delta_dim,
            run_manifest_path=manifest_path,
        ),
        projection_instance=InstanceProjectionHead(
            config=ProjectionHeadConfig(input_dim=graph_dim, hidden_dim=12, output_dim=6, dropout=0.0)
        ),
        projection_pair=PairProjectionHead(
            config=ProjectionHeadConfig(input_dim=z_delta_dim, hidden_dim=11, output_dim=5, dropout=0.0)
        ),
        pair_projection_source="z_delta",
    ).cpu()

    output = model(graph_mut=graph_mut.cpu(), graph_wt=graph_wt.cpu())

    assert output.z_delta is not None
    assert output.z_delta_status == "validated"
    assert output.z_delta_is_validated
    assert output.z_instance_pair is not None
    assert output.z_instance_pair.shape == (2, 5)
