from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from gnn_siamese.models.model import SharedSiameseEncoderModel
from gnn_siamese.models.relational import RDelta, RelationalRepresentation


def test_r_delta_matches_manual_concatenation_exactly() -> None:
    h_mut = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    h_wt = torch.tensor([[0.5, 1.0], [-1.5, 2.0]], dtype=torch.float32)

    r_delta = RDelta()(h_mut, h_wt)
    manual = torch.cat((h_mut, h_wt, h_mut - h_wt, (h_mut - h_wt).abs(), h_mut * h_wt), dim=-1)

    assert torch.equal(r_delta, manual)


def test_r_delta_has_no_parameters() -> None:
    r_delta_module = RDelta()

    assert list(r_delta_module.parameters()) == []


def test_r_delta_dimension_is_five_times_embedding_dim() -> None:
    h_mut = torch.randn(3, 4)
    h_wt = torch.randn(3, 4)

    r_delta = RDelta()(h_mut, h_wt)

    assert r_delta.shape == (3, 20)


def test_z_delta_is_absent_when_mlp_delta_is_disabled() -> None:
    module = RelationalRepresentation(embedding_dim=4, mlp_delta_enabled=False)
    h_mut = torch.randn(2, 4)
    h_wt = torch.randn(2, 4)

    output = module(h_mut, h_wt)

    assert output.z_delta is None
    assert output.z_delta_status == "inactive"
    assert not output.z_delta_is_validated
    assert "z_delta" not in output.to_dict()
    assert "z_delta_status" not in output.to_dict()


def test_z_delta_is_present_with_expected_shape_when_mlp_delta_is_enabled() -> None:
    module = RelationalRepresentation(
        embedding_dim=4,
        mlp_delta_enabled=True,
        mlp_delta_hidden_dim=7,
        mlp_delta_output_dim=3,
        mlp_delta_num_layers=2,
    )
    h_mut = torch.randn(5, 4)
    h_wt = torch.randn(5, 4)

    output = module(h_mut, h_wt)

    assert output.z_delta is not None
    assert output.z_delta.shape == (5, 3)
    assert output.z_delta_status == "unvalidated"
    assert not output.z_delta_is_validated
    assert "z_delta" in output.to_dict()
    assert output.to_dict()["z_delta_status"] == "unvalidated"


def test_severity_is_near_zero_when_h_mut_equals_h_wt() -> None:
    h_mut = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.25]], dtype=torch.float32)
    module = RelationalRepresentation(embedding_dim=3, severity_eps=1.0e-8)

    output = module(h_mut, h_mut.clone())

    assert torch.allclose(output.severity, torch.zeros(2), atol=1.0e-8, rtol=0.0)


def test_mechanism_direction_has_no_nan_or_inf_when_severity_is_zero_or_tiny() -> None:
    module = RelationalRepresentation(embedding_dim=3, severity_eps=1.0e-6)
    h_wt = torch.zeros(2, 3)
    h_mut = torch.tensor([[0.0, 0.0, 0.0], [1.0e-8, -1.0e-8, 2.0e-8]], dtype=torch.float32)

    output = module(h_mut, h_wt)

    assert torch.isfinite(output.mechanism_direction).all()


def test_mechanism_direction_matches_manual_normalization_when_delta_is_nonzero() -> None:
    h_mut = torch.tensor([[3.0, 4.0], [2.0, 3.0]], dtype=torch.float32)
    h_wt = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    module = RelationalRepresentation(embedding_dim=2, severity_eps=1.0e-8)

    output = module(h_mut, h_wt)
    delta = h_mut - h_wt
    manual = delta / torch.linalg.vector_norm(delta, ord=2, dim=-1, keepdim=True)

    assert torch.allclose(output.mechanism_direction, manual, atol=1.0e-6, rtol=1.0e-6)


def test_backward_propagates_gradients_to_h_mut_and_h_wt() -> None:
    module = RelationalRepresentation(
        embedding_dim=3,
        mlp_delta_enabled=True,
        mlp_delta_hidden_dim=5,
        mlp_delta_output_dim=4,
    )
    h_mut = torch.randn(2, 3, requires_grad=True)
    h_wt = torch.randn(2, 3, requires_grad=True)

    output = module(h_mut, h_wt)
    loss = (
        output.r_delta.square().mean()
        + output.severity.square().mean()
        + output.mechanism_direction.square().mean()
        + output.z_delta.square().mean()
    )
    loss.backward()

    assert h_mut.grad is not None
    assert h_wt.grad is not None
    assert torch.isfinite(h_mut.grad).all()
    assert torch.isfinite(h_wt.grad).all()
    assert h_mut.grad.abs().sum().item() > 0.0
    assert h_wt.grad.abs().sum().item() > 0.0


def test_z_delta_is_not_marked_validated_without_explicit_flag() -> None:
    module = RelationalRepresentation(
        embedding_dim=4,
        mlp_delta_enabled=True,
        mlp_delta_hidden_dim=6,
        mlp_delta_output_dim=4,
        z_delta_validated=False,
    )
    output = module(torch.randn(2, 4), torch.randn(2, 4))

    assert output.z_delta is not None
    assert output.z_delta_status == "unvalidated"
    assert not output.z_delta_is_validated


def test_shared_model_exposes_relational_outputs_only_when_module_is_attached() -> None:
    class DummySharedEncoder(nn.Module):
        def forward(self, graph: object) -> object:
            return type("BranchOutput", (), {"H": graph["H"], "h_encoder": graph["h_encoder"]})()

    graph_mut = {
        "H": torch.randn(4, 3),
        "h_encoder": torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
    }
    graph_wt = {
        "H": torch.randn(5, 3),
        "h_encoder": torch.tensor([[0.5, 1.0, 1.5]], dtype=torch.float32),
    }

    model_without_relational = SharedSiameseEncoderModel(DummySharedEncoder())
    plain_output = model_without_relational(graph_mut=graph_mut, graph_wt=graph_wt)
    assert plain_output.r_delta is None
    assert plain_output.z_delta is None
    assert plain_output.z_delta_status == "not_applicable"
    assert "z_delta" not in plain_output.to_dict()

    model_with_relational = SharedSiameseEncoderModel(
        DummySharedEncoder(),
        relational_module=RelationalRepresentation(embedding_dim=3, mlp_delta_enabled=False),
    )
    relational_output = model_with_relational(graph_mut=graph_mut, graph_wt=graph_wt)

    assert relational_output.r_delta is not None
    assert relational_output.r_delta.shape == (1, 15)
    assert relational_output.severity is not None
    assert relational_output.mechanism_direction is not None
    assert relational_output.z_delta is None
    assert relational_output.h_mut is relational_output.h_encoder_mut
    assert relational_output.h_wt is relational_output.h_encoder_wt
    assert "r_delta" in relational_output.to_dict()
    assert "z_delta" not in relational_output.to_dict()
