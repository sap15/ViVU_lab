from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.models.projection import (
    InstanceProjectionHead,
    PairProjectionHead,
    ProjectionHeadConfig,
)


def test_projection_instance_produces_expected_shape_on_cpu() -> None:
    head = InstanceProjectionHead(
        config=ProjectionHeadConfig(
            input_dim=6,
            hidden_dim=8,
            output_dim=4,
            dropout=0.0,
            use_layer_norm=True,
            normalize_output=True,
        )
    ).cpu()
    h_encoder_mut = torch.randn(3, 6, device="cpu")

    z_instance = head(h_encoder_mut, input_name="h_encoder_mut")

    assert z_instance.device.type == "cpu"
    assert z_instance.shape == (3, 4)
    assert torch.allclose(
        torch.linalg.vector_norm(z_instance, ord=2, dim=-1),
        torch.ones(3, device="cpu"),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_projection_pair_produces_expected_shape_on_cpu() -> None:
    head = PairProjectionHead(
        config=ProjectionHeadConfig(
            input_dim=10,
            hidden_dim=12,
            output_dim=5,
            dropout=0.0,
            use_layer_norm=True,
            normalize_output=False,
        )
    ).cpu()
    r_delta = torch.randn(4, 10, device="cpu")

    z_instance_pair = head(r_delta, input_name="r_delta")

    assert z_instance_pair.device.type == "cpu"
    assert z_instance_pair.shape == (4, 5)


def test_projection_heads_do_not_share_parameters() -> None:
    projection_instance = InstanceProjectionHead(
        config=ProjectionHeadConfig(input_dim=6, hidden_dim=8, output_dim=4)
    )
    projection_pair = PairProjectionHead(
        config=ProjectionHeadConfig(input_dim=30, hidden_dim=8, output_dim=4)
    )

    instance_param_ids = {id(parameter) for parameter in projection_instance.parameters()}
    pair_param_ids = {id(parameter) for parameter in projection_pair.parameters()}

    assert instance_param_ids
    assert pair_param_ids
    assert instance_param_ids.isdisjoint(pair_param_ids)


def test_projection_pair_rejects_h_encoder_dim_when_r_delta_is_expected() -> None:
    head = PairProjectionHead(
        config=ProjectionHeadConfig(input_dim=20, hidden_dim=10, output_dim=6)
    )
    h_encoder_mut = torch.randn(2, 4)

    with pytest.raises(ValueError, match="expects last dimension 20"):
        head(h_encoder_mut, input_name="r_delta")


def test_projection_instance_rejects_r_delta_dim_when_h_encoder_is_expected() -> None:
    head = InstanceProjectionHead(
        config=ProjectionHeadConfig(input_dim=4, hidden_dim=10, output_dim=6)
    )
    r_delta = torch.randn(2, 20)

    with pytest.raises(ValueError, match="expects last dimension 4"):
        head(r_delta, input_name="h_encoder_mut")


def test_projection_instance_rejects_pair_semantic_name() -> None:
    head = InstanceProjectionHead(
        config=ProjectionHeadConfig(input_dim=6, hidden_dim=8, output_dim=4)
    )

    with pytest.raises(ValueError, match="semantic input 'r_delta'"):
        head(torch.randn(2, 6), input_name="r_delta")


def test_projection_pair_rejects_instance_semantic_name() -> None:
    head = PairProjectionHead(
        config=ProjectionHeadConfig(input_dim=10, hidden_dim=12, output_dim=5)
    )

    with pytest.raises(ValueError, match="semantic input 'h_encoder_mut'"):
        head(torch.randn(2, 10), input_name="h_encoder_mut")


def test_backward_produces_nonzero_gradients_in_each_head() -> None:
    projection_instance = InstanceProjectionHead(
        config=ProjectionHeadConfig(input_dim=6, hidden_dim=8, output_dim=4, dropout=0.0)
    )
    projection_pair = PairProjectionHead(
        config=ProjectionHeadConfig(input_dim=30, hidden_dim=10, output_dim=5, dropout=0.0)
    )
    h_encoder_mut = torch.randn(3, 6, requires_grad=True)
    r_delta = torch.randn(3, 30, requires_grad=True)

    z_instance = projection_instance(h_encoder_mut, input_name="h_encoder_mut")
    z_instance_pair = projection_pair(r_delta, input_name="r_delta")
    loss = z_instance.square().mean() + z_instance_pair.square().mean()
    loss.backward()

    instance_grad_norms = [
        parameter.grad.norm().item()
        for parameter in projection_instance.parameters()
        if parameter.grad is not None
    ]
    pair_grad_norms = [
        parameter.grad.norm().item()
        for parameter in projection_pair.parameters()
        if parameter.grad is not None
    ]

    assert h_encoder_mut.grad is not None
    assert r_delta.grad is not None
    assert h_encoder_mut.grad.abs().sum().item() > 0.0
    assert r_delta.grad.abs().sum().item() > 0.0
    assert instance_grad_norms
    assert pair_grad_norms
    assert any(norm > 0.0 for norm in instance_grad_norms)
    assert any(norm > 0.0 for norm in pair_grad_norms)
