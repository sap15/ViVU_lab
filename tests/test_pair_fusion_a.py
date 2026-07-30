from __future__ import annotations

import pytest
import torch

from gnn_siamese.models.pair_fusion_a import MLPPairFusion


def test_exact_architecture_shape_gradients_and_weight_update() -> None:
    module = MLPPairFusion(
        input_dim=120, hidden_dim=32, output_dim=16, dropout=0.0
    )
    assert isinstance(module.network[0], torch.nn.Linear)
    assert isinstance(module.network[1], torch.nn.ReLU)
    assert isinstance(module.network[2], torch.nn.Dropout)
    assert isinstance(module.network[3], torch.nn.Linear)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(parameter) in optimizer_ids for parameter in module.parameters())

    before = [parameter.detach().clone() for parameter in module.parameters()]
    output = module(torch.randn(3, 120))
    assert output.shape == (3, 16)
    output.square().mean().backward()
    gradients = [parameter.grad for parameter in module.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, module.parameters())
    )


def test_identity_is_explicit_and_dimension_safe() -> None:
    module = MLPPairFusion(
        input_dim=12,
        output_dim=12,
        enabled=False,
        disabled_policy="identity",
    )
    value = torch.randn(2, 12)
    assert module.mode == "identity"
    assert module(value) is value
    assert list(module.parameters()) == []

    with pytest.raises(ValueError, match="identity fusion requires output_dim"):
        MLPPairFusion(input_dim=12, output_dim=7, enabled=False)
    with pytest.raises(ValueError, match="disabled_policy must be 'identity'"):
        MLPPairFusion(
            input_dim=12, output_dim=12, enabled=False, disabled_policy="none"
        )


def test_input_dimension_is_validated() -> None:
    module = MLPPairFusion(input_dim=12, hidden_dim=8, output_dim=4)
    with pytest.raises(ValueError, match="input_dim=12"):
        module(torch.randn(2, 11))
