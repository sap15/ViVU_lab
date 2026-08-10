from __future__ import annotations

import torch

from gnn_siamese.losses import NTXentLoss
from gnn_siamese.models import ModelAProjectionHead
from gnn_siamese.training.model_a_contrastive import ModelAContrastive
from test_model_a_two_views import _forward


def _parameter_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _assert_module_gradients(module: torch.nn.Module) -> None:
    parameters = list(module.parameters())
    assert parameters
    gradients = [parameter.grad for parameter in parameters]
    connected = [gradient for gradient in gradients if gradient is not None]
    assert connected
    assert all(torch.isfinite(gradient).all() for gradient in connected)
    assert sum(float(gradient.norm()) for gradient in connected) > 0.0


def test_a6_end_to_end_backward_gradient_flow_and_weight_changes() -> None:
    model, two_views = _forward(torch.device("cpu"))
    model.train()
    projection = ModelAProjectionHead(
        z_delta_pair_dim=16, hidden_dim=12, projection_dim=7
    )
    route = ModelAContrastive(
        projection_head=projection, nt_xent=NTXentLoss(temperature=0.2)
    )
    modules = {
        "encoder": model.one_view_model.shared_encoder,
        "DeltaBlock": model.one_view_model.node_delta_block,
        "MLP_pair_fusion": model.one_view_model.multiscale_relational.pair_fusion,
        "ModelAProjectionHead": projection,
    }
    snapshots = {name: _parameter_snapshot(module) for name, module in modules.items()}
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projection.parameters()), lr=1.0e-3
    )

    # A3 poolings and per-scale relational fusion are parameter-free. Retaining
    # these intermediate gradients audits flow instead of inventing weights.
    flow_tensors = [
        two_views.view1.h_mutation_MUT,
        two_views.view1.h_local_MUT,
        two_views.view1.h_global_MUT,
        two_views.view1.z_delta_mutation,
        two_views.view1.z_delta_local,
        two_views.view1.z_delta_global,
    ]
    for tensor in flow_tensors:
        assert tensor is not None
        tensor.retain_grad()

    optimizer.zero_grad()
    output = route(two_views, positions=[100, 200])
    assert output.z_instance_pair_view1.shape == (2, 7)
    assert output.z_instance_pair_view2.shape == (2, 7)
    assert two_views.h_pair_delta_view1.shape[-1] == 120
    assert two_views.z_delta_pair_view1.shape[-1] == 16
    assert torch.isfinite(output.loss)
    output.loss.backward()

    for module in modules.values():
        _assert_module_gradients(module)
    for tensor in flow_tensors:
        assert tensor is not None and tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.norm() > 0

    optimizer.step()
    for name, module in modules.items():
        assert any(
            not torch.equal(before, after.detach())
            for before, after in zip(snapshots[name], module.parameters(), strict=True)
        ), name


def test_public_a6_wiring_selects_view_z_delta_pair_and_one_shared_head() -> None:
    model, two_views = _forward(torch.device("cpu"))
    projection = ModelAProjectionHead(
        z_delta_pair_dim=16, hidden_dim=10, projection_dim=6
    )
    calls: list[torch.Tensor] = []
    hook = projection.register_forward_pre_hook(
        lambda _module, args: calls.append(args[0])
    )
    route = ModelAContrastive(
        projection_head=projection, nt_xent=NTXentLoss()
    )
    output = route(two_views, positions=[100, 200])
    hook.remove()
    assert calls[0] is two_views.view1.z_delta_pair
    assert calls[1] is two_views.view2.z_delta_pair
    assert all(value is not two_views.h_pair_delta_view1 for value in calls)
    assert [module for module in route.modules() if isinstance(module, ModelAProjectionHead)] == [projection]
    assert output.z_instance_pair_view1.shape[-1] == 6
