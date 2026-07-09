from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from gnn_siamese.training import TotalLossAssembler, training_step


class DummyContrastiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z1 = self.proj(batch["view1"])
        z2 = self.proj(batch["view2"])
        return {"z1": z1, "z2": z2}


def test_training_step_runs_forward_backward_and_updates_weights() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)
    batch = {
        "view1": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float32),
        "view2": torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, -0.5]], dtype=torch.float32),
    }

    before = model.proj.weight.detach().clone()
    result = training_step(model, batch, assembler, optimizer=optimizer)
    after = model.proj.weight.detach()

    assert result["did_backward"] is True
    assert result["did_step"] is True
    assert result["loss_output"].active_components == ["nt_xent"]
    assert model.proj.weight.grad is not None
    assert model.proj.weight.grad.abs().sum().item() > 0.0
    assert not torch.allclose(before, after)


def test_training_step_does_not_update_weights_when_all_weights_are_zero() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=0.0, relative_wt_weight=0.0, delta_weight=0.0)
    batch = {
        "view1": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float32),
        "view2": torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, -0.5]], dtype=torch.float32),
    }

    before = model.proj.weight.detach().clone()
    result = training_step(model, batch, assembler, optimizer=optimizer)
    after = model.proj.weight.detach()

    assert result["did_backward"] is False
    assert result["did_step"] is False
    assert result["audit_flags"]["all_components_inactive"] is True
    assert torch.allclose(before, after)
    assert model.proj.weight.grad is None


def test_training_step_exports_metrics_and_api_symbols() -> None:
    model = DummyContrastiveModel()
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)
    batch = {
        "view1": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float32),
        "view2": torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, -0.5]], dtype=torch.float32),
    }

    result = training_step(model, batch, assembler)

    assert "loss_total" in result["metrics"]
    assert "nt_xent" in result["components"]
    assert "component_status" in result["audit_flags"]
    from gnn_siamese.training import __all__ as training_public_api

    assert "training_step" in training_public_api
