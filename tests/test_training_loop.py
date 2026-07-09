from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from gnn_siamese.training import (
    TotalLossAssembler,
    TrainingLoopConfig,
    TrainingLoopOutput,
    build_run_manifest,
    fit,
)


class DummyContrastiveModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z1 = self.proj(batch["view1"])
        z2 = self.proj(batch["view2"])
        return {"z1": z1, "z2": z2}


class NonFiniteLossAssembler(nn.Module):
    def forward(self, **_: torch.Tensor) -> object:
        zero = torch.tensor(0.0, dtype=torch.float32)
        return type(
            "NonFiniteLossOutput",
            (),
            {
                "loss": torch.tensor(float("nan"), dtype=torch.float32),
                "components": {"nt_xent": zero},
                "weights": {"nt_xent": 1.0, "relative_wt": 0.0, "delta": 0.0},
                "active_components": ["nt_xent"],
                "inactive_components": [],
                "skipped_components": ["relative_wt", "delta"],
                "metrics": {"loss_total": torch.tensor(float("nan"), dtype=torch.float32)},
                "audit_flags": {
                    "component_status": {
                        "nt_xent": "active",
                        "relative_wt": "skipped_weight_zero",
                        "delta": "skipped_weight_zero",
                    }
                },
            },
        )()


class DummyScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1


def _make_dataloader() -> list[dict[str, torch.Tensor]]:
    batch_a = {
        "view1": torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float32),
        "view2": torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, -0.5]], dtype=torch.float32),
    }
    batch_b = {
        "view1": torch.tensor([[0.5, 0.2, 0.1], [0.2, 0.5, -0.1]], dtype=torch.float32),
        "view2": torch.tensor([[0.45, 0.25, 0.1], [0.25, 0.45, -0.1]], dtype=torch.float32),
    }
    return [batch_a, batch_b]


def _fit_config(**overrides: object) -> TrainingLoopConfig:
    config = TrainingLoopConfig(
        epochs=2,
        device="cpu",
        grad_clip_norm=None,
        log_every=0,
        stop_on_nonfinite_loss=True,
        run_name="test-loop",
        write_manifest=False,
        seed=123,
    )
    return TrainingLoopConfig(**(config.__dict__ | overrides))


def test_basic_loop_runs_two_epochs_and_returns_history() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)

    output = fit(model, _make_dataloader(), optimizer, assembler, _fit_config())

    assert isinstance(output, TrainingLoopOutput)
    assert output.epochs_completed == 2
    assert output.num_steps == 4
    assert len(output.history) == 2
    assert output.history[0]["epoch"] == 1
    assert "loss_total" in output.final_metrics


def test_fit_updates_weights_when_a_loss_component_is_active() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)
    before = model.proj.weight.detach().clone()

    fit(model, _make_dataloader(), optimizer, assembler, _fit_config())

    after = model.proj.weight.detach()
    assert not torch.allclose(before, after)


def test_baseline_flags_keep_only_nt_xent_active() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)

    output = fit(model, _make_dataloader(), optimizer, assembler, _fit_config())

    assert output.audit_flags["baseline_only_nt_xent"] is True
    assert output.audit_flags["relative_wt_active"] is False
    assert output.audit_flags["delta_active"] is False
    assert output.audit_flags["z_delta_not_trained"] is True
    assert output.audit_flags["loss_component_status"]["relative_wt"] == "skipped_weight_zero"
    assert output.audit_flags["loss_component_status"]["delta"] == "skipped_weight_zero"


def test_all_components_inactive_does_not_fail_and_does_not_update_weights() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=0.0, relative_wt_weight=0.0, delta_weight=0.0)
    before = model.proj.weight.detach().clone()

    output = fit(model, _make_dataloader(), optimizer, assembler, _fit_config())

    after = model.proj.weight.detach()
    assert output.audit_flags["all_components_inactive"] is True
    assert torch.allclose(before, after)


def test_nonfinite_loss_stops_early_with_explicit_reason() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    output = fit(
        model,
        _make_dataloader(),
        optimizer,
        NonFiniteLossAssembler(),
        _fit_config(epochs=3, stop_on_nonfinite_loss=True),
    )

    assert output.stopped_early is True
    assert output.stop_reason == "nonfinite_loss"
    assert output.audit_flags["nonfinite_loss_detected"] is True
    assert output.epochs_completed == 1


def test_grad_clipping_executes_without_breaking_training() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)

    output = fit(
        model,
        _make_dataloader(),
        optimizer,
        assembler,
        _fit_config(grad_clip_norm=0.5),
    )

    assert output.epochs_completed == 2
    assert output.stopped_early is False


def test_scheduler_is_called_once_per_epoch() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)
    scheduler = DummyScheduler()

    output = fit(model, _make_dataloader(), optimizer, assembler, _fit_config(), scheduler=scheduler)

    assert output.epochs_completed == 2
    assert scheduler.step_calls == 2


def test_build_run_manifest_returns_expected_fields_and_can_write(tmp_path) -> None:
    output = TrainingLoopOutput(
        history=[],
        final_metrics={"loss_total": 0.123},
        epochs_completed=2,
        num_steps=4,
        stopped_early=False,
        stop_reason=None,
        manifest=None,
        audit_flags={
            "active_components": ["nt_xent"],
            "inactive_components": [],
            "skipped_components": ["relative_wt", "delta"],
            "loss_component_status": {
                "nt_xent": "active",
                "relative_wt": "skipped_weight_zero",
                "delta": "skipped_weight_zero",
            },
            "all_components_inactive": False,
            "nonfinite_loss_detected": False,
            "relative_wt_active": False,
            "delta_active": False,
            "baseline_only_nt_xent": True,
        },
    )
    config = _fit_config(write_manifest=True, output_dir=tmp_path)

    manifest = build_run_manifest(config, output, output_dir=tmp_path, write_manifest=True)

    manifest_path = tmp_path / "run_manifest.json"
    assert manifest["run_name"] == "test-loop"
    assert manifest["loss_components"]["active"] == ["nt_xent"]
    assert manifest["reconstruction_status"] == "disabled/pending"
    assert manifest_path.is_file() is True
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["epochs_completed"] == 2
    assert payload["delta_active"] is False


def test_batch_adapter_can_prepare_loss_inputs_without_coupling_to_dataset() -> None:
    model = DummyContrastiveModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    assembler = TotalLossAssembler(nt_xent_weight=1.0, relative_wt_weight=0.0, delta_weight=0.0)
    raw_batches = [
        {
            "x1": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            "x2": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
        }
    ]

    def batch_adapter(batch: dict[str, torch.Tensor], *, device: torch.device) -> dict[str, object]:
        return {
            "loss_inputs": {
                "z1": batch["x1"].to(device),
                "z2": batch["x2"].to(device),
            }
        }

    output = fit(
        model,
        raw_batches,
        optimizer,
        assembler,
        _fit_config(epochs=1),
        batch_adapter=batch_adapter,
    )

    assert output.epochs_completed == 1
    assert output.num_steps == 1
    assert output.audit_flags["baseline_only_nt_xent"] is True


def test_training_loop_api_is_exported_from_training_package() -> None:
    from gnn_siamese.training import __all__ as training_public_api

    assert "fit" in training_public_api
    assert "TrainingLoopConfig" in training_public_api
    assert "TrainingLoopOutput" in training_public_api
    assert "build_run_manifest" in training_public_api
