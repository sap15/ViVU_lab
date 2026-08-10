from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.config import ConfigError, resolve_training_device, validate_c1_config
from gnn_siamese.training import load_checkpoint, train_model_b_pipeline
from gnn_siamese.training.checkpointing import move_optimizer_state_to_device
from gnn_siamese.training.loop import run_model_b_epoch
from tests.model_b_test_utils import (
    build_model_b_config,
    create_multi_pair_hdf5,
    write_schema_json,
)


def _config(tmp_path: Path, *, overrides: dict | None = None) -> dict:
    mutant = tmp_path / "mutants.hdf5"
    wt = tmp_path / "wt.hdf5"
    schema = tmp_path / "schema.json"
    create_multi_pair_hdf5(mutant, wt)
    write_schema_json(schema)
    return build_model_b_config(
        mutant,
        wt,
        schema,
        tmp_path / "split.json",
        overrides=overrides,
    )


def test_device_resolution_cpu_auto_and_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_training_device("cpu") == torch.device("cpu")
    assert resolve_training_device("auto") == torch.device("cpu")
    with pytest.raises(ConfigError, match="requested CUDA.*not available.*cpu.*auto"):
        resolve_training_device("cuda")


def test_auto_selects_cuda_logically(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_training_device("auto") == torch.device("cuda")


def test_pipeline_moves_model_before_optimizer_and_configures_loader(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(
        _config(tmp_path, overrides={"training": {"pin_memory": True, "persistent_workers": False}})
    )
    model_parameters = {id(parameter) for parameter in pipeline.model.parameters() if parameter.requires_grad}
    optimizer_parameters = {
        id(parameter)
        for group in pipeline.optimizer.param_groups
        for parameter in group["params"]
    }
    assert {parameter.device.type for parameter in pipeline.model.parameters()} == {"cpu"}
    assert optimizer_parameters == model_parameters
    assert pipeline.dataloaders.train_loader.pin_memory is True
    assert pipeline.dataloaders.train_loader.persistent_workers is False
    batch = next(iter(pipeline.dataloaders.train_loader)).to(pipeline.device)
    assert batch.graph_mut.x.device == next(pipeline.model.parameters()).device


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"persistent_workers": True, "num_workers": 0}, "persistent_workers=true"),
        ({"mixed_precision": {"enabled": True}}, "mixed_precision.enabled=true"),
        ({"gradient_accumulation_steps": 2}, "gradient_accumulation_steps"),
        ({"early_stopping": {"enabled": True}}, "early_stopping.enabled=true"),
    ],
)
def test_unsupported_or_invalid_training_options_fail_before_pipeline(
    tmp_path: Path,
    override: dict,
    message: str,
) -> None:
    config = _config(tmp_path)
    config["training"].update(override)
    with pytest.raises(ConfigError, match=message):
        build_training_pipeline(config)


@pytest.mark.parametrize("clip_norm, expected_calls", [(1.0, 1), (None, 0), (0.0, 0)])
def test_integrated_gradient_clipping_order_and_validation_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clip_norm: float | None,
    expected_calls: int,
) -> None:
    pipeline = build_training_pipeline(_config(tmp_path))
    events: list[str] = []
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = pipeline.optimizer.step

    def tracked_clip(parameters: object, max_norm: float) -> torch.Tensor:
        events.append("clip")
        return original_clip(parameters, max_norm)

    def tracked_step(*args: object, **kwargs: object) -> object:
        events.append("step")
        return original_step(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", tracked_clip)
    monkeypatch.setattr(pipeline.optimizer, "step", tracked_step)
    run_model_b_epoch(
        pipeline.model,
        [next(iter(pipeline.dataloaders.train_loader))],
        pipeline.total_loss_assembler,
        optimizer=pipeline.optimizer,
        device="cpu",
        augmenter=pipeline.augmenter,
        gradient_clip_norm=clip_norm,
    )
    assert events.count("clip") == expected_calls
    if expected_calls:
        assert events.index("clip") < events.index("step")
    events.clear()
    run_model_b_epoch(
        pipeline.model,
        [next(iter(pipeline.dataloaders.validation_loader))],
        pipeline.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
        gradient_clip_norm=1.0,
    )
    assert "clip" not in events


def test_optimizer_state_move_utility_moves_nested_tensors() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.state[next(model.parameters())]["nested"] = {"tensor": torch.ones(1)}
    move_optimizer_state_to_device(optimizer, "cpu")
    assert optimizer.state[next(model.parameters())]["nested"]["tensor"].device.type == "cpu"


def test_output_root_aliases_and_metrics_filename_validation(tmp_path: Path) -> None:
    external = tmp_path / "external" / "runs"
    canonical = validate_c1_config({"outputs": {"root_dir": str(external), "metrics_filename": "epochs.jsonl"}})
    assert canonical["outputs"]["root_dir"] == str(external)
    legacy = validate_c1_config({"paths": {"runs_root": str(external)}})
    assert legacy["outputs"]["root_dir"] == str(external)
    both = validate_c1_config({"paths": {"runs_root": str(external)}, "outputs": {"root_dir": str(external)}})
    assert both["outputs"]["root_dir"] == both["paths"]["runs_root"]
    with pytest.raises(ConfigError, match="must not disagree"):
        validate_c1_config({"paths": {"runs_root": "old"}, "outputs": {"root_dir": "new"}})
    with pytest.raises(ConfigError, match=r"ending in \.jsonl"):
        validate_c1_config({"outputs": {"metrics_filename": "metrics.csv"}})


def test_external_output_root_and_metrics_name_control_all_run_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "drive" / "project-runs"
    pipeline = build_training_pipeline(
        _config(
            tmp_path,
            overrides={
                "outputs": {
                    "root_dir": str(root),
                    "model_name": "c1",
                    "metrics_filename": "epoch_metrics.jsonl",
                }
            },
        )
    )
    output = train_model_b_pipeline(pipeline, config_path=tmp_path / "config.yaml")
    assert Path(output.run_dir).is_relative_to(root)
    assert Path(output.metrics_path).name == "epoch_metrics.jsonl"
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))
    assert Path(manifest["artifacts"]["metrics"]).name == "epoch_metrics.jsonl"
    for value in manifest["artifacts"].values():
        assert Path(value).is_relative_to(root)


@pytest.mark.parametrize("requested_epochs", [0, 1])
def test_resume_rejects_non_increasing_total_epochs(tmp_path: Path, requested_epochs: int) -> None:
    first = _config(tmp_path, overrides={"outputs": {"root_dir": str(tmp_path / "runs"), "model_name": "first"}})
    first_output = train_model_b_pipeline(build_training_pipeline(first), config_path=tmp_path / "first.yaml")
    resumed = deepcopy(first)
    resumed["training"]["epochs"] = requested_epochs
    resumed["outputs"]["model_name"] = f"resume-{requested_epochs}"
    pipeline = build_training_pipeline(resumed)
    with pytest.raises(ValueError, match="desired total historical epoch count"):
        train_model_b_pipeline(
            pipeline,
            config_path=tmp_path / "resume.yaml",
            resume_from=first_output.last_checkpoint_path,
        )
    manifests = list(
        (tmp_path / "runs" / f"resume-{requested_epochs}").glob("run_*/run_manifest.json")
    )
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == "resuming"
    assert manifest["error"]["type"] == "ValueError"


def test_resume_with_greater_total_epochs_runs_additional_epoch(tmp_path: Path) -> None:
    first = _config(tmp_path, overrides={"outputs": {"root_dir": str(tmp_path / "runs"), "model_name": "first"}})
    first_output = train_model_b_pipeline(build_training_pipeline(first), config_path=tmp_path / "first.yaml")
    resumed = deepcopy(first)
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = "resumed"
    output = train_model_b_pipeline(
        build_training_pipeline(resumed),
        config_path=tmp_path / "resume.yaml",
        resume_from=first_output.last_checkpoint_path,
    )
    assert output.epochs_completed == 2
    assert output.epochs_run_this_invocation == 1
    assert load_checkpoint(output.last_checkpoint_path)["epoch_completed"] == 2


def test_cli_controls_yaml_and_configuration_errors_without_traceback(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("training: [invalid\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "error=ConfigError: Invalid YAML" in result.stderr
    assert "Traceback" not in result.stderr
