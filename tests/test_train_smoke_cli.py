from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import gnn_siamese.builders as builders_module
from gnn_siamese.data import prepare_smoke_data
from tests.model_b_test_utils import build_model_b_config

SMOKE_FILE_NAMES = {
    "mutants_smoke.hdf5",
    "wt_companion_smoke.hdf5",
    "sample_schema.json",
    "split_leave_position_out.json",
}


def _build_smoke_fallback_config(tmp_path: Path) -> dict:
    empty_sample_root = tmp_path / "empty_sample_data"
    empty_sample_root.mkdir(parents=True, exist_ok=True)
    config = build_model_b_config(
        tmp_path / "unused_mutants.hdf5",
        tmp_path / "unused_wt.hdf5",
        tmp_path / "unused_schema.json",
        tmp_path / "unused_split.json",
        overrides={
            "paths": {
                "mutants_hdf5": None,
                "wt_companion_hdf5": None,
                "sample_data_root": str(empty_sample_root),
            },
            "training": {
                "smoke_test": {"enabled": True, "epochs": 1, "batch_size": 2},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    return config


def test_prepare_smoke_data_falls_back_to_synthetic_and_cleans_up(tmp_path: Path) -> None:
    config = _build_smoke_fallback_config(tmp_path)

    artifacts = prepare_smoke_data(config)
    temp_dir = Path(artifacts.temp_dir)
    try:
        assert artifacts.source == "synthetic_temporary"
        assert artifacts.pair_count == 8
        assert temp_dir.name.startswith("gnn_siamese_smoke_")
        assert Path(artifacts.mutants_hdf5).exists()
        assert Path(artifacts.wt_companion_hdf5).exists()
        assert Path(artifacts.schema_json).exists()
        assert Path(artifacts.split_json).parent == temp_dir
    finally:
        artifacts.cleanup()

    assert not temp_dir.exists()


def _list_tmp_runtime_entries(tmp_root: Path) -> list[str]:
    return sorted(path.relative_to(tmp_root).as_posix() for path in tmp_root.rglob("*"))


def _list_smoke_artifacts(tmp_root: Path) -> list[str]:
    smoke_artifacts: list[str] = []
    for relative_path in _list_tmp_runtime_entries(tmp_root):
        path = Path(relative_path)
        top_level = path.parts[0]
        if top_level.startswith("gnn_siamese_smoke_") or path.name in SMOKE_FILE_NAMES:
            smoke_artifacts.append(relative_path)
    return smoke_artifacts


def test_smoke_cli_runs_without_manual_pythonpath_and_uses_synthetic_fallback(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    pytest.importorskip("torch_geometric")

    repo_root = Path(__file__).resolve().parents[1]
    empty_sample_root = tmp_path / "empty_sample_data"
    empty_sample_root.mkdir(parents=True, exist_ok=True)
    tmp_root = tmp_path / "tmp_runtime"
    tmp_root.mkdir(parents=True, exist_ok=True)

    config_path = tmp_path / "smoke_cli.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "extends": str((repo_root / "configs" / "model_b_baseline.yaml").resolve()),
                "paths": {"sample_data_root": str(empty_sample_root)},
                "training": {"smoke_test": {"epochs": 1, "batch_size": 2}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["TMPDIR"] = str(tmp_root)

    result = subprocess.run(
        [
            "python3",
            "scripts/train.py",
            "--config",
            str(config_path),
            "--smoke-test",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    summary = dict(line.split("=", 1) for line in stdout_lines if "=" in line)

    assert summary["smoke_dataset_source"] == "synthetic_temporary"
    assert int(summary["train_examples"]) >= 2
    assert int(summary["validation_examples"]) >= 2
    assert int(summary["epochs_completed"]) == 1
    assert summary["device"] == "cpu"
    assert math.isfinite(float(summary["train_loss"]))
    assert math.isfinite(float(summary["validation_loss"]))
    smoke_artifacts = _list_smoke_artifacts(tmp_root)
    assert smoke_artifacts == [], f"Smoke artifacts left in TMPDIR: {smoke_artifacts}"


def test_build_training_pipeline_cleans_smoke_artifacts_when_builder_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    pytest.importorskip("torch_geometric")

    config = _build_smoke_fallback_config(tmp_path)
    config["model"]["projection_instance"]["num_layers"] = 3

    captured_temp_dir: dict[str, Path] = {}
    original_prepare_smoke_data = builders_module.prepare_smoke_data

    def tracked_prepare_smoke_data(config_payload: dict) -> object:
        artifacts = original_prepare_smoke_data(config_payload)
        captured_temp_dir["path"] = Path(artifacts.temp_dir)
        return artifacts

    monkeypatch.setattr(builders_module, "prepare_smoke_data", tracked_prepare_smoke_data)

    with pytest.raises(builders_module.BuilderError, match="num_layers=2"):
        builders_module.build_training_pipeline(config)

    temp_dir = captured_temp_dir["path"]
    assert temp_dir.name.startswith("gnn_siamese_smoke_")
    assert not temp_dir.exists()


def test_smoke_cli_cleans_smoke_artifacts_when_training_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("h5py")
    pytest.importorskip("torch_geometric")

    import scripts.train as train_script

    config_path = tmp_path / "smoke_cli.yaml"
    config_path.write_text(
        yaml.safe_dump(_build_smoke_fallback_config(tmp_path), sort_keys=False),
        encoding="utf-8",
    )

    captured_temp_dir: dict[str, Path] = {}
    original_build_training_pipeline = train_script.build_training_pipeline

    def tracked_build_training_pipeline(config_payload: dict) -> object:
        pipeline = original_build_training_pipeline(config_payload)
        assert pipeline.smoke_data is not None
        captured_temp_dir["path"] = Path(pipeline.smoke_data.temp_dir)
        return pipeline

    def failing_fit_model_b_baseline(*args: object, **kwargs: object) -> object:
        raise RuntimeError("forced training failure")

    monkeypatch.setattr(train_script, "build_training_pipeline", tracked_build_training_pipeline)
    monkeypatch.setattr(train_script, "fit_model_b_baseline", failing_fit_model_b_baseline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--smoke-test",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(RuntimeError, match="forced training failure"):
        train_script.main()

    temp_dir = captured_temp_dir["path"]
    assert temp_dir.name.startswith("gnn_siamese_smoke_")
    assert not temp_dir.exists()


def test_production_code_does_not_import_from_tests() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    production_files = list((repo_root / "src").rglob("*.py")) + [repo_root / "scripts" / "train.py"]
    offenders: list[str] = []
    for path in production_files:
        text = path.read_text(encoding="utf-8")
        if "from tests." in text or "import tests" in text:
            offenders.append(str(path.relative_to(repo_root)))
    assert offenders == []
