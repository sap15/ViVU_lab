from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.config import apply_runtime_overrides, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "model_b_end_to_end_smoke.yaml"


def test_versioned_model_b_end_to_end_smoke_config_loads() -> None:
    assert CONFIG_PATH.exists()
    assert CONFIG_PATH.stat().st_size > 0

    config = load_config(CONFIG_PATH)
    smoke_config = config["training"]["smoke_test"]
    assert smoke_config["max_pairs"] == 8
    assert smoke_config["epochs"] == 1
    assert smoke_config["resume_epochs"] == 1

    effective_config = apply_runtime_overrides(
        config,
        device="cpu",
        smoke_test=True,
    )
    assert effective_config["training"]["batch_size"] == 2
    assert effective_config["training"]["num_workers"] == 0


def test_versioned_model_b_end_to_end_smoke_config_has_no_personal_paths() -> None:
    raw_text = CONFIG_PATH.read_text(encoding="utf-8")
    forbidden_paths = ("/home/" + "sartesero", "/content/" + "drive")
    assert not any(path in raw_text for path in forbidden_paths)


def test_versioned_model_b_end_to_end_smoke_config_builds_synthetic_pipeline() -> None:
    config = apply_runtime_overrides(
        load_config(CONFIG_PATH),
        device="cpu",
        smoke_test=True,
    )
    config["__config_path__"] = str(CONFIG_PATH)
    pipeline = build_training_pipeline(config)
    try:
        assert pipeline.smoke_data is not None
        assert pipeline.smoke_data.source == "synthetic_temporary"
        assert len(pipeline.dataset) == 8
    finally:
        pipeline.smoke_data.cleanup()


def test_versioned_model_b_end_to_end_smoke_config_runs_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(CONFIG_PATH),
            "--smoke-test",
            "--device",
            "cpu",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    summary = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    assert summary["smoke_dataset_source"] == "synthetic_temporary"
    assert int(summary["train_examples"]) > 0
    assert int(summary["validation_examples"]) > 0
    assert int(summary["epochs_completed"]) == 1
    assert int(summary["resume_epochs_completed"]) == 2
    assert summary["encoder_status"] == "trained"
    assert summary["projection_instance_status"] == "trained"
    assert summary["mlp_delta_status"] in {"inactive", "not_applicable"}
    assert summary["z_delta_learned"] == "false"
