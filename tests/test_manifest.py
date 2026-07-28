from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn_siamese.utils.manifest import RunManifestWriter, build_run_layout, collect_git_metadata

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("torch_geometric")

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.training.loop import BaselineEpochOutput
from gnn_siamese.training import train_model_b_pipeline
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def test_manifest_writer_creates_and_updates_run_manifest(tmp_path: Path) -> None:
    layout = build_run_layout(root_dir=tmp_path, model_name="model_b_graph_level_relational", run_id="abc123")
    writer = RunManifestWriter(layout.manifest_path, resolved_config_path=layout.resolved_config_path)
    writer.initialize({"run_id": "abc123", "architecture": "model_b"})
    initial = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert initial["run_id"] == "abc123"
    assert initial["status"] == "running"
    assert "started_at_utc" in initial

    writer.finalize(status="completed", extra_updates={"training": {"epochs_completed": 2}})
    final = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert final["status"] == "completed"
    assert final["training"]["epochs_completed"] == 2
    assert "finished_at_utc" in final


def test_train_pipeline_writes_resolved_manifest_fields(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "training": {"epochs": 1},
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": "manifest_case",
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "resolved_custom.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)
    output = train_model_b_pipeline(pipeline, config_path=config["__config_path__"])

    manifest_path = Path(output.manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["run_id"]
    assert manifest["architecture"] == "model_b"
    assert manifest["model_name"] == "manifest_case"
    assert manifest["configuration"]["resolved_config"]["training"]["epochs"] == 1
    assert manifest["data"]["dataset_fingerprint"]
    assert manifest["data"]["split_fingerprint"]
    assert manifest["artifacts"]["best_checkpoint"].endswith("best.pt")
    assert manifest["artifacts"]["last_checkpoint"].endswith("last.pt")
    assert Path(manifest["artifacts"]["resolved_config"]).is_file()
    assert Path(manifest["artifacts"]["resolved_config"]).name == "resolved_custom.yaml"
    assert not (Path(output.run_dir) / "config_resolved.yaml").exists()
    assert Path(manifest["artifacts"]["split"]).is_file()
    assert Path(manifest["artifacts"]["metrics"]).is_file()
    assert Path(manifest["artifacts"]["gradient_audit"]).is_file()
    git_metadata = manifest["code"]
    assert "commit" in git_metadata
    assert git_metadata["working_tree_state"] in {"clean", "dirty", "unknown"}
    assert manifest["z_delta_learned"] is False


def test_two_consecutive_runs_get_distinct_run_ids_and_directories(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants_unique.hdf5"
    wt_path = tmp_path / "wt_unique.hdf5"
    schema_path = tmp_path / "schema_unique.json"
    split_path = tmp_path / "split_unique.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "training": {"epochs": 1},
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": "unique_runs",
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "config_resolved.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "unique.yaml")

    first_output = train_model_b_pipeline(build_training_pipeline(config), config_path=config["__config_path__"])
    second_output = train_model_b_pipeline(build_training_pipeline(config), config_path=config["__config_path__"])

    first_manifest = json.loads(Path(first_output.manifest_path).read_text(encoding="utf-8"))
    second_manifest = json.loads(Path(second_output.manifest_path).read_text(encoding="utf-8"))
    assert first_manifest["run_id"] != second_manifest["run_id"]
    assert first_output.run_dir != second_output.run_dir
    assert Path(first_output.run_dir).is_dir()
    assert Path(second_output.run_dir).is_dir()
    assert Path(first_output.run_dir, "run_manifest.json").is_file()
    assert Path(second_output.run_dir, "run_manifest.json").is_file()


def test_train_pipeline_marks_failed_manifest_and_reraises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mutant_path = tmp_path / "mutants_failed.hdf5"
    wt_path = tmp_path / "wt_failed.hdf5"
    schema_path = tmp_path / "schema_failed.json"
    split_path = tmp_path / "split_failed.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": "failed_case",
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "config_resolved.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "failed.yaml")
    pipeline = build_training_pipeline(config)

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("gnn_siamese.training.loop.run_model_b_epoch", _raise)
    with pytest.raises(RuntimeError, match="boom"):
        train_model_b_pipeline(pipeline, config_path=config["__config_path__"])

    manifests = list((tmp_path / "runs" / "failed_case").glob("run_*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["errors"] == ["RuntimeError: boom"]


def test_train_pipeline_marks_interrupted_manifest_and_reraises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mutant_path = tmp_path / "mutants_interrupted.hdf5"
    wt_path = tmp_path / "wt_interrupted.hdf5"
    schema_path = tmp_path / "schema_interrupted.json"
    split_path = tmp_path / "split_interrupted.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": "interrupted_case",
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "config_resolved.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "interrupted.yaml")
    pipeline = build_training_pipeline(config)
    call_count = {"value": 0}

    def _interrupt(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            return BaselineEpochOutput(
                phase="train",
                mean_loss=1.0,
                num_batches=2,
                num_examples=4,
                used_eval_mode=False,
                gradients_enabled=True,
                component_means={"nt_xent": 1.0},
                metrics={},
                active_components=["nt_xent"],
                inactive_components=[],
                skipped_components=[],
            )
        raise KeyboardInterrupt("ctrl-c")

    monkeypatch.setattr("gnn_siamese.training.loop.run_model_b_epoch", _interrupt)
    with pytest.raises(KeyboardInterrupt, match="ctrl-c"):
        train_model_b_pipeline(pipeline, config_path=config["__config_path__"])

    manifests = list((tmp_path / "runs" / "interrupted_case").glob("run_*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "interrupted"
    assert manifest["errors"] == ["KeyboardInterrupt: ctrl-c"]


def test_collect_git_metadata_returns_expected_shape() -> None:
    payload = collect_git_metadata()
    assert set(payload) == {"commit", "branch", "working_tree_state"}
