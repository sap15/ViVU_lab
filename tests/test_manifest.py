from __future__ import annotations

import json
import os
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
from gnn_siamese.utils.fingerprints import fingerprint_split_definition


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


def test_manifest_update_publication_failure_preserves_previous_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    writer = RunManifestWriter(manifest_path)
    writer.initialize({"run_id": "atomic-case", "training": {"epochs_completed": 0}})
    previous_content = manifest_path.read_text(encoding="utf-8")

    def _fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "replace", _fail_replace)
    with pytest.raises(OSError, match="simulated publication failure"):
        writer.update({"training": {"epochs_completed": 1}})

    assert manifest_path.read_text(encoding="utf-8") == previous_content
    assert json.loads(previous_content)["training"]["epochs_completed"] == 0
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []


def test_manifest_update_preserves_existing_fields_and_writes_valid_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    writer = RunManifestWriter(manifest_path)
    writer.initialize({"run_id": "update-case", "training": {"epochs_completed": 0}})

    writer.update({"training": {"epochs_completed": 1}, "metrics": {"loss": 0.5}})

    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["run_id"] == "update-case"
    assert updated["status"] == "running"
    assert updated["training"]["epochs_completed"] == 1
    assert updated["metrics"]["loss"] == 0.5
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []


def test_manifest_json_serialization_failure_preserves_previous_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run_manifest.json"
    writer = RunManifestWriter(manifest_path)
    writer.initialize({"run_id": "serialization-case"})
    previous_content = manifest_path.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        writer.update({"not_json_serializable": object()})

    assert manifest_path.read_text(encoding="utf-8") == previous_content
    assert json.loads(previous_content)["run_id"] == "serialization-case"
    assert list(tmp_path.glob(".run_manifest.json.*.tmp")) == []


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
    assert manifest["data"]["split_fingerprint"] == fingerprint_split_definition(
        pipeline.split_bundle.split
    )
    assert manifest["data"]["dataset_fingerprint"] == pipeline.split_bundle.split.dataset_fingerprint
    assert manifest["data"]["inventory"] == {
        "hdf5_mutant_input_groups": len(pipeline.dataset.pairs),
        "biological_variants": len(pipeline.dataset.pairs),
        "native_wt_controls": 0,
        "trainable_variants": len(pipeline.dataset.pairs),
        "native_wt_control_ids": [],
        "native_wt_control_policy": {
            "role": "evaluation_control",
            "used_for_split": False,
            "used_for_training": False,
            "used_for_validation": False,
            "used_for_test_loss": False,
            "available_for_inference": True,
        },
    }
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
    assert git_metadata["git_dirty"] in {True, False, None}
    assert manifest["data"]["split_source_path"] == str(split_path.resolve())
    assert manifest["configuration"]["seed_bundle"] == {
        "project": 123,
        "run": 123,
        "split": 123,
        "python": None,
        "numpy": None,
        "torch": None,
        "cuda": None,
        "model_initialization": None,
        "dataloader_configured": None,
        "dataloader_effective": 123,
    }
    assert manifest["artifacts"]["run_root"] == str(Path(output.run_dir).parent)
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
    assert set(payload) == {"commit", "branch", "working_tree_state", "git_dirty"}
