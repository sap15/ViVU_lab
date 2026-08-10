from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.training import load_checkpoint, train_model_b_pipeline
from gnn_siamese.training.checkpointing import (
    build_legacy_resume_compatibility_payload,
    save_checkpoint_payload_atomic,
    validate_resume_compatibility,
)
from gnn_siamese.utils.fingerprints import (
    combine_content_fingerprints,
    fingerprint_file,
    fingerprint_hdf5_inputs,
)
from gnn_siamese.utils.manifest import RunManifestWriter, build_run_layout
from tests.model_b_test_utils import (
    build_model_b_config,
    create_multi_pair_hdf5,
    write_schema_json,
)


def test_streaming_content_fingerprint_is_location_independent_and_byte_sensitive(tmp_path: Path) -> None:
    original = tmp_path / "local" / "mutants.hdf5"
    relocated = tmp_path / "content" / "renamed.hdf5"
    original.parent.mkdir()
    relocated.parent.mkdir()
    original.write_bytes(b"HDF5-synthetic-content")
    shutil.copy2(original, relocated)

    first = fingerprint_file(original, role="mutants", logical_identity="pkp2:mutants", chunk_size=3)
    second = fingerprint_file(relocated, role="mutants", logical_identity="pkp2:mutants", chunk_size=5)
    assert first == second
    assert first["algorithm"] == "sha256"
    assert first["version"] == 1
    assert first["scope"] == "raw_file_bytes"

    relocated.write_bytes(relocated.read_bytes() + b"x")
    changed = fingerprint_file(relocated, role="mutants", logical_identity="pkp2:mutants")
    assert changed["digest"] != first["digest"]


def test_combined_content_fingerprint_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    mutants = tmp_path / "mutants.hdf5"
    wt = tmp_path / "wt.hdf5"
    mutants.write_bytes(b"mutants")
    wt.write_bytes(b"wt")
    records = [
        fingerprint_file(mutants, role="mutants"),
        fingerprint_file(wt, role="wt_companion"),
    ]
    assert combine_content_fingerprints(records) == combine_content_fingerprints(reversed(records))

    bundle = fingerprint_hdf5_inputs(mutants_path=mutants, wt_companion_path=wt, dataset_id="pkp2")
    assert bundle["combined"]["digest"]
    assert {item["role"] for item in bundle["files"]} == {"mutants", "wt_companion"}


def test_manifest_lifecycle_transitions_and_portable_reference_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = build_run_layout(root_dir=tmp_path, model_name="model", run_id="one")
    layout.run_dir.mkdir(parents=True)
    writer = RunManifestWriter(layout.manifest_path)
    writer.initialize({"run_id": "one", "status": "initializing"})
    writer.set_stage("building_dataset")
    writer.transition("running", stage="training")
    writer.finalize(status="completed")
    payload = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["lifecycle"]["stage"] == "training"
    assert layout.relative_reference(layout.checkpoints_dir / "last.pt") == "checkpoints/last.pt"
    assert layout.resolve_reference("checkpoints/last.pt") == layout.checkpoints_dir / "last.pt"
    moved = tmp_path / "moved" / layout.run_dir.name
    shutil.copytree(layout.run_dir, moved)
    moved_layout = build_run_layout(root_dir=moved.parent.parent, model_name=moved.parent.name, run_id="one")
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    assert moved_layout.resolve_reference("checkpoints/last.pt") == moved / "checkpoints" / "last.pt"
    with pytest.raises(ValueError, match="must be relative"):
        layout.resolve_reference(tmp_path / "absolute.pt")
    for unsafe in ("../outside", "subdir/../../outside"):
        with pytest.raises(ValueError, match="escapes run_dir"):
            layout.resolve_reference(unsafe)
    with pytest.raises(ValueError, match="Invalid run lifecycle transition"):
        writer.transition("running")


@pytest.mark.parametrize(
    "stage",
    [
        "opening_mutants_hdf5",
        "opening_wt_hdf5",
        "building_dataset",
        "building_split",
        "building_dataloaders",
        "building_model",
        "building_optimizer",
        "building_scheduler",
    ],
)
def test_early_manifest_records_structured_builder_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    import scripts.train as train_script
    from gnn_siamese.utils.interruptions import InterruptionController

    config = {
        "outputs": {"root_dir": str(tmp_path / "runs"), "model_name": "failure"},
        "training": {},
    }

    def fail_builder(_config: dict, *, stage_callback) -> None:
        stage_callback(stage)
        raise RuntimeError(f"failure at {stage}")

    monkeypatch.setattr(train_script, "build_training_pipeline", fail_builder)
    with pytest.raises(RuntimeError, match="failure at"):
        train_script._build_with_early_manifest(
            config,
            config_path=str(tmp_path / "config.yaml"),
            controller=InterruptionController(),
        )
    manifests = list((tmp_path / "runs" / "failure").glob("run_*/run_manifest.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["lifecycle"]["stage"] == stage
    assert payload["error"]["type"] == "RuntimeError"
    assert stage in payload["error"]["message"]
    assert payload["error"]["timestamp_utc"]
    assert "RuntimeError" in payload["error"]["traceback"]


def _portable_config(tmp_path: Path, *, model_name: str, epochs: int = 1) -> dict:
    mutants = tmp_path / "data" / "mutants.hdf5"
    wt = tmp_path / "data" / "wt.hdf5"
    schema = tmp_path / "schema.json"
    split = tmp_path / f"{model_name}-split.json"
    mutants.parent.mkdir(parents=True, exist_ok=True)
    create_multi_pair_hdf5(mutants, wt)
    write_schema_json(schema)
    config = build_model_b_config(
        mutants,
        wt,
        schema,
        split,
        overrides={
            "training": {"epochs": epochs},
            "outputs": {"root_dir": str(tmp_path / "runs"), "model_name": model_name},
        },
    )
    config["__config_path__"] = str(tmp_path / f"{model_name}.yaml")
    return config


def _only_manifest(config: dict) -> tuple[Path, dict]:
    root = Path(config["outputs"]["root_dir"]) / str(config["outputs"]["model_name"])
    manifests = list(root.glob("run_*/run_manifest.json"))
    assert len(manifests) == 1
    return manifests[0], json.loads(manifests[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("role", "expected_stage"),
    [("mutants_hdf5", "opening_mutants_hdf5"), ("wt_companion_hdf5", "opening_wt_hdf5")],
)
def test_real_hdf5_open_failure_has_terminal_manifest(
    tmp_path: Path,
    role: str,
    expected_stage: str,
) -> None:
    import scripts.train as train_script
    from gnn_siamese.utils.interruptions import InterruptionController

    config = _portable_config(tmp_path, model_name=f"open-{role}")
    Path(config["paths"][role]).write_bytes(b"not-an-hdf5")
    with pytest.raises(OSError):
        train_script._build_with_early_manifest(
            config,
            config_path=config["__config_path__"],
            controller=InterruptionController(),
        )
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == expected_stage
    assert manifest["error"]["type"] == "OSError"


@pytest.mark.parametrize(
    ("target", "stage"),
    [
        ("MutWtPairDataset", "building_dataset"),
        ("build_split_bundle", "building_split"),
        ("build_dataloaders", "building_dataloaders"),
        ("build_model", "building_model"),
        ("build_optimizer", "building_optimizer"),
        ("build_scheduler", "building_scheduler"),
    ],
)
def test_real_builder_boundary_failure_has_terminal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    stage: str,
) -> None:
    import gnn_siamese.builders as builders
    import scripts.train as train_script
    from gnn_siamese.utils.interruptions import InterruptionController

    config = _portable_config(tmp_path, model_name=f"boundary-{target}")

    def fail(*args, **kwargs):
        raise RuntimeError(f"{target} boundary failure")

    monkeypatch.setattr(builders, target, fail)
    with pytest.raises(RuntimeError, match="boundary failure"):
        train_script._build_with_early_manifest(
            config,
            config_path=config["__config_path__"],
            controller=InterruptionController(),
        )
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == stage
    assert manifest["error"]["type"] == "RuntimeError"


@pytest.mark.parametrize(
    ("boundary", "stage"),
    [
        ("fingerprint", "fingerprinting_hdf5"),
        ("split_copy", "copying_split"),
        ("compatibility", "building_resume_compatibility"),
        ("gradient_audit", "initializing_gradient_audit"),
    ],
)
def test_real_training_boundary_failure_has_terminal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    stage: str,
) -> None:
    import gnn_siamese.training.loop as training_loop

    config = _portable_config(tmp_path, model_name=f"training-boundary-{boundary}")
    pipeline = build_training_pipeline(config)

    def fail(*args, **kwargs):
        raise RuntimeError(f"{boundary} boundary failure")

    if boundary == "fingerprint":
        monkeypatch.setattr(training_loop, "fingerprint_hdf5_inputs", fail)
    elif boundary == "compatibility":
        monkeypatch.setattr(training_loop, "build_resume_compatibility_payload", fail)
    elif boundary == "gradient_audit":
        monkeypatch.setattr(training_loop, "create_gradient_audit", fail)
    else:
        real_atomic = training_loop.atomic_write_text

        def fail_split(destination, content, **kwargs):
            if Path(destination).name == "split.json":
                fail()
            return real_atomic(destination, content, **kwargs)

        monkeypatch.setattr(training_loop, "atomic_write_text", fail_split)

    with pytest.raises(RuntimeError, match="boundary failure"):
        train_model_b_pipeline(pipeline, config_path=config["__config_path__"])
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == stage
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["status"] != "completed"


def test_resume_failure_has_terminal_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn_siamese.training.loop as training_loop

    initial = _portable_config(tmp_path, model_name="resume-failure-source")
    source = train_model_b_pipeline(
        build_training_pipeline(initial),
        config_path=initial["__config_path__"],
    )
    resumed = deepcopy(initial)
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = "resume-failure-target"

    def fail_resume(*args, **kwargs):
        raise RuntimeError("resume boundary failure")

    monkeypatch.setattr(training_loop, "resume_from_checkpoint", fail_resume)
    with pytest.raises(RuntimeError, match="resume boundary failure"):
        train_model_b_pipeline(
            build_training_pipeline(resumed),
            config_path=resumed["__config_path__"],
            resume_from=source.last_checkpoint_path,
        )
    _path, manifest = _only_manifest(resumed)
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == "resuming"


def test_smoke_post_bootstrap_failure_retains_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn_siamese.training.loop as training_loop
    import scripts.train as train_script
    from gnn_siamese.utils.interruptions import InterruptionController

    config = _portable_config(tmp_path, model_name="smoke-context")
    config["training"]["smoke_test"] = {"enabled": True, "epochs": 1, "batch_size": 2, "resume_epochs": 1}

    def fail_fingerprint(*args, **kwargs):
        raise RuntimeError("smoke fingerprint failure")

    monkeypatch.setattr(training_loop, "fingerprint_hdf5_inputs", fail_fingerprint)
    with pytest.raises(RuntimeError, match="smoke fingerprint failure"):
        train_script._run_smoke_end_to_end(
            config,
            config_path=config["__config_path__"],
            controller=InterruptionController(),
        )
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == "fingerprinting_hdf5"


def _write_smoke_cli_config(config: dict, path: Path) -> None:
    serializable = {key: value for key, value in config.items() if not key.startswith("__")}
    path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")


def test_initial_smoke_verification_failure_finishes_manifest_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train as train_script

    config = _portable_config(tmp_path, model_name="smoke-validation-initial")
    config["training"]["smoke_test"] = {
        "enabled": True,
        "epochs": 1,
        "resume_epochs": 1,
        "batch_size": 2,
        "max_pairs": 8,
    }
    config_path = tmp_path / "smoke-validation-initial.yaml"
    _write_smoke_cli_config(config, config_path)
    monkeypatch.setattr(
        train_script,
        "_parse_args",
        lambda: SimpleNamespace(config=str(config_path), device="cpu", smoke_test=True, resume_from=None),
    )

    def fail_verification(**kwargs):
        raise RuntimeError("initial smoke artifact verification failed")

    monkeypatch.setattr(train_script, "_verify_run_artifacts", fail_verification)
    assert train_script.main() == 1

    manifest_path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["lifecycle"]["stage"] == "validating_smoke_artifacts"
    assert manifest["error"]["type"] == "RuntimeError"
    assert manifest["error"]["message"] == "initial smoke artifact verification failed"
    assert manifest["error"]["timestamp_utc"]
    assert "RuntimeError" in manifest["error"]["traceback"]
    checkpoint_path = manifest_path.parent / manifest["training"]["last_valid_checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["epoch_completed"] == manifest["training"]["epochs_completed"] == 1
    assert checkpoint["global_step"] == manifest["training"]["global_step"] == 2
    assert load_checkpoint(manifest_path.parent / "checkpoints" / "best.pt")["format_version"] == 1
    assert not list(manifest_path.parent.rglob(".*.tmp"))


def test_resumed_smoke_verification_failure_only_fails_resumed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.train as train_script

    config = _portable_config(tmp_path, model_name="smoke-validation-resume")
    config["training"]["smoke_test"] = {
        "enabled": True,
        "epochs": 1,
        "resume_epochs": 1,
        "batch_size": 2,
        "max_pairs": 8,
    }
    config_path = tmp_path / "smoke-validation-resume.yaml"
    _write_smoke_cli_config(config, config_path)
    monkeypatch.setattr(
        train_script,
        "_parse_args",
        lambda: SimpleNamespace(config=str(config_path), device="cpu", smoke_test=True, resume_from=None),
    )
    real_verify = train_script._verify_run_artifacts
    verification_calls = 0

    def fail_resumed_verification(**kwargs):
        nonlocal verification_calls
        verification_calls += 1
        if verification_calls == 2:
            raise RuntimeError("resumed smoke artifact verification failed")
        return real_verify(**kwargs)

    monkeypatch.setattr(train_script, "_verify_run_artifacts", fail_resumed_verification)
    assert train_script.main() == 1

    manifests = sorted(
        (Path(config["outputs"]["root_dir"]) / config["outputs"]["model_name"]).glob(
            "run_*/run_manifest.json"
        )
    )
    assert len(manifests) == 2
    payloads = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in manifests]
    completed = [(path, payload) for path, payload in payloads if payload["status"] == "completed"]
    failed = [(path, payload) for path, payload in payloads if payload["status"] == "failed"]
    assert len(completed) == len(failed) == 1
    initial_path, initial_manifest = completed[0]
    resumed_path, resumed_manifest = failed[0]
    assert initial_manifest["training"]["epochs_completed"] == 1
    assert resumed_manifest["lifecycle"]["stage"] == "validating_smoke_artifacts"
    assert resumed_manifest["error"]["type"] == "RuntimeError"
    assert resumed_manifest["error"]["message"] == "resumed smoke artifact verification failed"
    resumed_checkpoint = load_checkpoint(
        resumed_path.parent / resumed_manifest["training"]["last_valid_checkpoint"]
    )
    assert resumed_checkpoint["epoch_completed"] == resumed_manifest["training"]["epochs_completed"] == 2
    assert resumed_checkpoint["global_step"] == resumed_manifest["training"]["global_step"] == 4
    assert load_checkpoint(initial_path.parent / "checkpoints" / "last.pt")["epoch_completed"] == 1
    assert not list(initial_path.parent.rglob(".*.tmp"))
    assert not list(resumed_path.parent.rglob(".*.tmp"))


def test_resume_accepts_relocated_hdf5_and_rejects_modified_content(tmp_path: Path) -> None:
    initial = _portable_config(tmp_path, model_name="initial")
    first = train_model_b_pipeline(
        build_training_pipeline(initial),
        config_path=initial["__config_path__"],
    )

    relocated_root = tmp_path / "relocated"
    relocated_root.mkdir()
    relocated_mutants = relocated_root / "renamed-mutants.hdf5"
    relocated_wt = relocated_root / "renamed-wt.hdf5"
    shutil.copy2(initial["paths"]["mutants_hdf5"], relocated_mutants)
    shutil.copy2(initial["paths"]["wt_companion_hdf5"], relocated_wt)
    resumed = deepcopy(initial)
    resumed["paths"]["mutants_hdf5"] = str(relocated_mutants)
    resumed["paths"]["wt_companion_hdf5"] = str(relocated_wt)
    resumed["split"]["persist_path"] = str(tmp_path / "relocated-split.json")
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = "relocated"
    output = train_model_b_pipeline(
        build_training_pipeline(resumed),
        config_path=resumed["__config_path__"],
        resume_from=first.last_checkpoint_path,
    )
    assert output.epochs_completed == 2
    checkpoint = load_checkpoint(output.last_checkpoint_path)
    assert checkpoint["hdf5_content_fingerprint"]["scope"] == "hdf5_content_set"

    relocated_mutants.write_bytes(relocated_mutants.read_bytes() + b"x")
    altered = deepcopy(resumed)
    altered["training"]["epochs"] = 3
    altered["outputs"]["model_name"] = "altered"
    with pytest.raises(ValueError, match="hdf5_content_fingerprint"):
        train_model_b_pipeline(
            build_training_pipeline(altered),
            config_path=altered["__config_path__"],
            resume_from=output.last_checkpoint_path,
        )


def test_each_hdf5_is_hashed_once_and_result_is_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn_siamese.training.loop as training_loop

    config = _portable_config(tmp_path, model_name="hash-once")
    real = training_loop.fingerprint_hdf5_inputs
    calls: list[tuple[str, str]] = []

    def counted(**kwargs):
        calls.append((str(kwargs["mutants_path"]), str(kwargs["wt_companion_path"])))
        return real(**kwargs)

    monkeypatch.setattr(training_loop, "fingerprint_hdf5_inputs", counted)
    output = train_model_b_pipeline(
        build_training_pipeline(config),
        config_path=config["__config_path__"],
    )
    assert calls == [(config["paths"]["mutants_hdf5"], config["paths"]["wt_companion_hdf5"])]
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(output.last_checkpoint_path)
    combined = manifest["data"]["hdf5_content_fingerprint"]["combined"]
    assert checkpoint["hdf5_content_fingerprint"] == combined
    assert checkpoint["compatibility"]["hdf5_content_fingerprint"] == combined


def test_complete_run_can_move_and_resolve_internal_references_from_new_root(tmp_path: Path) -> None:
    config = _portable_config(tmp_path, model_name="move-complete-run")
    output = train_model_b_pipeline(
        build_training_pipeline(config),
        config_path=config["__config_path__"],
    )
    source = Path(output.run_dir)
    moved = tmp_path / "copied-to-colab" / source.name
    moved.parent.mkdir()
    shutil.copytree(source, moved)
    payload = json.loads((moved / "run_manifest.json").read_text(encoding="utf-8"))
    layout = build_run_layout(
        root_dir=moved.parent.parent,
        model_name=moved.parent.name,
        run_id=moved.name.removeprefix("run_"),
    )
    for reference in payload["artifact_references"].values():
        assert layout.resolve_reference(reference).is_relative_to(moved)
        assert layout.resolve_reference(reference).exists()
    assert Path(config["paths"]["mutants_hdf5"]).is_absolute()
    assert payload["data"]["locators"]["mutants"]["path"] == config["paths"]["mutants_hdf5"]


def test_new_checkpoint_rejects_schema_change(tmp_path: Path) -> None:
    initial = _portable_config(tmp_path, model_name="schema-source")
    source = train_model_b_pipeline(
        build_training_pipeline(initial),
        config_path=initial["__config_path__"],
    )
    changed_schema = json.loads(Path(initial["paths"]["sample_schema"]).read_text(encoding="utf-8"))
    changed_schema["schema_version"] = "incompatible-test-version"
    Path(initial["paths"]["sample_schema"]).write_text(
        json.dumps(changed_schema),
        encoding="utf-8",
    )
    resumed = deepcopy(initial)
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = "schema-rejected"
    with pytest.raises(ValueError, match=r"compatibility\.schema\.schema_version"):
        train_model_b_pipeline(
            build_training_pipeline(resumed),
            config_path=resumed["__config_path__"],
            resume_from=source.last_checkpoint_path,
        )


def test_new_checkpoint_rejects_independently_modified_wt(tmp_path: Path) -> None:
    initial = _portable_config(tmp_path, model_name="wt-source")
    source = train_model_b_pipeline(
        build_training_pipeline(initial),
        config_path=initial["__config_path__"],
    )
    wt_path = Path(initial["paths"]["wt_companion_hdf5"])
    wt_path.write_bytes(wt_path.read_bytes() + b"x")
    resumed = deepcopy(initial)
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = "wt-rejected"
    with pytest.raises(ValueError, match="hdf5_content_fingerprint"):
        train_model_b_pipeline(
            build_training_pipeline(resumed),
            config_path=resumed["__config_path__"],
            resume_from=source.last_checkpoint_path,
        )


def _write_historical_checkpoint(
    path: Path,
    *,
    source_checkpoint: str | Path,
    pipeline,
    mutate_dataset_fingerprint: bool = False,
    mutate_split_fingerprint: bool = False,
) -> None:
    payload = load_checkpoint(source_checkpoint)
    legacy = build_legacy_resume_compatibility_payload(
        new_compatibility=payload["compatibility"],
        dataset=pipeline.dataset,
        split_bundle=pipeline.split_bundle,
    )
    if mutate_dataset_fingerprint:
        legacy["dataset_fingerprint"] = "historical-dataset-mismatch"
    if mutate_split_fingerprint:
        legacy["split_fingerprint"] = "historical-split-mismatch"
    payload["compatibility"] = legacy
    payload.pop("hdf5_content_fingerprint", None)
    save_checkpoint_payload_atomic(payload, path)


def test_real_historical_v1_resume_uses_explicit_weak_policy(tmp_path: Path) -> None:
    initial = _portable_config(tmp_path, model_name="legacy-source")
    source_pipeline = build_training_pipeline(initial)
    source = train_model_b_pipeline(source_pipeline, config_path=initial["__config_path__"])
    historical = tmp_path / "historical-v1.pt"
    _write_historical_checkpoint(
        historical,
        source_checkpoint=source.last_checkpoint_path,
        pipeline=source_pipeline,
    )
    resumed = deepcopy(initial)
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = "legacy-resumed"
    output = train_model_b_pipeline(
        build_training_pipeline(resumed),
        config_path=resumed["__config_path__"],
        resume_from=historical,
    )
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))
    assert manifest["resume_compatibility"] == {
        "content_verification": "legacy_unavailable_historical_controls_only",
        "strong_content_verification": False,
    }


@pytest.mark.parametrize("mismatch", ["dataset", "split"])
def test_real_historical_v1_resume_rejects_historical_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    initial = _portable_config(tmp_path, model_name=f"legacy-{mismatch}-source")
    source_pipeline = build_training_pipeline(initial)
    source = train_model_b_pipeline(source_pipeline, config_path=initial["__config_path__"])
    historical = tmp_path / f"historical-{mismatch}.pt"
    _write_historical_checkpoint(
        historical,
        source_checkpoint=source.last_checkpoint_path,
        pipeline=source_pipeline,
        mutate_dataset_fingerprint=mismatch == "dataset",
        mutate_split_fingerprint=mismatch == "split",
    )
    resumed = deepcopy(initial)
    resumed["training"]["epochs"] = 2
    resumed["outputs"]["model_name"] = f"legacy-{mismatch}-rejected"
    with pytest.raises(ValueError, match=f"{mismatch}_fingerprint"):
        train_model_b_pipeline(
            build_training_pipeline(resumed),
            config_path=resumed["__config_path__"],
            resume_from=historical,
        )


@pytest.mark.parametrize(
    ("signum", "expected_code"),
    [(signal.SIGINT, 130), (signal.SIGTERM, 143)],
)
def test_cli_signal_interruption_keeps_last_complete_epoch(
    tmp_path: Path,
    signum: signal.Signals,
    expected_code: int,
) -> None:
    config = _portable_config(tmp_path, model_name=f"signal-{signum.name}", epochs=1000)
    config_path = tmp_path / f"{signum.name}.yaml"
    serializable = {key: value for key, value in config.items() if not key.startswith("__")}
    config_path.write_text(yaml.safe_dump(serializable, sort_keys=False), encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )
    manifest_path: Path | None = None
    last_path: Path | None = None
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        manifests = list((tmp_path / "runs" / f"signal-{signum.name}").glob("run_*/run_manifest.json"))
        if manifests:
            manifest_path = manifests[0]
            candidate = manifest_path.parent / "checkpoints" / "last.pt"
            if candidate.exists():
                last_path = candidate
                break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    assert manifest_path is not None
    assert last_path is not None, process.communicate(timeout=5)
    process.send_signal(signum)
    stdout, stderr = process.communicate(timeout=20)
    assert process.returncode == expected_code, stderr or stdout

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(last_path)
    assert manifest["status"] == "interrupted"
    assert manifest["interruption"]["signal"] == int(signum)
    assert manifest["training"]["epochs_completed"] == checkpoint["epoch_completed"]
    assert manifest["training"]["global_step"] == checkpoint["global_step"]
    assert not list(manifest_path.parent.rglob(".*.tmp"))


def test_second_signal_only_restores_default_without_reentrant_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn_siamese.utils.interruptions import InterruptionController

    calls: list[tuple[int, object]] = []
    monkeypatch.setattr(signal, "signal", lambda signum, handler: calls.append((signum, handler)))
    controller = InterruptionController()
    controller.handler(signal.SIGTERM, None)
    controller.handler(signal.SIGTERM, None)
    assert controller.request_count == 2
    assert controller.requested
    assert calls == [(signal.SIGTERM, signal.SIG_DFL)]


@pytest.mark.parametrize(
    ("point", "expected_stage"),
    [
        ("gradient_audit", "initializing_gradient_audit"),
        ("resume", "resuming"),
        ("finalizing", "finalizing"),
    ],
)
def test_interruption_preserves_current_lifecycle_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    point: str,
    expected_stage: str,
) -> None:
    import gnn_siamese.training.loop as training_loop
    from gnn_siamese.utils.interruptions import InterruptionController, RunInterrupted

    config = _portable_config(tmp_path, model_name=f"interrupt-{point}")
    controller = InterruptionController()
    resume_from = None
    if point == "resume":
        source_config = _portable_config(tmp_path / "source", model_name="interrupt-resume-source")
        source = train_model_b_pipeline(
            build_training_pipeline(source_config),
            config_path=source_config["__config_path__"],
        )
        resume_from = source.last_checkpoint_path
        config["training"]["epochs"] = 2

    def interrupt(*args, **kwargs):
        controller.requested = True
        controller.signum = signal.SIGTERM
        controller.signal_name = "SIGTERM"
        raise RunInterrupted(signum=signal.SIGTERM, reason="SIGTERM")

    if point == "gradient_audit":
        monkeypatch.setattr(training_loop, "create_gradient_audit", interrupt)
    elif point == "resume":
        monkeypatch.setattr(training_loop, "resume_from_checkpoint", interrupt)
    else:
        real_finalize = training_loop.finalize_gradient_audit

        def request_during_finalize(trackers):
            result = real_finalize(trackers)
            controller.requested = True
            controller.signum = signal.SIGTERM
            controller.signal_name = "SIGTERM"
            return result

        monkeypatch.setattr(training_loop, "finalize_gradient_audit", request_during_finalize)

    with pytest.raises(KeyboardInterrupt):
        train_model_b_pipeline(
            build_training_pipeline(config),
            config_path=config["__config_path__"],
            resume_from=resume_from,
            interruption_controller=controller,
        )
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "interrupted"
    assert manifest["lifecycle"]["stage"] == expected_stage
    assert manifest["interruption"]["signal_name"] == "SIGTERM"


def test_interruption_during_initialization_preserves_stage(tmp_path: Path) -> None:
    import scripts.train as train_script
    from gnn_siamese.utils.interruptions import InterruptionController

    config = _portable_config(tmp_path, model_name="interrupt-initialization")
    controller = InterruptionController(
        requested=True,
        signum=signal.SIGINT,
        signal_name="SIGINT",
        request_count=1,
    )
    with pytest.raises(KeyboardInterrupt):
        train_script._build_with_early_manifest(
            config,
            config_path=config["__config_path__"],
            controller=controller,
        )
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "interrupted"
    assert manifest["lifecycle"]["stage"] == "resolving_paths"


def test_normal_failure_records_last_valid_checkpoint_only_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn_siamese.training.loop as training_loop

    config = _portable_config(tmp_path, model_name="failed-after-last", epochs=2)
    real_epoch = training_loop.run_model_b_epoch
    calls = 0

    def fail_on_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("failure after first last.pt")
        return real_epoch(*args, **kwargs)

    monkeypatch.setattr(training_loop, "run_model_b_epoch", fail_on_second_epoch)
    with pytest.raises(RuntimeError, match="failure after first"):
        train_model_b_pipeline(
            build_training_pipeline(config),
            config_path=config["__config_path__"],
        )
    manifest_path, manifest = _only_manifest(config)
    checkpoint_path = manifest_path.parent / manifest["training"]["last_valid_checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path)
    assert manifest["status"] == "failed"
    assert manifest["training"]["epochs_completed"] == checkpoint["epoch_completed"] == 1
    assert manifest["training"]["global_step"] == checkpoint["global_step"]


def test_failure_before_first_last_has_no_valid_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn_siamese.training.loop as training_loop

    config = _portable_config(tmp_path, model_name="failed-before-last")

    def fail_epoch(*args, **kwargs):
        raise RuntimeError("failure before last.pt")

    monkeypatch.setattr(training_loop, "run_model_b_epoch", fail_epoch)
    with pytest.raises(RuntimeError, match="before last"):
        train_model_b_pipeline(
            build_training_pipeline(config),
            config_path=config["__config_path__"],
        )
    _path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["training"]["last_valid_checkpoint"] is None
    assert manifest["training"]["epochs_completed"] == 0
    assert manifest["training"]["global_step"] == 0


def test_best_checkpoint_failure_before_last_is_not_reported_as_valid_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn_siamese.training.loop as training_loop

    config = _portable_config(tmp_path, model_name="best-fails-before-last")
    real_save = training_loop.save_checkpoint

    def fail_best(path, **kwargs):
        if Path(path).name == "best.pt":
            raise OSError("best publication failure")
        return real_save(path, **kwargs)

    monkeypatch.setattr(training_loop, "save_checkpoint", fail_best)
    with pytest.raises(OSError, match="best publication failure"):
        train_model_b_pipeline(
            build_training_pipeline(config),
            config_path=config["__config_path__"],
        )
    manifest_path, manifest = _only_manifest(config)
    assert manifest["status"] == "failed"
    assert manifest["training"]["last_valid_checkpoint"] is None
    assert not (manifest_path.parent / "checkpoints" / "last.pt").exists()
