"""Run the minimal end-to-end Model B baseline from YAML configuration."""

from __future__ import annotations

import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.config import apply_runtime_overrides, load_config
from gnn_siamese.training import (
    bootstrap_operational_run,
    complete_operational_run,
    load_checkpoint,
    record_run_failure,
    train_model_b_pipeline,
)
from gnn_siamese.utils.interruptions import InterruptionController


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--device", default=None, help="Override training.device.")
    parser.add_argument("--smoke-test", action="store_true", help="Run the reduced smoke-test setup.")
    parser.add_argument("--resume-from", default=None, help="Resume from an existing checkpoint.")
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _file_signature(path: str | Path) -> dict[str, int]:
    stat = Path(path).stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}.")
    return payload


def _count_train_batches(pipeline: object) -> int:
    loader = pipeline.dataloaders.train_loader
    return len(loader) if hasattr(loader, "__len__") else 0


def _all_numeric_values_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_all_numeric_values_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numeric_values_finite(item) for item in value)
    return True


def _verify_run_artifacts(
    *,
    pipeline: object,
    output: object,
    expected_total_epochs: int,
    expected_invocation_epochs: int,
    expected_resume_from: str | None,
    expected_manifest_status: str = "completed",
) -> dict:
    _require(str(output.device) == "cpu", f"Smoke test requires CPU; got {output.device!r}.")
    _require(len(pipeline.dataloaders.train_dataset) > 0, "Smoke split produced empty train dataset.")
    _require(len(pipeline.dataloaders.validation_dataset) > 0, "Smoke split produced empty validation dataset.")
    _require(_count_train_batches(pipeline) >= 1, "Smoke train loader produced zero batches.")
    _require(math.isfinite(output.final_train_loss), "Smoke train loss is not finite.")
    _require(math.isfinite(output.final_validation_loss), "Smoke validation loss is not finite.")

    split_train_positions = {int(pipeline.dataset.pairs[index].position) for index in pipeline.split_bundle.train_indices}
    split_validation_positions = {int(pipeline.dataset.pairs[index].position) for index in pipeline.split_bundle.validation_indices}
    _require(split_train_positions, "Smoke split produced empty train positions.")
    _require(split_validation_positions, "Smoke split produced empty validation positions.")
    _require(
        split_train_positions.isdisjoint(split_validation_positions),
        "Smoke split leaked positions between train and validation.",
    )

    required_paths = {
        "run_dir": output.run_dir,
        "best_checkpoint": output.best_checkpoint_path,
        "last_checkpoint": output.last_checkpoint_path,
        "manifest": output.manifest_path,
        "gradient_audit": output.gradient_audit_path,
        "metrics": output.metrics_path,
    }
    for label, raw_path in required_paths.items():
        _require(raw_path is not None, f"Smoke run did not populate {label}.")
        path = Path(str(raw_path))
        _require(path.exists(), f"Required smoke artifact is missing: {path}")
        _require(path.stat().st_size > 0, f"Required smoke artifact is empty: {path}")

    split_path = Path(output.run_dir) / "split.json"
    resolved_config_path = Path(output.run_dir) / "config_resolved.yaml"
    _require(split_path.exists() and split_path.stat().st_size > 0, f"Missing split artifact: {split_path}")
    _require(
        resolved_config_path.exists() and resolved_config_path.stat().st_size > 0,
        f"Missing resolved config artifact: {resolved_config_path}",
    )

    manifest = _load_json(output.manifest_path)
    gradient_audit = _load_json(output.gradient_audit_path)
    last_checkpoint = load_checkpoint(output.last_checkpoint_path)
    best_checkpoint = load_checkpoint(output.best_checkpoint_path)
    metrics_rows = [json.loads(line) for line in Path(output.metrics_path).read_text(encoding="utf-8").splitlines()]

    _require(
        manifest.get("status") == expected_manifest_status,
        f"Manifest status must be {expected_manifest_status!r}, got {manifest.get('status')!r}.",
    )
    _require(manifest.get("data", {}).get("dataset_fingerprint"), "Manifest is missing dataset_fingerprint.")
    _require(manifest.get("data", {}).get("split_fingerprint"), "Manifest is missing split_fingerprint.")
    content_fingerprint = manifest.get("data", {}).get("hdf5_content_fingerprint", {})
    _require(content_fingerprint.get("algorithm") == "sha256", "Manifest is missing the SHA-256 HDF5 fingerprint.")
    _require(content_fingerprint.get("version") == 1, "Manifest HDF5 fingerprint version is inconsistent.")
    _require(content_fingerprint.get("scope") == "raw_file_bytes", "Manifest HDF5 fingerprint scope is inconsistent.")
    _require(manifest.get("training", {}).get("epochs_completed") == expected_total_epochs, "Manifest epochs_completed is inconsistent.")
    _require(
        manifest.get("training", {}).get("epochs_run_this_invocation") == expected_invocation_epochs,
        "Manifest epochs_run_this_invocation is inconsistent.",
    )
    _require(
        manifest.get("training", {}).get("resume_from") == expected_resume_from,
        "Manifest resume_from is inconsistent.",
    )
    _require(Path(manifest["artifacts"]["best_checkpoint"]).exists(), "Manifest best checkpoint path does not exist.")
    _require(Path(manifest["artifacts"]["last_checkpoint"]).exists(), "Manifest last checkpoint path does not exist.")
    _require(Path(manifest["artifacts"]["metrics"]).exists(), "Manifest metrics path does not exist.")
    _require(Path(manifest["artifacts"]["gradient_audit"]).exists(), "Manifest gradient audit path does not exist.")
    _require(Path(manifest["artifacts"]["split"]).exists(), "Manifest split path does not exist.")

    run_dir = Path(output.run_dir).resolve()
    artifact_references = manifest.get("artifact_references", {})
    _require(bool(artifact_references), "Manifest is missing portable artifact references.")
    for label, reference in artifact_references.items():
        reference_path = Path(reference)
        _require(not reference_path.is_absolute(), f"Portable artifact reference {label!r} is absolute.")
        resolved = (run_dir / reference_path).resolve()
        _require(resolved.is_relative_to(run_dir), f"Portable artifact reference {label!r} escapes run_dir.")
        _require(resolved.exists(), f"Portable artifact reference {label!r} does not exist.")
    _require(not list(run_dir.rglob(".*.tmp")), "Smoke run left atomic temporary files behind.")
    _require(bool(metrics_rows), "Smoke metrics file contains no epoch records.")
    _require(_all_numeric_values_finite(metrics_rows), "Smoke metrics contain NaN or Inf.")

    split_manifest_fingerprint = manifest["data"]["split_fingerprint"]
    dataset_manifest_fingerprint = manifest["data"]["dataset_fingerprint"]
    _require(last_checkpoint["split_fingerprint"] == split_manifest_fingerprint, "last.pt split_fingerprint mismatch.")
    _require(best_checkpoint["split_fingerprint"] == split_manifest_fingerprint, "best.pt split_fingerprint mismatch.")
    _require(last_checkpoint["dataset_fingerprint"] == dataset_manifest_fingerprint, "last.pt dataset_fingerprint mismatch.")
    _require(best_checkpoint["dataset_fingerprint"] == dataset_manifest_fingerprint, "best.pt dataset_fingerprint mismatch.")
    _require(last_checkpoint["epoch_completed"] == expected_total_epochs, "last.pt epoch_completed is inconsistent.")
    _require(last_checkpoint["global_step"] == manifest["training"]["global_step"], "last.pt global_step mismatch.")
    _require(best_checkpoint["epoch_completed"] >= 1, "best.pt epoch_completed is invalid.")
    for label, checkpoint in (("last.pt", last_checkpoint), ("best.pt", best_checkpoint)):
        _require(checkpoint.get("format_version") == 1, f"{label} format_version is inconsistent.")
        compatibility = checkpoint.get("compatibility", {})
        _require(
            compatibility.get("compatibility_metadata", {}).get("version") == 2,
            f"{label} compatibility metadata version is inconsistent.",
        )
        _require(compatibility.get("schema", {}).get("schema_name"), f"{label} is missing schema_name.")
        _require(compatibility.get("schema", {}).get("schema_version"), f"{label} is missing schema_version.")
        _require(
            checkpoint.get("hdf5_content_fingerprint") == content_fingerprint.get("combined"),
            f"{label} HDF5 content fingerprint mismatch.",
        )

    encoder_audit = gradient_audit.get("encoder", {})
    projection_audit = gradient_audit.get("projection_instance", {})
    _require(encoder_audit.get("optimizer_group") is not None, "Encoder is not registered in the optimizer.")
    _require(projection_audit.get("optimizer_group") is not None, "projection_instance is not registered in the optimizer.")
    _require(encoder_audit.get("mean_gradient_norm", 0.0) > 0.0, "Encoder did not receive non-zero gradients.")
    _require(projection_audit.get("mean_gradient_norm", 0.0) > 0.0, "projection_instance did not receive non-zero gradients.")
    _require(not encoder_audit.get("has_nan_or_inf", False), "Encoder gradients contain NaN/Inf.")
    _require(not projection_audit.get("has_nan_or_inf", False), "projection_instance gradients contain NaN/Inf.")
    _require(encoder_audit.get("relative_weight_change", 0.0) > 0.0, "Encoder weights did not change.")
    _require(projection_audit.get("relative_weight_change", 0.0) > 0.0, "projection_instance weights did not change.")

    mlp_delta_status = gradient_audit.get("mlp_delta", {}).get("status")
    lambda_delta = float(pipeline.config.get("loss", {}).get("lambda_delta", 0.0))
    if lambda_delta <= 0.0:
        _require(mlp_delta_status in {"inactive", "not_applicable"}, f"mlp_delta should not be trained when lambda_delta=0, got {mlp_delta_status!r}.")
        _require(manifest.get("z_delta_learned") is False, "z_delta_learned must be false when lambda_delta=0.")

    return {
        "manifest": manifest,
        "gradient_audit": gradient_audit,
        "last_checkpoint": last_checkpoint,
        "best_checkpoint": best_checkpoint,
        "train_positions": sorted(split_train_positions),
        "validation_positions": sorted(split_validation_positions),
    }


def _build_with_early_manifest(
    config: dict,
    *,
    config_path: str,
    controller: InterruptionController,
) -> tuple[object, object]:
    context = bootstrap_operational_run(config, config_path=config_path)

    def stage_callback(stage: str) -> None:
        context.manifest_writer.set_stage(stage)
        controller.raise_if_requested()

    try:
        if "stage_callback" in inspect.signature(build_training_pipeline).parameters:
            pipeline = build_training_pipeline(config, stage_callback=stage_callback)
        else:
            pipeline = build_training_pipeline(config)
    except BaseException as exc:
        record_run_failure(
            context,
            exc,
            interrupted=isinstance(exc, KeyboardInterrupt),
            interruption=controller.metadata() if isinstance(exc, KeyboardInterrupt) else None,
        )
        raise
    return pipeline, context


def _run_smoke_end_to_end(
    config: dict,
    *,
    config_path: str,
    controller: InterruptionController,
) -> int:
    smoke_cfg = dict(config.get("training", {}).get("smoke_test", {}))
    initial_epochs = int(config.get("training", {}).get("epochs", 1))
    resume_epochs = int(smoke_cfg.get("resume_epochs", 1))
    _require(initial_epochs > 0, "Smoke test requires training.smoke_test.epochs > 0.")
    _require(resume_epochs > 0, "Smoke test requires training.smoke_test.resume_epochs > 0.")

    input_paths = {
        "mutant": config.get("paths", {}).get("mutants_hdf5"),
        "wt": config.get("paths", {}).get("wt_companion_hdf5"),
    }
    before_signatures = {
        name: None if raw_path in (None, "") else _file_signature(raw_path)
        for name, raw_path in input_paths.items()
    }

    pipeline, context = _build_with_early_manifest(
        config,
        config_path=config_path,
        controller=controller,
    )
    shared_smoke_data = pipeline.smoke_data
    try:
        try:
            output = train_model_b_pipeline(
                pipeline,
                config_path=config_path,
                resume_from=None,
                run_context=context,
                interruption_controller=controller,
                defer_completion=True,
            )
            initial_checks = _verify_run_artifacts(
                pipeline=pipeline,
                output=output,
                expected_total_epochs=initial_epochs,
                expected_invocation_epochs=initial_epochs,
                expected_resume_from=None,
                expected_manifest_status="running",
            )
            controller.raise_if_requested()
            initial_checks["manifest"] = complete_operational_run(context)
        except BaseException as exc:
            record_run_failure(
                context,
                exc,
                interrupted=isinstance(exc, KeyboardInterrupt),
                interruption=controller.metadata() if isinstance(exc, KeyboardInterrupt) else None,
            )
            raise
        resume_config = deepcopy(config)
        if shared_smoke_data is not None:
            resume_config.setdefault("paths", {})
            resume_config["paths"]["mutants_hdf5"] = shared_smoke_data.mutants_hdf5
            resume_config["paths"]["wt_companion_hdf5"] = shared_smoke_data.wt_companion_hdf5
            resume_config["paths"]["sample_schema"] = shared_smoke_data.schema_json
            resume_config.setdefault("split", {})
            resume_config["split"]["persist_path"] = shared_smoke_data.split_json
        resume_config["training"]["epochs"] = initial_epochs + resume_epochs
        resumed_pipeline, resumed_context = _build_with_early_manifest(
            resume_config,
            config_path=config_path,
            controller=controller,
        )
        try:
            try:
                resumed_output = train_model_b_pipeline(
                    resumed_pipeline,
                    config_path=config_path,
                    resume_from=output.last_checkpoint_path,
                    run_context=resumed_context,
                    interruption_controller=controller,
                    defer_completion=True,
                )
                resumed_checks = _verify_run_artifacts(
                    pipeline=resumed_pipeline,
                    output=resumed_output,
                    expected_total_epochs=initial_epochs + resume_epochs,
                    expected_invocation_epochs=resume_epochs,
                    expected_resume_from=output.last_checkpoint_path,
                    expected_manifest_status="running",
                )

                _require(output.run_dir != resumed_output.run_dir, "Smoke resume overwrote the original run directory.")
                _require(
                    resumed_checks["last_checkpoint"]["global_step"]
                    > initial_checks["last_checkpoint"]["global_step"],
                    "Smoke resume did not advance global_step.",
                )
                _require(
                    resumed_checks["last_checkpoint"]["epoch_completed"]
                    == initial_checks["last_checkpoint"]["epoch_completed"] + resume_epochs,
                    "Smoke resume did not advance epoch_completed.",
                )

                for name, raw_path in input_paths.items():
                    if raw_path in (None, ""):
                        continue
                    after_signature = _file_signature(raw_path)
                    _require(
                        before_signatures[name] == after_signature,
                        f"Input HDF5 changed during smoke test: {raw_path}",
                    )
                controller.raise_if_requested()
                resumed_checks["manifest"] = complete_operational_run(resumed_context)
            except BaseException as exc:
                record_run_failure(
                    resumed_context,
                    exc,
                    interrupted=isinstance(exc, KeyboardInterrupt),
                    interruption=controller.metadata() if isinstance(exc, KeyboardInterrupt) else None,
                )
                raise
        finally:
            if resumed_pipeline.smoke_data is not None and resumed_pipeline.smoke_data is not shared_smoke_data:
                resumed_pipeline.smoke_data.cleanup()
    finally:
        if shared_smoke_data is not None:
            shared_smoke_data.cleanup()

    smoke_source = "configured_paths"
    if pipeline.smoke_data is not None:
        smoke_source = pipeline.smoke_data.source
    elif pipeline.smoke_selection is not None:
        smoke_source = str(pipeline.smoke_selection.get("source", "configured_hdf5_subset"))

    print(f"smoke_dataset_source={smoke_source}")
    print(f"train_examples={len(pipeline.dataloaders.train_dataset)}")
    print(f"validation_examples={len(pipeline.dataloaders.validation_dataset)}")
    print(f"train_batches={_count_train_batches(pipeline)}")
    print(f"validation_batches={len(pipeline.dataloaders.validation_loader)}")
    print(f"node_dim={pipeline.dataset.node_input_dim}")
    print(f"epochs_completed={output.epochs_completed}")
    print(f"resume_epochs_completed={resumed_output.epochs_completed}")
    print(f"global_step={initial_checks['last_checkpoint']['global_step']}")
    print(f"resume_global_step={resumed_checks['last_checkpoint']['global_step']}")
    print(f"train_loss={output.final_train_loss:.6f}")
    print(f"validation_loss={output.final_validation_loss:.6f}")
    print(f"resume_train_loss={resumed_output.final_train_loss:.6f}")
    print(f"resume_validation_loss={resumed_output.final_validation_loss:.6f}")
    print(f"device={output.device}")
    print(f"run_dir={output.run_dir}")
    print(f"resume_run_dir={resumed_output.run_dir}")
    print(f"best_checkpoint={output.best_checkpoint_path}")
    print(f"last_checkpoint={output.last_checkpoint_path}")
    print(f"resume_last_checkpoint={resumed_output.last_checkpoint_path}")
    print(f"manifest_path={output.manifest_path}")
    print(f"resume_manifest_path={resumed_output.manifest_path}")
    print(f"split_fingerprint={initial_checks['manifest']['data']['split_fingerprint']}")
    print(f"dataset_fingerprint={initial_checks['manifest']['data']['dataset_fingerprint']}")
    print(f"train_positions={','.join(str(value) for value in initial_checks['train_positions'])}")
    print(f"validation_positions={','.join(str(value) for value in initial_checks['validation_positions'])}")
    print(f"encoder_status={initial_checks['gradient_audit']['encoder']['status']}")
    print(f"projection_instance_status={initial_checks['gradient_audit']['projection_instance']['status']}")
    print(f"mlp_delta_status={initial_checks['gradient_audit']['mlp_delta']['status']}")
    print(f"z_delta_learned={str(initial_checks['manifest']['z_delta_learned']).lower()}")
    print(f"manifest_status={initial_checks['manifest']['status']}")
    print(f"resume_manifest_status={resumed_checks['manifest']['status']}")
    if pipeline.smoke_selection is not None:
        print(f"selected_pair_count={pipeline.smoke_selection['selected_pair_count']}")
    for metric_name, metric_value in sorted(output.final_validation_metrics.items()):
        if isinstance(metric_value, bool):
            print(f"{metric_name}={str(metric_value).lower()}")
        elif isinstance(metric_value, str):
            print(f"{metric_name}={metric_value}")
        elif isinstance(metric_value, (int, float)):
            print(f"{metric_name}={metric_value:.6f}")
    return 0


def main() -> int:
    args = _parse_args()
    controller = InterruptionController()
    previous_handlers = controller.install()
    context = None
    try:
        # Invalid configuration before an output root can be resolved remains a CLI-only error.
        config = load_config(args.config)
        config["__config_path__"] = str(Path(args.config).resolve())
        config = apply_runtime_overrides(config, device=args.device, smoke_test=args.smoke_test)
        if args.smoke_test:
            return _run_smoke_end_to_end(
                config,
                config_path=args.config,
                controller=controller,
            )

        pipeline, context = _build_with_early_manifest(
            config,
            config_path=args.config,
            controller=controller,
        )
        try:
            output = train_model_b_pipeline(
                pipeline,
                config_path=args.config,
                resume_from=args.resume_from,
                run_context=context,
                interruption_controller=controller,
            )
        finally:
            if pipeline.smoke_data is not None:
                pipeline.smoke_data.cleanup()

        print(f"split_path={pipeline.split_bundle.split_path}")
        print(f"split_created={str(pipeline.split_bundle.created).lower()}")
        print(f"device={output.device}")
        print(f"train_loss={output.final_train_loss:.6f}")
        print(f"validation_loss={output.final_validation_loss:.6f}")
        print(f"epochs_completed={output.epochs_completed}")
        print(f"run_dir={output.run_dir}")
        print(f"best_checkpoint={output.best_checkpoint_path}")
        print(f"last_checkpoint={output.last_checkpoint_path}")
        print(f"manifest_path={output.manifest_path}")
        return 0
    except KeyboardInterrupt as exc:
        if context is not None:
            record_run_failure(
                context,
                exc,
                interrupted=True,
                interruption=controller.metadata(),
            )
        return controller.metadata()["exit_code"]
    except Exception as exc:
        if context is not None:
            record_run_failure(context, exc)
        print(f"error={type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        controller.restore(previous_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
