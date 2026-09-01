"""Fail-closed configuration and post-run acceptance checks for A9."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gnn_siamese.data.splits import load_leave_position_out_split
from gnn_siamese.training.gradient_audit import expected_a9_active_module_names
from gnn_siamese.utils.fingerprints import fingerprint_split_definition


MODEL_A_ARCHITECTURE = "model_a_nodal_multiscale_pair"
MODEL_B_ARCHITECTURE = "model_b_graph_level_relational"
FROZEN_SPLIT_PATH = "splits/leave_position_out_seed_42.json"
PRODUCTIVE_RUN_SEEDS = (11, 23, 37, 41, 53)
EXPECTED_PARTITIONS = {"train": 342, "validation": 78, "test": 63}


class A9ContractError(RuntimeError):
    """Raised when an A9 config, preflight, or completed run is not acceptable."""


_RUN_SEED_FIELDS = (
    "seed_python", "seed_numpy", "seed_torch", "seed_cuda", "seed_dataloader"
)


def apply_a9_run_seed(config: Mapping[str, Any], run_seed: int) -> dict[str, Any]:
    """Resolve one declared A9 run seed without ever changing the frozen split."""

    seed = int(run_seed)
    if seed not in PRODUCTIVE_RUN_SEEDS:
        raise A9ContractError(f"Run seed is not predeclared for A9: {seed}")
    resolved = deepcopy(dict(config))
    resolved.setdefault("project", {})["seed"] = seed
    reproducibility = resolved.setdefault("reproducibility", {})
    for field in _RUN_SEED_FIELDS:
        reproducibility[field] = seed
    split = resolved.setdefault("split", {})
    if int(split.get("seed", -1)) != 42 or bool(split.get("allow_create", True)):
        raise A9ContractError("A9 run seed resolution requires split.seed=42 and allow_create=false.")
    return resolved


def validate_a9_run_seed(config: Mapping[str, Any], run_seed: int) -> None:
    seed = int(run_seed)
    if seed not in PRODUCTIVE_RUN_SEEDS or int(_get(config, "project.seed")) != seed:
        raise A9ContractError(f"Invalid A9 run seed contract: {seed}")
    for field in _RUN_SEED_FIELDS:
        if int(_get(config, f"reproducibility.{field}")) != seed:
            raise A9ContractError(f"A9 reproducibility.{field} must equal run seed {seed}.")
    if int(_get(config, "split.seed")) != 42:
        raise A9ContractError("A9 split.seed must remain 42.")


_SHARED_PATHS = (
    "training.epochs",
    "training.batch_size",
    "training.optimizer",
    "training.learning_rate",
    "training.weight_decay",
    "training.scheduler",
    "training.early_stopping",
    "training.checkpointing",
    "training.mixed_precision.enabled",
    "loss.main",
    "loss.temperature",
    "loss.false_negative_mask.enabled",
    "loss.false_negative_mask.mode",
    "loss.false_negative_mask.same_position",
    "loss.false_negative_mask.strict",
    "loss.false_negative_mask.min_valid_negatives",
    "loss.false_negative_mask.min_valid_negative_fraction",
    "features.node_groups",
    "features.edge_groups",
    "features.graph_groups",
    "split.type",
    "split.seed",
    "split.persist_path",
    "split.allow_create",
    "a9.run_seeds",
    "a9.expected_hdf5_fingerprints",
    "a9.expected_split_fingerprint",
)


def validate_a9_config_pair(model_a: Mapping[str, Any], model_b: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen productive A/B protocol without loading data."""

    required = {
        "training.epochs": 100,
        "training.batch_size": 4,
        "training.optimizer": "adamw",
        "training.learning_rate": 0.001,
        "training.weight_decay": 0.0001,
        "training.scheduler": "cosine",
        "training.early_stopping.enabled": True,
        "training.early_stopping.monitor": "validation_loss",
        "training.early_stopping.mode": "min",
        "training.early_stopping.patience": 15,
        "training.early_stopping.min_delta": 0.0,
        "training.checkpointing.monitor": "validation_loss",
        "training.checkpointing.mode": "min",
        "training.checkpointing.save_best": True,
        "training.checkpointing.save_last": True,
        "training.mixed_precision.enabled": False,
        "loss.main": "nt_xent",
        "loss.temperature": 0.2,
        "loss.false_negative_mask.enabled": True,
        "loss.false_negative_mask.mode": "same_position",
        "loss.false_negative_mask.same_position": True,
        "loss.false_negative_mask.strict": True,
        "loss.false_negative_mask.min_valid_negatives": 1,
        "loss.false_negative_mask.min_valid_negative_fraction": 0.0,
        "split.type": "leave_position_out",
        "split.seed": 42,
        "split.persist_path": FROZEN_SPLIT_PATH,
        "split.allow_create": False,
        "a9.run_seeds": list(PRODUCTIVE_RUN_SEEDS),
    }
    mismatches: dict[str, Any] = {}
    for label, config in (("model_a", model_a), ("model_b", model_b)):
        for path, expected in required.items():
            actual = _get(config, path)
            if actual != expected:
                mismatches[f"{label}.{path}"] = {"expected": expected, "actual": actual}
        try:
            validate_a9_run_seed(config, int(_get(config, "project.seed")))
        except A9ContractError as exc:
            mismatches[f"{label}.run_seed"] = str(exc)
    if _get(model_a, "model.architecture") != MODEL_A_ARCHITECTURE:
        mismatches["model_a.model.architecture"] = MODEL_A_ARCHITECTURE
    if _get(model_b, "model.architecture") != MODEL_B_ARCHITECTURE:
        mismatches["model_b.model.architecture"] = MODEL_B_ARCHITECTURE
    if _get(model_b, "model.projection_instance.enabled") is not False:
        mismatches["model_b.model.projection_instance.enabled"] = False
    if _get(model_b, "model.mlp_delta.enabled") is not True:
        mismatches["model_b.model.mlp_delta.enabled"] = True
    if _get(model_b, "model.projection_pair.enabled") is not True:
        mismatches["model_b.model.projection_pair.enabled"] = True
    if _get(model_b, "model.projection_pair.input") != "z_delta":
        mismatches["model_b.model.projection_pair.input"] = "z_delta"
    if _get(model_a, "model.active_scales") != ["mutation", "local", "global"]:
        mismatches["model_a.model.active_scales"] = ["mutation", "local", "global"]
    if _get(model_a, "features.domains.enabled") is not False:
        mismatches["model_a.features.domains.enabled"] = False
    if _get(model_a, "model.projection_instance.enabled") is not False:
        mismatches["model_a.model.projection_instance.enabled"] = False
    if _get(model_a, "loss.lambda_wt") != 0.0 or _get(model_a, "loss.lambda_delta") != 0.0:
        mismatches["model_a.loss.auxiliary"] = {"lambda_wt": 0.0, "lambda_delta": 0.0}
    for path in _SHARED_PATHS:
        if _get(model_a, path) != _get(model_b, path):
            mismatches[f"shared.{path}"] = {
                "model_a": _get(model_a, path),
                "model_b": _get(model_b, path),
            }
    output_a = Path(str(_get(model_a, "outputs.root_dir")))
    output_b = Path(str(_get(model_b, "outputs.root_dir")))
    if output_a == output_b or "model_a_a9" not in output_a.parts or "model_b_a9" not in output_b.parts:
        mismatches["outputs.namespaces"] = {"model_a": str(output_a), "model_b": str(output_b)}
    if mismatches:
        raise A9ContractError(f"A9 configuration contract mismatch: {mismatches}")
    return {
        "status": "PASS",
        "architectures": [MODEL_A_ARCHITECTURE, MODEL_B_ARCHITECTURE],
        "shared_fields": list(_SHARED_PATHS),
        "run_seeds": list(PRODUCTIVE_RUN_SEEDS),
        "split_seed": 42,
        "output_roots": {"model_a": str(output_a), "model_b": str(output_b)},
    }


def validate_separate_output_roots(model_a_root: str | Path, model_b_root: str | Path) -> dict[str, str]:
    """Reject physical equality and nesting, including symlink resolution."""

    root_a = Path(model_a_root).expanduser().resolve()
    root_b = Path(model_b_root).expanduser().resolve()
    if root_a == root_b or root_a in root_b.parents or root_b in root_a.parents:
        raise A9ContractError(
            f"A9 output roots must be physically separate non-nested paths: A={root_a}, B={root_b}"
        )
    return {"model_a": str(root_a), "model_b": str(root_b)}


def validate_a9_run_acceptance(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    representations: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Accept only an objectively complete, finite, non-collapsed A9 run."""

    root = Path(run_dir)
    manifest_path = root / "run_manifest.json"
    gradient_path = root / "gradient_audit.json"
    best_path = root / "checkpoints" / "best.pt"
    last_path = root / "checkpoints" / "last.pt"
    for path in (manifest_path, gradient_path, best_path, last_path):
        if not path.is_file():
            raise A9ContractError(f"Required A9 artifact is missing: {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        gradients = json.loads(gradient_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A9ContractError("A9 manifest or gradient audit is unreadable.") from exc

    required_manifest_paths = (
        "status",
        "architecture",
        "configuration.seed",
        "configuration.seed_bundle.split",
        "data.hdf5_content_fingerprint.files",
        "data.hdf5_content_fingerprint.combined.digest",
        "data.split_fingerprint",
        "data.inventory.biological_variants",
        "data.inventory.native_wt_controls",
        "data.inventory.native_wt_control_ids",
        "data.inventory.native_wt_control_policy.used_for_training",
        "data.inventory.native_wt_control_policy.used_for_validation",
        "data.inventory.native_wt_control_policy.used_for_test_loss",
        "training.epochs_planned",
        "training.epochs_completed",
        "training.stopped_early",
        "training.stop_reason",
        "training.optimizer",
        "training.optimizer_class",
        "training.scheduler_details.class",
        "training.scheduler_details.config.T_max",
        "training.scheduler_details.config.step_semantics",
        "training.learning_rate",
        "training.weight_decay",
        "training.batch_size",
        "training.early_stopping.enabled",
        "training.early_stopping.monitor",
        "training.early_stopping.mode",
        "training.early_stopping.patience",
        "training.early_stopping.min_delta",
        "training.early_stopping.bad_epochs",
        "training.early_stopping.best_metric",
        "losses.main",
        "losses.temperature",
        "losses.false_negative_mask",
        "configuration.node_feature_names",
        "configuration.edge_feature_names",
        "configuration.graph_feature_names",
        "gradient_audit_identity.expected_active_modules",
        "artifacts.best_checkpoint",
        "artifacts.last_checkpoint",
    )
    for path in required_manifest_paths:
        _get(manifest, path)
    if manifest["status"] != "completed":
        raise A9ContractError(f"Run status is not completed: {manifest['status']!r}")
    if manifest["architecture"] != _get(config, "model.architecture"):
        raise A9ContractError("Run architecture does not match its frozen A9 config.")
    run_seed = int(_get(manifest, "configuration.seed"))
    validate_a9_run_seed(config, run_seed)
    if int(_get(manifest, "configuration.seed_bundle.split")) != 42:
        raise A9ContractError("Run manifest split seed is not 42.")
    _validate_hdf5_fingerprints(manifest, config)
    if _get(manifest, "data.split_fingerprint") != _get(config, "a9.expected_split_fingerprint"):
        raise A9ContractError("Run split fingerprint does not match A9.")
    if int(_get(manifest, "data.inventory.biological_variants")) != 483:
        raise A9ContractError("Run does not contain exactly 483 biological variants.")
    if int(_get(manifest, "data.inventory.native_wt_controls")) != 1:
        raise A9ContractError("Run does not contain exactly one native WT control.")
    _validate_manifest_training_contract(manifest)
    if _contains_nonfinite(manifest):
        raise A9ContractError("Manifest contains NaN or Inf numeric values.")

    native_wt_ids = {str(value) for value in _get(manifest, "data.inventory.native_wt_control_ids")}
    if len(native_wt_ids) != 1:
        raise A9ContractError("Manifest must identify exactly one native WT control.")
    for field in ("used_for_training", "used_for_validation", "used_for_test_loss"):
        if _get(manifest, f"data.inventory.native_wt_control_policy.{field}") is not False:
            raise A9ContractError(f"Native WT policy incorrectly enables {field}.")
    actual_split_fingerprint = _validate_frozen_split(root / "split.json", native_wt_ids=native_wt_ids)
    if actual_split_fingerprint != _get(config, "a9.expected_split_fingerprint"):
        raise A9ContractError("Stored split.json canonical fingerprint does not match A9.")
    expected_modules = expected_a9_active_module_names(config)
    declared_modules = tuple(sorted(_get(manifest, "gradient_audit_identity.expected_active_modules")))
    if declared_modules != tuple(sorted(expected_modules)):
        raise A9ContractError("Manifest active-module inventory does not match the A9 architecture.")
    _validate_active_modules(gradients, expected_modules=expected_modules)

    required_representations = list(
        _get(config, "a9.acceptance.required_pair_representations", default=[])
    )
    representation_metrics: dict[str, Any] = {}
    if required_representations:
        if representations is None:
            raise A9ContractError("Required A9 pair representations were not supplied for acceptance.")
        for name in required_representations:
            if name not in representations:
                raise A9ContractError(f"Required A9 representation is missing: {name}")
            representation_metrics[name] = _audit_representation(name, representations[name])
    return {
        "status": "accepted",
        "run_seed": run_seed,
        "architecture": manifest["architecture"],
        "representation_metrics": representation_metrics,
        "automatic_rejection_checks": "PASS",
    }


def _validate_frozen_split(path: Path, *, native_wt_ids: set[str]) -> str:
    if not path.is_file():
        raise A9ContractError(f"Run split artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments = payload.get("assignments", [])
    by_partition = {name: [] for name in EXPECTED_PARTITIONS}
    if len(assignments) != 483:
        raise A9ContractError(f"A9 split must contain 483 assignments, got {len(assignments)}.")
    for item in assignments:
        partition = item.get("partition")
        if partition not in by_partition:
            raise A9ContractError(f"Unknown A9 split partition: {partition!r}")
        by_partition[partition].append(item)
    counts = {name: len(values) for name, values in by_partition.items()}
    if counts != EXPECTED_PARTITIONS:
        raise A9ContractError(f"A9 split counts are invalid: {counts}")
    variant_sets = {name: {item["variant_id"] for item in values} for name, values in by_partition.items()}
    position_sets = {name: {int(item["position"]) for item in values} for name, values in by_partition.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if variant_sets[left] & variant_sets[right] or position_sets[left] & position_sets[right]:
            raise A9ContractError(f"A9 split leakage detected between {left} and {right}.")
    if native_wt_ids & set().union(*variant_sets.values()):
        raise A9ContractError("Native WT control leaked into an A9 split.")
    try:
        split = load_leave_position_out_split(path)
        return fingerprint_split_definition(split)
    except Exception as exc:
        raise A9ContractError(f"Stored A9 split is not canonically readable: {exc}") from exc


def _validate_active_modules(
    gradients: Mapping[str, Any], *, expected_modules: tuple[str, ...]
) -> None:
    if not gradients:
        raise A9ContractError("Gradient audit is empty.")
    missing = sorted(set(expected_modules) - set(gradients))
    if missing:
        raise A9ContractError(f"Gradient audit is missing active modules: {missing}")
    required_fields = (
        "parameter_count", "trainable_parameter_count", "optimizer_group", "connected_losses",
        "mean_gradient_norm", "max_gradient_norm", "none_gradient_fraction",
        "zero_gradient_fraction", "has_nan_or_inf", "relative_weight_change", "status",
    )
    for name in expected_modules:
        record = gradients[name]
        if not isinstance(record, Mapping):
            raise A9ContractError(f"Invalid gradient audit record: {name}")
        absent = [field for field in required_fields if field not in record]
        if absent:
            raise A9ContractError(f"Gradient audit record {name} is incomplete: {absent}")
        if record.get("status") != "trained":
            raise A9ContractError(f"Active module is not trained: {name}")
        if bool(record.get("has_nan_or_inf", False)):
            raise A9ContractError(f"Active module has NaN/Inf gradients: {name}")
        numeric_fields = (
            "mean_gradient_norm", "max_gradient_norm", "none_gradient_fraction",
            "zero_gradient_fraction", "relative_weight_change",
        )
        if any(not math.isfinite(float(record[field])) for field in numeric_fields):
            raise A9ContractError(f"Active module has non-finite audit metrics: {name}")
        if float(record["none_gradient_fraction"]) > 0.0:
            raise A9ContractError(f"Active module has parameters without gradients: {name}")
        if float(record["mean_gradient_norm"]) <= 0.0 or float(record["max_gradient_norm"]) <= 0.0:
            raise A9ContractError(f"Active module has no non-zero gradient: {name}")
        if float(record["relative_weight_change"]) <= 0.0:
            raise A9ContractError(f"Active module weights did not change: {name}")


def _validate_hdf5_fingerprints(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    files = _get(manifest, "data.hdf5_content_fingerprint.files")
    if not isinstance(files, list):
        raise A9ContractError("HDF5 fingerprint files must be a list.")
    by_role = {str(record.get("role")): record.get("digest") for record in files if isinstance(record, Mapping)}
    for role in ("mutants", "wt_companion"):
        expected = _get(config, f"a9.expected_hdf5_fingerprints.{role}")
        if by_role.get(role) != expected:
            raise A9ContractError(f"Run HDF5 {role} fingerprint does not match A9.")
    if _get(manifest, "data.hdf5_content_fingerprint.combined.digest") != _get(
        config, "a9.expected_hdf5_fingerprints.combined"
    ):
        raise A9ContractError("Run HDF5 combined fingerprint does not match A9.")


def _validate_manifest_training_contract(manifest: Mapping[str, Any]) -> None:
    expected = {
        "training.epochs_planned": 100,
        "training.optimizer": "adamw",
        "training.optimizer_class": "AdamW",
        "training.scheduler_details.class": "CosineAnnealingLR",
        "training.scheduler_details.config.T_max": 100,
        "training.scheduler_details.config.step_semantics": "once_per_epoch",
        "training.learning_rate": 0.001,
        "training.weight_decay": 0.0001,
        "training.batch_size": 4,
        "training.early_stopping.enabled": True,
        "training.early_stopping.monitor": "validation_loss",
        "training.early_stopping.mode": "min",
        "training.early_stopping.patience": 15,
        "training.early_stopping.min_delta": 0.0,
        "losses.main": "nt_xent",
        "losses.temperature": 0.2,
    }
    for path, value in expected.items():
        if _get(manifest, path) != value:
            raise A9ContractError(f"A9 manifest training contract mismatch at {path}.")
    mask = _get(manifest, "losses.false_negative_mask")
    required_mask = {
        "enabled": True, "mode": "same_position", "same_position": True, "strict": True,
        "min_valid_negatives": 1, "min_valid_negative_fraction": 0.0,
    }
    if any(mask.get(key) != value for key, value in required_mask.items()):
        raise A9ContractError("A9 manifest false-negative policy mismatch.")
    if int(_get(manifest, "training.epochs_completed")) < 1:
        raise A9ContractError("A9 manifest has no completed epochs.")


def _audit_representation(name: str, value: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[0] != 483 or matrix.shape[1] < 1:
        raise A9ContractError(f"{name} must be a 2D matrix for exactly 483 biological variants.")
    if not np.isfinite(matrix).all():
        raise A9ContractError(f"{name} contains NaN or Inf.")
    exact_unique_rows = int(np.unique(matrix, axis=0).shape[0])
    if exact_unique_rows == 1:
        raise A9ContractError(f"{name} has exact total collapse (all rows are identical).")
    centered = matrix.astype(np.float64, copy=False) - matrix.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    total = float(variance.sum())
    pc1_fraction = 0.0 if total == 0.0 else float(variance[0] / total)
    probabilities = variance / total if total > 0.0 else variance
    positive = probabilities[probabilities > 0.0]
    effective_rank = 0.0 if positive.size == 0 else float(np.exp(-(positive * np.log(positive)).sum()))
    return {
        "shape": list(matrix.shape),
        "finite": True,
        "exact_unique_rows": exact_unique_rows,
        "exact_total_collapse": False,
        "pc1_variance_fraction": pc1_fraction,
        "effective_rank": effective_rank,
    }


def _contains_nonfinite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite(item) for item in value)
    return isinstance(value, float) and not math.isfinite(value)


_MISSING = object()


def _get(mapping: Mapping[str, Any], dotted_path: str, *, default: Any = _MISSING) -> Any:
    value: Any = mapping
    try:
        for part in dotted_path.split("."):
            value = value[part]
    except (KeyError, TypeError):
        if default is not _MISSING:
            return default
        raise A9ContractError(f"Required A9 field is missing: {dotted_path}") from None
    return value
