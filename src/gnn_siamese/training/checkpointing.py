"""Checkpoint save/load helpers for the operational training pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import torch

from gnn_siamese.data import fingerprint_split_records
from gnn_siamese.utils.atomic_io import atomic_publish
from gnn_siamese.utils.fingerprints import (
    fingerprint_pairing_inventory,
    fingerprint_split_definition,
)

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    monitor: str
    mode: str

    def is_improved(self, candidate: float, best_so_far: float | None) -> bool:
        if best_so_far is None:
            return True
        if self.mode == "min":
            return candidate < best_so_far
        if self.mode == "max":
            return candidate > best_so_far
        raise ValueError(f"Unsupported checkpoint selection mode {self.mode!r}.")


@dataclass(frozen=True)
class ResumeState:
    epoch_completed: int
    next_epoch: int
    global_step: int
    best_metric: float | None
    checkpoint_path: str
    checkpoint_payload: dict[str, Any]
    content_verification: str


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": None,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": None,
    }
    if np is not None:
        state["numpy_random_state"] = np.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(payload: Mapping[str, Any], *, device: torch.device | str | None = None) -> None:
    random.setstate(tuple(payload["python_random_state"]))
    numpy_state = payload.get("numpy_random_state")
    if numpy_state is not None and np is not None:
        np.random.set_state(_normalize_numpy_state(numpy_state))
    torch.set_rng_state(_to_byte_tensor(payload["torch_cpu_rng_state"]))
    cuda_state = payload.get("torch_cuda_rng_state")
    requested_device = None if device is None else torch.device(device)
    if cuda_state is not None and torch.cuda.is_available() and (
        requested_device is None or requested_device.type == "cuda"
    ):
        torch.cuda.set_rng_state_all([_to_byte_tensor(item) for item in cuda_state])


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> None:
    """Move all tensors nested in optimizer state to the training device."""

    target = torch.device(device)

    def move(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(target)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        if isinstance(value, tuple):
            return tuple(move(item) for item in value)
        return value

    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = move(state)


def build_dataset_fingerprint(dataset: Any) -> str:
    return fingerprint_split_records(dataset.pairs)


def build_resume_compatibility_payload(
    *,
    config: Mapping[str, Any],
    dataset: Any,
    split_bundle: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    hdf5_content_fingerprint: Mapping[str, Any] | None = None,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_cfg = dict(config.get("model", {}))
    loss_cfg = dict(config.get("loss", {}))
    training_cfg = dict(config.get("training", {}))
    projection_instance_cfg = dict(model_cfg.get("projection_instance", {}))
    projection_pair_cfg = dict(model_cfg.get("projection_pair", {}))
    mlp_delta_cfg = dict(model_cfg.get("mlp_delta", {}))
    false_negative_mask_cfg = dict(loss_cfg.get("false_negative_mask", {}))
    relative_wt_cfg = dict(loss_cfg.get("relative_wt", {}))
    delta_cfg = dict(loss_cfg.get("delta", {}))
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    mask_mode = "none"
    if bool(false_negative_mask_cfg.get("enabled", False)):
        mask_mode = str(false_negative_mask_cfg.get("mode", "none"))
    payload = {
        "compatibility_metadata": {"version": 2},
        "schema": {
            "schema_name": None if schema is None else schema.get("schema_name"),
            "schema_version": None if schema is None else schema.get("schema_version"),
        },
        "architecture": {
            "name": str(model_cfg.get("architecture", "model_b")),
            "dimensions": {
                "graph_dim": int(model_cfg.get("graph_dim", 0)),
                "hidden_dim": int(model_cfg.get("hidden_dim", 0)),
                "num_layers": int(model_cfg.get("num_layers", 0)),
                "dropout": float(model_cfg.get("dropout", 0.0)),
            },
            "pooling": _json_safe(dict(model_cfg.get("pooling", {}))),
            "mlp_delta": {
                "enabled": bool(mlp_delta_cfg.get("enabled", False)),
                "hidden_dim": int(mlp_delta_cfg.get("hidden_dim", 0)),
                "output_dim": int(mlp_delta_cfg.get("output_dim", 0)),
                "num_layers": int(mlp_delta_cfg.get("num_layers", 0)),
                "dropout": float(mlp_delta_cfg.get("dropout", 0.0)),
            },
            "projection_instance": {
                "enabled": bool(projection_instance_cfg.get("enabled", False)),
                "hidden_dim": int(projection_instance_cfg.get("hidden_dim", 0)),
                "output_dim": int(projection_instance_cfg.get("output_dim", 0)),
                "num_layers": int(projection_instance_cfg.get("num_layers", 0)),
                "normalize_output": bool(projection_instance_cfg.get("normalize_output", False)),
            },
            "projection_pair": {
                "enabled": bool(projection_pair_cfg.get("enabled", False)),
                "input": str(projection_pair_cfg.get("input", "r_delta")),
                "hidden_dim": int(projection_pair_cfg.get("hidden_dim", 0)),
                "output_dim": int(projection_pair_cfg.get("output_dim", 0)),
                "normalize_output": bool(projection_pair_cfg.get("normalize_output", False)),
            },
        },
        "features": {
            "node_feature_names": list(dataset.node_feature_names),
            "edge_feature_names": list(dataset.edge_feature_names),
            "graph_feature_names": list(getattr(dataset, "graph_feature_names", [])),
        },
        "losses": {
            "main": str(loss_cfg.get("main", "nt_xent")),
            "temperature": float(loss_cfg.get("temperature", 0.2)),
            "false_negative_mask": {
                "enabled": bool(false_negative_mask_cfg.get("enabled", False)),
                "mode": mask_mode,
                "same_position": bool(false_negative_mask_cfg.get("same_position", False)),
                "strict": bool(false_negative_mask_cfg.get("strict", False)),
                "min_valid_negatives": float(false_negative_mask_cfg.get("min_valid_negatives", 8.0)),
                "min_valid_negative_fraction": float(false_negative_mask_cfg.get("min_valid_negative_fraction", 0.25)),
                "structural_soft": _json_safe(dict(false_negative_mask_cfg.get("structural_soft", {}))),
            },
            "lambda_wt": float(loss_cfg.get("lambda_wt", 0.0)),
            "relative_wt": _json_safe(relative_wt_cfg),
            "lambda_delta": float(loss_cfg.get("lambda_delta", 0.0)),
            "delta": _json_safe(delta_cfg),
        },
        "augmentations": _json_safe(dict(config.get("augmentation", {}))),
        "gradient_clipping": training_cfg.get("gradient_clip_norm"),
        "optimizer": {
            "class": optimizer.__class__.__name__,
            "config": _optimizer_config_from_instance(optimizer),
        },
        "scheduler": {
            "class": None if scheduler is None else scheduler.__class__.__name__,
            "config": _scheduler_config_from_instance(scheduler, scheduler_name=scheduler_name),
        },
        "dataset_fingerprint": (
            fingerprint_pairing_inventory(dataset.pairs)
            if hasattr(dataset, "mutant_h5_path")
            else build_dataset_fingerprint(dataset)
        ),
        "split_fingerprint": fingerprint_split_definition(split_bundle.split),
        "split_type": str(split_bundle.split.split_type),
    }
    if str(model_cfg.get("architecture", "model_b")) == "model_a_nodal_multiscale_pair":
        payload["architecture"]["model_a"] = {
            "active_scales": _json_safe(list(model_cfg.get("active_scales", []))),
            "encoder_a": _json_safe(dict(model_cfg.get("encoder_a", {}))),
            "node_delta": _json_safe(dict(model_cfg.get("node_delta", {}))),
            "pair_fusion": _json_safe(dict(model_cfg.get("pair_fusion", {}))),
            "projection_pair_a": _json_safe(dict(model_cfg.get("projection_pair_a", {}))),
        }
        payload["augmentations"] = _json_safe(dict(config.get("augmentation_pair_a", {})))
    if hdf5_content_fingerprint is not None:
        payload["hdf5_content_fingerprint"] = _json_safe(dict(hdf5_content_fingerprint))
    return payload


def build_legacy_resume_compatibility_payload(
    *,
    new_compatibility: Mapping[str, Any],
    dataset: Any,
    split_bundle: Any,
) -> dict[str, Any]:
    """Reconstruct the exact compatibility semantics used by historical v1 checkpoints."""

    payload = dict(_json_safe(new_compatibility))
    payload.pop("compatibility_metadata", None)
    payload.pop("schema", None)
    payload.pop("hdf5_content_fingerprint", None)
    payload["dataset_fingerprint"] = build_dataset_fingerprint(dataset)
    payload["split_fingerprint"] = str(split_bundle.split.dataset_fingerprint)
    return payload


def validate_resume_compatibility(
    checkpoint_payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    legacy_expected: Mapping[str, Any] | None = None,
) -> str:
    checkpoint_compat = checkpoint_payload.get("compatibility")
    if not isinstance(checkpoint_compat, Mapping):
        raise ValueError("Checkpoint is missing compatibility metadata required for resume.")
    checkpoint_normalized = _json_safe(checkpoint_compat)
    metadata = checkpoint_normalized.get("compatibility_metadata")
    is_new = isinstance(metadata, Mapping) and int(metadata.get("version", 0)) >= 2
    if not is_new:
        if legacy_expected is None:
            # Backwards-compatible utility behavior for callers that already
            # provide an expected payload expressed in historical semantics.
            expected_normalized = dict(_json_safe(expected))
            expected_normalized.pop("compatibility_metadata", None)
            expected_normalized.pop("schema", None)
            expected_normalized.pop("hdf5_content_fingerprint", None)
        else:
            expected_normalized = _json_safe(legacy_expected)
        if checkpoint_normalized != expected_normalized:
            mismatch = _find_first_mismatch(checkpoint_normalized, expected_normalized, path="compatibility")
            raise ValueError(f"Checkpoint resume incompatibility detected for {mismatch}.")
        return "legacy_unavailable_historical_controls_only"

    expected_normalized = _json_safe(expected)
    if checkpoint_normalized != expected_normalized:
        mismatch = _find_first_mismatch(checkpoint_normalized, expected_normalized, path="compatibility")
        raise ValueError(f"Checkpoint resume incompatibility detected for {mismatch}.")
    return "sha256_raw_file_bytes_verified"


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    epoch_completed: int,
    global_step: int,
    best_metric: float | None,
    train_metrics: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    seed: int | None,
    split_id: str,
    split_fingerprint: str,
    dataset_fingerprint: str,
    dataset_id: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    run_id: str,
    hdf5_content_fingerprint: Mapping[str, Any] | None = None,
    augmenter_state: Mapping[str, Any] | None = None,
    data_loader_state: Mapping[str, Any] | None = None,
) -> None:
    payload = {
        "format_version": 1,
        "run_id": run_id,
        "architecture": str(
            dict(resolved_config).get("model", {}).get("architecture", "model_b")
        ),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "epoch_completed": int(epoch_completed),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "train_metrics": dict(train_metrics),
        "validation_metrics": dict(validation_metrics),
        "resolved_config": _json_safe(dict(resolved_config)),
        "seed": seed,
        "split_id": split_id,
        "split_fingerprint": split_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "hdf5_content_fingerprint": None if hdf5_content_fingerprint is None else _json_safe(hdf5_content_fingerprint),
        "dataset_id": dict(dataset_id),
        "compatibility": _json_safe(dict(compatibility)),
        "rng_state": capture_rng_state(),
        "augmenter_state": None if augmenter_state is None else dict(augmenter_state),
        "data_loader_state": None if data_loader_state is None else dict(data_loader_state),
    }
    save_checkpoint_payload_atomic(payload, path)


_CHECKPOINT_V1_REQUIRED_KEYS = frozenset(
    {
        "format_version",
        "run_id",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "epoch_completed",
        "global_step",
        "best_metric",
        "train_metrics",
        "validation_metrics",
        "resolved_config",
        "seed",
        "split_id",
        "split_fingerprint",
        "dataset_fingerprint",
        "dataset_id",
        "compatibility",
        "rng_state",
        "augmenter_state",
        "data_loader_state",
    }
)


def save_checkpoint_payload_atomic(payload: Mapping[str, Any], path: str | Path) -> None:
    """Serialize and atomically publish an already-constructed v1 checkpoint."""

    expected_keys = frozenset(payload.keys())

    def write_payload(handle: Any) -> None:
        torch.save(payload, handle)

    def validate_payload(temporary_path: Path) -> None:
        loaded = torch.load(temporary_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise ValueError("Serialized checkpoint payload must be a dict.")
        if loaded.get("format_version") != 1:
            raise ValueError("Serialized checkpoint must retain format_version == 1.")
        missing_keys = _CHECKPOINT_V1_REQUIRED_KEYS.difference(loaded)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise ValueError(f"Serialized checkpoint is missing required keys: {missing}.")
        if frozenset(loaded.keys()) != expected_keys:
            raise ValueError("Serialized checkpoint keys differ from the expected payload.")

    atomic_publish(path, write_payload, validator=validate_payload)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict: {path}")
    return payload


def resume_from_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    expected_compatibility: Mapping[str, Any],
    legacy_expected_compatibility: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
) -> ResumeState:
    payload = load_checkpoint(path, map_location=map_location)
    _validate_scheduler_resume(scheduler=scheduler, checkpoint_payload=payload, expected_compatibility=expected_compatibility)
    content_verification = validate_resume_compatibility(
        payload,
        expected_compatibility,
        legacy_expected=legacy_expected_compatibility,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    training_device = torch.device(map_location if device is None else device)
    model.to(training_device)
    move_optimizer_state_to_device(optimizer, training_device)
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    restore_rng_state(payload["rng_state"], device=training_device)
    epoch_completed = int(payload.get("epoch_completed", 0))
    return ResumeState(
        epoch_completed=epoch_completed,
        next_epoch=epoch_completed + 1,
        global_step=int(payload.get("global_step", 0)),
        best_metric=None if payload.get("best_metric") is None else float(payload["best_metric"]),
        checkpoint_path=str(path),
        checkpoint_payload=payload,
        content_verification=content_verification,
    )


def _normalize_numpy_state(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[1], list):
            normalized = list(value)
            normalized[1] = np.array(normalized[1], dtype=np.uint32)
            return tuple(normalized)
        return tuple(value)
    raise TypeError(f"Unsupported numpy RNG state type: {type(value)!r}")


def _to_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value, dtype=torch.uint8)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        raise TypeError("Torch tensors are not JSON-safe metadata.")
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        normalized_items = sorted(((str(key), _json_safe(item)) for key, item in value.items()), key=lambda item: item[0])
        return {key: item for key, item in normalized_items}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported non-JSON metadata type: {type(value)!r}")


def _optimizer_config_from_instance(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    group = dict(optimizer.param_groups[0]) if optimizer.param_groups else {}
    group.pop("params", None)
    allowed = ("lr", "weight_decay", "betas", "eps", "amsgrad", "momentum", "dampening", "nesterov", "maximize")
    return {key: _json_safe(group[key]) for key in allowed if key in group}


def _scheduler_config_from_instance(scheduler: Any | None, *, scheduler_name: str) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    config: dict[str, Any] = {"name": scheduler_name}
    for key in ("gamma", "step_size", "eta_min"):
        if hasattr(scheduler, key):
            config[key] = _json_safe(getattr(scheduler, key))
    return config


def _validate_scheduler_resume(
    *,
    scheduler: Any | None,
    checkpoint_payload: Mapping[str, Any],
    expected_compatibility: Mapping[str, Any],
) -> None:
    checkpoint_scheduler_state = checkpoint_payload.get("scheduler_state_dict")
    checkpoint_scheduler = dict(checkpoint_payload.get("compatibility", {}).get("scheduler", {}))
    expected_scheduler = dict(expected_compatibility.get("scheduler", {}))
    checkpoint_has_scheduler = checkpoint_scheduler_state is not None
    current_has_scheduler = scheduler is not None
    if checkpoint_has_scheduler != current_has_scheduler:
        raise ValueError(
            "Checkpoint resume incompatibility detected for scheduler presence: "
            f"checkpoint_has_scheduler={checkpoint_has_scheduler!r} expected_has_scheduler={current_has_scheduler!r}."
        )
    if not checkpoint_has_scheduler:
        return
    checkpoint_class = checkpoint_scheduler.get("class")
    expected_class = expected_scheduler.get("class")
    if checkpoint_class != expected_class:
        raise ValueError(
            "Checkpoint resume incompatibility detected for scheduler class: "
            f"checkpoint={checkpoint_class!r} expected={expected_class!r}."
        )
    checkpoint_config = checkpoint_scheduler.get("config")
    expected_config = expected_scheduler.get("config")
    if checkpoint_config != expected_config:
        raise ValueError(
            "Checkpoint resume incompatibility detected for scheduler config: "
            f"checkpoint={checkpoint_config!r} expected={expected_config!r}."
        )


def _find_first_mismatch(checkpoint_value: Any, expected_value: Any, *, path: str) -> str:
    if isinstance(checkpoint_value, Mapping) and isinstance(expected_value, Mapping):
        checkpoint_keys = set(str(key) for key in checkpoint_value.keys())
        expected_keys = set(str(key) for key in expected_value.keys())
        for missing_key in sorted(checkpoint_keys ^ expected_keys):
            return f"{path}.{missing_key}: checkpoint={checkpoint_value.get(missing_key)!r} expected={expected_value.get(missing_key)!r}"
        for key in sorted(checkpoint_keys):
            nested = _find_first_mismatch(checkpoint_value.get(key), expected_value.get(key), path=f"{path}.{key}")
            if nested:
                return nested
        return ""
    if isinstance(checkpoint_value, list) and isinstance(expected_value, list):
        if len(checkpoint_value) != len(expected_value):
            return f"{path}.length: checkpoint={len(checkpoint_value)!r} expected={len(expected_value)!r}"
        for index, (checkpoint_item, expected_item) in enumerate(zip(checkpoint_value, expected_value, strict=False)):
            nested = _find_first_mismatch(checkpoint_item, expected_item, path=f"{path}[{index}]")
            if nested:
                return nested
        return ""
    if checkpoint_value != expected_value:
        return f"{path}: checkpoint={checkpoint_value!r} expected={expected_value!r}"
    return ""
