"""YAML-driven configuration helpers for the operational Model B baseline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised only when PyYAML is unavailable.
    raise ImportError(
        "gnn_siamese.config requires PyYAML to load YAML configuration files."
    ) from exc


class ConfigError(ValueError):
    """Raised when YAML configuration files cannot be resolved or validated."""


def _deep_merge_dicts(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_object(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in configuration file: {path}") from exc

    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Top-level YAML object must be a mapping: {path}")
    return dict(payload)


def _load_config_unvalidated(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    payload = _load_yaml_object(path)
    parent_value = payload.pop("extends", None)
    if parent_value is None:
        return payload
    if not isinstance(parent_value, str) or not parent_value:
        raise ConfigError("config.extends must be a non-empty string when provided.")

    parent_path = (path.parent / parent_value).resolve()
    parent_payload = _load_config_unvalidated(parent_path)
    return _deep_merge_dicts(parent_payload, payload)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load, inherit, and validate one YAML configuration."""

    return validate_c1_config(_load_config_unvalidated(config_path))


def resolve_training_device(requested: str | None) -> torch.device:
    """Resolve the C1 device policy before constructing data or a model."""

    value = "auto" if requested is None else str(requested).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise ConfigError(
                "training.device requested CUDA, but CUDA is not available; "
                "use training.device=cpu or training.device=auto."
            )
        return torch.device("cuda")
    raise ConfigError("training.device must be one of: auto, cpu, cuda.")


def _option_enabled(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value.get("enabled", False))
    raise ConfigError(f"{field_name} must be a boolean or a mapping with an enabled key.")


def validate_c1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize only the runtime options covered by Pre-Colab C1."""

    resolved = deepcopy(dict(config))
    training = resolved.setdefault("training", {})
    outputs = resolved.setdefault("outputs", {})
    paths = resolved.setdefault("paths", {})
    if not isinstance(training, dict):
        raise ConfigError("config.training must be a mapping.")
    if not isinstance(outputs, dict):
        raise ConfigError("config.outputs must be a mapping.")
    if not isinstance(paths, dict):
        raise ConfigError("config.paths must be a mapping.")

    device = str(training.get("device", "auto")).lower()
    resolve_training_device(device)
    training["device"] = device

    clip = training.get("gradient_clip_norm")
    if clip is not None:
        try:
            clip = float(clip)
        except (TypeError, ValueError) as exc:
            raise ConfigError("training.gradient_clip_norm must be null or a non-negative number.") from exc
        if clip < 0.0:
            raise ConfigError("training.gradient_clip_norm must be null or a non-negative number.")
        training["gradient_clip_norm"] = clip

    num_workers = int(training.get("num_workers", 0))
    if num_workers < 0:
        raise ConfigError("training.num_workers must be greater than or equal to zero.")
    training["num_workers"] = num_workers
    pin_memory = training.get("pin_memory", False)
    if pin_memory == "auto":
        pin_memory = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if not isinstance(pin_memory, bool):
        raise ConfigError("training.pin_memory must be true, false, or auto.")
    training["pin_memory"] = pin_memory
    persistent_workers = training.get("persistent_workers", False)
    if not isinstance(persistent_workers, bool):
        raise ConfigError("training.persistent_workers must be a boolean.")
    if persistent_workers and num_workers <= 0:
        raise ConfigError("training.persistent_workers=true requires training.num_workers > 0.")

    if _option_enabled(training.get("mixed_precision", False), field_name="training.mixed_precision"):
        raise ConfigError("training.mixed_precision.enabled=true is not supported in Pre-Colab C1.")
    accumulation = int(training.get("gradient_accumulation_steps", 1))
    if accumulation != 1:
        raise ConfigError("training.gradient_accumulation_steps must be 1; accumulation is not supported in Pre-Colab C1.")
    if _option_enabled(training.get("early_stopping", False), field_name="training.early_stopping"):
        raise ConfigError("training.early_stopping.enabled=true is not supported in Pre-Colab C1.")

    canonical_root = outputs.get("root_dir")
    legacy_root = paths.get("runs_root")
    if canonical_root is None and legacy_root is None:
        canonical_root = "runs"
    elif canonical_root is None:
        canonical_root = legacy_root
    elif legacy_root is not None and Path(str(canonical_root)) != Path(str(legacy_root)):
        raise ConfigError("outputs.root_dir and deprecated paths.runs_root must not disagree.")
    outputs["root_dir"] = str(canonical_root)
    if legacy_root is not None:
        paths["runs_root"] = str(canonical_root)

    metrics_filename = str(outputs.get("metrics_filename", "metrics.jsonl"))
    if Path(metrics_filename).name != metrics_filename or not metrics_filename.endswith(".jsonl"):
        raise ConfigError("outputs.metrics_filename must be a filename ending in .jsonl.")
    outputs["metrics_filename"] = metrics_filename
    return resolved


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")


def load_schema(schema_path: str | Path) -> dict[str, Any]:
    path = Path(schema_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Schema file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Schema file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Schema JSON must contain a top-level object: {path}")
    return payload


def apply_runtime_overrides(
    config: Mapping[str, Any],
    *,
    device: str | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Apply minimal CLI overrides without mutating the loaded config."""

    resolved = deepcopy(dict(config))
    training_cfg = resolved.setdefault("training", {})
    if not isinstance(training_cfg, dict):
        raise ConfigError("config.training must be a mapping.")
    if device is not None:
        training_cfg["device"] = device
    if smoke_test:
        smoke_cfg = training_cfg.setdefault("smoke_test", {})
        if not isinstance(smoke_cfg, dict):
            raise ConfigError("config.training.smoke_test must be a mapping.")
        smoke_cfg["enabled"] = True
        training_cfg["epochs"] = int(smoke_cfg.get("epochs", 2))
        training_cfg["batch_size"] = int(smoke_cfg.get("batch_size", 4))
    return validate_c1_config(resolved)
