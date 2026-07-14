"""YAML-driven configuration helpers for the operational Model B baseline."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

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


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load one YAML config file, optionally inheriting from a parent config."""

    path = Path(config_path).resolve()
    payload = _load_yaml_object(path)
    parent_value = payload.pop("extends", None)
    if parent_value is None:
        return payload
    if not isinstance(parent_value, str) or not parent_value:
        raise ConfigError("config.extends must be a non-empty string when provided.")

    parent_path = (path.parent / parent_value).resolve()
    parent_payload = load_config(parent_path)
    return _deep_merge_dicts(parent_payload, payload)


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
    return resolved
