"""Run manifest lifecycle helpers for reproducible Model B executions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
from uuid import uuid4

from gnn_siamese.config import save_config

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import torch_geometric
except ImportError:  # pragma: no cover
    torch_geometric = None  # type: ignore[assignment]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_git_metadata() -> dict[str, Any]:
    """Collect best-effort git metadata without failing offline environments."""

    payload: dict[str, Any] = {
        "commit": None,
        "branch": None,
        "working_tree_state": "unknown",
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload["commit"] = commit.stdout.strip() or None
        payload["branch"] = branch.stdout.strip() or None
        payload["working_tree_state"] = "dirty" if status.stdout.strip() else "clean"
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return payload


def collect_environment_metadata(device: str, *, include_cuda: bool = True) -> dict[str, Any]:
    """Collect runtime environment versions required by the manifest."""

    dependencies = {
        "python": platform.python_version(),
        "pytorch": None if torch is None else torch.__version__,
        "torch_geometric": None if torch_geometric is None else torch_geometric.__version__,
    }
    cuda_info: dict[str, Any] = {"available": False}
    if include_cuda and torch is not None and torch.cuda.is_available():
        cuda_info = {
            "available": True,
            "version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()),
        }

    return {
        "device": device,
        "dependencies": dependencies,
        "cuda": cuda_info,
    }


@dataclass(frozen=True)
class RunArtifactsLayout:
    """Filesystem layout for one operational run."""

    run_id: str
    run_dir: Path
    checkpoints_dir: Path
    manifest_path: Path
    resolved_config_path: Path
    metrics_path: Path
    gradient_audit_path: Path
    split_path: Path


def build_run_layout(
    *,
    root_dir: str | Path = "runs",
    model_name: str = "model_b_graph_level_relational",
    run_id: str,
    manifest_filename: str = "run_manifest.json",
    resolved_config_filename: str = "config_resolved.yaml",
    metrics_filename: str = "metrics.jsonl",
    gradient_audit_filename: str = "gradient_audit.json",
    split_filename: str = "split.json",
    checkpoints_dirname: str = "checkpoints",
) -> RunArtifactsLayout:
    run_dir = Path(root_dir) / model_name / f"run_{run_id}"
    return RunArtifactsLayout(
        run_id=run_id,
        run_dir=run_dir,
        checkpoints_dir=run_dir / checkpoints_dirname,
        manifest_path=run_dir / manifest_filename,
        resolved_config_path=run_dir / resolved_config_filename,
        metrics_path=run_dir / metrics_filename,
        gradient_audit_path=run_dir / gradient_audit_filename,
        split_path=run_dir / split_filename,
    )


class MetricsJsonlWriter:
    """Append one JSON payload per line for epoch-level metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


class RunManifestWriter:
    """Create, update and finalize `run_manifest.json` during one run."""

    def __init__(self, path: str | Path, *, resolved_config_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.resolved_config_path = (
            self.path.parent / "config_resolved.yaml"
            if resolved_config_path is None
            else Path(resolved_config_path)
        )
        self.payload: dict[str, Any] = {}

    def initialize(self, initial_payload: Mapping[str, Any]) -> dict[str, Any]:
        self.payload = dict(initial_payload)
        self.payload.setdefault("status", "running")
        self.payload.setdefault("started_at_utc", _utc_now_iso())
        self._write()
        return dict(self.payload)

    def update(self, updates: Mapping[str, Any]) -> dict[str, Any]:
        self.payload = _deep_update(self.payload, dict(updates))
        self._write()
        return dict(self.payload)

    def finalize(
        self,
        *,
        status: str,
        finished_at_utc: str | None = None,
        error: str | None = None,
        extra_updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "status": status,
            "finished_at_utc": finished_at_utc or _utc_now_iso(),
        }
        if error is not None:
            updates["errors"] = [error]
        if extra_updates is not None:
            updates = _deep_update(updates, dict(extra_updates))
        return self.update(updates)

    def save_resolved_config(self, config: Mapping[str, Any], *, strip_internal_keys: bool = True) -> None:
        payload = dict(config)
        if strip_internal_keys:
            payload = {key: value for key, value in payload.items() if not str(key).startswith("__")}
        save_config(payload, self.resolved_config_path)

    def _write(self) -> None:
        self.path.write_text(json.dumps(self.payload, indent=2, sort_keys=True), encoding="utf-8")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def generate_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{uuid4().hex[:8]}"
