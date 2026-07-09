"""Minimal multi-epoch training loop for the current production baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import subprocess
from typing import Any

import torch
from torch import Tensor, nn

from gnn_siamese.training.losses import TotalLossAssembler
from gnn_siamese.training.step import training_step


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingLoopConfig:
    """Minimal reproducible config for the current multi-epoch training loop."""

    epochs: int = 1
    device: str = "auto"
    grad_clip_norm: float | None = None
    log_every: int = 0
    stop_on_nonfinite_loss: bool = True
    run_name: str = "training-loop"
    output_dir: str | Path | None = None
    write_manifest: bool = False
    seed: int | None = None


@dataclass(frozen=True)
class TrainingLoopOutput:
    """Structured result returned by `fit`."""

    history: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    epochs_completed: int
    num_steps: int
    stopped_early: bool
    stop_reason: str | None
    manifest: dict[str, Any] | None
    audit_flags: dict[str, Any]


@dataclass(frozen=True)
class _PreparedBatch:
    batch: Any | None = None
    loss_inputs: Mapping[str, Any] | None = None


def _resolve_device(requested: str | torch.device | None) -> torch.device:
    if requested is None:
        return torch.device("cpu")
    if isinstance(requested, torch.device):
        return requested
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _call_batch_adapter(
    batch_adapter: Callable[..., Any] | None,
    raw_batch: Any,
    device: torch.device,
) -> Any:
    if batch_adapter is None:
        return raw_batch
    try:
        return batch_adapter(raw_batch, device=device)
    except TypeError:
        return batch_adapter(raw_batch)


def _prepare_batch(
    raw_batch: Any,
    *,
    batch_adapter: Callable[..., Any] | None,
    device: torch.device,
) -> _PreparedBatch:
    adapted = _call_batch_adapter(batch_adapter, raw_batch, device)
    if isinstance(adapted, Mapping):
        if "loss_inputs" in adapted:
            return _PreparedBatch(loss_inputs=_move_to_device(adapted["loss_inputs"], device))
        if "model_batch" in adapted:
            return _PreparedBatch(batch=_move_to_device(adapted["model_batch"], device))
    return _PreparedBatch(batch=_move_to_device(adapted, device))


def _git_code_version() -> dict[str, Any]:
    payload: dict[str, Any] = {"commit": None, "working_tree_dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload["commit"] = commit.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
        payload["working_tree_dirty"] = bool(status.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        payload["commit"] = "unknown"
        payload["working_tree_dirty"] = "unknown"
    return payload


def _default_output_dir(config: TrainingLoopConfig) -> Path:
    base_dir = Path("runs")
    return base_dir / config.run_name


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def build_run_manifest(
    config: TrainingLoopConfig,
    output: TrainingLoopOutput,
    *,
    output_dir: str | Path | None = None,
    write_manifest: bool | None = None,
) -> dict[str, Any]:
    """Build and optionally persist a lightweight run manifest."""

    component_status = dict(output.audit_flags.get("loss_component_status", {}))
    active_components = list(output.audit_flags.get("active_components", []))
    inactive_components = list(output.audit_flags.get("inactive_components", []))
    skipped_components = list(output.audit_flags.get("skipped_components", []))
    manifest = {
        "run_name": config.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_configured": config.epochs,
        "epochs_completed": output.epochs_completed,
        "num_steps": output.num_steps,
        "loss_components": {
            "active": active_components,
            "inactive": inactive_components,
            "skipped": skipped_components,
            "status_by_component": component_status,
        },
        "delta_active": bool(output.audit_flags.get("delta_active", False)),
        "relative_wt_active": bool(output.audit_flags.get("relative_wt_active", False)),
        "nonfinite_loss_detected": bool(output.audit_flags.get("nonfinite_loss_detected", False)),
        "all_components_inactive": bool(output.audit_flags.get("all_components_inactive", False)),
        "seed": config.seed,
        "reconstruction_status": "disabled/pending",
        "z_delta_space_status": "not_learned"
        if not output.audit_flags.get("delta_active", False)
        else "requires_module_audit",
        "baseline_only_nt_xent": bool(output.audit_flags.get("baseline_only_nt_xent", False)),
        "custom_structure_energy_primary_target": False,
        "stop_reason": output.stop_reason,
        "final_metrics": output.final_metrics,
        "code_version": _git_code_version(),
    }

    should_write = config.write_manifest if write_manifest is None else write_manifest
    if should_write:
        manifest_dir = Path(output_dir) if output_dir is not None else (
            Path(config.output_dir) if config.output_dir is not None else _default_output_dir(config)
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)

    return manifest


def fit(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    total_loss_assembler: TotalLossAssembler,
    config: TrainingLoopConfig,
    *,
    device: str | torch.device | None = None,
    scheduler: Any | None = None,
    batch_adapter: Callable[..., Any] | None = None,
) -> TrainingLoopOutput:
    """Run a minimal auditable multi-epoch training loop."""

    runtime_device = _resolve_device(device or config.device)
    _set_seed(config.seed)
    model.to(runtime_device)
    model.train()

    history: list[dict[str, Any]] = []
    num_steps = 0
    epochs_completed = 0
    stopped_early = False
    stop_reason: str | None = None
    nonfinite_loss_detected = False
    had_active_loss_without_grad_graph = False

    active_components_seen: set[str] = set()
    inactive_components_seen: set[str] = set()
    skipped_components_seen: set[str] = set()
    loss_component_status: dict[str, str] = {}
    all_components_inactive = True

    for epoch_index in range(config.epochs):
        epoch_losses: list[float] = []
        epoch_component_values: dict[str, list[float]] = defaultdict(list)
        epoch_active_components: set[str] = set()
        epoch_inactive_components: set[str] = set()
        epoch_skipped_components: set[str] = set()
        epoch_nonfinite_batches = 0

        for raw_batch in dataloader:
            prepared = _prepare_batch(raw_batch, batch_adapter=batch_adapter, device=runtime_device)
            optimizer.zero_grad(set_to_none=True)

            if prepared.loss_inputs is not None:
                assembled = total_loss_assembler(**prepared.loss_inputs)
                step_result = {
                    "loss": assembled.loss,
                    "loss_output": assembled,
                    "components": assembled.components,
                    "metrics": assembled.metrics,
                    "audit_flags": assembled.audit_flags,
                    "model_output": dict(prepared.loss_inputs),
                    "did_backward": False,
                    "did_step": False,
                }
            else:
                step_result = training_step(
                    model,
                    prepared.batch,
                    total_loss_assembler,
                    optimizer=None,
                    backward=False,
                )
                assembled = step_result["loss_output"]

            loss = assembled.loss
            loss_value = float(loss.detach().cpu().item())
            num_steps += 1

            epoch_active_components.update(assembled.active_components)
            epoch_inactive_components.update(assembled.inactive_components)
            epoch_skipped_components.update(assembled.skipped_components)
            active_components_seen.update(assembled.active_components)
            inactive_components_seen.update(assembled.inactive_components)
            skipped_components_seen.update(assembled.skipped_components)
            loss_component_status.update(assembled.audit_flags.get("component_status", {}))

            if assembled.active_components:
                all_components_inactive = False

            for name, component_tensor in assembled.components.items():
                epoch_component_values[name].append(float(component_tensor.detach().cpu().item()))

            if not torch.isfinite(loss).item():
                nonfinite_loss_detected = True
                epoch_nonfinite_batches += 1
                if config.stop_on_nonfinite_loss:
                    stopped_early = True
                    stop_reason = "nonfinite_loss"
                    logger.warning("Stopping run %s because a non-finite loss was detected.", config.run_name)
                    break
                continue

            epoch_losses.append(loss_value)
            if assembled.active_components and loss.requires_grad:
                loss.backward()
                if config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
                optimizer.step()
                step_result["did_backward"] = True
                step_result["did_step"] = True
            elif assembled.active_components:
                had_active_loss_without_grad_graph = True

            if config.log_every > 0 and num_steps % config.log_every == 0:
                logger.info(
                    "run=%s epoch=%d step=%d loss=%.6f active=%s",
                    config.run_name,
                    epoch_index + 1,
                    num_steps,
                    loss_value,
                    ",".join(assembled.active_components) or "none",
                )

        epoch_record = {
            "epoch": epoch_index + 1,
            "num_batches": len(epoch_losses) + epoch_nonfinite_batches,
            "loss_total": _mean(epoch_losses),
            "active_components": sorted(epoch_active_components),
            "inactive_components": sorted(epoch_inactive_components),
            "skipped_components": sorted(epoch_skipped_components),
            "component_means": {
                name: _mean(values) for name, values in sorted(epoch_component_values.items())
            },
            "nonfinite_batches": epoch_nonfinite_batches,
        }
        history.append(epoch_record)
        epochs_completed = epoch_index + 1

        if scheduler is not None:
            scheduler.step()

        if stopped_early:
            break

    final_metrics = dict(history[-1]) if history else {
        "epoch": 0,
        "num_batches": 0,
        "loss_total": 0.0,
        "active_components": [],
        "inactive_components": [],
        "skipped_components": [],
        "component_means": {},
        "nonfinite_batches": 0,
    }
    audit_flags = {
        "active_components": sorted(active_components_seen),
        "inactive_components": sorted(inactive_components_seen),
        "skipped_components": sorted(skipped_components_seen),
        "loss_component_status": loss_component_status,
        "all_components_inactive": all_components_inactive,
        "nonfinite_loss_detected": nonfinite_loss_detected,
        "relative_wt_active": "relative_wt" in active_components_seen,
        "delta_active": "delta" in active_components_seen,
        "baseline_only_nt_xent": active_components_seen == {"nt_xent"} and "relative_wt" not in active_components_seen and "delta" not in active_components_seen,
        "z_delta_not_trained": "delta" not in active_components_seen,
        "reconstruction_status": "disabled/pending",
        "had_active_loss_without_grad_graph": had_active_loss_without_grad_graph,
    }

    provisional_output = TrainingLoopOutput(
        history=history,
        final_metrics=final_metrics,
        epochs_completed=epochs_completed,
        num_steps=num_steps,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        manifest=None,
        audit_flags=audit_flags,
    )
    manifest = build_run_manifest(config, provisional_output) if config.write_manifest else None
    return TrainingLoopOutput(
        history=history,
        final_metrics=final_metrics,
        epochs_completed=epochs_completed,
        num_steps=num_steps,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        manifest=manifest,
        audit_flags=audit_flags,
    )
