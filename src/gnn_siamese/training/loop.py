"""Minimal multi-epoch training loop for the current production baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import subprocess
from typing import Any

import torch
from torch import Tensor, nn

from gnn_siamese.losses import NTXentLoss
from gnn_siamese.losses.false_negative_mask import build_false_negative_mask
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


@dataclass(frozen=True)
class BaselineEpochOutput:
    """One train or validation epoch summary for the integrated Model B baseline."""

    phase: str
    mean_loss: float
    num_batches: int
    num_examples: int
    used_eval_mode: bool
    gradients_enabled: bool
    component_means: dict[str, float]
    metrics: dict[str, Any]
    active_components: list[str]
    inactive_components: list[str]
    skipped_components: list[str]


@dataclass(frozen=True)
class ModelBTrainingOutput:
    """Structured output for the baseline end-to-end Model B training loop."""

    train_history: list[BaselineEpochOutput]
    validation_history: list[BaselineEpochOutput]
    final_train_loss: float
    final_validation_loss: float
    final_train_metrics: dict[str, Any]
    final_validation_metrics: dict[str, Any]
    device: str
    epochs_completed: int


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


def _move_graph_batch_to_device(graph_batch: Any, device: torch.device) -> Any:
    if hasattr(graph_batch, "to"):
        return graph_batch.to(device)
    return _move_to_device(graph_batch, device)


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


def _count_batch_examples(batch: Any) -> int:
    if hasattr(batch, "batch_size"):
        return int(batch.batch_size)
    raise ValueError("Expected a MutWtPairBatch-like object with batch_size.")


def _prepare_model_b_views(
    batch: Any,
    *,
    augmenter: Any | None,
    device: torch.device,
) -> dict[str, Any]:
    graph_mut = _move_graph_batch_to_device(batch.graph_mut, device)
    graph_wt = _move_graph_batch_to_device(batch.graph_wt, device)
    if augmenter is None:
        view1_graph_mut = graph_mut
        view2_graph_mut = graph_mut
    else:
        view1_graph_mut, view2_graph_mut = augmenter.create_two_views(graph_mut)
    return {
        "view1_graph_mut": view1_graph_mut,
        "view1_graph_wt": graph_wt,
        "view2_graph_mut": view2_graph_mut,
        "view2_graph_wt": graph_wt,
    }


def _require_batch_positions(batch: Any) -> list[int]:
    positions: list[int] = []
    for item in getattr(batch, "metadata", []):
        position = item.get("position")
        if position is None:
            raise ValueError("False-negative masking requires batch metadata with real positions.")
        positions.append(int(position))
    if len(positions) != int(getattr(batch, "batch_size", 0)):
        raise ValueError("False-negative masking requires one position per batch element.")
    return positions


def _build_model_b_mask_output(batch: Any, loss_assembler: TotalLossAssembler) -> Any | None:
    mask_cfg = dict(loss_assembler.false_negative_mask_kwargs)
    mode = str(mask_cfg.get("mode", "none"))
    if mode == "none":
        return None

    batch_size = _count_batch_examples(batch)
    build_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "mode": mode,
        "min_valid_negatives": float(mask_cfg.get("min_valid_negatives", 8.0)),
        "min_valid_fraction": float(
            mask_cfg.get("min_valid_negative_fraction", mask_cfg.get("min_valid_fraction", 0.25))
        ),
        "strict": bool(mask_cfg.get("strict", False)),
        "combine_same_position": bool(mask_cfg.get("combine_same_position", False)),
    }
    positions = _require_batch_positions(batch)
    if mode == "same_position" or build_kwargs["combine_same_position"]:
        build_kwargs["positions"] = positions
    if mode in {"structural_hard", "structural_soft"}:
        structural_neighbors = getattr(batch, "structural_neighbors", None)
        if structural_neighbors is None:
            raise ValueError(
                "False-negative masking mode "
                f"{mode!r} requires structural_neighbors in the batch; this metadata is not available yet."
            )
        build_kwargs["positions"] = positions
        build_kwargs["structural_neighbors"] = structural_neighbors
        build_kwargs["alpha"] = mask_cfg.get("alpha")
    return build_false_negative_mask(**build_kwargs)


def _select_batch_target(batch: Any, *, target_name: str, device: torch.device) -> Tensor:
    values: list[float] = []
    for item in getattr(batch, "metadata", []):
        if target_name not in item:
            raise ValueError(f"Configured auxiliary target {target_name!r} is missing from batch metadata.")
        raw_value = item[target_name]
        if raw_value is None:
            raise ValueError(f"Configured auxiliary target {target_name!r} contains missing values in batch metadata.")
        values.append(float(raw_value))
    return torch.tensor(values, dtype=torch.float32, device=device)


def _available_batch_metadata_keys(batch: Any) -> list[str]:
    keys: set[str] = set()
    for item in getattr(batch, "metadata", []):
        keys.update(str(key) for key in item.keys())
    return sorted(keys)


def _relative_wt_target_error(batch: Any, *, mode: str, target_name: str | None, reason: str) -> ValueError:
    available_metadata = _available_batch_metadata_keys(batch)
    return ValueError(
        "RelativeWTLoss target resolution failed: "
        f"mode={mode!r}; requested_target_name={target_name!r}; "
        f"available_metadata={available_metadata}; reason={reason}."
    )


def _build_model_b_loss_inputs(
    batch: Any,
    model_output: Any,
    *,
    loss_assembler: TotalLossAssembler,
    device: torch.device,
) -> dict[str, Any]:
    output_payload = model_output.to_dict() if hasattr(model_output, "to_dict") else dict(model_output)
    loss_inputs: dict[str, Any] = {
        "z1": output_payload["z1"],
        "z2": output_payload["z2"],
        "h_mut": output_payload["h_mut"],
        "h_wt": output_payload["h_wt"],
        "z_delta": output_payload.get("z_delta"),
        "z_delta_2": output_payload.get("z_delta_2"),
    }
    if loss_assembler.weights["nt_xent"] > 0.0:
        loss_inputs["mask_output"] = _build_model_b_mask_output(batch, loss_assembler)

    relative_mode = loss_assembler.relative_wt.mode
    relative_target_name = loss_assembler.relative_wt_target_name
    if loss_assembler.weights["relative_wt"] > 0.0 and relative_mode == "ranking":
        if relative_target_name is None:
            raise _relative_wt_target_error(
                batch,
                mode=relative_mode,
                target_name=relative_target_name,
                reason="ranking mode requires explicit loss.relative_wt.target_name in YAML and never falls back to model severity.",
            )
        try:
            loss_inputs["ranking_target"] = _select_batch_target(batch, target_name=relative_target_name, device=device)
        except ValueError as exc:
            raise _relative_wt_target_error(
                batch,
                mode=relative_mode,
                target_name=relative_target_name,
                reason=str(exc),
            ) from exc
        loss_inputs["relative_wt_target_name"] = relative_target_name
    elif loss_assembler.weights["relative_wt"] > 0.0 and relative_mode == "predictive":
        if relative_target_name is None:
            raise _relative_wt_target_error(
                batch,
                mode=relative_mode,
                target_name=relative_target_name,
                reason="predictive mode requires explicit loss.relative_wt.target_name in YAML.",
            )
        try:
            loss_inputs["auxiliary_target"] = _select_batch_target(batch, target_name=relative_target_name, device=device)
        except ValueError as exc:
            raise _relative_wt_target_error(
                batch,
                mode=relative_mode,
                target_name=relative_target_name,
                reason=str(exc),
            ) from exc
        loss_inputs["relative_wt_target_name"] = relative_target_name

    delta_mode = loss_assembler.delta.mode
    delta_target_name = loss_assembler.delta_target_name
    if loss_assembler.weights["delta"] > 0.0 and delta_mode != "none":
        if output_payload.get("z_delta") is None:
            raise ValueError("Delta loss requires z_delta, but model.mlp_delta.enabled=false or z_delta is unavailable.")
        if delta_mode == "descriptor":
            if delta_target_name is None:
                raise ValueError("DeltaLoss descriptor mode requires loss.delta.target_name in YAML.")
            if delta_target_name in {"severity", "severity_target"}:
                loss_inputs["delta_target"] = output_payload.get("severity")
            elif delta_target_name in {"mechanism_direction", "mechanism_direction_target"}:
                loss_inputs["delta_target"] = output_payload.get("mechanism_direction")
            else:
                loss_inputs["delta_target"] = _select_batch_target(batch, target_name=delta_target_name, device=device)
            loss_inputs["delta_target_name"] = delta_target_name

    return loss_inputs


def run_model_b_epoch(
    model: nn.Module,
    dataloader: Any,
    loss_fn: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device,
    augmenter: Any | None = None,
) -> BaselineEpochOutput:
    """Run one train or validation epoch for the integrated Model B baseline."""

    runtime_device = _resolve_device(device)
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.grad = None

    phase = "train" if is_training else "validation"
    total_loss = 0.0
    total_examples = 0
    total_batches = 0
    component_totals: dict[str, float] = defaultdict(float)
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_labels: dict[str, Any] = {}
    active_components_seen: set[str] = set()
    inactive_components_seen: set[str] = set()
    skipped_components_seen: set[str] = set()

    grad_context = torch.enable_grad() if is_training else torch.no_grad()
    with grad_context:
        for batch in dataloader:
            batch_size = _count_batch_examples(batch)
            if batch_size < 2:
                raise ValueError(
                    f"{phase} batch is contrastively degenerate: batch_size={batch_size}, expected >= 2."
                )

            model_inputs = _prepare_model_b_views(batch, augmenter=augmenter, device=runtime_device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            output = model(**model_inputs)
            if isinstance(loss_fn, TotalLossAssembler):
                assembled = loss_fn(**_build_model_b_loss_inputs(batch, output, loss_assembler=loss_fn, device=runtime_device))
                loss = assembled.loss
                for name, component in assembled.components.items():
                    component_totals[name] += float(component.detach().cpu().item()) * batch_size
                active_components_seen.update(assembled.active_components)
                inactive_components_seen.update(assembled.inactive_components)
                skipped_components_seen.update(assembled.skipped_components)
                for name, value in assembled.metrics.items():
                    if isinstance(value, Tensor) and value.ndim == 0:
                        metric_values[name].append(float(value.detach().cpu().item()))
                    elif isinstance(value, (float, int)):
                        metric_values[name].append(float(value))
                    elif isinstance(value, (str, bool)):
                        metric_labels[name] = value
            else:
                loss_output = loss_fn(output.z1, output.z2)
                loss = loss_output.loss
                component_totals["nt_xent"] += float(loss.detach().cpu().item()) * batch_size
                active_components_seen.add("nt_xent")
            if not torch.isfinite(loss):
                raise RuntimeError(f"{phase} loss became non-finite.")
            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_examples += batch_size
            total_batches += 1

    mean_loss = total_loss / max(total_examples, 1)
    return BaselineEpochOutput(
        phase=phase,
        mean_loss=mean_loss,
        num_batches=total_batches,
        num_examples=total_examples,
        used_eval_mode=not is_training and not model.training,
        gradients_enabled=is_training,
        component_means={
            name: value / max(total_examples, 1) for name, value in sorted(component_totals.items())
        },
        metrics=(
            {name: _mean(values) for name, values in sorted(metric_values.items())}
            | dict(sorted(metric_labels.items()))
        ),
        active_components=sorted(active_components_seen),
        inactive_components=sorted(inactive_components_seen),
        skipped_components=sorted(skipped_components_seen),
    )


def fit_model_b_baseline(
    model: nn.Module,
    *,
    train_dataloader: Any,
    validation_dataloader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: str | torch.device,
    augmenter: Any | None = None,
) -> ModelBTrainingOutput:
    """Run the minimal train/validation baseline required for B2."""

    runtime_device = _resolve_device(device)
    model.to(runtime_device)
    train_history: list[BaselineEpochOutput] = []
    validation_history: list[BaselineEpochOutput] = []

    for _ in range(int(epochs)):
        train_history.append(
            run_model_b_epoch(
                model,
                train_dataloader,
                loss_fn,
                optimizer=optimizer,
                device=runtime_device,
                augmenter=augmenter,
            )
        )
        validation_history.append(
            run_model_b_epoch(
                model,
                validation_dataloader,
                loss_fn,
                optimizer=None,
                device=runtime_device,
                augmenter=augmenter,
            )
        )

    return ModelBTrainingOutput(
        train_history=train_history,
        validation_history=validation_history,
        final_train_loss=train_history[-1].mean_loss,
        final_validation_loss=validation_history[-1].mean_loss,
        final_train_metrics=dict(train_history[-1].metrics),
        final_validation_metrics=dict(validation_history[-1].metrics),
        device=str(runtime_device),
        epochs_completed=int(epochs),
    )


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
