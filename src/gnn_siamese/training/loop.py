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
from typing import Any

import torch
from torch import Tensor, nn

from gnn_siamese.losses import NTXentLoss
from gnn_siamese.losses.false_negative_mask import build_false_negative_mask
from gnn_siamese.training.checkpointing import (
    CheckpointSelectionConfig,
    build_dataset_fingerprint,
    build_legacy_resume_compatibility_payload,
    build_resume_compatibility_payload,
    load_checkpoint,
    resume_from_checkpoint,
    save_checkpoint,
    save_checkpoint_payload_atomic,
)
from gnn_siamese.training.gradient_audit import create_gradient_audit, finalize_gradient_audit
from gnn_siamese.training.losses import TotalLossAssembler
from gnn_siamese.training.step import training_step
from gnn_siamese.utils.manifest import (
    MetricsJsonlWriter,
    RunArtifactsLayout,
    RunManifestWriter,
    build_run_layout,
    collect_environment_metadata,
    collect_git_metadata,
    generate_run_id,
)
from gnn_siamese.utils.atomic_io import atomic_write_text
from gnn_siamese.utils.fingerprints import fingerprint_hdf5_inputs, fingerprint_pairing_inventory
from gnn_siamese.utils.interruptions import InterruptionController


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
    epochs_run_this_invocation: int
    run_dir: str | None = None
    best_checkpoint_path: str | None = None
    last_checkpoint_path: str | None = None
    metrics_path: str | None = None
    gradient_audit_path: str | None = None
    manifest_path: str | None = None
    resumed_from: str | None = None
    best_metric: float | None = None


@dataclass
class OperationalRunContext:
    layout: RunArtifactsLayout
    manifest_writer: RunManifestWriter
    last_valid_checkpoint: str | None = None
    persisted_epoch_completed: int = 0
    persisted_global_step: int = 0


def bootstrap_operational_run(
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
) -> OperationalRunContext:
    """Create the run and its readable initializing manifest before pipeline construction."""

    outputs_cfg = dict(config.get("outputs", {}))
    layout = _create_unique_run_layout(outputs_cfg)
    layout.checkpoints_dir.mkdir(parents=True, exist_ok=False)
    writer = RunManifestWriter(layout.manifest_path, resolved_config_path=layout.resolved_config_path)
    writer.initialize(
        {
            "run_id": layout.run_id,
            "status": "initializing",
            "lifecycle": {"stage": "bootstrap"},
            "model_name": str(outputs_cfg.get("model_name", "model_b_graph_level_relational")),
            "configuration": {
                "config_path_provenance": str(Path(config_path).resolve()),
                "resolved_config": {key: value for key, value in config.items() if not str(key).startswith("__")},
            },
            "artifacts": {
                "run_dir": ".",
                "best_checkpoint": "checkpoints/best.pt",
                "last_checkpoint": "checkpoints/last.pt",
                "metrics": layout.relative_reference(layout.metrics_path),
                "gradient_audit": layout.relative_reference(layout.gradient_audit_path),
                "resolved_config": layout.relative_reference(layout.resolved_config_path),
                "split": layout.relative_reference(layout.split_path),
            },
        }
    )
    context = OperationalRunContext(layout=layout, manifest_writer=writer)
    try:
        writer.set_stage("saving_resolved_config")
        writer.save_resolved_config(config)
    except BaseException as exc:
        record_run_failure(context, exc)
        raise
    return context


def record_run_failure(
    context: OperationalRunContext,
    exc: BaseException,
    *,
    stage: str | None = None,
    interrupted: bool = False,
    interruption: Mapping[str, Any] | None = None,
) -> None:
    import traceback

    if context.manifest_writer.payload.get("status") in {"completed", "failed", "interrupted"}:
        return
    if stage is not None:
        context.manifest_writer.set_stage(stage)
    trace = traceback.format_exception(type(exc), exc, exc.__traceback__, limit=12)
    status = "interrupted" if interrupted else "failed"
    context.manifest_writer.finalize(
        status=status,
        error=f"{type(exc).__name__}: {exc}",
        extra_updates={
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "traceback": "".join(trace)[-12000:],
            },
            "training": {
                "epochs_completed": context.persisted_epoch_completed,
                "global_step": context.persisted_global_step,
                "last_valid_checkpoint": context.last_valid_checkpoint,
            },
            **({"interruption": dict(interruption or {})} if interrupted else {}),
        },
    )


def complete_operational_run(context: OperationalRunContext) -> dict[str, Any]:
    """Publish the successful terminal state after caller-owned validation."""

    return context.manifest_writer.finalize(status="completed")


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
    gradient_trackers: Mapping[str, Any] | None = None,
    gradient_clip_norm: float | None = None,
    stop_requested: Callable[[], None] | None = None,
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
            if stop_requested is not None:
                stop_requested()
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
                if gradient_trackers is not None:
                    for tracker in gradient_trackers.values():
                        tracker.record_step()
                if gradient_clip_norm is not None and gradient_clip_norm > 0.0:
                    parameters = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=gradient_clip_norm)
                optimizer.step()
            if stop_requested is not None:
                stop_requested()

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
        epochs_run_this_invocation=int(epochs),
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
        "code_version": collect_git_metadata(),
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


def _train_model_b_pipeline_impl(
    pipeline: Any,
    *,
    config_path: str | Path,
    resume_from: str | Path | None = None,
    run_context: OperationalRunContext,
    interruption_controller: InterruptionController | None = None,
    defer_completion: bool = False,
) -> ModelBTrainingOutput:
    """Run the operational train/validation loop with checkpointing and manifest updates."""

    config = dict(pipeline.config)
    training_cfg = dict(config.get("training", {}))
    outputs_cfg = dict(config.get("outputs", {}))
    loss_cfg = dict(config.get("loss", {}))
    model_cfg = dict(config.get("model", {}))
    split_cfg = dict(config.get("split", {}))
    project_cfg = dict(config.get("project", {}))
    reproducibility_cfg = dict(config.get("reproducibility", {}))

    context = run_context
    layout = context.layout
    manifest_writer = context.manifest_writer
    manifest_writer.set_stage("initializing_training")

    if resume_from is not None:
        manifest_writer.set_stage("resuming")
        checkpoint_epoch = int(load_checkpoint(resume_from, map_location="cpu").get("epoch_completed", 0))
        requested_epochs = int(training_cfg.get("epochs", 1))
        if requested_epochs <= checkpoint_epoch:
            raise ValueError(
                "training.epochs must be greater than checkpoint.epoch_completed when resuming; "
                "training.epochs is the desired total historical epoch count "
                f"(training.epochs={requested_epochs}, checkpoint.epoch_completed={checkpoint_epoch})."
            )

    run_id = layout.run_id
    manifest_writer.set_stage("copying_split")
    atomic_write_text(
        layout.split_path,
        json.dumps(pipeline.split_bundle.split.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metrics_writer = MetricsJsonlWriter(layout.metrics_path)

    active_loss_components = ["nt_xent"]
    if float(loss_cfg.get("lambda_wt", 0.0)) > 0.0:
        active_loss_components.append("relative_wt")
    if float(loss_cfg.get("lambda_delta", 0.0)) > 0.0:
        active_loss_components.append("delta")
    manifest_writer.set_stage("fingerprinting_hdf5")
    hdf5_content_fingerprint = fingerprint_hdf5_inputs(
        mutants_path=pipeline.dataset.mutant_h5_path,
        wt_companion_path=pipeline.dataset.wt_h5_path,
        dataset_id=str(project_cfg.get("name", "dataset")),
    )
    manifest_writer.set_stage("building_resume_compatibility")
    compatibility = build_resume_compatibility_payload(
        config=config,
        dataset=pipeline.dataset,
        split_bundle=pipeline.split_bundle,
        optimizer=pipeline.optimizer,
        scheduler=pipeline.scheduler,
        hdf5_content_fingerprint=hdf5_content_fingerprint["combined"],
        schema=pipeline.schema,
    )
    legacy_compatibility = build_legacy_resume_compatibility_payload(
        new_compatibility=compatibility,
        dataset=pipeline.dataset,
        split_bundle=pipeline.split_bundle,
    )
    dataset_fingerprint = build_dataset_fingerprint(pipeline.dataset)
    split_fingerprint = str(pipeline.split_bundle.split.dataset_fingerprint)
    selection = CheckpointSelectionConfig(
        monitor=str(training_cfg.get("checkpointing", {}).get("monitor", "validation_loss")),
        mode=str(training_cfg.get("checkpointing", {}).get("mode", "min")),
    )
    manifest_writer.set_stage("expanding_manifest")
    manifest_writer.update(
        {
            "run_id": run_id,
            "architecture": str(model_cfg.get("architecture", "model_b")),
            "model_name": str(outputs_cfg.get("model_name", "model_b_graph_level_relational")),
            "code": collect_git_metadata(),
            "environment": collect_environment_metadata(str(pipeline.device)),
            "data": {
                "dataset_id": {
                    "mutants_hdf5": Path(pipeline.dataset.mutant_h5_path).name,
                    "wt_companion_hdf5": Path(pipeline.dataset.wt_h5_path).name,
                },
                "hdf5_files": [
                    Path(pipeline.dataset.mutant_h5_path).name,
                    Path(pipeline.dataset.wt_h5_path).name,
                ],
                "dataset_fingerprint": dataset_fingerprint,
                "pairing_inventory_fingerprint": fingerprint_pairing_inventory(pipeline.dataset.pairs),
                "dataset_identity": {
                    "dataset_id": str(project_cfg.get("name", "dataset")),
                    "schema_name": pipeline.schema.get("schema_name"),
                    "schema_version": pipeline.schema.get("schema_version"),
                    "roles": ["mutants", "wt_companion"],
                },
                "hdf5_content_fingerprint": hdf5_content_fingerprint,
                "locators": {
                    "mutants": {"path": str(Path(pipeline.dataset.mutant_h5_path).resolve()), "provenance": "current_execution"},
                    "wt_companion": {"path": str(Path(pipeline.dataset.wt_h5_path).resolve()), "provenance": "current_execution"},
                },
                "hdf5_schema": {
                    "schema_name": pipeline.schema.get("schema_name"),
                    "schema_version": pipeline.schema.get("schema_version"),
                },
                "split_id": Path(pipeline.split_bundle.split_path).name,
                "split_seed": split_cfg.get("seed"),
                "split_fingerprint": split_fingerprint,
                "split_path": layout.relative_reference(layout.split_path),
                "train_examples": len(pipeline.dataloaders.train_dataset),
                "validation_examples": len(pipeline.dataloaders.validation_dataset),
                "test_examples": len(pipeline.dataloaders.test_dataset),
                "smoke_test": dict(pipeline.smoke_selection or {}),
            },
            "configuration": {
                "config_path": str(Path(config_path)),
                "resolved_config": {key: value for key, value in config.items() if not str(key).startswith("__")},
                "seed": project_cfg.get("seed"),
                "seed_bundle": {
                    "project": project_cfg.get("seed"),
                    "python": reproducibility_cfg.get("seed_python"),
                    "numpy": reproducibility_cfg.get("seed_numpy"),
                    "torch": reproducibility_cfg.get("seed_torch"),
                    "cuda": reproducibility_cfg.get("seed_cuda"),
                },
                "node_feature_names": list(pipeline.dataset.node_feature_names),
                "edge_feature_names": list(pipeline.dataset.edge_feature_names),
                "graph_feature_names": list(getattr(pipeline.dataset, "graph_feature_names", [])),
                "augmentations": dict(config.get("augmentation", {})),
                "pooling": dict(model_cfg.get("pooling", {})),
                "model": {
                    "hidden_dim": model_cfg.get("hidden_dim"),
                    "graph_dim": model_cfg.get("graph_dim"),
                    "num_layers": model_cfg.get("num_layers"),
                    "projection_instance_enabled": model_cfg.get("projection_instance", {}).get("enabled"),
                    "projection_pair_enabled": model_cfg.get("projection_pair", {}).get("enabled"),
                    "mlp_delta_enabled": model_cfg.get("mlp_delta", {}).get("enabled"),
                },
            },
            "training": {
                "optimizer": str(training_cfg.get("optimizer", "adamw")),
                "scheduler": str(training_cfg.get("scheduler", "none")),
                "learning_rate": float(training_cfg.get("learning_rate", 0.0)),
                "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
                "batch_size": int(training_cfg.get("batch_size", 0)),
                "epochs_planned": int(training_cfg.get("epochs", 0)),
                "epochs_completed": 0,
                "epochs_run_this_invocation": 0,
                "epochs_completed_semantics": "total historical epochs completed by the run, including resumed epochs",
                "mixed_precision": dict(training_cfg.get("mixed_precision", {})),
                "gradient_clipping": training_cfg.get("gradient_clip_norm"),
                "best_selection": {"monitor": selection.monitor, "mode": selection.mode},
                "resume_from": None if resume_from is None else str(resume_from),
            },
            "losses": {
                "main": str(loss_cfg.get("main", "nt_xent")),
                "temperature": float(loss_cfg.get("temperature", 0.2)),
                "false_negative_mask": dict(loss_cfg.get("false_negative_mask", {})),
                "lambda_wt": float(loss_cfg.get("lambda_wt", 0.0)),
                "relative_wt": dict(loss_cfg.get("relative_wt", {})),
                "lambda_delta": float(loss_cfg.get("lambda_delta", 0.0)),
                "delta": dict(loss_cfg.get("delta", {})),
                "active_components": active_loss_components,
                "weights": dict(pipeline.total_loss_assembler.weights),
            },
            "artifacts": {
                "run_dir": str(layout.run_dir),
                "best_checkpoint": str(layout.checkpoints_dir / "best.pt"),
                "last_checkpoint": str(layout.checkpoints_dir / "last.pt"),
                "metrics": str(layout.metrics_path),
                "gradient_audit": str(layout.gradient_audit_path),
                "resolved_config": str(layout.resolved_config_path),
                "split": str(layout.split_path),
            },
            "artifact_references": {
                "best_checkpoint": "checkpoints/best.pt",
                "last_checkpoint": "checkpoints/last.pt",
                "metrics": layout.relative_reference(layout.metrics_path),
                "gradient_audit": layout.relative_reference(layout.gradient_audit_path),
                "resolved_config": layout.relative_reference(layout.resolved_config_path),
                "split": layout.relative_reference(layout.split_path),
            },
        }
    )
    manifest_writer.transition("running", stage="initializing_training")

    manifest_writer.set_stage("initializing_gradient_audit")
    loss_weights = {
        "nt_xent": float(pipeline.total_loss_assembler.weights["nt_xent"]),
        "relative_wt": float(pipeline.total_loss_assembler.weights["relative_wt"]),
        "delta": float(pipeline.total_loss_assembler.weights["delta"]),
    }
    gradient_trackers: Mapping[str, Any] = {}

    start_epoch = 1
    global_step = 0
    best_metric: float | None = None
    resumed_from_path: str | None = None
    resumed_epoch_completed = 0
    completed_epoch = 0
    train_history: list[BaselineEpochOutput] = []
    validation_history: list[BaselineEpochOutput] = []
    try:
        gradient_trackers = create_gradient_audit(
            pipeline.model,
            pipeline.optimizer,
            loss_weights=loss_weights,
        )
        if interruption_controller is not None:
            interruption_controller.raise_if_requested()
        if resume_from is not None:
            manifest_writer.set_stage("resuming")
            resume_state = resume_from_checkpoint(
                resume_from,
                model=pipeline.model,
                optimizer=pipeline.optimizer,
                scheduler=pipeline.scheduler,
                expected_compatibility=compatibility,
                legacy_expected_compatibility=legacy_compatibility,
                map_location="cpu",
                device=pipeline.device,
            )
            start_epoch = resume_state.next_epoch
            global_step = resume_state.global_step
            best_metric = resume_state.best_metric
            resumed_from_path = resume_state.checkpoint_path
            resumed_epoch_completed = resume_state.epoch_completed
            completed_epoch = resume_state.epoch_completed
            _restore_pipeline_random_state(pipeline, resume_state.checkpoint_payload)
            legacy_content_fingerprint = (
                resume_state.content_verification
                == "legacy_unavailable_historical_controls_only"
            )
            manifest_writer.update(
                {
                    "resume_compatibility": {
                        "content_verification": (
                            "legacy_unavailable_historical_controls_only"
                            if legacy_content_fingerprint
                            else "sha256_raw_file_bytes_verified"
                        ),
                        "strong_content_verification": not legacy_content_fingerprint,
                    },
                    "training": {
                        "resume_from": resumed_from_path,
                        "resume_epoch": resume_state.epoch_completed,
                        "epochs_completed": resume_state.epoch_completed,
                        "global_step": global_step,
                        "best_metric": best_metric,
                    }
                }
            )
            if best_metric is not None:
                save_checkpoint_payload_atomic(
                    resume_state.checkpoint_payload,
                    layout.checkpoints_dir / "best.pt",
                )

        epochs = int(training_cfg.get("epochs", 1))
        manifest_writer.set_stage("training")
        for epoch in range(start_epoch, epochs + 1):
            train_epoch = run_model_b_epoch(
                pipeline.model,
                pipeline.dataloaders.train_loader,
                pipeline.total_loss_assembler,
                optimizer=pipeline.optimizer,
                device=pipeline.device,
                augmenter=pipeline.augmenter,
                gradient_trackers=gradient_trackers,
                gradient_clip_norm=training_cfg.get("gradient_clip_norm"),
                stop_requested=None if interruption_controller is None else interruption_controller.raise_if_requested,
            )
            train_history.append(train_epoch)

            validation_epoch = run_model_b_epoch(
                pipeline.model,
                pipeline.dataloaders.validation_loader,
                pipeline.total_loss_assembler,
                optimizer=None,
                device=pipeline.device,
                augmenter=pipeline.augmenter,
                stop_requested=None if interruption_controller is None else interruption_controller.raise_if_requested,
            )
            validation_history.append(validation_epoch)
            global_step += train_epoch.num_batches
            if pipeline.scheduler is not None:
                pipeline.scheduler.step()

            epoch_metrics = {
                "epoch": epoch,
                "global_step": global_step,
                "train": _epoch_output_to_dict(train_epoch),
                "validation": _epoch_output_to_dict(validation_epoch),
            }
            metrics_writer.append(epoch_metrics)

            monitor_value = _select_monitor_value(selection.monitor, train_epoch, validation_epoch)
            if selection.is_improved(monitor_value, best_metric):
                best_metric = float(monitor_value)
                save_checkpoint(
                    layout.checkpoints_dir / "best.pt",
                    model=pipeline.model,
                    optimizer=pipeline.optimizer,
                    scheduler=pipeline.scheduler,
                    epoch_completed=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    train_metrics=_epoch_output_to_dict(train_epoch),
                    validation_metrics=_epoch_output_to_dict(validation_epoch),
                    resolved_config=config,
                    seed=project_cfg.get("seed"),
                    split_id=Path(pipeline.split_bundle.split_path).name,
                    split_fingerprint=split_fingerprint,
                    dataset_fingerprint=dataset_fingerprint,
                    hdf5_content_fingerprint=hdf5_content_fingerprint["combined"],
                    dataset_id={
                        "mutants_hdf5": Path(pipeline.dataset.mutant_h5_path).name,
                        "wt_companion_hdf5": Path(pipeline.dataset.wt_h5_path).name,
                    },
                    compatibility=compatibility,
                    run_id=run_id,
                    augmenter_state=_capture_augmenter_state(pipeline.augmenter),
                    data_loader_state=_capture_data_loader_state(pipeline.dataloaders),
                )
            save_checkpoint(
                layout.checkpoints_dir / "last.pt",
                model=pipeline.model,
                optimizer=pipeline.optimizer,
                scheduler=pipeline.scheduler,
                epoch_completed=epoch,
                global_step=global_step,
                best_metric=best_metric,
                train_metrics=_epoch_output_to_dict(train_epoch),
                validation_metrics=_epoch_output_to_dict(validation_epoch),
                resolved_config=config,
                seed=project_cfg.get("seed"),
                split_id=Path(pipeline.split_bundle.split_path).name,
                split_fingerprint=split_fingerprint,
                dataset_fingerprint=dataset_fingerprint,
                hdf5_content_fingerprint=hdf5_content_fingerprint["combined"],
                dataset_id={
                    "mutants_hdf5": Path(pipeline.dataset.mutant_h5_path).name,
                    "wt_companion_hdf5": Path(pipeline.dataset.wt_h5_path).name,
                },
                compatibility=compatibility,
                run_id=run_id,
                augmenter_state=_capture_augmenter_state(pipeline.augmenter),
                data_loader_state=_capture_data_loader_state(pipeline.dataloaders),
            )
            completed_epoch = epoch
            context.last_valid_checkpoint = "checkpoints/last.pt"
            context.persisted_epoch_completed = epoch
            context.persisted_global_step = global_step

            manifest_writer.update(
                {
                    "training": {
                        "epochs_completed": epoch,
                        "epochs_run_this_invocation": epoch - resumed_epoch_completed,
                        "global_step": global_step,
                        "best_metric": best_metric,
                        "last_epoch_metrics": epoch_metrics,
                    }
                }
            )

        manifest_writer.set_stage("finalizing")
        if interruption_controller is not None:
            interruption_controller.raise_if_requested()
        module_audit = finalize_gradient_audit(gradient_trackers)
        atomic_write_text(layout.gradient_audit_path, json.dumps(module_audit, indent=2, sort_keys=True), encoding="utf-8")
        if interruption_controller is not None:
            interruption_controller.raise_if_requested()
        z_delta_learned, z_delta_reason = _resolve_z_delta_learned(
            module_audit.get("mlp_delta", {}),
            config=config,
        )
        total_parameters = sum(parameter.numel() for parameter in pipeline.model.parameters())
        trainable_parameters = sum(parameter.numel() for parameter in pipeline.model.parameters() if parameter.requires_grad)
        if interruption_controller is not None:
            interruption_controller.raise_if_requested()
        final_updates = {
            "training": {
                "epochs_completed": completed_epoch,
                "epochs_run_this_invocation": completed_epoch - resumed_epoch_completed,
                "best_metric": best_metric,
            },
            "modules": module_audit,
            "module_summary": {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            },
            "z_delta_learned": z_delta_learned,
            "z_delta_reason": z_delta_reason,
        }
        if defer_completion:
            manifest_writer.update(final_updates)
            manifest_writer.set_stage("validating_smoke_artifacts")
        else:
            manifest_writer.finalize(status="completed", extra_updates=final_updates)
    except KeyboardInterrupt as exc:
        interruption = (
            interruption_controller.metadata()
            if interruption_controller is not None
            else {"reason": "KeyboardInterrupt", "signal": None, "signal_name": None, "exit_code": 130}
        )
        record_run_failure(
            context,
            exc,
            interrupted=True,
            interruption=interruption,
        )
        raise
    except Exception as exc:
        record_run_failure(context, exc)
        raise

    total_epochs_completed = completed_epoch
    return ModelBTrainingOutput(
        train_history=train_history,
        validation_history=validation_history,
        final_train_loss=train_history[-1].mean_loss,
        final_validation_loss=validation_history[-1].mean_loss,
        final_train_metrics=dict(train_history[-1].metrics),
        final_validation_metrics=dict(validation_history[-1].metrics),
        device=str(pipeline.device),
        epochs_completed=total_epochs_completed,
        epochs_run_this_invocation=len(train_history),
        run_dir=str(layout.run_dir),
        best_checkpoint_path=str(layout.checkpoints_dir / "best.pt"),
        last_checkpoint_path=str(layout.checkpoints_dir / "last.pt"),
        metrics_path=str(layout.metrics_path),
        gradient_audit_path=str(layout.gradient_audit_path),
        manifest_path=str(layout.manifest_path),
        resumed_from=resumed_from_path,
        best_metric=best_metric,
    )


def train_model_b_pipeline(
    pipeline: Any,
    *,
    config_path: str | Path,
    resume_from: str | Path | None = None,
    run_context: OperationalRunContext | None = None,
    interruption_controller: InterruptionController | None = None,
    defer_completion: bool = False,
) -> ModelBTrainingOutput:
    """Run Model B with one lifecycle guard covering every post-bootstrap operation."""

    context = run_context or bootstrap_operational_run(
        dict(pipeline.config),
        config_path=config_path,
    )
    try:
        return _train_model_b_pipeline_impl(
            pipeline,
            config_path=config_path,
            resume_from=resume_from,
            run_context=context,
            interruption_controller=interruption_controller,
            defer_completion=defer_completion,
        )
    except KeyboardInterrupt as exc:
        record_run_failure(
            context,
            exc,
            interrupted=True,
            interruption=(
                interruption_controller.metadata()
                if interruption_controller is not None
                else {"reason": "KeyboardInterrupt", "signal": None, "signal_name": None, "exit_code": 130}
            ),
        )
        raise
    except BaseException as exc:
        record_run_failure(context, exc)
        raise


def _epoch_output_to_dict(epoch: BaselineEpochOutput) -> dict[str, Any]:
    return {
        "phase": epoch.phase,
        "mean_loss": epoch.mean_loss,
        "num_batches": epoch.num_batches,
        "num_examples": epoch.num_examples,
        "used_eval_mode": epoch.used_eval_mode,
        "gradients_enabled": epoch.gradients_enabled,
        "component_means": dict(epoch.component_means),
        "metrics": dict(epoch.metrics),
        "active_components": list(epoch.active_components),
        "inactive_components": list(epoch.inactive_components),
        "skipped_components": list(epoch.skipped_components),
    }


def _select_monitor_value(
    monitor: str,
    train_epoch: BaselineEpochOutput,
    validation_epoch: BaselineEpochOutput,
) -> float:
    if monitor == "validation_loss":
        return float(validation_epoch.mean_loss)
    if monitor == "train_loss":
        return float(train_epoch.mean_loss)
    if monitor in validation_epoch.metrics:
        return float(validation_epoch.metrics[monitor])
    if monitor in train_epoch.metrics:
        return float(train_epoch.metrics[monitor])
    raise KeyError(f"Configured checkpoint monitor {monitor!r} is unavailable.")


def _create_unique_run_layout(outputs_cfg: Mapping[str, Any]) -> RunArtifactsLayout:
    root_dir = outputs_cfg.get("root_dir", "runs")
    model_name = str(outputs_cfg.get("model_name", "model_b_graph_level_relational"))
    manifest_filename = str(outputs_cfg.get("manifest_filename", "run_manifest.json"))
    resolved_config_filename = str(outputs_cfg.get("resolved_config_filename", "config_resolved.yaml"))
    gradient_audit_filename = str(outputs_cfg.get("gradient_audit_filename", "gradient_audit.json"))
    metrics_filename = str(outputs_cfg.get("metrics_filename", "metrics.jsonl"))
    checkpoints_dirname = str(outputs_cfg.get("directories", {}).get("checkpoints", "checkpoints"))
    last_error: FileExistsError | None = None
    for _ in range(8):
        layout = build_run_layout(
            root_dir=root_dir,
            model_name=model_name,
            run_id=generate_run_id(),
            manifest_filename=manifest_filename,
            resolved_config_filename=resolved_config_filename,
            metrics_filename=metrics_filename,
            gradient_audit_filename=gradient_audit_filename,
            split_filename="split.json",
            checkpoints_dirname=checkpoints_dirname,
        )
        try:
            layout.run_dir.mkdir(parents=True, exist_ok=False)
            return layout
        except FileExistsError as exc:
            last_error = exc
    raise RuntimeError("Failed to allocate a unique run directory without reusing an existing run.") from last_error


def _capture_augmenter_state(augmenter: Any | None) -> dict[str, Any] | None:
    if augmenter is None:
        return None
    if hasattr(augmenter, "_call_index"):
        return {"call_index": int(getattr(augmenter, "_call_index"))}
    return None


def _capture_data_loader_state(dataloaders: Any) -> dict[str, Any] | None:
    generator = getattr(dataloaders, "train_generator", None)
    if generator is None:
        return None
    return {"train_generator_state": generator.get_state()}


def _restore_pipeline_random_state(pipeline: Any, checkpoint_payload: Mapping[str, Any]) -> None:
    augmenter_state = checkpoint_payload.get("augmenter_state")
    if isinstance(augmenter_state, Mapping) and hasattr(pipeline.augmenter, "_call_index"):
        pipeline.augmenter._call_index = int(augmenter_state.get("call_index", 0))
    data_loader_state = checkpoint_payload.get("data_loader_state")
    generator = getattr(pipeline.dataloaders, "train_generator", None)
    if isinstance(data_loader_state, Mapping) and generator is not None:
        generator_state = data_loader_state.get("train_generator_state")
        if generator_state is not None:
            generator.set_state(generator_state)


def _resolve_z_delta_learned(module_record: Mapping[str, Any], *, config: Mapping[str, Any]) -> tuple[bool, str]:
    mlp_delta_enabled = bool(config.get("model", {}).get("mlp_delta", {}).get("enabled", False))
    lambda_delta = float(config.get("loss", {}).get("lambda_delta", 0.0))
    if not mlp_delta_enabled:
        return False, "model.mlp_delta.enabled=false"
    if lambda_delta <= 0.0:
        return False, "loss.lambda_delta=0"
    if not module_record:
        return False, "mlp_delta audit missing"
    if module_record.get("status") != "trained":
        return False, f"mlp_delta status={module_record.get('status')}"
    if not module_record.get("optimizer_group"):
        return False, "mlp_delta missing optimizer group"
    if "delta" not in list(module_record.get("connected_losses", [])):
        return False, "mlp_delta not connected to L_delta"
    if bool(module_record.get("has_nan_or_inf", False)):
        return False, "mlp_delta gradients invalid"
    if float(module_record.get("mean_gradient_norm", 0.0)) <= 0.0 and float(module_record.get("max_gradient_norm", 0.0)) <= 0.0:
        return False, "mlp_delta gradients are zero"
    if float(module_record.get("relative_weight_change", 0.0)) <= 0.0:
        return False, "mlp_delta weights did not change"
    return True, "mlp_delta trained and audited"
