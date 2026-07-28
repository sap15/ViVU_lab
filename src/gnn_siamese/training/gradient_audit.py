"""Per-module gradient and weight-change audit helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch


def _flatten_trainable_parameters(module: torch.nn.Module | None) -> list[tuple[str, torch.nn.Parameter]]:
    if module is None:
        return []
    return [(name, parameter) for name, parameter in module.named_parameters() if parameter is not None]


def _parameter_count(parameters: Sequence[tuple[str, torch.nn.Parameter]], *, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for _, parameter in parameters)


def _weight_vector(parameters: Sequence[tuple[str, torch.nn.Parameter]]) -> torch.Tensor:
    if not parameters:
        return torch.zeros(0, dtype=torch.float32)
    flattened = [parameter.detach().float().reshape(-1).cpu() for _, parameter in parameters]
    return torch.cat(flattened) if flattened else torch.zeros(0, dtype=torch.float32)


@dataclass
class ModuleAuditTracker:
    name: str
    module: torch.nn.Module | None
    connected_losses: list[str]
    optimizer_group: str | None
    status_hint: str
    parameters: list[tuple[str, torch.nn.Parameter]] = field(init=False)
    initial_weight_vector: torch.Tensor = field(init=False)
    gradient_norms: list[float] = field(default_factory=list)
    none_grad_steps: int = 0
    zero_grad_steps: int = 0
    total_steps: int = 0
    has_nan_or_inf: bool = False

    def __post_init__(self) -> None:
        self.parameters = _flatten_trainable_parameters(self.module)
        self.initial_weight_vector = _weight_vector(self.parameters)

    @property
    def parameter_count(self) -> int:
        return _parameter_count(self.parameters)

    @property
    def trainable_parameter_count(self) -> int:
        return _parameter_count(self.parameters, trainable_only=True)

    def record_step(self) -> None:
        self.total_steps += 1
        grads: list[torch.Tensor] = []
        missing_grad = False
        for _, parameter in self.parameters:
            if not parameter.requires_grad:
                continue
            grad = parameter.grad
            if grad is None:
                missing_grad = True
                continue
            grad_cpu = grad.detach().float().cpu()
            if not torch.isfinite(grad_cpu).all():
                self.has_nan_or_inf = True
            grads.append(grad_cpu.reshape(-1))

        if missing_grad:
            self.none_grad_steps += 1
        if not grads:
            self.zero_grad_steps += 1
            return

        stacked = torch.cat(grads)
        norm = float(torch.linalg.vector_norm(stacked, ord=2).item())
        self.gradient_norms.append(norm)
        if norm == 0.0:
            self.zero_grad_steps += 1

    def finalize(self) -> dict[str, Any]:
        final_weight_vector = _weight_vector(self.parameters)
        initial_norm = float(torch.linalg.vector_norm(self.initial_weight_vector, ord=2).item())
        final_norm = float(torch.linalg.vector_norm(final_weight_vector, ord=2).item())
        delta_norm = float(torch.linalg.vector_norm(final_weight_vector - self.initial_weight_vector, ord=2).item())
        relative_weight_change = delta_norm / (initial_norm + 1.0e-12)
        gradient_norms = list(self.gradient_norms)
        status = self._classify_status(relative_weight_change)
        return {
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "optimizer_group": self.optimizer_group,
            "connected_losses": list(self.connected_losses),
            "mean_gradient_norm": float(sum(gradient_norms) / len(gradient_norms)) if gradient_norms else 0.0,
            "median_gradient_norm": _median(gradient_norms),
            "max_gradient_norm": max(gradient_norms) if gradient_norms else 0.0,
            "none_gradient_fraction": self.none_grad_steps / max(self.total_steps, 1),
            "zero_gradient_fraction": self.zero_grad_steps / max(self.total_steps, 1),
            "has_nan_or_inf": self.has_nan_or_inf,
            "initial_weight_norm": initial_norm,
            "final_weight_norm": final_norm,
            "relative_weight_change": relative_weight_change,
            "status": status,
        }

    def _classify_status(self, relative_weight_change: float) -> str:
        if self.status_hint == "not_applicable":
            return "not_applicable"
        if self.status_hint == "inactive":
            return "inactive"
        if self.optimizer_group is None or self.trainable_parameter_count == 0:
            return "failed" if self.status_hint == "active" else "inactive"
        if self.has_nan_or_inf:
            return "failed"
        if not self.gradient_norms:
            return "failed"
        if max(self.gradient_norms) <= 0.0:
            return "failed"
        if relative_weight_change <= 0.0:
            return "failed"
        return "trained"


def build_module_registry(model: torch.nn.Module, loss_weights: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    siamese_model = getattr(model, "siamese_model", None)
    shared_encoder = getattr(siamese_model, "shared_encoder", None)
    projection_instance = getattr(siamese_model, "projection_instance", None)
    projection_pair = getattr(siamese_model, "projection_pair", None)
    relational_module = getattr(siamese_model, "relational_module", None)
    mlp_delta = None if relational_module is None else getattr(relational_module, "mlp_delta", None)
    pooling_fusion = None if shared_encoder is None else getattr(shared_encoder, "fusion_mlp", None)
    bio_head = getattr(siamese_model, "bio_head", None)
    relative_wt_head = getattr(siamese_model, "relative_wt_head", None)
    reconstruction_decoder = getattr(siamese_model, "reconstruction_decoder", None)

    return {
        "encoder": {
            "module": shared_encoder,
            "connected_losses": _connected_losses(["nt_xent", "relative_wt", "delta"], loss_weights),
            "status_hint": "active",
        },
        "pooling_fusion": {
            "module": pooling_fusion,
            "connected_losses": _connected_losses(["nt_xent", "relative_wt", "delta"], loss_weights),
            "status_hint": "active" if pooling_fusion is not None else "not_applicable",
        },
        "projection_instance": {
            "module": projection_instance,
            "connected_losses": _connected_losses(["nt_xent"], loss_weights),
            "status_hint": "active",
        },
        "projection_pair": {
            "module": projection_pair,
            "connected_losses": _connected_losses(["delta"], loss_weights),
            "status_hint": "inactive" if loss_weights.get("delta", 0.0) <= 0.0 else "active",
        },
        "mlp_delta": {
            "module": mlp_delta,
            "connected_losses": _connected_losses(["delta"], loss_weights),
            "status_hint": "inactive" if loss_weights.get("delta", 0.0) <= 0.0 else "active",
        },
        "relative_wt_head": {
            "module": relative_wt_head,
            "connected_losses": _connected_losses(["relative_wt"], loss_weights),
            "status_hint": "not_applicable" if relative_wt_head is None else ("inactive" if loss_weights.get("relative_wt", 0.0) <= 0.0 else "active"),
        },
        "bio_head": {
            "module": bio_head,
            "connected_losses": [],
            "status_hint": "not_applicable" if bio_head is None else "inactive",
        },
        "reconstruction_decoder": {
            "module": reconstruction_decoder,
            "connected_losses": [],
            "status_hint": "not_applicable" if reconstruction_decoder is None else "inactive",
        },
    }


def create_gradient_audit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    loss_weights: Mapping[str, float],
) -> dict[str, ModuleAuditTracker]:
    optimizer_group_by_param_id: dict[int, str] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group.get("params", []):
            optimizer_group_by_param_id[id(parameter)] = f"group_{group_index}"

    registry = build_module_registry(model, loss_weights)
    trackers: dict[str, ModuleAuditTracker] = {}
    for name, metadata in registry.items():
        module = metadata["module"]
        optimizer_group = _resolve_optimizer_group(module, optimizer_group_by_param_id)
        trackers[name] = ModuleAuditTracker(
            name=name,
            module=module,
            connected_losses=list(metadata["connected_losses"]),
            optimizer_group=optimizer_group,
            status_hint=str(metadata["status_hint"]),
        )
    return trackers


def finalize_gradient_audit(trackers: Mapping[str, ModuleAuditTracker]) -> dict[str, Any]:
    return {name: tracker.finalize() for name, tracker in trackers.items()}


def _connected_losses(names: Sequence[str], loss_weights: Mapping[str, float]) -> list[str]:
    return [name for name in names if float(loss_weights.get(name, 0.0)) > 0.0]


def _resolve_optimizer_group(
    module: torch.nn.Module | None,
    optimizer_group_by_param_id: Mapping[int, str],
) -> str | None:
    if module is None:
        return None
    groups = {
        optimizer_group_by_param_id.get(id(parameter))
        for parameter in module.parameters()
        if parameter.requires_grad
    }
    groups.discard(None)
    if not groups:
        return None
    if len(groups) == 1:
        return next(iter(groups))
    return ",".join(sorted(groups))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)
