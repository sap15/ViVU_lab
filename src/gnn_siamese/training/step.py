"""Minimal single-step training helper for the current loss assembly phase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from gnn_siamese.training.losses import TotalLossAssembler, TotalLossOutput


def _materialize_outputs(model_output: Any) -> dict[str, Any]:
    if hasattr(model_output, "to_dict"):
        return dict(model_output.to_dict())
    if isinstance(model_output, Mapping):
        return dict(model_output)
    raise TypeError("training_step expects model outputs to be a mapping or expose to_dict().")


def training_step(
    model: nn.Module | Any,
    batch: Any,
    loss_assembler: TotalLossAssembler,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    backward: bool | None = None,
) -> dict[str, Any]:
    """Run a minimal forward/loss/backward step without production training loops."""

    if isinstance(batch, Mapping):
        try:
            model_output = model(**batch)
        except TypeError:
            model_output = model(batch)
    else:
        model_output = model(batch)

    assembled = loss_assembler(**_materialize_outputs(model_output))
    should_backward = optimizer is not None if backward is None else backward
    did_backward = False
    did_step = False

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    if should_backward and optimizer is not None and assembled.active_components:
        assembled.loss.backward()
        did_backward = True
        optimizer.step()
        did_step = True

    return {
        "loss": assembled.loss,
        "loss_output": assembled,
        "components": assembled.components,
        "metrics": assembled.metrics,
        "audit_flags": assembled.audit_flags,
        "model_output": _materialize_outputs(model_output),
        "did_backward": did_backward,
        "did_step": did_step,
    }
