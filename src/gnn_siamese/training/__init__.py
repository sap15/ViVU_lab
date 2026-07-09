"""Minimal training utilities for loss assembly, step execution and loop control."""

from gnn_siamese.training.losses import (
    TotalLossAssembler,
    TotalLossConfig,
    TotalLossOutput,
)
from gnn_siamese.training.loop import (
    TrainingLoopConfig,
    TrainingLoopOutput,
    build_run_manifest,
    fit,
)
from gnn_siamese.training.step import training_step

__all__ = [
    "TotalLossAssembler",
    "TotalLossConfig",
    "TotalLossOutput",
    "TrainingLoopConfig",
    "TrainingLoopOutput",
    "build_run_manifest",
    "fit",
    "training_step",
]
