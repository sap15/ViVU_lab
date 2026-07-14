"""Minimal training utilities for loss assembly, step execution and loop control."""

from gnn_siamese.training.losses import (
    TotalLossAssembler,
    TotalLossConfig,
    TotalLossOutput,
)
from gnn_siamese.training.loop import (
    BaselineEpochOutput,
    ModelBTrainingOutput,
    TrainingLoopConfig,
    TrainingLoopOutput,
    build_run_manifest,
    fit,
    fit_model_b_baseline,
    run_model_b_epoch,
)
from gnn_siamese.training.step import training_step

__all__ = [
    "TotalLossAssembler",
    "TotalLossConfig",
    "TotalLossOutput",
    "BaselineEpochOutput",
    "ModelBTrainingOutput",
    "TrainingLoopConfig",
    "TrainingLoopOutput",
    "build_run_manifest",
    "fit",
    "fit_model_b_baseline",
    "run_model_b_epoch",
    "training_step",
]
