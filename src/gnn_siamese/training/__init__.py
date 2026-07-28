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
    train_model_b_pipeline,
)
from gnn_siamese.training.checkpointing import load_checkpoint, resume_from_checkpoint, save_checkpoint
from gnn_siamese.training.gradient_audit import create_gradient_audit, finalize_gradient_audit
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
    "create_gradient_audit",
    "fit",
    "fit_model_b_baseline",
    "finalize_gradient_audit",
    "load_checkpoint",
    "run_model_b_epoch",
    "resume_from_checkpoint",
    "save_checkpoint",
    "train_model_b_pipeline",
    "training_step",
]
