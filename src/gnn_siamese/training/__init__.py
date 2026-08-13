"""Minimal training utilities for loss assembly, step execution and loop control."""

from gnn_siamese.training.losses import (
    TotalLossAssembler,
    TotalLossConfig,
    TotalLossOutput,
)
from gnn_siamese.training.loop import (
    BaselineEpochOutput,
    ModelBTrainingOutput,
    OperationalRunContext,
    TrainingLoopConfig,
    TrainingLoopOutput,
    build_run_manifest,
    bootstrap_operational_run,
    complete_operational_run,
    fit,
    fit_model_b_baseline,
    run_model_b_epoch,
    run_contrastive_epoch,
    record_run_failure,
    train_model_b_pipeline,
    train_contrastive_pipeline,
)
from gnn_siamese.training.checkpointing import (
    build_legacy_resume_compatibility_payload,
    load_checkpoint,
    move_optimizer_state_to_device,
    resume_from_checkpoint,
    save_checkpoint,
)
from gnn_siamese.training.gradient_audit import create_gradient_audit, finalize_gradient_audit
from gnn_siamese.training.model_a_contrastive import (
    ModelAContrastive,
    ModelAContrastiveOutput,
)
from gnn_siamese.training.step import training_step
from gnn_siamese.training.architecture import ContrastiveBatchOutput, forward_contrastive_batch

__all__ = [
    "TotalLossAssembler",
    "TotalLossConfig",
    "TotalLossOutput",
    "BaselineEpochOutput",
    "ModelBTrainingOutput",
    "ModelAContrastive",
    "ModelAContrastiveOutput",
    "OperationalRunContext",
    "TrainingLoopConfig",
    "TrainingLoopOutput",
    "build_run_manifest",
    "build_legacy_resume_compatibility_payload",
    "bootstrap_operational_run",
    "complete_operational_run",
    "create_gradient_audit",
    "fit",
    "fit_model_b_baseline",
    "finalize_gradient_audit",
    "load_checkpoint",
    "move_optimizer_state_to_device",
    "run_model_b_epoch",
    "run_contrastive_epoch",
    "record_run_failure",
    "resume_from_checkpoint",
    "save_checkpoint",
    "train_model_b_pipeline",
    "train_contrastive_pipeline",
    "training_step",
    "ContrastiveBatchOutput",
    "forward_contrastive_batch",
]
