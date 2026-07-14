from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.training import fit_model_b_baseline, run_model_b_epoch
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def test_model_b_pipeline_runs_end_to_end_on_cpu_and_updates_parameters(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")

    pipeline = build_training_pipeline(config)
    first_batch = next(iter(pipeline.dataloaders.train_loader))
    assert pipeline.model.siamese_model.shared_encoder.input_projection.in_features == first_batch.graph_mut.x.shape[1]
    assert first_batch.graph_mut.x.shape[1] == first_batch.graph_wt.x.shape[1]
    view1_graph_mut, view2_graph_mut = pipeline.augmenter.create_two_views(first_batch.graph_mut)
    output = pipeline.model(
        view1_graph_mut=view1_graph_mut,
        view1_graph_wt=first_batch.graph_wt,
        view2_graph_mut=view2_graph_mut,
        view2_graph_wt=first_batch.graph_wt,
    )

    assert output.z1.shape[0] == first_batch.batch_size
    assert output.z2.shape == output.z1.shape
    loss_output = pipeline.loss_fn(output.z1, output.z2)
    assert loss_output.loss.ndim == 0
    assert torch.isfinite(loss_output.loss)

    parameter = next(pipeline.model.parameters())
    before = parameter.detach().clone()
    pipeline.optimizer.zero_grad(set_to_none=True)
    loss_output.loss.backward()
    assert any(
        param.grad is not None and param.grad.abs().sum().item() > 0.0
        for param in pipeline.model.parameters()
        if param.requires_grad
    )
    pipeline.optimizer.step()
    after = parameter.detach()
    assert not torch.allclose(before, after)


def test_validation_runs_in_eval_mode_without_gradients(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    train_epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.train_loader,
        pipeline.loss_fn,
        optimizer=pipeline.optimizer,
        device="cpu",
        augmenter=pipeline.augmenter,
    )
    validation_epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.validation_loader,
        pipeline.loss_fn,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert train_epoch.phase == "train"
    assert validation_epoch.phase == "validation"
    assert validation_epoch.used_eval_mode is True
    assert validation_epoch.gradients_enabled is False
    assert all(parameter.grad is None for parameter in pipeline.model.parameters())


def test_fit_model_b_baseline_completes_with_model_b_as_default_architecture(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    output = fit_model_b_baseline(
        pipeline.model,
        train_dataloader=pipeline.dataloaders.train_loader,
        validation_dataloader=pipeline.dataloaders.validation_loader,
        optimizer=pipeline.optimizer,
        loss_fn=pipeline.loss_fn,
        epochs=1,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert pipeline.model.architecture_name == "model_b"
    assert output.epochs_completed == 1
    assert output.final_train_loss == pytest.approx(output.train_history[-1].mean_loss)
    assert output.final_validation_loss == pytest.approx(output.validation_history[-1].mean_loss)
