from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import pytest
import torch

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.data import collate_mut_wt_pairs
from gnn_siamese.training import fit_model_b_baseline, forward_contrastive_batch, run_model_b_epoch
from gnn_siamese.training import loop as training_loop
from tests.model_b_test_utils import (
    build_model_b_config,
    create_multi_pair_hdf5,
    create_multi_pair_hdf5_with_variants,
    write_schema_json,
)


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
    assembled = pipeline.total_loss_assembler(
        z1=output.z1,
        z2=output.z2,
        h_mut=output.view1.h_mut,
        h_wt=output.view1.h_wt,
    )
    assert loss_output.loss.ndim == 0
    assert torch.isfinite(loss_output.loss)
    assert assembled.loss.item() == pytest.approx(loss_output.loss.item(), abs=1.0e-6)
    assert assembled.active_components == ["nt_xent"]
    assert assembled.components["relative_wt"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert assembled.components["delta"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert assembled.metrics["weighted_loss_relative_wt"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert assembled.metrics["weighted_loss_delta"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert assembled.metrics["loss_total"].item() == pytest.approx(assembled.metrics["weighted_loss_nt_xent"].item(), abs=1.0e-6)

    parameter = next(pipeline.model.parameters())
    before = parameter.detach().clone()
    pipeline.optimizer.zero_grad(set_to_none=True)
    assembled.loss.backward()
    assert any(
        param.grad is not None and param.grad.abs().sum().item() > 0.0
        for param in pipeline.model.parameters()
        if param.requires_grad
    )
    pipeline.optimizer.step()
    after = parameter.detach()
    assert not torch.allclose(before, after)


def test_model_b_shared_dispatch_preserves_previous_productive_contract(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants_parity.hdf5"
    wt_path = tmp_path / "wt_parity.hdf5"
    schema_path = tmp_path / "schema_parity.json"
    split_path = tmp_path / "split_parity.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "training": {"batch_size": 2, "num_workers": 0},
            "loss": {
                "false_negative_mask": {
                    "enabled": True,
                    "mode": "same_position",
                    "min_valid_negatives": 0,
                    "min_valid_negative_fraction": 0.0,
                }
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "parity.yaml")
    direct = build_training_pipeline(deepcopy(config))
    shared = build_training_pipeline(deepcopy(config))

    direct_batch = next(iter(direct.dataloaders.train_loader))
    shared_batch = next(iter(shared.dataloaders.train_loader))
    direct_views = training_loop._prepare_model_b_views(direct_batch, augmenter=direct.augmenter, device=torch.device("cpu"))
    captured_inputs: dict[str, object] = {}

    def capture_inputs(_module: object, _args: object, kwargs: dict[str, object]) -> None:
        captured_inputs.update(kwargs)

    hook = shared.model.register_forward_pre_hook(capture_inputs, with_kwargs=True)
    dispatched = forward_contrastive_batch(
        shared.model,
        shared_batch,
        augmenter=shared.augmenter,
        loss_fn=shared.loss_fn,
        run_seed=123,
        epoch=1,
    )
    hook.remove()
    assert dispatched.loss is None
    for name, expected in direct_views.items():
        actual = captured_inputs[name]
        torch.testing.assert_close(actual.x, expected.x, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual.edge_index, expected.edge_index, rtol=0.0, atol=0.0)
        torch.testing.assert_close(actual.edge_attr, expected.edge_attr, rtol=0.0, atol=0.0)

    direct_output = direct.model(**direct_views)
    dispatched_payload = dispatched.model_output.to_dict()
    direct_payload = direct_output.to_dict()
    for name in ("z1", "z2", "h_mut", "h_wt"):
        torch.testing.assert_close(
            dispatched_payload[name],
            direct_payload[name],
            rtol=0.0,
            atol=0.0,
        )
    direct_inputs = training_loop._build_model_b_loss_inputs(
        direct_batch, direct_output, loss_assembler=direct.total_loss_assembler, device=torch.device("cpu")
    )
    shared_inputs = training_loop._build_model_b_loss_inputs(
        shared_batch, dispatched.model_output, loss_assembler=shared.total_loss_assembler, device=torch.device("cpu")
    )
    torch.testing.assert_close(
        direct_inputs["mask_output"].negative_weights,
        shared_inputs["mask_output"].negative_weights,
        rtol=0.0,
        atol=0.0,
    )
    direct_loss = direct.total_loss_assembler(**direct_inputs)
    shared_loss = shared.total_loss_assembler(**shared_inputs)
    torch.testing.assert_close(direct_loss.loss, shared_loss.loss, rtol=0.0, atol=0.0)
    assert direct_loss.active_components == shared_loss.active_components
    for name in direct_loss.components:
        torch.testing.assert_close(direct_loss.components[name], shared_loss.components[name], rtol=0.0, atol=0.0)

    # Reset both complete pipelines so the following comparison starts from
    # identical model, optimizer, loader and augmenter states.
    direct = build_training_pipeline(deepcopy(config))
    shared = build_training_pipeline(deepcopy(config))

    def previous_epoch(pipeline: object, loader: object, *, optimizer: object | None) -> object:
        is_training = optimizer is not None
        pipeline.model.train(is_training)
        total = 0.0
        examples = 0
        batches = 0
        components: dict[str, float] = {}
        metrics: dict[str, list[float]] = {}
        metric_labels: dict[str, object] = {}
        active: set[str] = set()
        inactive: set[str] = set()
        skipped: set[str] = set()
        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch in loader:
                views = training_loop._prepare_model_b_views(batch, augmenter=pipeline.augmenter, device=torch.device("cpu"))
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                output = pipeline.model(**views)
                assembled = pipeline.total_loss_assembler(
                    **training_loop._build_model_b_loss_inputs(
                        batch, output, loss_assembler=pipeline.total_loss_assembler, device=torch.device("cpu")
                    )
                )
                if optimizer is not None:
                    assembled.loss.backward()
                    clip_norm = config["training"].get("gradient_clip_norm")
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in pipeline.model.parameters() if p.requires_grad and p.grad is not None],
                            max_norm=float(clip_norm),
                        )
                    optimizer.step()
                size = batch.batch_size
                total += float(assembled.loss.detach()) * size
                examples += size
                batches += 1
                for name, value in assembled.components.items():
                    components[name] = components.get(name, 0.0) + float(value.detach()) * size
                for name, value in assembled.metrics.items():
                    if isinstance(value, torch.Tensor) and value.ndim == 0:
                        metrics.setdefault(name, []).append(float(value.detach()))
                    elif isinstance(value, (int, float)):
                        metrics.setdefault(name, []).append(float(value))
                    elif isinstance(value, (str, bool)):
                        metric_labels[name] = value
                active.update(assembled.active_components)
                inactive.update(assembled.inactive_components)
                skipped.update(assembled.skipped_components)
        return {
            "loss": total / examples,
            "batches": batches,
            "examples": examples,
            "components": {name: value / examples for name, value in sorted(components.items())},
            "metrics": {
                **{name: sum(values) / len(values) for name, values in sorted(metrics.items())},
                **metric_labels,
            },
            "active": sorted(active),
            "inactive": sorted(inactive),
            "skipped": sorted(skipped),
        }

    expected_train = previous_epoch(direct, direct.dataloaders.train_loader, optimizer=direct.optimizer)
    actual_train = run_model_b_epoch(
        shared.model,
        shared.dataloaders.train_loader,
        shared.total_loss_assembler,
        optimizer=shared.optimizer,
        device="cpu",
        augmenter=shared.augmenter,
        gradient_clip_norm=config["training"].get("gradient_clip_norm"),
        contrastive_loss_fn=shared.loss_fn,
        run_seed=123,
        epoch=1,
    )
    assert actual_train.mean_loss == pytest.approx(expected_train["loss"], rel=0.0, abs=0.0)
    assert actual_train.component_means == expected_train["components"]
    assert actual_train.metrics == expected_train["metrics"]
    assert actual_train.active_components == expected_train["active"]
    assert actual_train.inactive_components == expected_train["inactive"]
    assert actual_train.skipped_components == expected_train["skipped"]
    for name, value in direct.model.state_dict().items():
        torch.testing.assert_close(value, shared.model.state_dict()[name], rtol=0.0, atol=0.0)
    for direct_parameter, shared_parameter in zip(direct.model.parameters(), shared.model.parameters(), strict=True):
        if direct_parameter.grad is None:
            assert shared_parameter.grad is None
        else:
            torch.testing.assert_close(direct_parameter.grad, shared_parameter.grad, rtol=0.0, atol=0.0)

    expected_validation = previous_epoch(direct, direct.dataloaders.validation_loader, optimizer=None)
    actual_validation = run_model_b_epoch(
        shared.model,
        shared.dataloaders.validation_loader,
        shared.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=shared.augmenter,
        contrastive_loss_fn=shared.loss_fn,
        run_seed=123,
        epoch=1,
    )
    assert actual_validation.mean_loss == pytest.approx(expected_validation["loss"], rel=0.0, abs=0.0)
    assert actual_validation.component_means == expected_validation["components"]
    assert actual_validation.metrics == expected_validation["metrics"]


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
        pipeline.total_loss_assembler,
        optimizer=pipeline.optimizer,
        device="cpu",
        augmenter=pipeline.augmenter,
    )
    validation_epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.validation_loader,
        pipeline.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert train_epoch.phase == "train"
    assert validation_epoch.phase == "validation"
    assert validation_epoch.used_eval_mode is True
    assert validation_epoch.gradients_enabled is False
    assert validation_epoch.active_components == ["nt_xent"]
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
        loss_fn=pipeline.total_loss_assembler,
        epochs=1,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert pipeline.model.architecture_name == "model_b"
    assert output.epochs_completed == 1
    assert output.final_train_loss == pytest.approx(output.train_history[-1].mean_loss)
    assert output.final_validation_loss == pytest.approx(output.validation_history[-1].mean_loss)
    assert output.train_history[-1].active_components == ["nt_xent"]


def test_same_position_mask_is_built_from_real_batch_metadata_and_reaches_nt_xent(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "false_negative_mask": {
                    "enabled": True,
                    "mode": "same_position",
                    "min_valid_negatives": 0,
                    "min_valid_negative_fraction": 0.0,
                }
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)
    duplicated_batch = collate_mut_wt_pairs([pipeline.dataset[0], pipeline.dataset[0]])

    epoch = run_model_b_epoch(
        pipeline.model,
        [duplicated_batch],
        pipeline.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert epoch.metrics["false_negative_mask_mode"] == "same_position"
    assert "false_negative_mask_mean_valid_negatives" in epoch.metrics
    assert "false_negative_mask_min_valid_fraction" in epoch.metrics
    assert epoch.active_components == ["nt_xent"]


def test_same_position_mask_uses_real_positions_and_modifies_negative_weights(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5_with_variants(
        mutant_path,
        wt_path,
        variants=[
            {"position": 100, "wt_full": "Glycine", "mut_full": "Aspartate", "wt_aa": "G", "mut_aa": "D"},
            {"position": 100, "wt_full": "Glycine", "mut_full": "Serine", "wt_aa": "G", "mut_aa": "S"},
            {"position": 220, "wt_full": "Alanine", "mut_full": "Valine", "wt_aa": "A", "mut_aa": "V"},
            {"position": 221, "wt_full": "Alanine", "mut_full": "Threonine", "wt_aa": "A", "mut_aa": "T"},
        ],
    )
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "false_negative_mask": {
                    "enabled": True,
                    "mode": "same_position",
                    "min_valid_negatives": 1,
                    "min_valid_negative_fraction": 0.25,
                    "strict": True,
                }
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)
    batch = collate_mut_wt_pairs([pipeline.dataset[0], pipeline.dataset[1], pipeline.dataset[2]])

    epoch = run_model_b_epoch(
        pipeline.model,
        [batch],
        pipeline.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert [item["position"] for item in batch.metadata] == [100, 100, 220]
    assert epoch.metrics["false_negative_mask_mode"] == "same_position"
    assert epoch.metrics["false_negative_mask_number_degenerate_anchors"] == pytest.approx(0.0, abs=1.0e-8)
    assert epoch.metrics["false_negative_mask_has_degenerate_anchors"] == pytest.approx(0.0, abs=1.0e-8)
    assert epoch.metrics["false_negative_mask_mean_valid_negatives"] == pytest.approx(8.0 / 3.0, abs=1.0e-8)
    assert epoch.metrics["false_negative_mask_min_valid_fraction"] == pytest.approx(0.5, abs=1.0e-8)


def test_relative_wt_loss_contributes_and_keeps_encoder_trainable(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "lambda_wt": 0.25,
                "relative_wt": {"mode": "margin", "margin": 0.0, "distance": "euclidean"},
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.train_loader,
        pipeline.total_loss_assembler,
        optimizer=pipeline.optimizer,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    encoder_grads = [
        parameter.grad
        for parameter in pipeline.model.siamese_model.shared_encoder.parameters()
        if parameter.requires_grad
    ]
    assert "relative_wt" in epoch.active_components
    assert epoch.component_means["relative_wt"] > 0.0
    assert epoch.metrics["weighted_loss_relative_wt"] > 0.0
    assert any(grad is not None and grad.abs().sum().item() > 0.0 for grad in encoder_grads)


def test_delta_loss_updates_mlp_delta_weights_when_enabled(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "model": {"mlp_delta": {"enabled": True}},
            "loss": {"lambda_delta": 0.2, "delta": {"mode": "consistency", "consistency_loss": "mse"}},
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    mlp_delta = pipeline.model.siamese_model.relational_module.mlp_delta
    assert mlp_delta is not None
    before = next(mlp_delta.parameters()).detach().clone()

    epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.train_loader,
        pipeline.total_loss_assembler,
        optimizer=pipeline.optimizer,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    grads = [parameter.grad for parameter in mlp_delta.parameters()]
    after = next(mlp_delta.parameters()).detach()
    assert "delta" in epoch.active_components
    assert epoch.component_means["delta"] > 0.0
    assert epoch.metrics["weighted_loss_delta"] > 0.0
    assert any(grad is not None and grad.abs().sum().item() > 0.0 for grad in grads)
    assert not torch.allclose(before, after)


def test_relative_wt_margin_mode_runs_without_target_name(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "lambda_wt": 0.25,
                "relative_wt": {"mode": "margin", "margin": 0.0},
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.train_loader,
        pipeline.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert epoch.metrics["relative_wt_mode"] == "margin"
    assert epoch.metrics["weighted_loss_relative_wt"] > 0.0


def test_relative_wt_ranking_fails_without_explicit_target_name_and_never_uses_model_severity(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "lambda_wt": 0.25,
                "relative_wt": {"mode": "ranking", "margin": 0.1},
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    with pytest.raises(ValueError, match="mode='ranking'.*requested_target_name=None.*available_metadata=.*position"):
        run_model_b_epoch(
            pipeline.model,
            pipeline.dataloaders.train_loader,
            pipeline.total_loss_assembler,
            optimizer=None,
            device="cpu",
            augmenter=pipeline.augmenter,
        )


def test_relative_wt_ranking_consumes_external_batch_target(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "lambda_wt": 0.25,
                "relative_wt": {"mode": "ranking", "margin": 0.1, "target_name": "external_rank"},
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)
    batch = next(iter(pipeline.dataloaders.train_loader))
    for index, item in enumerate(batch.metadata):
        item["external_rank"] = float(index)

    epoch = run_model_b_epoch(
        pipeline.model,
        [batch],
        pipeline.total_loss_assembler,
        optimizer=None,
        device="cpu",
        augmenter=pipeline.augmenter,
    )

    assert epoch.metrics["relative_wt_mode"] == "ranking"
    assert epoch.metrics["relative_wt_target_name"] == "external_rank"
    assert epoch.metrics["weighted_loss_relative_wt"] >= 0.0


def test_relative_wt_predictive_fails_when_target_is_missing(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "loss": {
                "lambda_wt": 0.25,
                "relative_wt": {"mode": "predictive", "target_name": "external_descriptor"},
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    with pytest.raises(
        ValueError,
        match="mode='predictive'.*requested_target_name='external_descriptor'.*available_metadata=.*position",
    ):
        run_model_b_epoch(
            pipeline.model,
            pipeline.dataloaders.train_loader,
            pipeline.total_loss_assembler,
            optimizer=None,
            device="cpu",
            augmenter=pipeline.augmenter,
        )
