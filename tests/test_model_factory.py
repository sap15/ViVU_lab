from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn_siamese.builders import BuilderError, build_training_pipeline
from gnn_siamese.config import load_config
from gnn_siamese.models import (
    ModelANodalMultiscalePair,
    ModelAProjectionHead,
    ModelBContrastiveBaseline,
)
from gnn_siamese.training import (
    bootstrap_operational_run,
    forward_contrastive_batch,
    load_checkpoint,
    save_checkpoint,
    train_contrastive_pipeline,
)
from gnn_siamese.training.checkpointing import (
    build_resume_compatibility_payload,
    resume_from_checkpoint,
)
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def _config(tmp_path: Path, architecture: str) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["model"]["architecture"] = architecture
    config["__config_path__"] = str(tmp_path / "config.yaml")
    if architecture == "model_a_nodal_multiscale_pair":
        config["model"].update({
            "hidden_dim": 8,
            "graph_dim": 6,
            "num_layers": 1,
            "dropout": 0.0,
            "active_scales": ["mutation", "local", "global"],
            "encoder_a": {"edge_mlp_hidden_dim": 7, "fusion_hidden_dim": 9},
            "node_delta": {"hidden_dim": 11, "output_dim": 8, "dropout": 0.0},
            "pair_fusion": {"enabled": True, "input": "h_pair_delta", "hidden_dim": 13, "output_dim": 10, "dropout": 0.0},
            "projection_pair_a": {"enabled": True, "input": "z_delta_pair", "hidden_dim": 9, "output_dim": 5},
        })
        config["augmentation_pair_a"] = {
            "enabled": True,
            "feature_mask_probability": 0.5,
            "allowed_feature_names": ["bsa"],
            "masked_value": 0.0,
        }
        config["loss"]["temperature"] = 0.2
        config["loss"]["false_negative_mask"] = {
            "enabled": True,
            "mode": "same_position",
            "same_position": True,
            "strict": True,
            "min_valid_negatives": 1,
            "min_valid_negative_fraction": 0.0,
        }
    return config


def test_factory_builds_distinct_final_a_and_b_classes(tmp_path: Path) -> None:
    a_config = _config(tmp_path / "a", "model_a_nodal_multiscale_pair")
    a_yaml = tmp_path / "a" / "model_a.yaml"
    a_yaml.write_text(
        yaml.safe_dump(
            {key: value for key, value in a_config.items() if key != "__config_path__"},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    loaded_a_config = load_config(a_yaml)
    loaded_a_config["__config_path__"] = str(a_yaml)
    a = build_training_pipeline(loaded_a_config)
    b = build_training_pipeline(_config(tmp_path / "b", "model_b_graph_level_relational"))

    assert isinstance(a.model, ModelANodalMultiscalePair)
    assert isinstance(b.model, ModelBContrastiveBaseline)
    assert type(a.model) is not type(b.model)
    assert b.model.architecture_name == "model_b"
    assert a.model.two_view_model.one_view_model.multiscale_pooling.enabled_scales == (
        "mutation", "local", "global"
    )


def test_historical_model_b_selector_keeps_historical_type_and_contract(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(_config(tmp_path, "model_b"))
    assert isinstance(pipeline.model, ModelBContrastiveBaseline)
    assert pipeline.model.architecture_name == "model_b"
    batch = next(iter(pipeline.dataloaders.train_loader))
    output = forward_contrastive_batch(
        pipeline.model,
        batch,
        augmenter=pipeline.augmenter,
        loss_fn=pipeline.loss_fn,
        run_seed=123,
        epoch=0,
    )
    assert output.architecture == "model_b_graph_level_relational"
    assert output.loss is None
    assert {"z1", "z2"}.issubset(output.to_dict())


def test_model_a_trainer_interface_preserves_all_semantic_outputs(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(_config(tmp_path, "model_a_nodal_multiscale_pair"))
    batch = next(iter(pipeline.dataloaders.train_loader))
    output = forward_contrastive_batch(
        pipeline.model,
        batch,
        augmenter=pipeline.augmenter,
        loss_fn=pipeline.loss_fn,
        run_seed=123,
        epoch=0,
    )
    payload = output.to_dict()
    assert output.architecture == "model_a_nodal_multiscale_pair"
    assert torch.isfinite(output.loss)
    assert {
        "h_pair_delta", "z_delta_pair", "z_instance_pair",
        "h_pair_delta_view1", "h_pair_delta_view2",
        "z_delta_pair_view1", "z_delta_pair_view2",
        "z_instance_pair_view1", "z_instance_pair_view2",
    }.issubset(payload)
    assert isinstance(pipeline.model.projection_pair_a, ModelAProjectionHead)
    assert pipeline.model.projection_pair_a.z_delta_pair_dim == 10
    assert pipeline.model.projection_pair_a.projection_dim == 5
    one_view = output.model_output.two_view_output.view1
    assert one_view.z_delta_local is not None
    assert one_view.h_local_MUT is not None
    assert one_view.h_local_WT is not None
    assert one_view.h_local_delta is not None
    assert payload["active_scales"] == ("mutation", "local", "global")
    assert payload["scale_order"] == ("mutation", "local", "domain", "global")
    assert payload["z_delta_local_view1"] is one_view.z_delta_local
    expected_local_counts = torch.diff(batch.local_alignment_ptr)
    torch.testing.assert_close(
        one_view.scale_counts["local"]["MUT"], expected_local_counts
    )
    torch.testing.assert_close(
        one_view.scale_counts["local"]["WT"], expected_local_counts
    )
    torch.testing.assert_close(
        one_view.scale_counts["local"]["delta"], expected_local_counts
    )
    expected = torch.cat(
        (one_view.z_delta_mutation, one_view.z_delta_local, one_view.z_delta_global),
        dim=-1,
    )
    torch.testing.assert_close(one_view.h_pair_delta, expected)
    assert one_view.pair_dimension == 3 * 5 * 8


def test_model_a_local_can_be_disabled_and_domain_requires_real_annotation(tmp_path: Path) -> None:
    without_local = _config(tmp_path / "without_local", "model_a_nodal_multiscale_pair")
    without_local["model"]["active_scales"] = ["mutation", "global"]
    pipeline = build_training_pipeline(without_local)
    batch = next(iter(pipeline.dataloaders.train_loader))
    output = forward_contrastive_batch(
        pipeline.model,
        batch,
        augmenter=pipeline.augmenter,
        loss_fn=pipeline.loss_fn,
        run_seed=123,
        epoch=0,
    )
    assert output.model_output.active_scales == ("mutation", "global")
    assert output.model_output.z_delta_local_view1 is None
    assert output.model_output.two_view_output.view1.pair_dimension == 2 * 5 * 8

    with_domain = _config(tmp_path / "with_domain", "model_a_nodal_multiscale_pair")
    with_domain["model"]["active_scales"] = ["mutation", "local", "domain", "global"]
    with pytest.raises(BuilderError, match="domain must remain disabled"):
        build_training_pipeline(with_domain)


def test_model_a_yaml_dimensions_really_change_pair_fusion_and_projection(tmp_path: Path) -> None:
    base = _config(tmp_path / "base", "model_a_nodal_multiscale_pair")
    changed = deepcopy(_config(tmp_path / "changed", "model_a_nodal_multiscale_pair"))
    changed["model"]["pair_fusion"].update({"hidden_dim": 17, "output_dim": 12})
    changed["model"]["projection_pair_a"].update({"hidden_dim": 14, "output_dim": 7})
    base_model = build_training_pipeline(base).model
    changed_model = build_training_pipeline(changed).model

    assert base_model.pair_fusion.hidden_dim == 13
    assert changed_model.pair_fusion.hidden_dim == 17
    assert base_model.pair_fusion.output_dim == 10
    assert changed_model.pair_fusion.output_dim == 12
    assert base_model.projection_pair_a.hidden_dim == 9
    assert changed_model.projection_pair_a.hidden_dim == 14
    assert base_model.projection_pair_a.projection_dim == 5
    assert changed_model.projection_pair_a.projection_dim == 7
    assert changed_model.projection_pair_a.z_delta_pair_dim == 12


def test_unknown_architecture_is_informative_and_manifest_records_a(tmp_path: Path) -> None:
    unknown = _config(tmp_path / "unknown", "not_a_model")
    with pytest.raises(BuilderError, match="supported architectures"):
        build_training_pipeline(unknown)

    config = _config(tmp_path / "manifest", "model_a_nodal_multiscale_pair")
    config.setdefault("outputs", {})["root_dir"] = str(tmp_path / "runs")
    config["outputs"]["model_name"] = "model_a_nodal_multiscale_pair"
    context = bootstrap_operational_run(config, config_path=config["__config_path__"])
    payload = json.loads(context.layout.manifest_path.read_text(encoding="utf-8"))
    assert payload["architecture"] == "model_a_nodal_multiscale_pair"
    assert payload["architecture_details"]["pair_fusion_semantics"] == {
        "input": "h_pair_delta", "output": "z_delta_pair"
    }
    assert payload["architecture_details"]["projection_pair_a_class"] == "ModelAProjectionHead"
    assert payload["architecture_details"]["projection_pair_a_semantics"] == {
        "input": "z_delta_pair", "output": "z_instance_pair"
    }
    resolved = payload["configuration"]["resolved_config"]["model"]
    assert resolved["pair_fusion"]["input"] == "h_pair_delta"
    assert resolved["projection_pair_a"]["output_dim"] == 5

    pipeline = build_training_pipeline(config)
    checkpoint_path = tmp_path / "model_a.pt"
    save_checkpoint(
        checkpoint_path,
        model=pipeline.model,
        optimizer=pipeline.optimizer,
        scheduler=pipeline.scheduler,
        epoch_completed=0,
        global_step=0,
        best_metric=None,
        train_metrics={},
        validation_metrics={},
        resolved_config=config,
        seed=123,
        split_id="split.json",
        split_fingerprint="split",
        dataset_fingerprint="dataset",
        dataset_id={},
        compatibility={},
        run_id="a7",
    )
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["architecture"] == "model_a_nodal_multiscale_pair"


def test_model_a_uses_shared_productive_train_validation_audit_and_resume(tmp_path: Path) -> None:
    config = _config(tmp_path / "productive_a", "model_a_nodal_multiscale_pair")
    config["training"]["epochs"] = 1
    config.setdefault("outputs", {}).update({
        "root_dir": str(tmp_path / "runs"),
        "model_name": "shared_productive_a",
    })
    pipeline = build_training_pipeline(config)
    before = {
        name: [parameter.detach().clone() for parameter in module.parameters()]
        for name, module in {
            "encoder": pipeline.model.two_view_model.one_view_model.shared_encoder,
            "node_delta_block": pipeline.model.two_view_model.one_view_model.node_delta_block,
            "pair_fusion": pipeline.model.pair_fusion,
            "projection_pair_a": pipeline.model.projection_pair_a,
        }.items()
    }

    output = train_contrastive_pipeline(
        pipeline,
        config_path=config["__config_path__"],
    )

    assert output.epochs_completed == 1
    assert output.train_history[0].phase == "train"
    assert output.train_history[0].gradients_enabled is True
    assert output.validation_history[0].phase == "validation"
    assert output.validation_history[0].used_eval_mode is True
    assert torch.isfinite(torch.tensor(output.final_train_loss))
    assert torch.isfinite(torch.tensor(output.final_validation_loss))
    for name, module in {
        "encoder": pipeline.model.two_view_model.one_view_model.shared_encoder,
        "node_delta_block": pipeline.model.two_view_model.one_view_model.node_delta_block,
        "pair_fusion": pipeline.model.pair_fusion,
        "projection_pair_a": pipeline.model.projection_pair_a,
    }.items():
        assert any(
            not torch.equal(old, new.detach())
            for old, new in zip(before[name], module.parameters(), strict=True)
        ), name

    audit = json.loads(Path(output.gradient_audit_path).read_text(encoding="utf-8"))
    assert set(audit) == {"encoder", "node_delta_block", "pair_fusion", "projection_pair_a"}
    for record in audit.values():
        assert record["status"] == "trained"
        assert record["optimizer_group"] is not None
        assert record["connected_losses"] == ["nt_xent"]
        assert record["mean_gradient_norm"] > 0.0
        assert record["min_gradient_norm"] >= 0.0
        assert record["none_gradient_fraction"] == 0.0
        assert record["relative_weight_change"] > 0.0
        assert record["has_nan_or_inf"] is False

    checkpoint = load_checkpoint(output.last_checkpoint_path)
    assert checkpoint["architecture"] == "model_a_nodal_multiscale_pair"
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))
    assert manifest["architecture"] == "model_a_nodal_multiscale_pair"
    assert manifest["modules"]["projection_pair_a"]["status"] == "trained"
    assert manifest["configuration"]["augmentations"] == config["augmentation_pair_a"]
    assert manifest["configuration"]["model"]["active_scales"] == config["model"]["active_scales"]
    assert manifest["configuration"]["model"]["encoder_a"] == config["model"]["encoder_a"]
    assert manifest["configuration"]["model"]["node_delta"] == config["model"]["node_delta"]
    assert manifest["losses"]["active_components"] == ["nt_xent"]
    assert manifest["data"]["dataset_fingerprint"]
    assert manifest["data"]["split_fingerprint"]
    assert manifest["configuration"]["seed"] == config["project"]["seed"]

    resume_config = deepcopy(config)
    resume_config["training"]["epochs"] = 2
    resumed_pipeline = build_training_pipeline(resume_config)
    resumed = train_contrastive_pipeline(
        resumed_pipeline,
        config_path=resume_config["__config_path__"],
        resume_from=output.last_checkpoint_path,
    )
    assert resumed.epochs_completed == 2
    assert resumed.epochs_run_this_invocation == 1
    assert resumed.resumed_from == output.last_checkpoint_path
    assert load_checkpoint(resumed.last_checkpoint_path)["epoch_completed"] == 2


@pytest.mark.parametrize("loss_name", ["lambda_wt", "lambda_delta"])
def test_model_a_rejects_unimplemented_auxiliary_losses_before_training(
    tmp_path: Path, loss_name: str
) -> None:
    config = _config(tmp_path / loss_name, "model_a_nodal_multiscale_pair")
    config["loss"][loss_name] = 0.5
    with pytest.raises(BuilderError, match=rf"loss\.{loss_name} must be 0"):
        build_training_pipeline(config)


def _model_a_compatibility(pipeline: object) -> dict:
    return build_resume_compatibility_payload(
        config=pipeline.config,
        dataset=pipeline.dataset,
        split_bundle=pipeline.split_bundle,
        optimizer=pipeline.optimizer,
        scheduler=pipeline.scheduler,
        schema=pipeline.schema,
    )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("active_scales", lambda config: config["model"].__setitem__("active_scales", ["mutation", "global"])),
        ("encoder_a", lambda config: config["model"]["encoder_a"].__setitem__("fusion_hidden_dim", 10)),
        ("node_delta", lambda config: config["model"]["node_delta"].__setitem__("activation", "gelu")),
        ("pair_fusion", lambda config: config["model"]["pair_fusion"].__setitem__("dropout", 0.2)),
        ("projection_pair_a", lambda config: config["model"]["projection_pair_a"].__setitem__("hidden_dim", 10)),
        ("augmentation_pair_a", lambda config: config["augmentation_pair_a"].__setitem__("feature_mask_probability", 0.25)),
    ],
)
def test_model_a_resume_rejects_each_scientific_signature_change(
    tmp_path: Path, field: str, mutation: object
) -> None:
    base = _config(tmp_path / "base", "model_a_nodal_multiscale_pair")
    base["training"]["epochs"] = 1
    base.setdefault("outputs", {}).update({"root_dir": str(tmp_path / "runs"), "model_name": "source"})
    source = build_training_pipeline(base)
    source_output = train_contrastive_pipeline(source, config_path=base["__config_path__"])

    changed = deepcopy(base)
    mutation(changed)
    changed["outputs"]["model_name"] = f"changed_{field}"
    target = build_training_pipeline(changed)
    with pytest.raises(ValueError, match="resume incompatibility"):
        resume_from_checkpoint(
            source_output.last_checkpoint_path,
            model=target.model,
            optimizer=target.optimizer,
            scheduler=target.scheduler,
            expected_compatibility=_model_a_compatibility(target),
        )


def test_model_a_without_model_name_uses_architecture_artifact_identity(tmp_path: Path) -> None:
    config = _config(tmp_path / "identity", "model_a_nodal_multiscale_pair")
    config.setdefault("outputs", {})["root_dir"] = str(tmp_path / "runs")
    config["outputs"].pop("model_name", None)
    context = bootstrap_operational_run(config, config_path=config["__config_path__"])
    assert context.layout.run_dir.parent.name == "model_a_nodal_multiscale_pair"
    payload = json.loads(context.layout.manifest_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "model_a_nodal_multiscale_pair"


def test_model_a_gradient_audit_rejects_optimizer_missing_trainable_parameters(tmp_path: Path) -> None:
    from gnn_siamese.training import create_gradient_audit

    pipeline = build_training_pipeline(_config(tmp_path, "model_a_nodal_multiscale_pair"))
    incomplete_optimizer = torch.optim.SGD(pipeline.model.projection_pair_a.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="optimizer must contain every trainable parameter"):
        create_gradient_audit(
            pipeline.model,
            incomplete_optimizer,
            loss_weights={"nt_xent": 1.0, "relative_wt": 0.0, "delta": 0.0},
        )


def test_model_a_continuous_training_equals_interrupted_resume(tmp_path: Path) -> None:
    shared = _config(tmp_path / "shared", "model_a_nodal_multiscale_pair")
    shared["training"].update({"epochs": 2, "scheduler": "none", "num_workers": 0})
    shared["model"].update({"dropout": 0.0})
    shared["model"]["pair_fusion"]["dropout"] = 0.0
    shared.setdefault("outputs", {})["root_dir"] = str(tmp_path / "runs")

    continuous_config = deepcopy(shared)
    continuous_config["outputs"]["model_name"] = "continuous_a"
    continuous = build_training_pipeline(continuous_config)
    continuous_output = train_contrastive_pipeline(continuous, config_path=continuous_config["__config_path__"])
    continuous_last = load_checkpoint(continuous_output.last_checkpoint_path)

    first_config = deepcopy(shared)
    first_config["training"]["epochs"] = 1
    first_config["outputs"]["model_name"] = "first_a"
    first = build_training_pipeline(first_config)
    first_output = train_contrastive_pipeline(first, config_path=first_config["__config_path__"])

    resumed_config = deepcopy(shared)
    resumed_config["outputs"]["model_name"] = "resumed_a"
    resumed = build_training_pipeline(resumed_config)
    resumed_output = train_contrastive_pipeline(
        resumed,
        config_path=resumed_config["__config_path__"],
        resume_from=first_output.last_checkpoint_path,
    )
    resumed_last = load_checkpoint(resumed_output.last_checkpoint_path)

    assert continuous_last["epoch_completed"] == resumed_last["epoch_completed"] == 2
    assert continuous_last["global_step"] == resumed_last["global_step"]
    assert continuous_last["scheduler_state_dict"] == resumed_last["scheduler_state_dict"]
    assert continuous_last["train_metrics"] == resumed_last["train_metrics"]
    assert continuous_last["validation_metrics"] == resumed_last["validation_metrics"]
    assert continuous_last["augmenter_state"] == resumed_last["augmenter_state"]
    continuous_rng = continuous_last["rng_state"]
    resumed_rng = resumed_last["rng_state"]
    assert continuous_rng["python_random_state"] == resumed_rng["python_random_state"]
    for expected, actual in zip(
        continuous_rng["numpy_random_state"], resumed_rng["numpy_random_state"], strict=True
    ):
        if isinstance(expected, np.ndarray):
            assert np.array_equal(expected, actual)
        else:
            assert expected == actual
    assert torch.equal(continuous_rng["torch_cpu_rng_state"], resumed_rng["torch_cpu_rng_state"])
    assert continuous_rng["torch_cuda_rng_state"] == resumed_rng["torch_cuda_rng_state"]
    assert torch.equal(
        continuous_last["data_loader_state"]["train_generator_state"],
        resumed_last["data_loader_state"]["train_generator_state"],
    )
    for key, value in continuous_last["model_state_dict"].items():
        assert torch.equal(value, resumed_last["model_state_dict"][key]), key
    assert continuous_last["optimizer_state_dict"]["param_groups"] == resumed_last["optimizer_state_dict"]["param_groups"]
    for parameter_id, state in continuous_last["optimizer_state_dict"]["state"].items():
        resumed_state = resumed_last["optimizer_state_dict"]["state"][parameter_id]
        for state_name, state_value in state.items():
            other = resumed_state[state_name]
            if isinstance(state_value, torch.Tensor):
                assert torch.equal(state_value, other)
            else:
                assert state_value == other
