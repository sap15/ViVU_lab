from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

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
