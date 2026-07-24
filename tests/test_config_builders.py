from __future__ import annotations

from pathlib import Path

import h5py
import pytest
import yaml

torch = pytest.importorskip("torch")

from gnn_siamese.builders import BuilderError, build_training_pipeline
from gnn_siamese.config import load_config
from gnn_siamese.losses import NTXentLoss
from gnn_siamese.models import ModelBContrastiveBaseline
from gnn_siamese.data.augmentations import AugmentationConfigError
from gnn_siamese.data.dataset import MutWtPairDatasetError
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_load_config_extends_and_builder_consumes_changed_yaml_option(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)

    base_config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    child_config = {
        "extends": "base.yaml",
        "model": {"projection_instance": {"output_dim": 7}},
        "training": {"optimizer": "adam"},
    }
    base_yaml = tmp_path / "base.yaml"
    child_yaml = tmp_path / "child.yaml"
    _write_yaml(base_yaml, base_config)
    _write_yaml(child_yaml, child_config)

    loaded = load_config(child_yaml)
    loaded["__config_path__"] = str(child_yaml)
    pipeline = build_training_pipeline(loaded)

    assert isinstance(pipeline.model, ModelBContrastiveBaseline)
    assert pipeline.model.architecture_name == "model_b"
    assert pipeline.model.siamese_model.projection_instance.output_dim == 7
    assert isinstance(pipeline.loss_fn, NTXentLoss)
    assert pipeline.total_loss_assembler.weights["nt_xent"] == pytest.approx(1.0)
    assert pipeline.total_loss_assembler.weights["relative_wt"] == pytest.approx(0.0)
    assert pipeline.total_loss_assembler.weights["delta"] == pytest.approx(0.0)
    assert isinstance(pipeline.optimizer, torch.optim.Adam)


def test_invalid_builder_configuration_is_rejected(tmp_path: Path) -> None:
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
        overrides={"model": {"architecture": "unknown_model"}},
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")

    with pytest.raises(BuilderError, match="Unsupported model.architecture"):
        build_training_pipeline(config)


def test_split_is_reloaded_from_persisted_json_without_re_shuffling(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)

    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")
    first_pipeline = build_training_pipeline(config)

    reloaded_config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={"split": {"seed": 999}},
    )
    reloaded_config["__config_path__"] = str(tmp_path / "config.yaml")
    second_pipeline = build_training_pipeline(reloaded_config)

    assert first_pipeline.split_bundle.created is True
    assert second_pipeline.split_bundle.created is False
    assert first_pipeline.split_bundle.split.assignments == second_pipeline.split_bundle.split.assignments


def test_builder_infers_final_node_input_dim_from_graph_x_and_counts_is_mutation(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)

    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")
    pipeline = build_training_pipeline(config)

    sample = pipeline.dataset[0]
    encoder = pipeline.model.siamese_model.shared_encoder

    assert pipeline.dataset.configured_node_feature_names == (
        "bsa",
        "res_mass",
        "diff_mass",
        "diff_charge",
        "diff_pI",
        "diff_size",
    )
    assert pipeline.dataset.node_feature_names == (
        "bsa",
        "res_mass",
        "diff_mass",
        "diff_charge",
        "diff_pI",
        "diff_size",
        "is_mutation",
    )
    assert sample.graph_mut.x.shape[1] == 7
    assert sample.graph_wt.x.shape[1] == 7
    assert pipeline.dataset.node_input_dim == sample.graph_mut.x.shape[1]
    assert pipeline.dataset.node_input_dim == sample.graph_wt.x.shape[1]
    assert encoder.input_projection.in_features == sample.graph_mut.x.shape[1]
    assert torch.equal(
        torch.as_tensor(sample.graph_mut.x[:, -1]),
        torch.as_tensor(sample.graph_mut.is_mutation),
    )


def test_builder_updates_input_projection_when_feature_selection_changes(tmp_path: Path) -> None:
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
            "features": {
                "node_groups": ["structure", "diff_bioq"],
                "biochemistry": {"enabled": False, "names": ["res_mass"]},
            },
            "augmentation": {
                "feature_dropout": {"allowed_feature_names": ["bsa"]},
                "feature_jitter": {"allowed_feature_names": ["bsa"]},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")

    pipeline = build_training_pipeline(config)
    sample = pipeline.dataset[0]

    assert pipeline.dataset.node_feature_names == (
        "bsa",
        "diff_mass",
        "diff_charge",
        "diff_pI",
        "diff_size",
        "is_mutation",
    )
    assert sample.graph_mut.x.shape[1] == 6
    assert sample.graph_wt.x.shape[1] == 6
    assert pipeline.dataset.node_input_dim == 6
    assert pipeline.model.siamese_model.shared_encoder.input_projection.in_features == 6


def test_builder_rejects_augmentations_that_reference_unknown_node_features(tmp_path: Path) -> None:
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
            "augmentation": {
                "feature_dropout": {"allowed_feature_names": ["missing_feature"]},
            }
        },
    )
    config["__config_path__"] = str(tmp_path / "config.yaml")

    with pytest.raises((AugmentationConfigError, BuilderError), match="Augmentation references unknown node features"):
        build_training_pipeline(config)


def test_builder_raises_clear_error_when_mutant_and_wt_final_dimensions_differ(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)

    wt_graph_key = "residue-srv:A:100:Glycine->Glycine:PKP2_WT"
    with h5py.File(wt_path, "a") as handle:
        node_group = handle[wt_graph_key]["node_features"]
        del node_group["diff_size"]
        node_group.create_dataset(
            "diff_size",
            data=[
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        )

    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")

    with pytest.raises(MutWtPairDatasetError) as exc_info:
        build_training_pipeline(config)

    message = str(exc_info.value)
    assert "mutant has 7 columns" in message
    assert "WT has 8 columns" in message
    assert "configured encoder selection has 6 columns" in message
    assert "Configured node features:" in message
    assert "Mutant final node features:" in message
    assert "WT final node features:" in message
