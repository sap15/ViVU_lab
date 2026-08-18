from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.config import load_config
from gnn_siamese.training.checkpointing import (
    build_resume_compatibility_payload,
    validate_resume_compatibility,
)
from gnn_siamese.training import train_model_b_pipeline
from tests.model_b_test_utils import (
    build_model_b_config,
    create_multi_pair_hdf5,
    write_schema_json,
)


def _model_a_config(tmp_path: Path) -> dict:
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt_companion.hdf5"
    schema_path = tmp_path / "schema.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)

    declared = load_config("configs/model_a_smoke.yaml")
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        tmp_path / "split.json",
    )
    config["model"] = deepcopy(declared["model"])
    config["augmentation_pair_a"] = deepcopy(declared["augmentation_pair_a"])
    config["loss"].update(deepcopy(declared["loss"]))
    config.setdefault("outputs", {}).update(deepcopy(declared["outputs"]))
    config["__config_path__"] = str(tmp_path / "model_a_smoke.yaml")
    config["paths"].update(
        {
            "mutants_hdf5": str(mutant_path),
            "wt_companion_hdf5": str(wt_path),
            "sample_schema": str(schema_path),
        }
    )
    config["split"].update(
        {"persist_path": str(tmp_path / "split.json"), "allow_create": True}
    )
    config["outputs"]["root_dir"] = str(tmp_path / "runs")
    return config


def test_model_a_smoke_config_builds_only_model_a_trainable_route(tmp_path: Path) -> None:
    declared_smoke = load_config("configs/model_a_smoke.yaml")
    declared_pilot = load_config("configs/model_a_pilot.yaml")
    assert declared_smoke["split"]["allow_create"] is True
    assert declared_pilot["split"]["allow_create"] is False
    assert declared_pilot["split"]["persist_path"] == "splits/leave_position_out_seed_42.json"
    assert declared_pilot["split"]["validation_fraction"] == pytest.approx(0.15)
    assert declared_pilot["split"]["test_fraction"] == pytest.approx(0.15)
    assert declared_pilot["model"]["projection_instance"]["enabled"] is False

    pipeline = build_training_pipeline(_model_a_config(tmp_path))

    assert pipeline.model.architecture_name == "model_a_nodal_multiscale_pair"
    assert pipeline.config["loss"]["main"] == "nt_xent"
    assert pipeline.config["loss"]["lambda_wt"] == 0.0
    assert pipeline.config["loss"]["lambda_delta"] == 0.0
    assert pipeline.config["outputs"]["model_name"] == "model_a_nodal_multiscale_pair"
    assert pipeline.model.pair_fusion is not pipeline.model.projection_pair_a


def test_resume_rejects_model_b_checkpoint_for_model_a(tmp_path: Path) -> None:
    config = _model_a_config(tmp_path)
    pipeline = build_training_pipeline(config)
    expected = build_resume_compatibility_payload(
        config=config,
        dataset=pipeline.dataset,
        split_bundle=pipeline.split_bundle,
        optimizer=pipeline.optimizer,
        scheduler=pipeline.scheduler,
        schema=pipeline.schema,
    )
    incompatible = deepcopy(expected)
    incompatible["architecture"]["name"] = "model_b_graph_level_relational"

    with pytest.raises(ValueError, match="architecture"):
        validate_resume_compatibility({"compatibility": incompatible}, expected)


def test_model_a_manifest_records_scale_and_effective_seed_contract(tmp_path: Path) -> None:
    config = _model_a_config(tmp_path)
    config["training"]["epochs"] = 1
    config["training"]["batch_size"] = 4
    config["augmentation_pair_a"]["allowed_feature_names"] = ["bsa", "res_mass"]
    config["augmentation_pair_a"]["feature_mask_probability"] = 0.0
    config["loss"]["false_negative_mask"].update(
        {"min_valid_negatives": 0, "min_valid_negative_fraction": 0.0}
    )
    pipeline = build_training_pipeline(config)
    output = train_model_b_pipeline(pipeline, config_path=config["__config_path__"])
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))

    assert manifest["configuration"]["model"]["active_scales"] == [
        "mutation",
        "local",
        "global",
    ]
    assert manifest["configuration"]["model"]["domain_scale_status"] == "DISABLED_DECLARED"
    assert manifest["configuration"]["seed_bundle"]["split"] == 123
    assert manifest["configuration"]["seed_bundle"]["run"] == 123
    assert manifest["configuration"]["seed_bundle"]["dataloader_effective"] == 123
