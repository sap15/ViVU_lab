from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("torch_geometric")

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.training import create_gradient_audit, finalize_gradient_audit, train_model_b_pipeline
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


class AuditHarness(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        siamese_model = torch.nn.Module()
        siamese_model.shared_encoder = torch.nn.Linear(4, 4, bias=False)
        siamese_model.projection_instance = torch.nn.Linear(4, 4, bias=False)
        siamese_model.projection_pair = torch.nn.Linear(4, 4, bias=False)
        siamese_model.relational_module = torch.nn.Module()
        siamese_model.relational_module.mlp_delta = torch.nn.Linear(4, 4, bias=False)
        self.siamese_model = siamese_model


def test_gradient_audit_classifies_trained_inactive_failed_and_not_applicable() -> None:
    model = AuditHarness()
    optimizer = torch.optim.SGD(
        list(model.siamese_model.shared_encoder.parameters()) + list(model.siamese_model.projection_instance.parameters()),
        lr=0.1,
    )
    trackers = create_gradient_audit(
        model,
        optimizer,
        loss_weights={"nt_xent": 1.0, "relative_wt": 0.0, "delta": 0.0},
    )

    for parameter in model.siamese_model.shared_encoder.parameters():
        parameter.grad = torch.ones_like(parameter)
    for parameter in model.siamese_model.projection_instance.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for tracker in trackers.values():
        tracker.record_step()

    audit = finalize_gradient_audit(trackers)
    assert audit["encoder"]["status"] == "trained"
    assert audit["projection_instance"]["status"] == "trained"
    assert audit["projection_pair"]["status"] == "inactive"
    assert audit["mlp_delta"]["status"] == "inactive"
    assert audit["bio_head"]["status"] == "not_applicable"


def test_gradient_audit_detects_failed_module_and_nan_inf() -> None:
    model = AuditHarness()
    optimizer = torch.optim.SGD(model.siamese_model.projection_instance.parameters(), lr=0.1)
    trackers = create_gradient_audit(
        model,
        optimizer,
        loss_weights={"nt_xent": 1.0, "relative_wt": 0.0, "delta": 1.0},
    )

    for parameter in model.siamese_model.projection_instance.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    trackers["projection_instance"].record_step()
    trackers["mlp_delta"].record_step()
    audit = finalize_gradient_audit(trackers)
    assert audit["projection_instance"]["has_nan_or_inf"] is True
    assert audit["projection_instance"]["status"] == "failed"
    assert audit["mlp_delta"]["status"] == "failed"


def test_mlp_delta_is_not_marked_learned_when_lambda_delta_is_zero(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants_zero.hdf5"
    wt_path = tmp_path / "wt_zero.hdf5"
    schema_path = tmp_path / "schema_zero.json"
    split_path = tmp_path / "split_zero.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "model": {"mlp_delta": {"enabled": True}},
            "loss": {"lambda_delta": 0.0, "delta": {"mode": "variance"}},
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": "delta_zero",
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "config_resolved.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "delta_zero.yaml")
    pipeline = build_training_pipeline(config)
    output = train_model_b_pipeline(pipeline, config_path=config["__config_path__"])
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))
    assert manifest["z_delta_learned"] is False
    assert manifest["modules"]["mlp_delta"]["status"] == "inactive"


def test_mlp_delta_is_marked_learned_when_connected_and_trained(tmp_path: Path) -> None:
    mutant_path = tmp_path / "mutants_trained.hdf5"
    wt_path = tmp_path / "wt_trained.hdf5"
    schema_path = tmp_path / "schema_trained.json"
    split_path = tmp_path / "split_trained.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "model": {"mlp_delta": {"enabled": True}},
            "loss": {"lambda_delta": 0.5, "delta": {"mode": "variance"}},
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": "delta_trained",
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "config_resolved.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / "delta_trained.yaml")
    pipeline = build_training_pipeline(config)
    output = train_model_b_pipeline(pipeline, config_path=config["__config_path__"])
    manifest = json.loads(Path(output.manifest_path).read_text(encoding="utf-8"))
    assert manifest["z_delta_learned"] is True
    assert manifest["modules"]["mlp_delta"]["status"] == "trained"
