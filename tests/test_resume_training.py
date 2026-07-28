from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("h5py")
pytest.importorskip("torch_geometric")

from gnn_siamese.builders import build_training_pipeline
from gnn_siamese.training import load_checkpoint, train_model_b_pipeline
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def _make_deterministic_config(
    tmp_path: Path,
    *,
    epochs: int,
    root_name: str,
    mutant_path: Path | None = None,
    wt_path: Path | None = None,
    schema_path: Path | None = None,
    split_path: Path | None = None,
) -> dict:
    mutant_path = mutant_path or (tmp_path / f"{root_name}_mutants.hdf5")
    wt_path = wt_path or (tmp_path / f"{root_name}_wt.hdf5")
    schema_path = schema_path or (tmp_path / f"{root_name}_schema.json")
    split_path = split_path or (tmp_path / f"{root_name}_split.json")
    if not mutant_path.exists() or not wt_path.exists():
        create_multi_pair_hdf5(mutant_path, wt_path)
    if not schema_path.exists():
        write_schema_json(schema_path)
    config = build_model_b_config(
        mutant_path,
        wt_path,
        schema_path,
        split_path,
        overrides={
            "training": {
                "epochs": epochs,
                "batch_size": 2,
                "scheduler": "step",
                "num_workers": 0,
            },
            "augmentation": {
                "feature_dropout": {"enabled": True, "probability": 0.2},
                "feature_jitter": {"enabled": True, "std": 0.05},
                "edge_dropout": {"enabled": True, "probability": 0.1},
            },
            "outputs": {
                "root_dir": str(tmp_path / "runs"),
                "model_name": root_name,
                "manifest_filename": "run_manifest.json",
                "resolved_config_filename": "config_resolved.yaml",
                "gradient_audit_filename": "gradient_audit.json",
                "directories": {"checkpoints": "checkpoints"},
            },
        },
    )
    config["__config_path__"] = str(tmp_path / f"{root_name}.yaml")
    return config


def test_continuous_training_matches_interrupted_and_resumed_training(tmp_path: Path) -> None:
    shared_mutant_path = tmp_path / "shared_mutants.hdf5"
    shared_wt_path = tmp_path / "shared_wt.hdf5"
    shared_schema_path = tmp_path / "shared_schema.json"
    shared_split_path = tmp_path / "shared_split.json"
    base_config = _make_deterministic_config(
        tmp_path,
        epochs=2,
        root_name="continuous",
        mutant_path=shared_mutant_path,
        wt_path=shared_wt_path,
        schema_path=shared_schema_path,
        split_path=shared_split_path,
    )
    continuous_pipeline = build_training_pipeline(base_config)
    continuous_output = train_model_b_pipeline(continuous_pipeline, config_path=base_config["__config_path__"])
    continuous_last = load_checkpoint(continuous_output.last_checkpoint_path)

    first_half_config = _make_deterministic_config(
        tmp_path,
        epochs=1,
        root_name="first_half",
        mutant_path=shared_mutant_path,
        wt_path=shared_wt_path,
        schema_path=shared_schema_path,
        split_path=shared_split_path,
    )
    first_half_pipeline = build_training_pipeline(first_half_config)
    first_half_output = train_model_b_pipeline(first_half_pipeline, config_path=first_half_config["__config_path__"])

    resumed_config = _make_deterministic_config(
        tmp_path,
        epochs=2,
        root_name="resumed",
        mutant_path=shared_mutant_path,
        wt_path=shared_wt_path,
        schema_path=shared_schema_path,
        split_path=shared_split_path,
    )
    resumed_pipeline = build_training_pipeline(resumed_config)
    resumed_output = train_model_b_pipeline(
        resumed_pipeline,
        config_path=resumed_config["__config_path__"],
        resume_from=first_half_output.last_checkpoint_path,
    )
    resumed_last = load_checkpoint(resumed_output.last_checkpoint_path)

    assert continuous_output.epochs_completed == 2
    assert continuous_output.epochs_run_this_invocation == 2
    assert resumed_output.epochs_completed == 2
    assert resumed_output.epochs_run_this_invocation == 1
    assert resumed_output.resumed_from == first_half_output.last_checkpoint_path
    assert continuous_last["epoch_completed"] == 2
    assert resumed_last["epoch_completed"] == 2
    assert continuous_last["global_step"] == resumed_last["global_step"]
    assert continuous_last["best_metric"] == pytest.approx(resumed_last["best_metric"], rel=0.0, abs=1.0e-8)
    assert continuous_last["scheduler_state_dict"] == resumed_last["scheduler_state_dict"]
    assert continuous_last["optimizer_state_dict"]["param_groups"] == resumed_last["optimizer_state_dict"]["param_groups"]
    assert continuous_last["augmenter_state"] == resumed_last["augmenter_state"]
    assert torch.equal(
        continuous_last["data_loader_state"]["train_generator_state"],
        resumed_last["data_loader_state"]["train_generator_state"],
    )

    continuous_lr = continuous_last["optimizer_state_dict"]["param_groups"][0]["lr"]
    resumed_lr = resumed_last["optimizer_state_dict"]["param_groups"][0]["lr"]
    assert continuous_lr == pytest.approx(resumed_lr, rel=0.0, abs=1.0e-12)

    for key, tensor in continuous_last["model_state_dict"].items():
        assert torch.equal(tensor, resumed_last["model_state_dict"][key]), key
    for param_id, state in continuous_last["optimizer_state_dict"]["state"].items():
        resumed_state = resumed_last["optimizer_state_dict"]["state"][param_id]
        assert state.keys() == resumed_state.keys()
        for state_key, state_value in state.items():
            resumed_value = resumed_state[state_key]
            if isinstance(state_value, torch.Tensor):
                assert torch.equal(state_value, resumed_value)
            else:
                assert state_value == resumed_value

    continuous_manifest = json.loads(Path(continuous_output.manifest_path).read_text(encoding="utf-8"))
    resumed_manifest = json.loads(Path(resumed_output.manifest_path).read_text(encoding="utf-8"))
    assert continuous_manifest["status"] == "completed"
    assert resumed_manifest["status"] == "completed"
    assert resumed_manifest["training"]["resume_from"] == first_half_output.last_checkpoint_path
    assert resumed_manifest["training"]["epochs_completed"] == 2
    assert resumed_manifest["training"]["epochs_run_this_invocation"] == 1
    assert resumed_manifest["training"]["global_step"] == resumed_last["global_step"]
    assert resumed_manifest["training"]["best_metric"] == pytest.approx(resumed_last["best_metric"], rel=0.0, abs=1.0e-8)
