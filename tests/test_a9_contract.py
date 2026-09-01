from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from gnn_siamese.config import load_config
from gnn_siamese.training.a9_contract import (
    A9ContractError,
    PRODUCTIVE_RUN_SEEDS,
    validate_a9_config_pair,
    validate_a9_run_acceptance,
    validate_separate_output_roots,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_A = REPO_ROOT / "configs" / "model_a_a9.yaml"
CONFIG_B = REPO_ROOT / "configs" / "model_b_a9.yaml"
FROZEN_SPLIT = REPO_ROOT / "splits" / "leave_position_out_seed_42.json"


def test_a9_configs_freeze_the_same_productive_protocol() -> None:
    model_a = load_config(CONFIG_A)
    model_b = load_config(CONFIG_B)

    result = validate_a9_config_pair(model_a, model_b)

    assert result["status"] == "PASS"
    assert result["run_seeds"] == list(PRODUCTIVE_RUN_SEEDS)
    assert model_a["model"]["projection_instance"]["enabled"] is False
    assert model_a["model"]["active_scales"] == ["mutation", "local", "global"]
    assert model_a["features"]["domains"]["enabled"] is False
    assert model_a["loss"]["lambda_wt"] == model_a["loss"]["lambda_delta"] == 0.0
    assert model_b["loss"]["lambda_wt"] == model_b["loss"]["lambda_delta"] == 0.0
    assert model_b["model"]["projection_instance"]["enabled"] is False
    assert model_b["model"]["mlp_delta"]["enabled"] is True
    assert model_b["model"]["projection_pair"]["enabled"] is True
    assert model_b["model"]["projection_pair"]["input"] == "z_delta"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("split.allow_create", True),
        ("split.seed", 11),
        ("training.epochs", 99),
        ("training.scheduler", "none"),
        ("training.early_stopping.patience", 14),
        ("training.checkpointing.monitor", "train_loss"),
        ("loss.temperature", 0.1),
        ("loss.false_negative_mask.mode", "none"),
        ("features.node_groups", ["structure"]),
        ("a9.run_seeds", [11]),
        ("outputs.root_dir", "runs/model_a/runs/model_a_a9"),
    ],
)
def test_a9_config_pair_rejects_protocol_drift(path: str, value: object) -> None:
    model_a = load_config(CONFIG_A)
    model_b = load_config(CONFIG_B)
    target = model_b if path != "outputs.root_dir" else model_b
    _set(target, path, value)

    with pytest.raises(A9ContractError, match="contract mismatch"):
        validate_a9_config_pair(model_a, model_b)


def test_a9_early_stopping_is_accepted_and_normalized() -> None:
    config = load_config(CONFIG_A)
    early = config["training"]["early_stopping"]
    assert early == {
        "enabled": True,
        "monitor": "validation_loss",
        "mode": "min",
        "patience": 15,
        "min_delta": 0.0,
    }


def test_a9_output_layout_has_one_root_namespace(tmp_path: Path) -> None:
    from gnn_siamese.utils.manifest import build_run_layout

    model_a = load_config(CONFIG_A)
    model_b = load_config(CONFIG_B)
    assert model_a["outputs"]["root_dir"] == "runs/model_a_a9"
    assert model_b["outputs"]["root_dir"] == "runs/model_b_a9"
    a_layout = build_run_layout(
        root_dir=model_a["outputs"]["root_dir"],
        model_name=model_a["outputs"]["model_name"], run_id="fixed",
    )
    b_layout = build_run_layout(
        root_dir=model_b["outputs"]["root_dir"],
        model_name=model_b["outputs"]["model_name"], run_id="fixed",
    )
    assert a_layout.run_dir == Path("runs/model_a_a9/model_a_nodal_multiscale_pair/run_fixed")
    assert b_layout.run_dir == Path("runs/model_b_a9/model_b_graph_level_relational/run_fixed")
    validate_separate_output_roots(tmp_path / "a", tmp_path / "b")


def test_a9_output_roots_reject_equal_nested_and_symlink_equivalent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    for left, right in ((real, real), (real, real / "child"), (alias, real)):
        with pytest.raises(A9ContractError, match="physically separate"):
            validate_separate_output_roots(left, right)


def _valid_run(tmp_path: Path) -> tuple[Path, dict, dict[str, np.ndarray]]:
    config = load_config(CONFIG_A)
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "best.pt").write_bytes(b"best")
    (run_dir / "checkpoints" / "last.pt").write_bytes(b"last")
    split = json.loads(FROZEN_SPLIT.read_text(encoding="utf-8"))
    (run_dir / "split.json").write_text(json.dumps(split), encoding="utf-8")
    module_record = {
            "status": "trained",
            "has_nan_or_inf": False,
            "none_gradient_fraction": 0.0,
            "zero_gradient_fraction": 0.0,
            "mean_gradient_norm": 1.0,
            "max_gradient_norm": 1.5,
            "relative_weight_change": 0.1,
            "parameter_count": 10,
            "trainable_parameter_count": 10,
            "optimizer_group": "default",
            "connected_losses": ["nt_xent"],
    }
    expected_modules = ["encoder", "node_delta_block", "pair_fusion", "projection_pair_a"]
    gradient = {name: dict(module_record) for name in expected_modules}
    (run_dir / "gradient_audit.json").write_text(json.dumps(gradient), encoding="utf-8")
    manifest = {
        "status": "completed",
        "architecture": "model_a_nodal_multiscale_pair",
        "data": {
            "hdf5_content_fingerprint": {
                "files": [
                    {"role": "mutants", "digest": config["a9"]["expected_hdf5_fingerprints"]["mutants"]},
                    {"role": "wt_companion", "digest": config["a9"]["expected_hdf5_fingerprints"]["wt_companion"]},
                ],
                "combined": {"digest": config["a9"]["expected_hdf5_fingerprints"]["combined"]}
            },
            "split_fingerprint": config["a9"]["expected_split_fingerprint"],
            "inventory": {
                "biological_variants": 483, "native_wt_controls": 1,
                "native_wt_control_ids": ["PKP2_WT"],
                "native_wt_control_policy": {
                    "used_for_training": False, "used_for_validation": False, "used_for_test_loss": False,
                },
            },
        },
        "configuration": {
            "seed": 11, "seed_bundle": {"split": 42},
            "node_feature_names": ["bsa"], "edge_feature_names": ["distance"], "graph_feature_names": [],
        },
        "training": {
            "epochs_planned": 100, "epochs_completed": 10, "stopped_early": True,
            "stop_reason": "early_stopping", "optimizer": "adamw", "optimizer_class": "AdamW",
            "scheduler_details": {"class": "CosineAnnealingLR", "config": {"T_max": 100, "step_semantics": "once_per_epoch"}},
            "learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 4,
            "early_stopping": {"enabled": True, "monitor": "validation_loss", "mode": "min", "patience": 15, "min_delta": 0.0, "bad_epochs": 15, "best_metric": 0.5},
            "last_epoch_metrics": {"loss": 1.0},
        },
        "losses": {"main": "nt_xent", "temperature": 0.2, "false_negative_mask": {"enabled": True, "mode": "same_position", "same_position": True, "strict": True, "min_valid_negatives": 1, "min_valid_negative_fraction": 0.0}},
        "gradient_audit_identity": {"expected_active_modules": expected_modules},
        "artifacts": {"best_checkpoint": "checkpoints/best.pt", "last_checkpoint": "checkpoints/last.pt"},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = np.arange(483 * 4, dtype=np.float32).reshape(483, 4)
    representations = {
        "h_pair_delta": base,
        "z_delta_pair": base + 1,
        "z_instance_pair": base + 2,
    }
    return run_dir, config, representations


def test_a9_acceptance_passes_and_records_diagnostics_without_geometric_thresholds(
    tmp_path: Path,
) -> None:
    run_dir, config, representations = _valid_run(tmp_path)

    result = validate_a9_run_acceptance(run_dir, config, representations=representations)

    assert result["status"] == "accepted"
    for metrics in result["representation_metrics"].values():
        assert metrics["finite"] is True
        assert metrics["exact_total_collapse"] is False
        assert "effective_rank" in metrics
        assert "pc1_variance_fraction" in metrics


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("status", "not completed"),
        ("architecture", "architecture"),
        ("seed", "run seed contract"),
        ("dataset_fingerprint", "HDF5"),
        ("mutant_fingerprint", "mutants"),
        ("split_fingerprint", "split fingerprint"),
        ("split_artifact", "canonical fingerprint"),
        ("native_wt", "Native WT"),
        ("leakage", "leakage"),
        ("module_gradient", "non-zero gradient"),
        ("module_missing_gradient", "parameters without gradients"),
        ("module_nonfinite_metric", "non-finite audit metrics"),
        ("module_omitted", "missing active modules"),
        ("module_incomplete", "incomplete"),
        ("module_weights", "weights did not change"),
        ("nonfinite_loss", "NaN or Inf"),
        ("nonfinite_representation", "contains NaN or Inf"),
        ("exact_collapse", "exact total collapse"),
    ],
)
def test_a9_acceptance_rejects_objective_failures(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    run_dir, config, representations = _valid_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gradient_path = run_dir / "gradient_audit.json"
    gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
    if mutation == "status":
        manifest["status"] = "failed"
    elif mutation == "architecture":
        manifest["architecture"] = "model_b"
    elif mutation == "seed":
        manifest["configuration"]["seed"] = 42
    elif mutation == "dataset_fingerprint":
        manifest["data"]["hdf5_content_fingerprint"]["combined"]["digest"] = "0" * 64
    elif mutation == "mutant_fingerprint":
        manifest["data"]["hdf5_content_fingerprint"]["files"][0]["digest"] = "0" * 64
    elif mutation == "split_fingerprint":
        manifest["data"]["split_fingerprint"] = "0" * 64
    elif mutation == "split_artifact":
        split_path = run_dir / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["assignments"][0]["variant_id"] = "changed-but-non-wt"
        split_path.write_text(json.dumps(split), encoding="utf-8")
    elif mutation == "native_wt":
        split_path = run_dir / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["assignments"][0]["variant_id"] = "PKP2_WT"
        split_path.write_text(json.dumps(split), encoding="utf-8")
    elif mutation == "leakage":
        split_path = run_dir / "split.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_position = next(x["position"] for x in split["assignments"] if x["partition"] == "train")
        next(x for x in split["assignments"] if x["partition"] == "validation")["position"] = train_position
        split_path.write_text(json.dumps(split), encoding="utf-8")
    elif mutation == "module_gradient":
        gradient["encoder"]["mean_gradient_norm"] = 0.0
    elif mutation == "module_missing_gradient":
        gradient["encoder"]["none_gradient_fraction"] = 0.5
    elif mutation == "module_nonfinite_metric":
        gradient["encoder"]["mean_gradient_norm"] = float("nan")
        gradient["encoder"]["has_nan_or_inf"] = False
    elif mutation == "module_omitted":
        del gradient["pair_fusion"]
    elif mutation == "module_incomplete":
        del gradient["encoder"]["relative_weight_change"]
    elif mutation == "module_weights":
        gradient["encoder"]["relative_weight_change"] = 0.0
    elif mutation == "nonfinite_loss":
        manifest["training"]["last_epoch_metrics"]["loss"] = float("nan")
    elif mutation == "nonfinite_representation":
        representations["z_delta_pair"][0, 0] = np.nan
    elif mutation == "exact_collapse":
        representations["z_instance_pair"][:] = 1.0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gradient_path.write_text(json.dumps(gradient), encoding="utf-8")

    with pytest.raises(A9ContractError, match=match):
        validate_a9_run_acceptance(run_dir, config, representations=representations)


@pytest.mark.parametrize("artifact", ["run_manifest.json", "gradient_audit.json", "checkpoints/best.pt", "checkpoints/last.pt"])
def test_a9_acceptance_rejects_missing_required_artifacts(tmp_path: Path, artifact: str) -> None:
    run_dir, config, representations = _valid_run(tmp_path)
    (run_dir / artifact).unlink()

    with pytest.raises(A9ContractError, match="missing"):
        validate_a9_run_acceptance(run_dir, config, representations=representations)


def test_a9_acceptance_rejects_incomplete_manifest(tmp_path: Path) -> None:
    run_dir, config, representations = _valid_run(tmp_path)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["data"]["inventory"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(A9ContractError, match="Required A9 field is missing"):
        validate_a9_run_acceptance(run_dir, config, representations=representations)


def test_a9_acceptance_supports_only_relational_b_outputs_and_modules(tmp_path: Path) -> None:
    run_dir, _config_a, representations_a = _valid_run(tmp_path)
    config_b = load_config(CONFIG_B)
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["architecture"] = "model_b_graph_level_relational"
    expected = ["encoder", "pooling_fusion", "mlp_delta", "projection_pair"]
    manifest["gradient_audit_identity"]["expected_active_modules"] = expected
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    gradient_path = run_dir / "gradient_audit.json"
    source = json.loads(gradient_path.read_text(encoding="utf-8"))["encoder"]
    gradient_path.write_text(json.dumps({name: source for name in expected}), encoding="utf-8")
    representations = {
        "r_delta": representations_a["h_pair_delta"],
        "z_delta": representations_a["z_delta_pair"],
        "z_instance_pair": representations_a["z_instance_pair"],
    }
    result = validate_a9_run_acceptance(run_dir, config_b, representations=representations)
    assert result["architecture"] == "model_b_graph_level_relational"


def _set(mapping: dict, dotted_path: str, value: object) -> None:
    current = mapping
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value
