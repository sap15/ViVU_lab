from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path

import pytest

from gnn_siamese.config import load_config, save_config
from gnn_siamese.data.splits import LeavePositionOutSplit
from scripts.colab_preflight import (
    ColabPreflightError,
    MODEL_A_EXPECTED_PARTITIONS,
    MODEL_A_PILOT_SPLIT,
    validate_model_a_dataset_identity,
    validate_model_a_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks/model_a_colab_master.ipynb"
PILOT_CONFIG = REPO_ROOT / "configs/model_a_pilot.yaml"
FROZEN_SPLIT = REPO_ROOT / MODEL_A_PILOT_SPLIT


def _frozen_content_fingerprint() -> dict:
    payload = json.loads(FROZEN_SPLIT.read_text(encoding="utf-8"))
    return deepcopy(payload["audit_metadata"]["hdf5_content_fingerprint"])


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def test_model_a_notebook_is_valid_clean_and_python_parses() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_model_a_notebook_is_thin_sha_pinned_and_has_explicit_modes() -> None:
    source = _source()
    assert "configs/model_a_pilot.yaml" in source
    assert "model_a_nodal_multiscale_pair" in source
    assert "GIT_COMMIT = ''" in source
    assert "{40}" in source and "rev-parse','HEAD" in source
    assert "RUN_FRESH = False" in source and "RUN_RESUME = False" in source
    assert "RESUME_CHECKPOINT = ''" in source
    assert "build_train_command" in source and "scripts/train.py" not in source
    assert "/home/sartesero/" not in source
    for forbidden in ("class ModelA", "def training_step", "def train", "DeltaBlock(", "NTXentLoss("):
        assert forbidden not in source


def test_model_a_notebook_preserves_frozen_split_and_separate_run_namespace() -> None:
    source = _source()
    assert "split.persist_path" not in source
    assert "splits/leave_position_out_seed_42.json" in source
    assert "allow_create'] is False" in source
    assert "DRIVE_MODEL_A_RUNS_ROOT" in source
    assert "model_a_a8_5" in source
    assert "model_b" in source  # explicit guard, never an output destination


def test_model_a_pilot_contract_and_frozen_split_counts() -> None:
    config = load_config(PILOT_CONFIG)
    assert config["model"]["architecture"] == "model_a_nodal_multiscale_pair"
    assert config["training"]["epochs"] == 5
    assert config["training"]["batch_size"] == 4
    assert config["model"]["active_scales"] == ["mutation", "local", "global"]
    assert config["split"]["type"] == "leave_position_out"
    assert config["split"]["seed"] == 42
    assert config["split"]["persist_path"] == MODEL_A_PILOT_SPLIT
    assert config["split"]["allow_create"] is False
    assert config["loss"]["main"] == "nt_xent"
    assert config["loss"]["lambda_wt"] == config["loss"]["lambda_delta"] == 0.0
    split = LeavePositionOutSplit.load_json(FROZEN_SPLIT)
    counts = {key: len(value) for key, value in split.assignments_by_partition().items()}
    assert counts == MODEL_A_EXPECTED_PARTITIONS
    assert len(split.assignments) == 483
    assert all("PKP2_WT" not in assignment.variant_id for assignment in split.assignments)


def test_model_a_frozen_dataset_identity_passes_and_reports_expected_vs_actual() -> None:
    actual = _frozen_content_fingerprint()
    result = validate_model_a_dataset_identity(actual, frozen_split_path=FROZEN_SPLIT)

    assert result["dataset_identity_status"] == "PASS"
    assert result["expected_mutant_sha256"] == result["actual_mutant_sha256"]
    assert result["expected_wt_sha256"] == result["actual_wt_sha256"]
    assert result["expected_combined_fingerprint"] == result["actual_combined_fingerprint"]


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("mutants", "mutants"),
        ("wt_companion", "wt_companion"),
        ("combined", "combined"),
    ],
)
def test_model_a_frozen_dataset_identity_rejects_each_digest(
    target: str,
    match: str,
) -> None:
    actual = _frozen_content_fingerprint()
    if target == "combined":
        actual["combined"]["digest"] = "0" * 64
    else:
        next(record for record in actual["files"] if record["role"] == target)["digest"] = "0" * 64

    with pytest.raises(ColabPreflightError, match=match):
        validate_model_a_dataset_identity(actual, frozen_split_path=FROZEN_SPLIT)


def test_model_a_preflight_rejects_scientific_mismatch_before_loading_data(
    tmp_path: Path,
) -> None:
    payload = load_config(PILOT_CONFIG)
    payload["split"]["allow_create"] = True
    path = tmp_path / "bad.yaml"
    save_config(payload, path)
    with pytest.raises(ColabPreflightError, match="scientific contract mismatch"):
        validate_model_a_preflight(path, repo_root=REPO_ROOT, mode="fresh")


def test_model_a_preflight_requires_explicit_last_checkpoint_for_resume(tmp_path: Path) -> None:
    payload = load_config(PILOT_CONFIG)
    payload["training"]["epochs"] = 6
    path = tmp_path / "resume.yaml"
    save_config(payload, path)
    with pytest.raises(ColabPreflightError, match="explicit last.pt"):
        validate_model_a_preflight(path, repo_root=REPO_ROOT, mode="resume")
