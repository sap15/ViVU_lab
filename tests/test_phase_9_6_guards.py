from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gnn_siamese.losses import __all__ as losses_public_api
from gnn_siamese.losses.contrastive import NTXentLoss


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TRAINING_PACKAGE_PATH = REPO_ROOT / "src" / "gnn_siamese" / "training"


def test_false_negative_masking_is_not_operational_yet_in_base_config() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "false_negative_mask:" in config_text
    assert "enabled: false" in config_text
    assert "mode: none" in config_text


def test_relative_wt_is_not_operational_yet_in_base_config() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "relative_wt:" in config_text
    assert "mode: none" in config_text
    assert "lambda_wt: 0.0" in config_text
    assert "use_wt_as_strong_positive: false" in config_text


def test_l_delta_is_not_operational_yet_in_base_config() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "delta:" in config_text
    assert "enabled: false" in config_text
    assert "lambda_delta: 0.0" in config_text
    assert "mlp_delta:" in config_text
    assert "require_explicit_loss: true" in config_text


def test_no_real_l_total_or_training_package_exists_yet() -> None:
    assert TRAINING_PACKAGE_PATH.exists() is False
    assert importlib.util.find_spec("gnn_siamese.training") is None


def test_losses_public_api_exposes_only_nt_xent_baseline() -> None:
    assert losses_public_api == ["NTXentLoss", "NTXentLossOutput"]


def test_nt_xent_docstring_explicitly_excludes_phase_9_6_extensions() -> None:
    doc = NTXentLoss.__doc__
    assert doc is not None
    assert "false-negative masking" in doc
    assert "L_relative_WT" in doc
    assert "relational losses" in doc
    assert "intentionally excluded from this baseline" in doc


def test_masked_reconstruction_remains_declared_but_disabled() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "reconstruction:" in config_text
    assert "priority: short_term_after_reproducible_baseline" in config_text
    assert "enabled: false" in config_text
