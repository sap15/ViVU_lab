from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from gnn_siamese.losses import __all__ as losses_public_api
from gnn_siamese.losses.contrastive import NTXentLoss
from gnn_siamese.losses.delta import DeltaLoss
from gnn_siamese.losses.relative_wt import RelativeWTLoss
from gnn_siamese.training import TotalLossAssembler, training_step


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
TRAINING_PACKAGE_PATH = REPO_ROOT / "src" / "gnn_siamese" / "training"


def test_false_negative_masking_is_declared_but_disabled_by_default_in_base_config() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "false_negative_mask:" in config_text
    assert "enabled: false" in config_text
    assert "mode: none" in config_text
    assert "mask_scope: within_batch_only" in config_text
    assert "does_not_replace_split: leave_neighborhood_out" in config_text


def test_relative_wt_is_declared_conservatively_in_base_config() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "relative_wt:" in config_text
    assert "mode: none" in config_text
    assert "lambda_wt: 0.0" in config_text
    assert "use_wt_as_strong_positive: false" in config_text
    assert "distance: euclidean" in config_text
    assert "stop_gradient: false" in config_text


def test_l_delta_is_declared_conservatively_in_base_config() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "delta:" in config_text
    assert "enabled: false" in config_text
    assert "lambda_delta: 0.0" in config_text
    assert "mlp_delta:" in config_text
    assert "require_explicit_loss: true" in config_text


def test_minimal_l_total_and_training_package_now_exist() -> None:
    assert TRAINING_PACKAGE_PATH.exists() is True
    assert importlib.util.find_spec("gnn_siamese.training") is not None
    assert TotalLossAssembler is not None
    assert training_step is not None


def test_losses_public_api_exposes_nt_xent_and_false_negative_masking_utilities() -> None:
    assert losses_public_api == [
        "DeltaLoss",
        "DeltaLossOutput",
        "FalseNegativeAnchorStats",
        "FalseNegativeBatchStats",
        "FalseNegativeMaskDegenerateError",
        "FalseNegativeMaskOutput",
        "NTXentLoss",
        "NTXentLossOutput",
        "RelativeWTLoss",
        "RelativeWTLossOutput",
        "build_false_negative_mask",
    ]


def test_nt_xent_docstring_keeps_relative_and_relational_extensions_out_of_scope() -> None:
    doc = NTXentLoss.__doc__
    assert doc is not None
    assert "L_relative_WT" in doc
    assert "relational losses" in doc
    assert "False-negative masking is optional" in doc


def test_relative_wt_loss_exists_without_making_wt_a_strong_positive() -> None:
    criterion = RelativeWTLoss(mode="none")

    assert criterion.mode == "none"
    assert criterion.stop_gradient_wt is False


def test_delta_loss_and_minimal_l_total_training_package_exist() -> None:
    criterion = DeltaLoss(mode="none")

    assert criterion.mode == "none"
    assert TRAINING_PACKAGE_PATH.exists() is True


def test_reconstruction_and_full_production_training_remain_pending() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "reconstruction:" in config_text
    assert "enabled: false" in config_text
    assert "scheduler: cosine" in config_text


def test_masked_reconstruction_remains_declared_but_disabled() -> None:
    config_text = BASE_CONFIG_PATH.read_text(encoding="utf-8")

    assert "reconstruction:" in config_text
    assert "priority: short_term_after_reproducible_baseline" in config_text
    assert "enabled: false" in config_text
