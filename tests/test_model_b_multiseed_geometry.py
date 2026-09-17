from __future__ import annotations

import numpy as np
import pytest

from scripts.model_b_analysis_common import ModelBAnalysisError, canonical_ids, official_a9_metrics, validate_matrix, validate_reextraction
from scripts.model_b_export_best_representations import _accepted_run_dirs
from scripts.model_b_multiseed_geometry import correlation, knn_indices, linear_cka, load_validated_embeddings


def test_canonical_ids_rejects_duplicates_and_wrong_count() -> None:
    with pytest.raises(ModelBAnalysisError, match="Duplicate"):
        canonical_ids(["x"] * 483)
    with pytest.raises(ModelBAnalysisError, match="483"):
        canonical_ids(["x"])


def test_validate_matrix_rejects_bad_shape_and_nonfinite() -> None:
    with pytest.raises(ModelBAnalysisError, match="shape"):
        validate_matrix("r_delta", np.zeros((483, 1)))
    value = np.zeros((483, 640)); value[0, 0] = np.nan
    with pytest.raises(ModelBAnalysisError, match="NaN"):
        validate_matrix("r_delta", value)


def test_distance_correlation_knn_jaccard_and_cka() -> None:
    left = np.asarray([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    assert correlation(np.arange(6), np.arange(6)) == pytest.approx(1.0)
    assert correlation(np.arange(6), np.arange(6), spearman=True) == pytest.approx(1.0)
    neighbors = knn_indices(left, 2)
    assert neighbors.shape == (4, 2)
    assert linear_cka(left, left) == pytest.approx(1.0)


def test_embeddings_missing_seed_fails(tmp_path) -> None:
    with pytest.raises(ModelBAnalysisError, match="Missing"):
        load_validated_embeddings(tmp_path)


def test_fast_b6_requires_exactly_one_run_per_seed(tmp_path) -> None:
    rows = [{"seed": seed, "run_dir": str(tmp_path / str(seed))} for seed in (11, 23, 37, 41, 53)]
    assert set(_accepted_run_dirs({"runs": rows}, tmp_path)) == {11, 23, 37, 41, 53}
    with pytest.raises(ModelBAnalysisError, match="exactly one"):
        _accepted_run_dirs({"runs": rows + [rows[0]]}, tmp_path)


def test_reextraction_requires_official_acceptance_metrics() -> None:
    matrices = {
        "r_delta": np.arange(483 * 640, dtype=np.float32).reshape(483, 640),
        "z_delta": np.arange(483 * 128, dtype=np.float32).reshape(483, 128),
        "z_instance_pair": np.arange(483 * 64, dtype=np.float32).reshape(483, 64),
    }
    acceptance = {"representation_metrics": {name: official_a9_metrics(name, value) for name, value in matrices.items()}}
    assert all(row["status"] == "PASS" for row in validate_reextraction(acceptance, matrices))
    acceptance["representation_metrics"]["r_delta"]["effective_rank"] = 0.0
    with pytest.raises(ModelBAnalysisError, match="REEXTRACTION"):
        validate_reextraction(acceptance, matrices)
