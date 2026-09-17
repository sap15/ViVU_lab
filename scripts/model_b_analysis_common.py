"""Read-only utilities shared by the Model B multiseed analysis scripts.

These helpers deliberately do not import the training loop or create an
optimizer.  They are kept in ``scripts`` so the scientific package remains
unchanged by this post-training audit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.training.a9_contract import A9ContractError, _audit_representation


RUN_SEEDS = (11, 23, 37, 41, 53)
REPRESENTATIONS = ("r_delta", "z_delta", "z_instance_pair")
EXPECTED_SHAPES = {
    "r_delta": (483, 640),
    "z_delta": (483, 128),
    "z_instance_pair": (483, 64),
}


class ModelBAnalysisError(RuntimeError):
    """Raised when a read-only analysis identity gate fails."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelBAnalysisError(f"Unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ModelBAnalysisError(f"JSON object expected: {path}")
    return value


def require_accepted_a9(run_dir: str | Path) -> dict[str, Any]:
    """Validate the immutable A9 acceptance record, without rewriting it."""
    path = Path(run_dir) / "a9_acceptance.json"
    acceptance = load_json(path)
    if acceptance.get("status") != "accepted":
        raise ModelBAnalysisError(f"A9 acceptance is not accepted: {path}")
    if acceptance.get("automatic_rejection_checks") != "PASS":
        raise ModelBAnalysisError(f"A9 automatic rejection checks did not pass: {path}")
    return acceptance


def canonical_ids(ids: Sequence[Any]) -> list[str]:
    values = [str(item) for item in ids]
    if len(values) != 483:
        raise ModelBAnalysisError(f"Expected 483 variant IDs, got {len(values)}.")
    if len(set(values)) != len(values):
        raise ModelBAnalysisError("Duplicate variant IDs are not permitted.")
    return values


def validate_matrix(name: str, value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.shape != EXPECTED_SHAPES[name]:
        raise ModelBAnalysisError(
            f"{name} has shape {matrix.shape}; expected {EXPECTED_SHAPES[name]}.")
    if not np.isfinite(matrix).all():
        raise ModelBAnalysisError(f"{name} contains NaN or Inf.")
    return matrix


def official_a9_metrics(name: str, matrix: np.ndarray) -> dict[str, Any]:
    """Use the exact A9 implementation, rather than a parallel definition."""
    return _audit_representation(name, matrix)


def validate_reextraction(
    acceptance: Mapping[str, Any], representations: Mapping[str, np.ndarray], *,
    tolerance: float = 1.0e-8,
) -> list[dict[str, Any]]:
    expected_all = acceptance.get("representation_metrics")
    if not isinstance(expected_all, Mapping):
        raise ModelBAnalysisError("A9 acceptance has no representation_metrics.")
    rows: list[dict[str, Any]] = []
    for name in REPRESENTATIONS:
        expected = expected_all.get(name)
        if not isinstance(expected, Mapping):
            raise ModelBAnalysisError(f"A9 acceptance has no metrics for {name}.")
        actual = official_a9_metrics(name, representations[name])
        for field in ("finite", "exact_unique_rows", "effective_rank", "pc1_variance_fraction"):
            if field not in expected:
                raise ModelBAnalysisError(f"A9 metric {name}.{field} is absent.")
            before, after = expected[field], actual[field]
            if isinstance(before, bool):
                difference, passed = float(before != after), before == after
            else:
                difference = abs(float(before) - float(after))
                passed = difference <= tolerance
            rows.append({
                "representation": name, "metric": field, "acceptance_value": before,
                "reextracted_value": after, "absolute_difference": difference,
                "tolerance": tolerance, "status": "PASS" if passed else "FAIL",
            })
    if any(row["status"] != "PASS" for row in rows):
        raise ModelBAnalysisError("REPRESENTATION_REEXTRACTION_GATE=FAIL")
    return rows


def write_tsv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
