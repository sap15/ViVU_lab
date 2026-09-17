#!/usr/bin/env python3
"""Descriptive, read-only geometric stability analysis of validated Model B NPZs."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_b_analysis_common import (
    REPRESENTATIONS, RUN_SEEDS, ModelBAnalysisError, canonical_ids, official_a9_metrics, validate_matrix, write_tsv,
)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, sufficient for deterministic Spearman without scipy."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, spearman: bool = False) -> float:
    left, right = np.asarray(left, dtype=float).ravel(), np.asarray(right, dtype=float).ravel()
    if left.shape != right.shape or left.size < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ModelBAnalysisError("Invalid vectors for correlation.")
    if spearman:
        left, right = _rank(left), _rank(right)
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        raise ModelBAnalysisError("Degenerate vector for correlation.")
    return float(np.corrcoef(left, right)[0, 1])


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, np.finfo(float).eps)
    return 1.0 - normalized @ normalized.T


def euclidean_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    standardized = (matrix - matrix.mean(axis=0, keepdims=True)) / np.maximum(matrix.std(axis=0, keepdims=True), np.finfo(float).eps)
    squares = np.sum(standardized * standardized, axis=1, keepdims=True)
    return np.sqrt(np.maximum(squares + squares.T - 2.0 * standardized @ standardized.T, 0.0))


def knn_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    distance = cosine_distance_matrix(matrix)
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1, kind="mergesort")[:, :k]


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64) - np.mean(left, axis=0, keepdims=True)
    y = np.asarray(right, dtype=np.float64) - np.mean(right, axis=0, keepdims=True)
    numerator = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    if denominator == 0.0:
        raise ModelBAnalysisError("Linear CKA is undefined for a collapsed matrix.")
    return float(numerator / denominator)


def _summary(rows: Iterable[Mapping[str, Any]], value: str, group: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in group), []).append(float(row[value]))
    result = []
    for key, values in sorted(buckets.items()):
        array = np.asarray(values)
        result.append(dict(zip(group, key)) | {"n_seed_pairs": int(array.size), "mean": float(array.mean()), "sd": float(array.std(ddof=0)), "median": float(np.median(array)), "min": float(array.min()), "max": float(array.max())})
    return result


def load_validated_embeddings(embeddings_dir: str | Path) -> dict[int, dict[str, np.ndarray]]:
    root = Path(embeddings_dir)
    data: dict[int, dict[str, np.ndarray]] = {}
    canonical: set[str] | None = None
    for seed in RUN_SEEDS:
        path = root / f"seed_{seed}_representations.npz"
        if not path.is_file():
            raise ModelBAnalysisError(f"Missing validated NPZ for seed {seed}: {path}")
        with np.load(path, allow_pickle=False) as npz:
            if not all(name in npz for name in ("variant_id", *REPRESENTATIONS)):
                raise ModelBAnalysisError(f"NPZ lacks required fields: {path}")
            ids = canonical_ids(npz["variant_id"])
            current = set(ids)
            if canonical is None:
                canonical = current
            elif current != canonical:
                raise ModelBAnalysisError("GEOMETRIC_ID_ALIGNMENT_GATE=FAIL")
            order = np.argsort(np.asarray(ids, dtype=str), kind="mergesort")
            data[seed] = {"variant_id": np.asarray(ids, dtype="U")[order]}
            for name in REPRESENTATIONS:
                data[seed][name] = validate_matrix(name, npz[name])[order].astype(np.float64, copy=False)
    return data


def geometry_analysis(*, embeddings_dir: str | Path, output_dir: str | Path, knn_k: Iterable[int] = (5, 10, 20)) -> dict[str, Any]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

    ks = tuple(int(k) for k in knn_k)
    if not ks or any(k <= 0 or k >= 483 for k in ks):
        raise ModelBAnalysisError("kNN k values must be between 1 and 482.")
    data, root = load_validated_embeddings(embeddings_dir), Path(output_dir)
    distance_rows: list[dict[str, Any]] = []
    knn_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    cluster_metrics: list[dict[str, Any]] = []
    cluster_pairs: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    upper = np.triu_indices(483, k=1)
    for name in REPRESENTATIONS:
        labels: dict[tuple[int, int], np.ndarray] = {}
        for left_seed, right_seed in itertools.combinations(RUN_SEEDS, 2):
            for distance_name, function in (("cosine_l2_normalized", cosine_distance_matrix), ("euclidean_feature_standardized", euclidean_distance_matrix)):
                left, right = function(data[left_seed][name])[upper], function(data[right_seed][name])[upper]
                distance_rows.append({"representation": name, "seed_left": left_seed, "seed_right": right_seed, "distance": distance_name, "pearson": correlation(left, right), "spearman": correlation(left, right, spearman=True)})
            cka_rows.append({"representation": name, "seed_left": left_seed, "seed_right": right_seed, "linear_cka": linear_cka(data[left_seed][name], data[right_seed][name])})
        for k in ks:
            neighbor_sets = {seed: knn_indices(data[seed][name], k) for seed in RUN_SEEDS}
            per_variant: dict[int, list[dict[str, Any]]] = {index: [] for index in range(483)}
            for left_seed, right_seed in itertools.combinations(RUN_SEEDS, 2):
                for index in range(483):
                    a, b = set(neighbor_sets[left_seed][index]), set(neighbor_sets[right_seed][index])
                    intersection = len(a & b)
                    row = {"representation": name, "k": k, "seed_left": left_seed, "seed_right": right_seed, "variant_id": data[left_seed]["variant_id"][index], "overlap_count": intersection, "overlap_fraction": intersection / k, "jaccard": intersection / len(a | b)}
                    knn_rows.append(row); per_variant[index].append(row)
            # retained locally then materialized below from all pairwise rows
        for k in range(2, 11):
            for seed in RUN_SEEDS:
                matrix = data[seed][name]
                labels[(seed, k)] = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(matrix)
                counts = np.bincount(labels[(seed, k)], minlength=k)
                cluster_metrics.append({"representation": name, "seed": seed, "k": k, "silhouette": float(silhouette_score(matrix, labels[(seed, k)])), "davies_bouldin": float(davies_bouldin_score(matrix, labels[(seed, k)])), "calinski_harabasz": float(calinski_harabasz_score(matrix, labels[(seed, k)])), "cluster_sizes": json.dumps(counts.tolist()), "singletons": int((counts == 1).sum())})
                assignments.extend({"representation": name, "seed": seed, "k": k, "variant_id": data[seed]["variant_id"][i], "cluster": int(label)} for i, label in enumerate(labels[(seed, k)]))
            for left_seed, right_seed in itertools.combinations(RUN_SEEDS, 2):
                cluster_pairs.append({"representation": name, "k": k, "seed_left": left_seed, "seed_right": right_seed, "ari": float(adjusted_rand_score(labels[(left_seed, k)], labels[(right_seed, k)])), "nmi": float(normalized_mutual_info_score(labels[(left_seed, k)], labels[(right_seed, k)]))})
    distance_summary = []
    for metric in ("pearson", "spearman"):
        distance_summary.extend([row | {"metric": metric} for row in _summary(distance_rows, metric, ("representation", "distance"))])
    knn_per_variant = []
    for name in REPRESENTATIONS:
        for k in ks:
            relevant = [row for row in knn_rows if row["representation"] == name and row["k"] == k]
            for variant in sorted({row["variant_id"] for row in relevant}):
                rows = [row for row in relevant if row["variant_id"] == variant]
                knn_per_variant.append({"representation": name, "k": k, "variant_id": variant, "mean_overlap_fraction": float(np.mean([row["overlap_fraction"] for row in rows])), "mean_jaccard": float(np.mean([row["jaccard"] for row in rows]))})
    knn_summary = _summary(knn_rows, "jaccard", ("representation", "k"))
    # Preserve the requested distribution statistics for all kNN metrics.
    for metric in ("overlap_count", "overlap_fraction"):
        knn_summary.extend([row | {"metric": metric} for row in _summary(knn_rows, metric, ("representation", "k"))])
    for row in knn_summary:
        row.setdefault("metric", "jaccard")
    write_tsv(root / "02_distance_stability" / "pairwise_distance_correlations.tsv", distance_rows)
    write_tsv(root / "02_distance_stability" / "distance_summary.tsv", distance_summary)
    write_tsv(root / "03_knn_stability" / "knn_pairwise.tsv", knn_rows)
    write_tsv(root / "03_knn_stability" / "knn_per_variant.tsv", knn_per_variant)
    write_tsv(root / "03_knn_stability" / "knn_summary.tsv", knn_summary)
    write_tsv(root / "04_geometry" / "linear_cka.tsv", cka_rows)
    write_tsv(root / "05_clustering" / "clustering_metrics.tsv", cluster_metrics)
    write_tsv(root / "05_clustering" / "cluster_pairwise_ari_nmi.tsv", cluster_pairs)
    write_tsv(root / "05_clustering" / "cluster_assignments.tsv", assignments)
    summary: list[dict[str, Any]] = []
    for name in REPRESENTATIONS:
        row: dict[str, Any] = {"representation": name, "distance_spearman_mean": float(np.mean([x["spearman"] for x in distance_rows if x["representation"] == name and x["distance"] == "cosine_l2_normalized"])), "distance_pearson_mean": float(np.mean([x["pearson"] for x in distance_rows if x["representation"] == name and x["distance"] == "cosine_l2_normalized"])), "linear_CKA_mean": float(np.mean([x["linear_cka"] for x in cka_rows if x["representation"] == name]))}
        for k in ks:
            row[f"knn_jaccard_k{k}_mean"] = float(np.mean([x["jaccard"] for x in knn_rows if x["representation"] == name and x["k"] == k]))
        metrics = [official_a9_metrics(name, data[seed][name]) for seed in RUN_SEEDS]
        row["effective_rank_mean"] = float(np.mean([x["effective_rank"] for x in metrics])); row["PC1_fraction_mean"] = float(np.mean([x["pc1_variance_fraction"] for x in metrics]))
        for k in range(2, 11):
            selected = [x for x in cluster_pairs if x["representation"] == name and x["k"] == k]
            row[f"ARI_k{k}_mean"] = float(np.mean([x["ari"] for x in selected])); row[f"NMI_k{k}_mean"] = float(np.mean([x["nmi"] for x in selected]))
        summary.append(row)
    write_tsv(root / "06_summary" / "representation_stability_summary.tsv", summary)
    report = {"gates": {"GEOMETRIC_EXTRACTION_GATE": "PASS", "GEOMETRIC_ID_ALIGNMENT_GATE": "PASS", "GEOMETRIC_FINITE_VALUES_GATE": "PASS", "GEOMETRIC_STABILITY_GATE": "PASS_DESCRIPTIVE"}, "summary": summary}
    (root / "06_summary" / "representation_stability_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "06_summary" / "representation_stability_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "06_summary" / "MODEL_B_GEOMETRIC_STABILITY_REPORT.md").write_text("# Model B geometric stability\n\nDescriptive read-only multiseed analysis; no scientific stability threshold was imposed. `z_instance_pair` is a technical contrastive space, not the primary biological space.\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--knn-k", default="5,10,20")
    args = parser.parse_args()
    try:
        values = tuple(int(part) for part in args.knn_k.split(",") if part)
        print(json.dumps(geometry_analysis(embeddings_dir=args.embeddings_dir, output_dir=args.output_dir, knn_k=values), indent=2, sort_keys=True))
    except (ModelBAnalysisError, OSError, ValueError, ImportError) as exc:
        print(f"GEOMETRIC_STABILITY_GATE=FAIL\nreason={exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
