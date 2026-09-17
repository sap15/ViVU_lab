#!/usr/bin/env python3
"""Re-extract accepted Model B representations from immutable ``best.pt`` files.

This is inference-only: it never invokes training, resume, or checkpoint save.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.builders import build_dataset_bundle, build_model
from gnn_siamese.data import collate_mut_wt_pairs, load_leave_position_out_split
from gnn_siamese.training.a9_contract import MODEL_B_ARCHITECTURE
from gnn_siamese.utils.fingerprints import fingerprint_hdf5_inputs, fingerprint_split_definition, resolve_hdf5_dataset_id
from scripts.model_b_analysis_common import (
    EXPECTED_SHAPES, REPRESENTATIONS, RUN_SEEDS, ModelBAnalysisError, canonical_ids,
    load_json, require_accepted_a9, sha256_file, validate_matrix, validate_reextraction, write_tsv,
)


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _accepted_run_dirs(summary: Mapping[str, Any], runs_root: Path) -> dict[int, Path]:
    """Accept a few documented FAST-B6 layouts but never guess a missing run."""
    candidates: list[Mapping[str, Any]] = []
    for key in ("runs", "per_seed", "accepted_runs", "seed_results", "results"):
        value = summary.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, Mapping))
        elif isinstance(value, Mapping):
            candidates.extend(item for item in value.values() if isinstance(item, Mapping))
    if not candidates:
        raise ModelBAnalysisError("FAST-B6 summary contains no readable per-seed run records.")
    found: dict[int, list[Path]] = {}
    for record in candidates:
        seed = record.get("seed", record.get("run_seed"))
        raw = record.get("run_dir", record.get("run_path", record.get("run_directory", record.get("path"))))
        if seed is None or raw is None:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = (runs_root / path).resolve()
        found.setdefault(int(seed), []).append(path)
    if set(found) != set(RUN_SEEDS) or any(len(paths) != 1 for paths in found.values()):
        raise ModelBAnalysisError("FAST-B6 must identify exactly one accepted run for each required seed.")
    return {seed: paths[0] for seed, paths in found.items()}


def _fast_b6_acceptance_hash(record: Mapping[str, Any]) -> str:
    for key in ("a9_acceptance_sha256", "acceptance_sha256", "a9_acceptance_hash"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    raise ModelBAnalysisError("FAST-B6 per-seed record lacks the required A9 acceptance SHA256.")


def _resolved_config(run_dir: Path, mutants_hdf5: str, wt_hdf5: str, device: str) -> dict[str, Any]:
    from gnn_siamese.config import load_config

    config = deepcopy(load_config(run_dir / "config_resolved.yaml"))
    # The only historical locators intentionally replaced are the raw HDF5 paths.
    config.setdefault("paths", {})["mutants_hdf5"] = str(Path(mutants_hdf5).resolve())
    config.setdefault("paths", {})["wt_companion_hdf5"] = str(Path(wt_hdf5).resolve())
    config.setdefault("training", {})["device"] = device
    config["__config_path__"] = str((run_dir / "config_resolved.yaml").resolve())
    return config


def _validate_identity(
    run_dir: Path, config: Mapping[str, Any], acceptance: Mapping[str, Any], *, mutants_hdf5: str, wt_hdf5: str,
) -> tuple[Any, dict[str, Any]]:
    manifest = load_json(run_dir / "run_manifest.json")
    if manifest.get("architecture") != MODEL_B_ARCHITECTURE or _nested(config, "model.architecture") != MODEL_B_ARCHITECTURE:
        raise ModelBAnalysisError(f"Model B architecture identity mismatch: {run_dir}")
    serialized = json.dumps(manifest, sort_keys=True)
    if "ed9dc23d7936a77a33bc40b2ed3400d43f1f9111" not in serialized:
        raise ModelBAnalysisError(f"Scientific training commit is not frozen in manifest: {run_dir}")
    if int(_nested(manifest, "configuration.seed_bundle.split")) != 42 or int(_nested(config, "split.seed")) != 42:
        raise ModelBAnalysisError(f"Split seed is not frozen at 42: {run_dir}")
    seed = int(acceptance.get("run_seed", -1))
    if seed not in RUN_SEEDS or int(_nested(manifest, "configuration.seed")) != seed:
        raise ModelBAnalysisError(f"Run seed identity mismatch: {run_dir}")
    expected = _nested(manifest, "data.hdf5_content_fingerprint")
    actual = fingerprint_hdf5_inputs(mutants_path=mutants_hdf5, wt_companion_path=wt_hdf5, dataset_id=resolve_hdf5_dataset_id(config))
    if not isinstance(expected, Mapping) or expected.get("combined") != actual.get("combined"):
        raise ModelBAnalysisError(f"HDF5 fingerprint mismatch: {run_dir}")
    return manifest, actual


def _extract_one(run_dir: Path, config: Mapping[str, Any], device: torch.device) -> tuple[list[str], list[Any], dict[str, np.ndarray], str]:
    bundle = build_dataset_bundle(config)
    if len(bundle.dataset) != 483:
        raise ModelBAnalysisError(f"Dataset must resolve 483 variants, got {len(bundle.dataset)}.")
    # Extraction is deliberately over the complete biological inventory.  The
    # persisted split is read only to verify its frozen identity, never to
    # select a training partition or recreate a historical split path.
    split = load_leave_position_out_split(run_dir / "split.json")
    expected_split = _nested(load_json(run_dir / "run_manifest.json"), "data.split_fingerprint")
    if fingerprint_split_definition(split) != expected_split:
        raise ModelBAnalysisError("SPLIT_FINGERPRINT_GATE=FAIL")
    loader = DataLoader(bundle.dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=0, collate_fn=collate_mut_wt_pairs)
    model = build_model(config, bundle.dataset).to(device)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise ModelBAnalysisError(f"Missing or empty best.pt: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        raise ModelBAnalysisError(f"Invalid best.pt payload: {checkpoint_path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    # z_delta is scientifically valid only because the immutable A9 gradient audit was checked first.
    model.siamese_model.relational_module.validation_audit = {"modules": load_json(run_dir / "gradient_audit.json")}
    model.eval()
    ids: list[str] = []
    positions: list[Any] = []
    collected = {name: [] for name in REPRESENTATIONS}
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model.siamese_model(graph_mut=batch.graph_mut, graph_wt=batch.graph_wt, allow_trainable_z_delta=True)
            if output.z_delta_status != "validated":
                raise ModelBAnalysisError("mlp_delta is not validated for scientific extraction.")
            values = {"r_delta": output.r_delta, "z_delta": output.z_delta, "z_instance_pair": output.z_instance_pair}
            for name, tensor in values.items():
                if tensor is None:
                    raise ModelBAnalysisError(f"Model did not expose {name}.")
                collected[name].append(tensor.detach().cpu().numpy())
            ids.extend(batch.variant_ids)
            positions.extend(item.get("position") for item in batch.metadata)
    ids = canonical_ids(ids)
    matrices = {name: validate_matrix(name, np.concatenate(parts, axis=0)) for name, parts in collected.items()}
    return ids, positions, matrices, str(checkpoint_path)


def export_best_representations(*, runs_root: str | Path, mutants_hdf5: str, wt_hdf5: str, output_dir: str | Path, device: str = "cpu") -> dict[str, Any]:
    if device != "cpu" and not torch.cuda.is_available():
        raise ModelBAnalysisError(f"Requested unavailable device: {device}")
    root, out = Path(runs_root).resolve(), Path(output_dir).resolve()
    summary_path = root / "fast_b6_multiseed_aggregation" / "model_b_a9_multiseed_summary.json"
    summary = load_json(summary_path)
    if str(summary.get("status", "PASS")).upper() != "PASS":
        raise ModelBAnalysisError("FAST-B6 is not PASS.")
    run_dirs = _accepted_run_dirs(summary, root)
    # Read the same records a second time so a FAST-B6 acceptance hash is a
    # required identity field, not an optional diagnostic.
    records = []
    for key in ("runs", "per_seed", "accepted_runs", "seed_results", "results"):
        value = summary.get(key)
        records.extend(value if isinstance(value, list) else (value.values() if isinstance(value, Mapping) else ()))
    fast_hashes = {int(item.get("seed", item.get("run_seed"))): _fast_b6_acceptance_hash(item) for item in records if isinstance(item, Mapping) and item.get("seed", item.get("run_seed")) is not None}
    if set(fast_hashes) != set(RUN_SEEDS):
        raise ModelBAnalysisError("FAST-B6 acceptance hashes are incomplete or ambiguous.")
    embeddings = out / "01_embeddings"
    embeddings.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    canonical: set[str] | None = None
    for seed in RUN_SEEDS:
        run_dir = run_dirs[seed]
        acceptance = require_accepted_a9(run_dir)
        if sha256_file(run_dir / "a9_acceptance.json") != fast_hashes[seed]:
            raise ModelBAnalysisError(f"FAST-B6 acceptance SHA256 mismatch for seed {seed}.")
        config = _resolved_config(run_dir, mutants_hdf5, wt_hdf5, device)
        run_manifest, hdf5_fingerprint = _validate_identity(run_dir, config, acceptance, mutants_hdf5=mutants_hdf5, wt_hdf5=wt_hdf5)
        ids, positions, matrices, best_path = _extract_one(run_dir, config, torch.device(device))
        if canonical is None:
            canonical = set(ids)
        elif set(ids) != canonical:
            raise ModelBAnalysisError("REPRESENTATION_ID_ALIGNMENT_GATE=FAIL")
        validation_rows.extend([{"seed": seed, **row} for row in validate_reextraction(acceptance, matrices)])
        npz_path = embeddings / f"seed_{seed}_representations.npz"
        # Position is optional metadata; use a Unicode array so NPZ never needs pickle.
        np.savez_compressed(npz_path, variant_id=np.asarray(ids, dtype="U"), position=np.asarray(["" if value is None else value for value in positions], dtype="U"), **matrices)
        for name, matrix in matrices.items():
            manifest_rows.append({"seed": seed, "run_id": _nested(run_manifest, "run_id"), "run_dir": str(run_dir), "best_pt_path": best_path, "best_pt_sha256": sha256_file(best_path), "acceptance_sha256": sha256_file(run_dir / "a9_acceptance.json"), "scientific_commit": _nested(run_manifest, "git.commit"), "hdf5_fingerprints": json.dumps(hdf5_fingerprint, sort_keys=True), "split_fingerprint": _nested(run_manifest, "data.split_fingerprint"), "device": device, "torch_version": torch.__version__, "torch_geometric_version": __import__("torch_geometric").__version__, "representation": name, "shape": json.dumps(list(matrix.shape)), "number_variants": 483, "npz_sha256": sha256_file(npz_path)})
    write_tsv(embeddings / "representation_export_manifest.tsv", manifest_rows)
    write_tsv(embeddings / "representation_reextraction_validation.tsv", validation_rows)
    report = {"gates": {"BEST_CHECKPOINT_IDENTITY_GATE": "PASS", "DATASET_FINGERPRINT_GATE": "PASS", "SPLIT_FINGERPRINT_GATE": "PASS", "REPRESENTATION_EXTRACTION_GATE": "PASS", "REPRESENTATION_ID_ALIGNMENT_GATE": "PASS", "REPRESENTATION_FINITE_VALUES_GATE": "PASS", "REPRESENTATION_REEXTRACTION_GATE": "PASS"}, "records": manifest_rows}
    (embeddings / "representation_export_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--mutants-hdf5", required=True)
    parser.add_argument("--wt-hdf5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    try:
        print(json.dumps(export_best_representations(**vars(args)), indent=2, sort_keys=True))
    except (ModelBAnalysisError, A9ContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"REPRESENTATION_REEXTRACTION_GATE=FAIL\nreason={exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
