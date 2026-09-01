"""Fail-closed productive A9 Colab preflight; never starts training."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.training.a9_contract import (  # noqa: E402
    A9ContractError, EXPECTED_PARTITIONS, MODEL_A_ARCHITECTURE, MODEL_B_ARCHITECTURE,
    apply_a9_run_seed, validate_a9_config_pair, validate_a9_run_seed,
    validate_separate_output_roots,
)
from gnn_siamese.utils.manifest import build_run_layout, generate_run_id  # noqa: E402
from scripts.colab_preflight import (  # noqa: E402
    ColabPreflightError, git_revision, preflight_output_root, runtime_summary,
    validate_model_a_dataset_identity,
)


def prospective_a9_run_dir(
    config: dict[str, Any], *, run_id: str | None = None
) -> Path:
    """Use the trainer's layout function and reject an existing fresh target."""
    candidate = build_run_layout(
        root_dir=config["outputs"]["root_dir"], model_name=config["outputs"]["model_name"],
        run_id=run_id or generate_run_id(),
    ).run_dir.resolve()
    if candidate.exists():
        raise ColabPreflightError(f"Prospective fresh A9 run already exists: {candidate}")
    return candidate


def resolve_a9_runtime_configs(
    config_path: str | Path, *, architecture: str, run_seed: int,
    mutants_hdf5: str | Path, wt_hdf5: str | Path,
    output_root: str | Path, peer_output_root: str | Path,
    repo_root: str | Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve runtime-only paths and one run seed without editing YAML."""
    from gnn_siamese.config import load_config

    root = Path(repo_root).resolve()
    if architecture not in {MODEL_A_ARCHITECTURE, MODEL_B_ARCHITECTURE}:
        raise A9ContractError(f"Unsupported A9 architecture: {architecture!r}")
    selected = apply_a9_run_seed(load_config(config_path), run_seed)
    if selected["model"]["architecture"] != architecture:
        raise A9ContractError("--architecture does not match --config.")
    peer_path = root / "configs" / (
        "model_b_a9.yaml" if architecture == MODEL_A_ARCHITECTURE else "model_a_a9.yaml"
    )
    peer = apply_a9_run_seed(load_config(peer_path), run_seed)
    config_a, config_b = (selected, peer) if architecture == MODEL_A_ARCHITECTURE else (peer, selected)
    for config in (config_a, config_b):
        config["paths"]["mutants_hdf5"] = str(Path(mutants_hdf5).expanduser().resolve())
        config["paths"]["wt_companion_hdf5"] = str(Path(wt_hdf5).expanduser().resolve())
        config["training"]["device"] = "cuda"
        config["__config_path__"] = str(Path(config_path).resolve() if config is selected else peer_path.resolve())
        validate_a9_run_seed(config, run_seed)
    selected["outputs"]["root_dir"] = str(Path(output_root).expanduser().resolve())
    peer["outputs"]["root_dir"] = str(Path(peer_output_root).expanduser().resolve())
    validate_a9_config_pair(config_a, config_b)
    validate_separate_output_roots(config_a["outputs"]["root_dir"], config_b["outputs"]["root_dir"])
    return config_a, config_b


def validate_a9_colab_preflight(
    config_path: str | Path, *, architecture: str, run_seed: int,
    mutants_hdf5: str | Path, wt_hdf5: str | Path,
    output_root: str | Path, peer_output_root: str | Path,
    repo_root: str | Path, expected_commit: str, allowed_output_root: str | Path,
    prospective_run_id: str | None = None,
) -> dict[str, Any]:
    """Validate code, GPU, immutable data/split, protocol and fresh storage."""
    from gnn_siamese.builders import build_dataset_bundle, build_split_bundle
    from gnn_siamese.utils.fingerprints import (
        fingerprint_hdf5_inputs, fingerprint_split_definition, resolve_hdf5_dataset_id,
    )

    root = Path(repo_root).resolve()
    config_a, config_b = resolve_a9_runtime_configs(
        config_path, architecture=architecture, run_seed=run_seed,
        mutants_hdf5=mutants_hdf5, wt_hdf5=wt_hdf5,
        output_root=output_root, peer_output_root=peer_output_root, repo_root=root,
    )
    git = git_revision(root, expected_commit)
    runtime = runtime_summary("cuda")
    if not runtime["cuda_visible"] or runtime["selected_device"] != "cuda":
        raise ColabPreflightError("A9 requires a recognized CUDA GPU.")
    frozen_split = root / "splits" / "leave_position_out_seed_42.json"
    inventories: dict[str, Any] = {}
    variant_sets: dict[str, set[str]] = {}
    for label, config in (("model_a", config_a), ("model_b", config_b)):
        runtime_config = deepcopy(config)
        runtime_config["split"]["persist_path"] = str(frozen_split)
        dataset_bundle = build_dataset_bundle(runtime_config)
        split_bundle = build_split_bundle(runtime_config, dataset_bundle.dataset)
        if split_bundle.created:
            raise ColabPreflightError(f"{label} unexpectedly created the frozen A9 split.")
        counts = {"train": len(split_bundle.train_indices), "validation": len(split_bundle.validation_indices), "test": len(split_bundle.test_indices)}
        if counts != EXPECTED_PARTITIONS:
            raise ColabPreflightError(f"{label} partition counts mismatch: {counts}")
        split_digest = fingerprint_split_definition(split_bundle.split)
        if split_digest != config["a9"]["expected_split_fingerprint"]:
            raise ColabPreflightError(f"{label} frozen split fingerprint mismatch: {split_digest}")
        dataset = dataset_bundle.dataset
        if len(dataset.pairs) != 483 or len(dataset.native_wt_controls) != 1:
            raise ColabPreflightError(f"{label} inventory is not 483 biological variants plus one WT.")
        variants = {pair.variant_id for pair in dataset.pairs}
        controls = {control.variant_id for control in dataset.native_wt_controls}
        assigned = {item.variant_id for item in split_bundle.split.assignments}
        if controls & assigned:
            raise ColabPreflightError(f"Native WT leaked into {label} split assignments.")
        fingerprints = fingerprint_hdf5_inputs(
            mutants_path=dataset.mutant_h5_path, wt_companion_path=dataset.wt_h5_path,
            dataset_id=resolve_hdf5_dataset_id(config),
        )
        identity = validate_model_a_dataset_identity(fingerprints, frozen_split_path=frozen_split)
        inventories[label] = {"biological_variants": 483, "native_wt_controls": 1, "partitions": counts, "split_fingerprint": split_digest, "dataset_identity": identity}
        variant_sets[label] = variants
    intersection = variant_sets["model_a"] & variant_sets["model_b"]
    if len(intersection) != 483 or variant_sets["model_a"] != variant_sets["model_b"]:
        raise ColabPreflightError(f"A/B valid intersection is not exactly 483: {len(intersection)}")
    roots = validate_separate_output_roots(config_a["outputs"]["root_dir"], config_b["outputs"]["root_dir"])
    output_checks = {label: preflight_output_root(path, allowed_root=allowed_output_root) for label, path in roots.items()}
    selected = config_a if architecture == MODEL_A_ARCHITECTURE else config_b
    prospective = prospective_a9_run_dir(selected, run_id=prospective_run_id)
    return {"status": "PASS", "git": git, "runtime": runtime, "run_seed": run_seed, "split_seed": 42, "inventories": inventories, "ab_intersection": 483, "output_roots": output_checks, "prospective_run_dir": str(prospective), "prospective_exists": False}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "mutants-hdf5", "wt-hdf5", "output-root", "peer-output-root", "allowed-output-root", "expected-commit", "architecture"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--run-seed", required=True, type=int)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()
    try:
        result = validate_a9_colab_preflight(
            args.config, architecture=args.architecture, run_seed=args.run_seed,
            mutants_hdf5=args.mutants_hdf5, wt_hdf5=args.wt_hdf5,
            output_root=args.output_root, peer_output_root=args.peer_output_root,
            repo_root=args.repo_root, expected_commit=args.expected_commit,
            allowed_output_root=args.allowed_output_root,
        )
    except (A9ContractError, ColabPreflightError, OSError, ValueError) as exc:
        print(f"A9_PREFLIGHT=FAIL\nreason={exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
