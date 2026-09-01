"""Evaluate the fail-closed A9 post-run acceptance contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.config import load_config  # noqa: E402
from gnn_siamese.training.a9_contract import A9ContractError, validate_a9_run_acceptance  # noqa: E402
from gnn_siamese.utils.atomic_io import atomic_write_text  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--pair-representations",
        help="NPZ containing h_pair_delta, z_delta_pair and z_instance_pair (required by Model A).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = load_config(args.config)
    representations = None
    if args.pair_representations:
        with np.load(args.pair_representations, allow_pickle=False) as payload:
            representations = {name: payload[name] for name in payload.files}
    try:
        result = validate_a9_run_acceptance(run_dir, config, representations=representations)
    except (A9ContractError, OSError, ValueError) as exc:
        failure = {"status": "rejected", "reason": str(exc)}
        atomic_write_text(run_dir / "a9_acceptance.json", json.dumps(failure, indent=2, sort_keys=True))
        print(f"A9_ACCEPTANCE=rejected\nreason={exc}", file=sys.stderr)
        return 2
    atomic_write_text(run_dir / "a9_acceptance.json", json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

