#!/usr/bin/env python3
"""Audit spatial locality of paired Mut--WT residue alignment, read-only."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.data.node_pair_alignment_spatial_audit import (
    DEFAULT_RADII,
    audit_spatial_hdf5_pairs,
    write_spatial_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mut-hdf5", required=True, type=Path)
    parser.add_argument("--wt-hdf5", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_a/a1/node_pair_alignment_spatial_audit"),
    )
    parser.add_argument("--radii", nargs="+", type=float, default=list(DEFAULT_RADII))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = audit_spatial_hdf5_pairs(args.mut_hdf5, args.wt_hdf5, radii=args.radii)
    for name, path in write_spatial_artifacts(args.output_dir, rows).items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
