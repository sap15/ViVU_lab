#!/usr/bin/env python3
"""Audit residue-level Mut--WT correspondence without modifying input HDF5."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.data.node_pair_alignment_audit import (
    audit_hdf5_pairs,
    write_alignment_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mut-hdf5", required=True, type=Path)
    parser.add_argument("--wt-hdf5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = audit_hdf5_pairs(args.mut_hdf5, args.wt_hdf5)
    paths = write_alignment_artifacts(args.output_dir, records)
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
