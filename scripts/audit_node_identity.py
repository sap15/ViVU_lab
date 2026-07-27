#!/usr/bin/env python3
"""CLI for the descriptive HDF5 node-identity audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from gnn_siamese.data.node_identity_audit import (
    audit_hdf5,
    build_fingerprint,
    write_audit_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mut-hdf5", required=True, type=Path)
    parser.add_argument("--wt-hdf5", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = audit_hdf5(args.mut_hdf5, "mutant")
    records.extend(audit_hdf5(args.wt_hdf5, "wt_companion"))
    command = [sys.executable, *sys.argv]
    fingerprint = build_fingerprint(
        {"mutant": args.mut_hdf5, "wt_companion": args.wt_hdf5},
        command=command,
        cwd=Path.cwd(),
    )
    paths = write_audit_artifacts(args.output_dir, records, fingerprint)
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

