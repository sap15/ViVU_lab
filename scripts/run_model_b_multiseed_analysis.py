#!/usr/bin/env python3
"""Fail-closed orchestration of read-only Model B multiseed analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.model_b_analysis_common import ModelBAnalysisError
from scripts.model_b_export_best_representations import export_best_representations
from scripts.model_b_multiseed_geometry import geometry_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--mutants-hdf5", required=True)
    parser.add_argument("--wt-hdf5", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--knn-k", default="5,10,20")
    args = parser.parse_args()
    try:
        export = export_best_representations(runs_root=args.runs_root, mutants_hdf5=args.mutants_hdf5, wt_hdf5=args.wt_hdf5, output_dir=args.output_dir, device=args.device)
        geometry = geometry_analysis(embeddings_dir=Path(args.output_dir) / "01_embeddings", output_dir=args.output_dir, knn_k=tuple(int(item) for item in args.knn_k.split(",") if item))
    except (ModelBAnalysisError, OSError, ValueError, RuntimeError) as exc:
        print(f"MODEL_B_MULTISEED_ANALYSIS=FAIL\nreason={exc}", file=sys.stderr)
        return 2
    print(json.dumps({"export": export["gates"], "geometry": geometry["gates"], "TRAINING_TRIGGERED": "NO", "RESUME_TRIGGERED": "NO"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
