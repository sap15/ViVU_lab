#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extrae una muestra mínima desde:
  - local_data/hdf5/proc_483p.hdf5
  - local_data/hdf5/wt_companion.hdf5

Genera HDF5 individuales en:
  - sample_data/examples/

También genera:
  - reports/sample_hdf5_candidates.tsv
  - reports/sample_hdf5_extraction_report.tsv
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd


SRV_RE = re.compile(
    r"(?i)^(?:srv|residue-graph|residue-srv):([^:]+):(\d+):([A-Za-z]+|STOP|TER|X)->([A-Za-z]+|STOP|TER|X):(.+)$"
)

TAIL_RE = re.compile(
    r"pos_(\d+)_([A-Z]{1,3})_([A-Z]{1,3}|STOP|TER|X)",
    re.IGNORECASE,
)

STOP_TOKENS = {"STOP", "TER", "X", "*"}

DIFF_PROBES = [
    "diff_mass",
    "diff_charge",
    "diff_pI",
    "diff_size",
    "diff_hb_donors",
    "diff_hb_acceptors",
    "diff_polarity",
    "diff_hydrophobicity",
    "diff_hbond_count",
]


def parse_variant(key: str) -> tuple[Optional[int], Optional[str], Optional[str]]:
    m = SRV_RE.match(str(key))
    if m:
        return int(m.group(2)), m.group(3).upper(), m.group(4).upper()

    m = TAIL_RE.search(str(key))
    if m:
        return int(m.group(1)), m.group(2).upper(), m.group(3).upper()

    return None, None, None


def safe_array(obj):
    try:
        return np.asarray(obj[()])
    except Exception:
        return None


def infer_n_nodes(group: h5py.Group) -> int:
    nf = group.get("node_features")
    if nf is None:
        return -1

    if "_name" in nf:
        return int(nf["_name"].shape[0])

    for name in nf.keys():
        arr = safe_array(nf[name])
        if arr is not None and arr.ndim >= 1:
            return int(arr.shape[0])

    return -1


def infer_n_edges(group: h5py.Group) -> int:
    ef = group.get("edge_features")
    if ef is None:
        return -1

    if "_index" in ef:
        arr = safe_array(ef["_index"])
        if arr is not None:
            if arr.ndim == 2 and arr.shape[1] == 2:
                return int(arr.shape[0])
            if arr.ndim == 2 and arr.shape[0] == 2:
                return int(arr.shape[1])

    return -1


def has_nonzero_diff(group: h5py.Group) -> bool:
    nf = group.get("node_features")
    if nf is None:
        return False

    for name in DIFF_PROBES:
        if name not in nf:
            continue
        arr = safe_array(nf[name])
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if np.any(np.isfinite(arr) & (np.abs(arr) > 1e-12)):
            return True

    return False


def has_mask_or_missing(group: h5py.Group) -> bool:
    nf = group.get("node_features")
    if nf is None:
        return False

    for name in nf.keys():
        low = name.lower()

        if "missing" in low:
            arr = safe_array(nf[name])
            if arr is not None:
                return True

        if low.startswith("mask_"):
            arr = safe_array(nf[name])
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=float)
            if arr.size > 0 and np.any(arr == 0):
                return True

    return False


def classify_record(h5_path: Path, key: str, group: h5py.Group, source: str) -> dict:
    pos, wt, mut = parse_variant(key)
    is_stop = str(mut).upper() in STOP_TOKENS if mut is not None else False

    return {
        "source": source,
        "h5": str(h5_path),
        "key": key,
        "pos": pos,
        "wt": wt,
        "mut": mut,
        "is_stop": is_stop,
        "n_nodes": infer_n_nodes(group),
        "n_edges": infer_n_edges(group),
        "has_nonzero_diff": has_nonzero_diff(group),
        "has_mask_or_missing": has_mask_or_missing(group),
    }


def scan_hdf5(h5_path: Path, source: str) -> pd.DataFrame:
    rows = []
    with h5py.File(h5_path, "r") as h5:
        for key in h5.keys():
            obj = h5[key]
            if not isinstance(obj, h5py.Group):
                continue
            rows.append(classify_record(h5_path, key, obj, source))
    return pd.DataFrame(rows)


def sanitize_name(text: str) -> str:
    text = str(text)
    text = text.replace(":", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")
    text = text.replace("->", "_to_")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def copy_single_group(src_h5: Path, key: str, out_h5: Path) -> None:
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    if out_h5.exists():
        out_h5.unlink()

    with h5py.File(src_h5, "r") as src, h5py.File(out_h5, "w") as dst:
        src.copy(src[key], dst, name=key)


def choose_examples(mut_df: pd.DataFrame, wt_df: pd.DataFrame) -> list[dict]:
    wt_positions = set(wt_df["pos"].dropna().astype(int).tolist())

    mut_ok = mut_df[
        (mut_df["pos"].notna())
        & (~mut_df["is_stop"])
        & (mut_df["has_nonzero_diff"])
        & (mut_df["pos"].astype(int).isin(wt_positions))
    ].copy()

    if mut_ok.empty:
        raise RuntimeError("No se encontraron mutantes missense con diff_* no nulos y WT companion disponible.")

    mut_ok = mut_ok.sort_values(["n_nodes", "n_edges", "pos"], ascending=[False, False, True])

    selected = []

    # Mutante completo 1
    selected.append({
        "role": "mutante_1",
        **mut_ok.iloc[0].to_dict(),
    })

    used_positions = {int(mut_ok.iloc[0]["pos"])}

    # Mutante completo 2 en otra posición
    second_candidates = mut_ok[~mut_ok["pos"].astype(int).isin(used_positions)]
    if second_candidates.empty:
        second_candidates = mut_ok.iloc[1:]

    if not second_candidates.empty:
        selected.append({
            "role": "mutante_2",
            **second_candidates.iloc[0].to_dict(),
        })
        used_positions.add(int(second_candidates.iloc[0]["pos"]))

    # Caso con máscara o missing
    missing_candidates = mut_df[
        (mut_df["pos"].notna())
        & (mut_df["has_mask_or_missing"])
        & (mut_df["pos"].astype(int).isin(wt_positions))
    ].copy()

    if not missing_candidates.empty:
        missing_candidates = missing_candidates.sort_values(
            ["is_stop", "n_nodes", "n_edges", "pos"],
            ascending=[False, True, True, True],
        )
        row = missing_candidates.iloc[0]
        selected.append({
            "role": "caso_con_missing_o_stop",
            **row.to_dict(),
        })
    else:
        # Fallback: caso límite por menor número de nodos
        fallback = mut_df[
            (mut_df["pos"].notna())
            & (mut_df["pos"].astype(int).isin(wt_positions))
        ].sort_values(["n_nodes", "n_edges", "pos"], ascending=[True, True, True])

        if not fallback.empty:
            row = fallback.iloc[0]
            selected.append({
                "role": "caso_limite_sin_missing_detectado",
                **row.to_dict(),
            })

    return selected


def find_wt_for_position(wt_df: pd.DataFrame, pos: int) -> Optional[dict]:
    hits = wt_df[wt_df["pos"].astype("Int64") == int(pos)]
    if hits.empty:
        return None
    return hits.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mut-h5", default="local_data/hdf5/proc_483p.hdf5")
    parser.add_argument("--wt-h5", default="local_data/hdf5/wt_companion.hdf5")
    parser.add_argument("--out-dir", default="sample_data/examples")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    mut_h5 = Path(args.mut_h5)
    wt_h5 = Path(args.wt_h5)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)

    if not mut_h5.exists():
        raise FileNotFoundError(f"No existe mut_h5: {mut_h5}")
    if not wt_h5.exists():
        raise FileNotFoundError(f"No existe wt_h5: {wt_h5}")

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mut_df = scan_hdf5(mut_h5, "mutante")
    wt_df = scan_hdf5(wt_h5, "wt_companion")

    all_df = pd.concat([mut_df, wt_df], ignore_index=True)
    candidates_tsv = reports_dir / "sample_hdf5_candidates.tsv"
    all_df.to_csv(candidates_tsv, sep="\t", index=False)

    selected_mutants = choose_examples(mut_df, wt_df)

    extraction_rows = []
    copied = set()

    for rec in selected_mutants:
        pos = int(rec["pos"])
        wt = rec["wt"]
        mut = rec["mut"]
        role = rec["role"]
        key = rec["key"]

        mut_name = f"pos_{pos}_{wt}_{mut}_{role}.hdf5"
        mut_out = out_dir / sanitize_name(mut_name)

        copy_single_group(Path(rec["h5"]), key, mut_out)

        extraction_rows.append({
            "role": role,
            "source_h5": rec["h5"],
            "source_key": key,
            "output_file": str(mut_out),
            "pos": pos,
            "wt": wt,
            "mut": mut,
            "n_nodes": rec["n_nodes"],
            "n_edges": rec["n_edges"],
            "has_mask_or_missing": rec["has_mask_or_missing"],
            "is_stop": rec["is_stop"],
        })

        copied.add(str(mut_out))

        wt_rec = find_wt_for_position(wt_df, pos)
        if wt_rec is not None:
            wt_name = f"pos_{pos}_{wt}_{mut}_wt_companion.hdf5"
            wt_out = out_dir / sanitize_name(wt_name)

            if str(wt_out) not in copied:
                copy_single_group(Path(wt_rec["h5"]), wt_rec["key"], wt_out)
                copied.add(str(wt_out))

                extraction_rows.append({
                    "role": f"wt_companion_for_{role}",
                    "source_h5": wt_rec["h5"],
                    "source_key": wt_rec["key"],
                    "output_file": str(wt_out),
                    "pos": pos,
                    "wt": wt_rec["wt"],
                    "mut": wt_rec["mut"],
                    "n_nodes": wt_rec["n_nodes"],
                    "n_edges": wt_rec["n_edges"],
                    "has_mask_or_missing": wt_rec["has_mask_or_missing"],
                    "is_stop": wt_rec["is_stop"],
                })

    report_df = pd.DataFrame(extraction_rows)
    report_tsv = reports_dir / "sample_hdf5_extraction_report.tsv"
    report_df.to_csv(report_tsv, sep="\t", index=False)

    print(f"[OK] Inventario guardado en: {candidates_tsv}")
    print(f"[OK] Reporte de extracción guardado en: {report_tsv}")
    print("[OK] Archivos generados:")
    for path in sorted(out_dir.glob("*.hdf5")):
        print(f"  - {path}")


if __name__ == "__main__":
    main()
