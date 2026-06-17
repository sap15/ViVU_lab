#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np


EXAMPLES_DIR = Path("sample_data/examples")
README_PATH = Path("sample_data/README.md")
SCHEMA_PATH = Path("sample_data/sample_schema.json")
EXTRACTION_REPORT = Path("reports/sample_hdf5_extraction_report.tsv")


def classify_file(path: Path) -> str:
    name = path.name
    if "wt_companion" in name:
        return "WT companion"
    if "caso_con_missing" in name or "stop" in name:
        return "Caso con máscara/missing o caso límite"
    if "mutante_1" in name:
        return "Mutante completo 1"
    if "mutante_2" in name:
        return "Mutante completo 2"
    return "Ejemplo HDF5"


def parse_filename(path: Path) -> tuple[str, str, str]:
    m = re.search(r"pos_(\d+)_([A-Z]+)_([A-Z]+)", path.name)
    if not m:
        return "-", "-", "-"
    return m.group(1), m.group(2), m.group(3)


def first_group_key(path: Path) -> str:
    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        if not keys:
            return ""
        return keys[0]


def count_group_items(path: Path, group_name: str) -> int:
    with h5py.File(path, "r") as h5:
        key = list(h5.keys())[0]
        grp = h5[key].get(group_name)
        if grp is None:
            return 0
        return len(grp.keys())


def infer_n_nodes(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        key = list(h5.keys())[0]
        nf = h5[key].get("node_features")
        if nf is None:
            return -1
        if "_name" in nf:
            return int(nf["_name"].shape[0])
        for feat in nf.keys():
            arr = np.asarray(nf[feat])
            if arr.ndim >= 1:
                return int(arr.shape[0])
    return -1


def infer_n_edges(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        key = list(h5.keys())[0]
        ef = h5[key].get("edge_features")
        if ef is None or "_index" not in ef:
            return -1
        arr = np.asarray(ef["_index"])
        if arr.ndim == 2 and arr.shape[1] == 2:
            return int(arr.shape[0])
        if arr.ndim == 2 and arr.shape[0] == 2:
            return int(arr.shape[1])
    return -1


def describe_hdf5(path: Path) -> dict:
    description = {}

    with h5py.File(path, "r") as handle:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                description[name] = {
                    "type": "dataset",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }
            elif isinstance(obj, h5py.Group):
                description[name] = {"type": "group"}

        handle.visititems(visitor)

    return description


def collect_schema() -> dict:
    files = sorted(EXAMPLES_DIR.glob("*.hdf5"))
    schema = {
        "description": (
            "Schema inferred from minimal HDF5 examples extracted from the full project HDF5 files. "
            "The full files are not included in Git."
        ),
        "source_full_hdf5": {
            "mutants": "local_data/hdf5/proc_483p.hdf5",
            "wt_companions": "local_data/hdf5/wt_companion.hdf5",
            "note": "These full HDF5 files are ignored by .gitignore and must not be committed.",
        },
        "examples_dir": "sample_data/examples",
        "examples": {},
    }

    for path in files:
        schema["examples"][path.name] = {
            "role": classify_file(path),
            "root_key": first_group_key(path),
            "n_nodes": infer_n_nodes(path),
            "n_edges": infer_n_edges(path),
            "hierarchy": describe_hdf5(path),
        }

    return schema


def write_schema() -> None:
    schema = collect_schema()
    SCHEMA_PATH.write_text(
        json.dumps(schema, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_readme() -> None:
    files = sorted(EXAMPLES_DIR.glob("*.hdf5"))

    rows = []
    for path in files:
        pos, wt, mut = parse_filename(path)
        rows.append({
            "file": path.name,
            "role": classify_file(path),
            "pos": pos,
            "wt": wt,
            "mut": mut,
            "key": first_group_key(path),
            "n_nodes": infer_n_nodes(path),
            "n_edges": infer_n_edges(path),
            "node_features": count_group_items(path, "node_features"),
            "edge_features": count_group_items(path, "edge_features"),
            "graph_features": count_group_items(path, "graph_features"),
        })

    lines = []
    lines.append("# sample_data")
    lines.append("")
    lines.append("Este directorio contiene una muestra mínima de archivos HDF5 del proyecto GNN siamés Mutante–WT.")
    lines.append("")
    lines.append("La muestra está pensada para documentar el esquema real de los HDF5, permitir pruebas unitarias, smoke tests y facilitar la revisión del formato sin incluir los datos completos de entrenamiento.")
    lines.append("")
    lines.append("## Origen de los ejemplos")
    lines.append("")
    lines.append("Los ejemplos de `sample_data/examples/` se han extraído de los HDF5 completos locales:")
    lines.append("")
    lines.append("- `local_data/hdf5/proc_483p.hdf5`: HDF5 global con grafos mutantes.")
    lines.append("- `local_data/hdf5/wt_companion.hdf5`: HDF5 global con grafos WT companion.")
    lines.append("")
    lines.append("Estos archivos completos están excluidos por `.gitignore` y no deben subirse al repositorio.")
    lines.append("")
    lines.append("## Archivos incluidos")
    lines.append("")
    lines.append("| Archivo | Rol | Posición | WT | Mutante | Nodos | Aristas | Key HDF5 |")
    lines.append("|---|---|---:|---|---|---:|---:|---|")

    for row in rows:
        lines.append(
            f"| `{row['file']}` | {row['role']} | {row['pos']} | "
            f"{row['wt']} | {row['mut']} | {row['n_nodes']} | {row['n_edges']} | "
            f"`{row['key']}` |"
        )

    lines.append("")
    lines.append("## Interpretación de la muestra mínima")
    lines.append("")
    lines.append("- `pos_563_CYSTEINE_TRYPTOPHAN_mutante_1.hdf5` representa un mutante missense completo con su WT companion correspondiente.")
    lines.append("- `pos_458_ISOLEUCINE_LYSINE_mutante_2.hdf5` representa un segundo mutante missense completo, útil para probar batching y emparejamiento Mutante–WT.")
    lines.append("- `pos_327_GLYCINE_GLUTAMATE_caso_con_missing_o_stop.hdf5` representa el caso con máscara, valor ausente o caso límite seleccionado automáticamente durante la extracción.")
    lines.append("- Cada posición incluida tiene su WT companion correspondiente.")
    lines.append("")
    lines.append("## Estructura esperada de cada HDF5 individual")
    lines.append("")
    lines.append("Cada archivo HDF5 individual contiene un único grupo raíz correspondiente a una query `residue-srv`.")
    lines.append("")
    lines.append("Dentro de cada grupo raíz se esperan, como mínimo:")
    lines.append("")
    lines.append("- `node_features/`: atributos de nodo a nivel de residuo.")
    lines.append("- `edge_features/`: atributos de arista y conectividad mediante `_index`.")
    lines.append("- `graph_features/`: atributos globales del grafo.")
    lines.append("")
    lines.append("El archivo `sample_data/sample_schema.json` describe la jerarquía, shapes y tipos de datos observados en estos ejemplos.")
    lines.append("")
    lines.append("## Archivos auxiliares relacionados")
    lines.append("")
    lines.append("- `reports/sample_hdf5_candidates.tsv`: inventario de candidatos detectados en los HDF5 completos.")
    lines.append("- `reports/sample_hdf5_extraction_report.tsv`: trazabilidad de los ejemplos extraídos.")
    lines.append("- `scripts/extract_minimal_hdf5_examples.py`: script usado para extraer los grupos individuales.")
    lines.append("")
    lines.append("## Nota importante")
    lines.append("")
    lines.append("La muestra reducida no debe utilizarse como dataset final de entrenamiento. El entrenamiento real debe ejecutarse con los HDF5 completos del proyecto, almacenados fuera del repositorio o en rutas ignoradas por Git.")

    README_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not EXAMPLES_DIR.exists():
        raise FileNotFoundError(f"No existe {EXAMPLES_DIR}")

    hdf5_files = sorted(EXAMPLES_DIR.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No hay archivos .hdf5 en {EXAMPLES_DIR}")

    write_readme()
    write_schema()

    print(f"[OK] Actualizado: {README_PATH}")
    print(f"[OK] Actualizado: {SCHEMA_PATH}")
    print(f"[OK] Ejemplos documentados: {len(hdf5_files)}")


if __name__ == "__main__":
    main()
