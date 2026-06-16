#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import h5py


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


def verify_schema_documentation(schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    text = json.dumps(schema, ensure_ascii=False).lower()

    required = {
        "node_features": "node_features" in text,
        "edge_features": "edge_features" in text,
        "graph_features": "graph_features" in text,
        "edge_index_or_index": ("edge_index" in text) or ("_index" in text),
        "orientation_or_pyg_conversion": (
            "orientation" in text
            or "orientación" in text
            or "pyg_conversion" in text
            or "transpose" in text
        ),
        "variant_id_or_graph_key": ("variant_id" in text) or ("graph_key_pattern" in text),
        "position": "position" in text,
        "wt_aa": ("wt_aa" in text) or ("wt_aa_full" in text),
        "mut_aa": ("mut_aa" in text) or ("mut_aa_full" in text),
        "wt_companion": "wt_companion" in text,
        "is_mutation": "is_mutation" in text,
        "diff_features": "diff_" in text,
        "mask": "mask" in text,
        "targets_or_custom_features": ("target" in text) or ("custom_" in text),
    }

    print("=== Verificación documental de sample_schema.json ===")
    for key, ok in required.items():
        print(f"{key}: {'OK' if ok else 'FALTA'}")

    missing = [key for key, ok in required.items() if not ok]
    if missing:
        raise SystemExit(f"ERROR: faltan elementos documentales: {missing}")

    print("OK: sample_schema.json cubre los elementos principales del apartado 8.2.")


def verify_hdf5_blocks(mut_h5: Path, wt_h5: Path) -> None:
    for path in [mut_h5, wt_h5]:
        if not path.exists():
            raise SystemExit(f"ERROR: no existe {path}")

        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
            if not keys:
                raise SystemExit(f"ERROR: {path.name} no contiene grafos")

            first_key = keys[0]
            grp = handle[first_key]

            print(f"\n=== {path.name} ===")
            print("n_grafos:", len(keys))
            print("primer_grafo:", first_key)

            for block in ["node_features", "edge_features", "graph_features"]:
                status = "OK" if block in grp else "FALTA"
                print(f"{block}: {status}")

            if "edge_features" in grp and "_index" in grp["edge_features"]:
                edge_index = grp["edge_features"]["_index"][()]
                print("edge_features/_index shape:", edge_index.shape)
                print("edge_features/_index dtype:", edge_index.dtype)

                if edge_index.ndim != 2:
                    raise SystemExit(f"ERROR: {path.name} edge index no es 2D")

                if edge_index.shape[1] == 2:
                    print("edge_index_orientation: stored (E,2), PyG requires transpose to (2,E)")
                elif edge_index.shape[0] == 2:
                    print("edge_index_orientation: stored (2,E), PyG direct")
                else:
                    raise SystemExit(f"ERROR: orientación no reconocida: {edge_index.shape}")
            else:
                raise SystemExit(f"ERROR: {path.name} no contiene edge_features/_index")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema",
        default="sample_data/sample_schema.json",
        help="Ruta a sample_schema.json",
    )
    parser.add_argument(
        "--mut-h5",
        default="/home/sartesero/modelo_optimized_gnn/local_data/hdf5/proc_483p.hdf5",
        help="Ruta al HDF5 de mutantes",
    )
    parser.add_argument(
        "--wt-h5",
        default="/home/sartesero/modelo_optimized_gnn/local_data/hdf5/wt_companion.hdf5",
        help="Ruta al HDF5 WT companion",
    )
    args = parser.parse_args()

    schema_path = Path(args.schema)
    mut_h5 = Path(args.mut_h5)
    wt_h5 = Path(args.wt_h5)

    verify_schema_documentation(schema_path)
    verify_hdf5_blocks(mut_h5, wt_h5)

    print("\nRESULTADO: OK. El apartado 8.2 queda verificado a nivel documental y estructural.")


if __name__ == "__main__":
    main()
