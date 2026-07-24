from __future__ import annotations

from pathlib import Path
from typing import Any

from gnn_siamese.data.smoke_data import create_synthetic_mut_wt_hdf5, write_smoke_schema_json


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def _base_data_config() -> dict[str, Any]:
    return {
        "features": {
            "node_groups": ["structure", "biochemistry", "diff_bioq"],
            "edge_groups": ["distance", "contact"],
            "graph_groups": [],
            "excluded_from_encoder_base": ["diff_polarity"],
            "confounders": ["custom_structure_energy"],
            "node_metadata": ["_chain_id", "_name", "_position"],
            "edge_metadata": ["_index"],
            "structure": {"enabled": True, "names": ["bsa"]},
            "biochemistry": {"enabled": True, "names": ["res_mass"]},
            "diff_bioq": {
                "enabled": True,
                "names": ["diff_mass", "diff_charge", "diff_pI", "diff_size", "diff_polarity"],
                "require_masks": True,
                "mask_prefix": "mask_",
            },
            "distance": {"enabled": True, "names": ["distance"]},
            "contact": {"enabled": True, "names": ["covalent"]},
        },
        "data": {
            "mutation_node": {
                "source": "diff_features",
                "probes": ["diff_mass", "diff_charge", "diff_pI", "diff_size"],
                "epsilon": 1.0e-12,
                "require_exactly_one_for_missense": True,
                "wt_expected_count": 0,
                "create_is_mutation_channel": True,
            }
        },
    }


def write_schema_json(path: Path) -> None:
    write_smoke_schema_json(path, _base_data_config())


def create_multi_pair_hdf5(mutant_path: Path, wt_path: Path) -> None:
    create_synthetic_mut_wt_hdf5(mutant_path, wt_path, _base_data_config())


def create_multi_pair_hdf5_with_variants(
    mutant_path: Path,
    wt_path: Path,
    *,
    variants: list[dict[str, Any]],
) -> None:
    create_synthetic_mut_wt_hdf5(mutant_path, wt_path, _base_data_config(), variants=variants)


def build_model_b_config(
    mutant_path: Path,
    wt_path: Path,
    schema_path: Path,
    split_path: Path,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = _base_data_config()
    config: dict[str, Any] = {
        **base,
        "project": {"seed": 123},
        "paths": {
            "mutants_hdf5": str(mutant_path),
            "wt_companion_hdf5": str(wt_path),
            "sample_schema": str(schema_path),
            "sample_data_root": "sample_data",
        },
        "model": {
            "architecture": "model_b",
            "hidden_dim": 16,
            "graph_dim": 12,
            "num_layers": 2,
            "dropout": 0.0,
            "pooling": {"fusion": {"output_dim": 16}},
            "projection_instance": {
                "enabled": True,
                "hidden_dim": 10,
                "output_dim": 8,
                "num_layers": 2,
                "normalize_output": True,
            },
            "relational": {"severity": {"epsilon": 1.0e-8}},
            "mlp_delta": {
                "enabled": False,
                "hidden_dim": 24,
                "output_dim": 12,
                "num_layers": 2,
                "dropout": 0.0,
            },
            "projection_pair": {
                "enabled": False,
                "input": "r_delta",
                "hidden_dim": 12,
                "output_dim": 6,
                "normalize_output": True,
            },
        },
        "augmentation": {
            "enabled": True,
            "num_views": 2,
            "preserve_mutation_node": True,
            "feature_dropout": {
                "enabled": True,
                "probability": 0.2,
                "allowed_feature_names": ["bsa", "res_mass"],
                "protect": ["is_mutation", "diff_", "mask_"],
            },
            "feature_jitter": {
                "enabled": True,
                "std": 0.05,
                "allowed_feature_names": ["bsa", "res_mass"],
                "protect": ["is_mutation", "diff_", "mask_"],
            },
            "edge_dropout": {"enabled": True, "probability": 0.2},
        },
        "training": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "optimizer": "adamw",
            "num_workers": 0,
            "smoke_test": {"enabled": False, "epochs": 1, "batch_size": 2},
        },
        "loss": {
            "main": "nt_xent",
            "temperature": 0.2,
            "positive_pair": "same_mutant_augmentations",
            "use_wt_as_strong_positive": False,
        },
        "split": {
            "type": "leave_position_out",
            "validation_fraction": 0.25,
            "test_fraction": 0.25,
            "group_key": "position",
            "shuffle_groups": True,
            "seed": 123,
            "enforce_no_position_overlap": True,
            "enforce_no_variant_overlap": True,
            "persist_path": str(split_path),
            "allow_create": True,
        },
    }
    if overrides:
        config = _deep_update(config, overrides)
    return config
