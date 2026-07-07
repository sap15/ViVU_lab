from __future__ import annotations

import pytest

from gnn_siamese.data.feature_selection import (
    MissingFeatureGroupError,
    MissingSchemaFeatureError,
    resolve_edge_feature_names,
    resolve_node_feature_names,
    split_encoder_inputs_and_auxiliary_features,
)


def _build_config() -> dict:
    return {
        "data": {
            "use_global_energy": False,
            "global_features_as_input": [],
        },
        "features": {
            "node_groups": ["structure", "biochemistry", "diff_bioq"],
            "edge_groups": ["distance", "contact", "sequential_relation"],
            "graph_groups": [],
            "excluded_from_encoder_base": ["diff_polarity"],
            "node_metadata": ["_chain_id", "_name", "_position"],
            "edge_metadata": ["_index", "_name"],
            "confounders": [
                "custom_structure_energy",
                "custom_complex_energy_phenotype",
                "graph_num_nodes",
                "graph_num_edges",
                "delta_nodes",
                "res_depth_missing",
            ],
            "structure": {
                "enabled": True,
                "names": ["bsa", "hse", "res_depth", "rsa", "sasa", "sec_struct"],
            },
            "biochemistry": {
                "enabled": True,
                "names": [
                    "hb_acceptors",
                    "hb_donors",
                    "hydrophobicity",
                    "polarity",
                    "res_charge",
                    "res_mass",
                    "res_pI",
                    "res_size",
                    "res_type",
                ],
            },
            "diff_bioq": {
                "enabled": True,
                "names": [
                    "diff_mass",
                    "diff_charge",
                    "diff_pI",
                    "diff_size",
                    "diff_hb_donors",
                    "diff_hb_acceptors",
                    "diff_polarity",
                ],
                "require_masks": True,
                "mask_prefix": "mask_",
            },
            "distance": {
                "enabled": True,
                "names": ["distance"],
            },
            "contact": {
                "enabled": True,
                "names": ["covalent", "electrostatic", "vanderwaals"],
            },
            "sequential_relation": {
                "enabled": True,
                "names": ["seq_sep"],
            },
        },
        "targets": {
            "supervised_phenotype": {
                "enabled": False,
                "name": "custom_complex_energy_phenotype",
                "required": False,
            }
        },
    }


def _build_schema() -> dict:
    return {
        "graph_layout": {
            "node_features": {
                "feature_datasets": {
                    "_chain_id": {},
                    "_name": {},
                    "_position": {},
                    "bsa": {},
                    "hse": {},
                    "res_depth": {},
                    "rsa": {},
                    "sasa": {},
                    "sec_struct": {},
                    "hb_acceptors": {},
                    "hb_donors": {},
                    "hydrophobicity": {},
                    "polarity": {},
                    "res_charge": {},
                    "res_mass": {},
                    "res_pI": {},
                    "res_size": {},
                    "res_type": {},
                    "diff_mass": {},
                    "diff_charge": {},
                    "diff_pI": {},
                    "diff_size": {},
                    "diff_hb_donors": {},
                    "diff_hb_acceptors": {},
                    "diff_polarity": {},
                    "mask_diff_mass": {},
                    "mask_diff_charge": {},
                    "mask_diff_pI": {},
                    "mask_diff_size": {},
                    "mask_diff_hb_donors": {},
                    "mask_diff_hb_acceptors": {},
                    "var_HSE": {},
                }
            },
            "edge_features": {
                "feature_datasets": {
                    "_index": {},
                    "distance": {},
                    "covalent": {},
                    "electrostatic": {},
                    "seq_sep": {},
                    "vanderwaals": {},
                }
            },
            "graph_features": {
                "feature_datasets": {
                    "custom_structure_energy": {},
                    "custom_complex_energy_phenotype": {},
                    "graph_num_nodes": {},
                    "graph_num_edges": {},
                    "delta_nodes": {},
                }
            },
        }
    }


def test_resolve_node_features_from_config_groups() -> None:
    selected = resolve_node_feature_names(_build_config(), _build_schema())

    assert selected == [
        "bsa",
        "hse",
        "res_depth",
        "rsa",
        "sasa",
        "sec_struct",
        "hb_acceptors",
        "hb_donors",
        "hydrophobicity",
        "polarity",
        "res_charge",
        "res_mass",
        "res_pI",
        "res_size",
        "res_type",
        "diff_mass",
        "diff_charge",
        "diff_pI",
        "diff_size",
        "diff_hb_donors",
        "diff_hb_acceptors",
    ]


def test_excludes_masks_metadata_and_confounders_from_encoder_inputs() -> None:
    split = split_encoder_inputs_and_auxiliary_features(_build_config(), _build_schema())

    assert "_name" not in split["encoder_node_features"]
    assert "_position" not in split["encoder_node_features"]
    assert "mask_diff_mass" not in split["encoder_node_features"]
    assert "diff_polarity" not in split["encoder_node_features"]
    assert split["encoder_graph_features"] == []
    assert "custom_structure_energy" in split["auxiliary_graph_features"]


def test_resolve_edge_features_from_config_groups() -> None:
    selected = resolve_edge_feature_names(_build_config(), _build_schema())

    assert selected == ["distance", "covalent", "electrostatic", "vanderwaals", "seq_sep"]
    assert "_index" not in selected


def test_missing_group_or_missing_feature_fails_clearly() -> None:
    bad_group_config = _build_config()
    bad_group_config["features"]["node_groups"] = ["structure", "unknown_group"]

    with pytest.raises(MissingFeatureGroupError, match="unknown_group"):
        resolve_node_feature_names(bad_group_config, _build_schema())

    bad_feature_schema = _build_schema()
    del bad_feature_schema["graph_layout"]["node_features"]["feature_datasets"]["diff_mass"]

    with pytest.raises(MissingSchemaFeatureError, match="diff_mass"):
        resolve_node_feature_names(_build_config(), bad_feature_schema)


def test_availability_masks_are_returned_as_auxiliary_not_encoder_inputs() -> None:
    split = split_encoder_inputs_and_auxiliary_features(_build_config(), _build_schema())

    assert split["node_availability_masks"]["diff_mass"] == "mask_diff_mass"
    assert split["node_availability_masks"]["diff_hb_acceptors"] == "mask_diff_hb_acceptors"
    assert "mask_diff_mass" not in split["encoder_node_features"]
