"""Data loading and validation helpers."""

from gnn_siamese.data.feature_selection import (
    FeatureSelectionError,
    MissingFeatureGroupError,
    MissingSchemaFeatureError,
    resolve_edge_feature_names,
    resolve_node_feature_names,
    split_encoder_inputs_and_auxiliary_features,
)
from gnn_siamese.data.hdf5_loader import (
    HDF5GraphComponents,
    HDF5GraphLoadError,
    build_is_mutation_channel,
    extract_variant_metadata,
    load_hdf5_graph_components,
    normalize_edge_index,
    validate_graph_components,
)
from gnn_siamese.data.pairing import (
    AmbiguousWTCompanionError,
    IncompleteSignatureError,
    MissingWTCompanionError,
    PairingError,
    PairingKey,
    SignatureParseError,
    VariantSignature,
    build_wt_index,
    pair_mutants_with_wt,
    parse_variant_signature,
    resolve_wt_companion,
)

__all__ = [
    "AmbiguousWTCompanionError",
    "FeatureSelectionError",
    "HDF5GraphComponents",
    "HDF5GraphLoadError",
    "IncompleteSignatureError",
    "MissingFeatureGroupError",
    "MissingWTCompanionError",
    "MissingSchemaFeatureError",
    "PairingError",
    "PairingKey",
    "SignatureParseError",
    "VariantSignature",
    "build_wt_index",
    "build_is_mutation_channel",
    "extract_variant_metadata",
    "load_hdf5_graph_components",
    "normalize_edge_index",
    "pair_mutants_with_wt",
    "parse_variant_signature",
    "resolve_edge_feature_names",
    "resolve_node_feature_names",
    "resolve_wt_companion",
    "split_encoder_inputs_and_auxiliary_features",
    "validate_graph_components",
]
