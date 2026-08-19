"""Minimal paired mutant-WT dataset built on top of the HDF5 loaders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py

from gnn_siamese.data.feature_selection import split_encoder_inputs_and_auxiliary_features
from gnn_siamese.data.hdf5_loader import HDF5GraphComponents, load_hdf5_graph_components
from gnn_siamese.data.node_pair_alignment import NodePairAlignment, align_node_pair
from gnn_siamese.data.pairing import (
    BIOLOGICAL_VARIANT,
    NATIVE_WT_CONTROL,
    PairingKey,
    classify_variant_record,
    pair_mutants_with_wt,
)


class MutWtPairDatasetError(ValueError):
    """Raised when the paired dataset cannot resolve or validate a sample."""


@dataclass(frozen=True)
class MutWtPairRecord:
    """Stable record describing one resolved mutant-WT pair."""

    pair_key: PairingKey
    variant_id: str
    mutant_key: str
    wt_key: str
    position: int
    wt_aa: str
    mut_aa: str
    chain_id: str | None
    mutant_source_h5: str
    wt_source_h5: str
    record_type: str = BIOLOGICAL_VARIANT
    role: str = "training_variant"
    trainable: bool = True
    available_for_inference: bool = True


@dataclass(frozen=True)
class MutWtPairSample:
    """One paired sample returned by :class:`MutWtPairDataset`."""

    graph_mut: HDF5GraphComponents
    graph_wt: HDF5GraphComponents
    metadata: dict[str, Any]
    pair_key: PairingKey
    variant_id: str
    mutant_key: str
    wt_key: str
    node_pair_alignment: NodePairAlignment | None = None

    @property
    def mut_aligned_index(self) -> tuple[int, ...]:
        return () if self.node_pair_alignment is None else self.node_pair_alignment.mut_aligned_index

    @property
    def wt_aligned_index(self) -> tuple[int, ...]:
        return () if self.node_pair_alignment is None else self.node_pair_alignment.wt_aligned_index

    @property
    def exists_MUT(self) -> tuple[bool, ...]:
        return () if self.node_pair_alignment is None else self.node_pair_alignment.exists_MUT

    @property
    def exists_WT(self) -> tuple[bool, ...]:
        return () if self.node_pair_alignment is None else self.node_pair_alignment.exists_WT

    @property
    def local_mut_aligned_index(self) -> tuple[int, ...]:
        if self.node_pair_alignment is None or not self.node_pair_alignment.local_views:
            return ()
        return self.node_pair_alignment.local_view(
            self.node_pair_alignment.technical_radius_angstrom
        ).local_mut_aligned_index

    @property
    def local_wt_aligned_index(self) -> tuple[int, ...]:
        if self.node_pair_alignment is None or not self.node_pair_alignment.local_views:
            return ()
        return self.node_pair_alignment.local_view(
            self.node_pair_alignment.technical_radius_angstrom
        ).local_wt_aligned_index


@dataclass(frozen=True)
class GraphInputSpec:
    """Final encoder input metadata inferred from loaded paired graphs."""

    configured_node_feature_names: tuple[str, ...]
    final_node_feature_names: tuple[str, ...]
    edge_feature_names: tuple[str, ...]
    node_input_dim: int
    edge_input_dim: int


def _list_graph_keys(h5_path: str | Path) -> list[str]:
    with h5py.File(h5_path, "r") as handle:
        return sorted(str(key) for key in handle.keys())


def _normalize_graph_keys(
    h5_path: str | Path,
    graph_keys: Sequence[str] | None,
    *,
    role: str,
) -> list[str]:
    if graph_keys is None:
        return _list_graph_keys(h5_path)
    normalized = [str(key) for key in graph_keys]
    if not normalized:
        raise MutWtPairDatasetError(f"{role} graph keys cannot be empty.")
    return normalized


def _build_pair_records(
    mutant_h5_path: str | Path,
    wt_h5_path: str | Path,
    mutant_graph_keys: Sequence[str] | None,
    wt_graph_keys: Sequence[str] | None,
) -> tuple[list[MutWtPairRecord], list[MutWtPairRecord]]:
    mutant_source = str(mutant_h5_path)
    wt_source = str(wt_h5_path)
    mutant_records = [
        {"variant_id": key, "graph_id": key, "source_path": mutant_source}
        for key in _normalize_graph_keys(mutant_h5_path, mutant_graph_keys, role="Mutant")
    ]
    wt_records = [
        {"variant_id": key, "graph_id": key, "source_path": wt_source}
        for key in _normalize_graph_keys(wt_h5_path, wt_graph_keys, role="WT")
    ]

    native_control_ids = {
        str(record["variant_id"])
        for record in mutant_records
        if classify_variant_record(record) == NATIVE_WT_CONTROL
    }
    # ``PKP2_WT`` is also the stable suffix of every record in the dedicated
    # WT-companion file.  Exactly one WT->WT query in the mutant-input
    # inventory identifies the deliberate native control; multiple candidates
    # indicate a swapped/misdeclared WT-companion role and must not be silently
    # reclassified as controls.
    if len(native_control_ids) != 1:
        native_control_ids = set()

    paired = pair_mutants_with_wt(mutant_records, wt_records)
    ordered = sorted(
        paired,
        key=lambda item: (
            "" if item["chain_id"] is None else str(item["chain_id"]),
            int(item["position"]),
            str(item["wt_aa"]),
            str(item["mut_aa"]),
            str(item["variant_id"]),
            str(item["wt_companion_id"]),
        ),
    )
    records: list[MutWtPairRecord] = []
    for item in ordered:
        record_type = (
            NATIVE_WT_CONTROL
            if str(item["variant_id"]) in native_control_ids
            else BIOLOGICAL_VARIANT
        )
        records.append(
            MutWtPairRecord(
                pair_key=item["pairing_signature"],
                variant_id=str(item["variant_id"]),
                mutant_key=str(item["graph_id"]),
                wt_key=str(item["wt_companion_id"]),
                position=int(item["position"]),
                wt_aa=str(item["wt_aa"]),
                mut_aa=str(item["mut_aa"]),
                chain_id=None
                if item["chain_id"] is None
                else str(item["chain_id"]),
                mutant_source_h5=mutant_source,
                wt_source_h5=wt_source,
                record_type=record_type,
                role=(
                    "evaluation_control"
                    if record_type == NATIVE_WT_CONTROL
                    else "training_variant"
                ),
                trainable=record_type == BIOLOGICAL_VARIANT,
            )
        )
    return (
        [record for record in records if record.record_type == BIOLOGICAL_VARIANT],
        [record for record in records if record.record_type == NATIVE_WT_CONTROL],
    )


def _validate_pair_compatibility(
    pair: MutWtPairRecord,
    graph_mut: HDF5GraphComponents,
    graph_wt: HDF5GraphComponents,
    *,
    configured_node_feature_names: tuple[str, ...] | None = None,
) -> None:
    if graph_mut.x.shape[1] != graph_wt.x.shape[1]:
        configured_names = tuple(configured_node_feature_names or ())
        raise MutWtPairDatasetError(
            f"Incompatible node feature dimensions for pair {pair.variant_id!r}: "
            f"mutant has {graph_mut.x.shape[1]} columns, WT has {graph_wt.x.shape[1]} columns, "
            f"configured encoder selection has {len(configured_names)} columns. "
            f"Configured node features: {configured_names!r}. "
            f"Mutant final node features: {graph_mut.node_feature_names!r}. "
            f"WT final node features: {graph_wt.node_feature_names!r}."
        )
    if graph_mut.edge_attr.shape[1] != graph_wt.edge_attr.shape[1]:
        raise MutWtPairDatasetError(
            f"Incompatible edge feature dimensions for pair {pair.variant_id!r}: "
            f"mutant has {graph_mut.edge_attr.shape[1]} columns, WT has {graph_wt.edge_attr.shape[1]}."
        )
    if graph_mut.node_feature_names != graph_wt.node_feature_names:
        raise MutWtPairDatasetError(
            f"Incompatible node feature names for pair {pair.variant_id!r}: "
            f"{graph_mut.node_feature_names!r} != {graph_wt.node_feature_names!r}."
        )
    if graph_mut.edge_feature_names != graph_wt.edge_feature_names:
        raise MutWtPairDatasetError(
            f"Incompatible edge feature names for pair {pair.variant_id!r}: "
            f"{graph_mut.edge_feature_names!r} != {graph_wt.edge_feature_names!r}."
        )


def _align_pair_nodes(pair: MutWtPairRecord) -> NodePairAlignment:
    """Run the A1.4 aligner directly on the documented raw HDF5 fields."""

    with h5py.File(pair.mutant_source_h5, "r") as mutant_handle, h5py.File(
        pair.wt_source_h5, "r"
    ) as wt_handle:
        mutant_nodes = mutant_handle[pair.mutant_key]["node_features"]
        wt_nodes = wt_handle[pair.wt_key]["node_features"]
        return align_node_pair(
            mutant_nodes["_chain_id"][()],
            mutant_nodes["res_id"][()],
            mutant_nodes["_position"][()],
            wt_nodes["_chain_id"][()],
            wt_nodes["res_id"][()],
            wt_nodes["_position"][()],
            anchor_key=(pair.chain_id, pair.position),
        )


class MutWtPairDataset:
    """Minimal deterministic dataset of paired mutant and WT graph components."""

    def __init__(
        self,
        *,
        mutant_h5_path: str | Path,
        wt_h5_path: str | Path,
        config: Mapping[str, Any],
        schema: Mapping[str, Any],
        mutant_graph_keys: Sequence[str] | None = None,
        wt_graph_keys: Sequence[str] | None = None,
    ) -> None:
        self.mutant_h5_path = str(mutant_h5_path)
        self.wt_h5_path = str(wt_h5_path)
        self.config = config
        self.schema = schema

        selection = split_encoder_inputs_and_auxiliary_features(config, schema)
        self.configured_node_feature_names = tuple(selection["encoder_node_features"])
        self.node_feature_names = self.configured_node_feature_names
        self.edge_feature_names = tuple(selection["encoder_edge_features"])
        self.node_availability_masks = dict(selection["node_availability_masks"])
        self.pairs, self.native_wt_controls = _build_pair_records(
            mutant_h5_path=mutant_h5_path,
            wt_h5_path=wt_h5_path,
            mutant_graph_keys=mutant_graph_keys,
            wt_graph_keys=wt_graph_keys,
        )
        self.biological_variant_count = len(self.pairs)
        self.hdf5_mutant_input_group_count = (
            self.biological_variant_count + len(self.native_wt_controls)
        )
        self.input_spec = self._infer_input_spec()
        self.node_feature_names = self.input_spec.final_node_feature_names
        self.edge_feature_names = self.input_spec.edge_feature_names
        self.node_input_dim = self.input_spec.node_input_dim
        self.edge_input_dim = self.input_spec.edge_input_dim

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair_graphs(self, pair: MutWtPairRecord) -> tuple[HDF5GraphComponents, HDF5GraphComponents]:
        graph_mut = load_hdf5_graph_components(
            pair.mutant_source_h5,
            pair.mutant_key,
            node_feature_names=self.configured_node_feature_names,
            edge_feature_names=self.edge_feature_names,
            node_availability_masks=self.node_availability_masks,
            config=self.config,
        )
        graph_wt = load_hdf5_graph_components(
            pair.wt_source_h5,
            pair.wt_key,
            node_feature_names=self.configured_node_feature_names,
            edge_feature_names=self.edge_feature_names,
            node_availability_masks=self.node_availability_masks,
            config=self.config,
        )
        _validate_pair_compatibility(
            pair,
            graph_mut,
            graph_wt,
            configured_node_feature_names=self.configured_node_feature_names,
        )
        return graph_mut, graph_wt

    def _infer_input_spec(self) -> GraphInputSpec:
        if not self.pairs:
            raise MutWtPairDatasetError("MutWtPairDataset resolved zero mutant-WT pairs.")

        graph_mut, graph_wt = self._load_pair_graphs(self.pairs[0])
        return GraphInputSpec(
            configured_node_feature_names=self.configured_node_feature_names,
            final_node_feature_names=graph_mut.node_feature_names,
            edge_feature_names=graph_mut.edge_feature_names,
            node_input_dim=int(graph_mut.x.shape[1]),
            edge_input_dim=int(graph_mut.edge_attr.shape[1]),
        )

    def __getitem__(self, index: int) -> MutWtPairSample:
        pair = self.pairs[index]
        return self._load_sample(pair)

    def get_native_wt_control(self, index: int = 0) -> MutWtPairSample:
        """Load one preserved native-WT control outside every split/loss dataset."""

        try:
            pair = self.native_wt_controls[index]
        except IndexError as exc:
            raise MutWtPairDatasetError(
                f"Native WT control index {index} is unavailable; "
                f"inventory contains {len(self.native_wt_controls)} control(s)."
            ) from exc
        return self._load_sample(pair)

    def _load_sample(self, pair: MutWtPairRecord) -> MutWtPairSample:
        graph_mut, graph_wt = self._load_pair_graphs(pair)
        node_pair_alignment = _align_pair_nodes(pair)

        metadata = {
            "variant_id": pair.variant_id,
            "position": pair.position,
            "wt_aa": pair.wt_aa,
            "mut_aa": pair.mut_aa,
            "chain_id": pair.chain_id,
            "mutant_key": pair.mutant_key,
            "wt_key": pair.wt_key,
            "mutant_source_h5": pair.mutant_source_h5,
            "wt_source_h5": pair.wt_source_h5,
            "source_h5": {
                "mutant": pair.mutant_source_h5,
                "wt": pair.wt_source_h5,
            },
            "record_type": pair.record_type,
            "role": pair.role,
            "trainable": pair.trainable,
            "used_for_split": pair.trainable,
            "used_for_training": pair.trainable,
            "used_for_validation": pair.trainable,
            "used_for_test_loss": pair.trainable,
            "available_for_inference": pair.available_for_inference,
        }
        return MutWtPairSample(
            graph_mut=graph_mut,
            graph_wt=graph_wt,
            metadata=metadata,
            pair_key=pair.pair_key,
            variant_id=pair.variant_id,
            mutant_key=pair.mutant_key,
            wt_key=pair.wt_key,
            node_pair_alignment=node_pair_alignment,
        )

    def subset_with_pairs(self, pairs: Sequence[MutWtPairRecord]) -> "MutWtPairDataset":
        """Return a shallow dataset clone restricted to already validated pair records."""

        clone = object.__new__(MutWtPairDataset)
        clone.mutant_h5_path = self.mutant_h5_path
        clone.wt_h5_path = self.wt_h5_path
        clone.config = self.config
        clone.schema = self.schema
        clone.configured_node_feature_names = self.configured_node_feature_names
        clone.node_feature_names = self.node_feature_names
        clone.edge_feature_names = self.edge_feature_names
        clone.node_availability_masks = self.node_availability_masks
        clone.pairs = list(pairs)
        clone.native_wt_controls = list(self.native_wt_controls)
        clone.biological_variant_count = self.biological_variant_count
        clone.hdf5_mutant_input_group_count = self.hdf5_mutant_input_group_count
        clone.input_spec = self.input_spec
        clone.node_input_dim = self.node_input_dim
        clone.edge_input_dim = self.edge_input_dim
        if not clone.pairs:
            raise MutWtPairDatasetError("Smoke subset resolved zero mutant-WT pairs.")
        return clone
