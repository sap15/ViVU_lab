"""Conservative paired feature masking for two complete Model A MUT-WT views."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from gnn_siamese.data.collate import MutWtPairBatch
from gnn_siamese.data.hdf5_loader import (
    HDF5GraphLoadError,
    NodeFeatureSlice,
    validate_node_feature_slices,
)
from gnn_siamese.data.pairing import PairingKey


class ModelAPairAugmentationError(ValueError):
    """Raised when a Model A paired augmentation contract is invalid."""


DEFAULT_MODEL_A_ALLOWED_FEATURE_NAMES = (
    "bsa",
    "hse",
    "hydrophobicity",
    "res_charge",
    "res_depth",
    "res_mass",
    "res_pI",
    "rsa",
    "sasa",
)

_PROTECTED_EXACT = {
    "is_mutation",
    "res_id",
    "res_id_norm",
    "res_type",
    "variant_res",
    "polarity",
    "sec_struct",
}
_PROTECTED_PREFIXES = ("diff_", "mask_")
_ALLOWED_FEATURE_NAMES = frozenset(DEFAULT_MODEL_A_ALLOWED_FEATURE_NAMES)


@dataclass(frozen=True)
class ModelAPairAugmentationConfig:
    """Only the conservative A5 paired channel-mask transform is supported.

    ``masked_value=0`` may coincide with a real biological value. The separate
    augmentation mask is audit metadata only and is never appended to encoder
    inputs. Its suitability therefore depends on a future normalization audit.
    """

    enabled: bool = True
    feature_mask_probability: float = 0.10
    allowed_feature_names: tuple[str, ...] = DEFAULT_MODEL_A_ALLOWED_FEATURE_NAMES
    masked_value: float = 0.0

    def __post_init__(self) -> None:
        probability = float(self.feature_mask_probability)
        if not 0.0 <= probability <= 1.0:
            raise ModelAPairAugmentationError(
                "feature_mask_probability must be in [0, 1]."
            )
        names = tuple(str(name) for name in self.allowed_feature_names)
        if self.enabled and not names:
            raise ModelAPairAugmentationError(
                "allowed_feature_names must be non-empty when augmentation is enabled."
            )
        if len(names) != len(set(names)):
            raise ModelAPairAugmentationError(
                "allowed_feature_names must not contain duplicates."
            )
        protected = sorted(name for name in names if _is_protected(name))
        if protected:
            raise ModelAPairAugmentationError(
                f"Protected features cannot be augmented: {protected!r}."
            )
        unknown = sorted(set(names).difference(_ALLOWED_FEATURE_NAMES))
        if unknown:
            raise ModelAPairAugmentationError(
                f"Unknown or unsupported Model A augmentation features: {unknown!r}."
            )
        if not math.isfinite(float(self.masked_value)):
            raise ModelAPairAugmentationError("masked_value must be finite.")
        object.__setattr__(self, "feature_mask_probability", probability)
        object.__setattr__(self, "allowed_feature_names", names)
        object.__setattr__(self, "masked_value", float(self.masked_value))


@dataclass(frozen=True)
class ModelAAugmentationExampleMetadata:
    """One example's auditable group-mask decision."""

    variant_id: str
    pair_key: PairingKey
    view_id: int
    effective_seed: int
    transform_id: str
    applied: bool
    status: str
    selected_feature_names: tuple[str, ...]
    masked_column_indices_MUT: tuple[int, ...]
    masked_column_indices_WT: tuple[int, ...]
    masked_value: float
    diversity_adjusted: bool
    diversity_adjustment_seed: int | None

    @property
    def masked_feature_names(self) -> tuple[str, ...]:
        """Backward-compatible alias for selected feature groups."""

        return self.selected_feature_names


@dataclass(frozen=True)
class ModelAPairView:
    """A deep-cloned pair batch with identity derived from one source of truth."""

    pair_batch: MutWtPairBatch
    view_id: int
    augmentation_metadata: tuple[ModelAAugmentationExampleMetadata, ...]

    @property
    def graph_mut(self) -> object:
        return self.pair_batch.graph_mut

    @property
    def graph_wt(self) -> object:
        return self.pair_batch.graph_wt

    @property
    def variant_ids(self) -> tuple[str, ...]:
        return tuple(self.pair_batch.variant_ids)

    @property
    def pair_keys(self) -> tuple[PairingKey, ...]:
        return tuple(self.pair_batch.pair_keys)

    @property
    def effective_seeds(self) -> tuple[int, ...]:
        return tuple(item.effective_seed for item in self.augmentation_metadata)

    @property
    def transformation_names(self) -> tuple[str, ...]:
        if any(item.applied for item in self.augmentation_metadata):
            return (ModelAPairAugmenter.transform_id,)
        return ()


def _is_protected(name: str) -> bool:
    lower = name.lower()
    return (
        name in _PROTECTED_EXACT
        or lower in _PROTECTED_EXACT
        or lower.startswith(_PROTECTED_PREFIXES)
        or "quality" in lower
        or "truncation" in lower
        or lower.endswith("_id")
        or lower.startswith("global_")
        or lower.startswith("chain")
    )


def _canonical_pair_key(pair_key: PairingKey) -> dict[str, object]:
    if not isinstance(pair_key, PairingKey):
        raise ModelAPairAugmentationError("pair_key must be a PairingKey.")
    return {
        "chain_id": None if pair_key.chain_id is None else str(pair_key.chain_id),
        "position": int(pair_key.position),
        "wt_aa": str(pair_key.wt_aa),
    }


def stable_seed(
    *,
    run_seed: int,
    epoch: int,
    variant_id: str,
    pair_key: PairingKey,
    mutant_key: str,
    wt_key: str,
    view_id: int,
    transform_id: str,
) -> int:
    """Return a process-stable seed from the complete pair identity."""

    payload = json.dumps(
        {
            "run_seed": int(run_seed),
            "epoch": int(epoch),
            "variant_id": str(variant_id),
            "pair_key": _canonical_pair_key(pair_key),
            "mutant_key": str(mutant_key),
            "wt_key": str(wt_key),
            "view_id": int(view_id),
            "transform_id": str(transform_id),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _resolved_layout(graph: object, *, role: str) -> dict[str, NodeFeatureSlice]:
    if not hasattr(graph, "node_feature_slices"):
        raise ModelAPairAugmentationError(
            f"{role} graph is missing required node_feature_slices metadata."
        )
    try:
        slices = validate_node_feature_slices(
            tuple(graph.node_feature_slices), width=int(graph.x.shape[1])
        )
    except (HDF5GraphLoadError, TypeError) as exc:
        raise ModelAPairAugmentationError(
            f"Invalid {role} node_feature_slices: {exc}"
        ) from exc
    names = tuple(getattr(graph, "node_feature_names", ()))
    slice_names = tuple(item.name for item in slices)
    if names != slice_names:
        raise ModelAPairAugmentationError(
            f"{role} node_feature_names must exactly match node_feature_slices."
        )
    return {item.name: item for item in slices}


def _columns(item: NodeFeatureSlice) -> tuple[int, ...]:
    return tuple(range(item.start, item.stop))


class ModelAPairAugmenter:
    """Create reproducible, storage-independent complete MUT-WT pair views.

    Graph batches must come from the repository's official pair collate, or a
    manual batch must already expose one normalized ``node_feature_slices``
    layout. Raw ``Batch.from_data_list`` may nest non-tensor metadata once per
    graph; the official collate validates those layouts and replaces that
    nesting with the common layout consumed here.
    """

    transform_id = "paired_feature_mask"
    diversity_transform_id = "paired_feature_mask_diversity"

    def __init__(self, config: ModelAPairAugmentationConfig | None = None) -> None:
        self.config = config or ModelAPairAugmentationConfig()

    def _validate_identity(self, pair_batch: MutWtPairBatch) -> None:
        if not isinstance(pair_batch, MutWtPairBatch):
            raise TypeError("pair_batch must be a MutWtPairBatch.")
        expected = pair_batch.batch_size
        fields = {
            "variant_ids": pair_batch.variant_ids,
            "pair_keys": pair_batch.pair_keys,
            "mutant_keys": pair_batch.mutant_keys,
            "wt_keys": pair_batch.wt_keys,
        }
        for name, values in fields.items():
            if len(values) != expected:
                raise ModelAPairAugmentationError(
                    f"batch_size must match the number of {name}."
                )
        if int(pair_batch.graph_mut.ptr.numel()) != expected + 1:
            raise ModelAPairAugmentationError("MUT ptr is incompatible with batch_size.")
        if int(pair_batch.graph_wt.ptr.numel()) != expected + 1:
            raise ModelAPairAugmentationError("WT ptr is incompatible with batch_size.")

    def _validate_and_resolve(
        self, pair_batch: MutWtPairBatch
    ) -> tuple[dict[str, NodeFeatureSlice], dict[str, NodeFeatureSlice]]:
        self._validate_identity(pair_batch)
        if not self.config.enabled:
            return {}, {}
        mut_layout = _resolved_layout(pair_batch.graph_mut, role="MUT")
        wt_layout = _resolved_layout(pair_batch.graph_wt, role="WT")
        for name in self.config.allowed_feature_names:
            if name not in mut_layout:
                raise ModelAPairAugmentationError(
                    f"Requested feature {name!r} is absent from MUT layout."
                )
            if name not in wt_layout:
                raise ModelAPairAugmentationError(
                    f"Requested feature {name!r} is absent from WT layout."
                )
            mut_width = mut_layout[name].stop - mut_layout[name].start
            wt_width = wt_layout[name].stop - wt_layout[name].start
            if mut_width != wt_width:
                raise ModelAPairAugmentationError(
                    f"Feature {name!r} has incompatible MUT/WT widths "
                    f"({mut_width} != {wt_width})."
                )
        return mut_layout, wt_layout

    def _seed(
        self,
        pair_batch: MutWtPairBatch,
        pair_index: int,
        *,
        run_seed: int,
        epoch: int,
        view_id: int,
        transform_id: str,
    ) -> int:
        return stable_seed(
            run_seed=run_seed,
            epoch=epoch,
            variant_id=pair_batch.variant_ids[pair_index],
            pair_key=pair_batch.pair_keys[pair_index],
            mutant_key=pair_batch.mutant_keys[pair_index],
            wt_key=pair_batch.wt_keys[pair_index],
            view_id=view_id,
            transform_id=transform_id,
        )

    def _sample_decisions(
        self,
        pair_batch: MutWtPairBatch,
        *,
        run_seed: int,
        epoch: int,
        view_id: int,
    ) -> tuple[tuple[bool, ...], ...]:
        decisions: list[tuple[bool, ...]] = []
        for pair_index in range(pair_batch.batch_size):
            if not self.config.enabled:
                decisions.append(tuple(False for _ in self.config.allowed_feature_names))
                continue
            seed = self._seed(
                pair_batch,
                pair_index,
                run_seed=run_seed,
                epoch=epoch,
                view_id=view_id,
                transform_id=self.transform_id,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            sampled = torch.rand(
                len(self.config.allowed_feature_names), generator=generator
            ) < self.config.feature_mask_probability
            decisions.append(tuple(bool(item) for item in sampled))
        return tuple(decisions)

    def _status(
        self, selected: Sequence[str], *, diversity_adjusted: bool
    ) -> str:
        if not self.config.enabled:
            return "disabled"
        probability = self.config.feature_mask_probability
        if probability == 0.0:
            return "degenerate_p0_no_op"
        if probability == 1.0:
            return "degenerate_p1_applied"
        if diversity_adjusted:
            return (
                "diversity_adjusted_applied"
                if selected
                else "diversity_adjusted_no_op"
            )
        return "applied" if selected else "no_groups_selected"

    def _build_view(
        self,
        pair_batch: MutWtPairBatch,
        *,
        run_seed: int,
        epoch: int,
        view_id: int,
        decisions: Sequence[Sequence[bool]],
        diversity_adjusted: Sequence[bool],
        diversity_adjustment_seeds: Sequence[int | None],
        mut_layout: Mapping[str, NodeFeatureSlice],
        wt_layout: Mapping[str, NodeFeatureSlice],
    ) -> ModelAPairView:
        cloned = deepcopy(pair_batch)
        mut_mask = torch.zeros_like(cloned.graph_mut.x, dtype=torch.bool)
        wt_mask = torch.zeros_like(cloned.graph_wt.x, dtype=torch.bool)
        metadata: list[ModelAAugmentationExampleMetadata] = []

        for pair_index in range(pair_batch.batch_size):
            selected = tuple(
                name
                for name, masked in zip(
                    self.config.allowed_feature_names, decisions[pair_index]
                )
                if masked
            )
            mut_columns = tuple(
                column
                for name in selected
                for column in _columns(mut_layout[name])
            )
            wt_columns = tuple(
                column
                for name in selected
                for column in _columns(wt_layout[name])
            )
            mut_start = int(cloned.graph_mut.ptr[pair_index].item())
            mut_end = int(cloned.graph_mut.ptr[pair_index + 1].item())
            wt_start = int(cloned.graph_wt.ptr[pair_index].item())
            wt_end = int(cloned.graph_wt.ptr[pair_index + 1].item())
            if mut_columns:
                cloned.graph_mut.x[mut_start:mut_end, mut_columns] = (
                    self.config.masked_value
                )
                cloned.graph_wt.x[wt_start:wt_end, wt_columns] = (
                    self.config.masked_value
                )
                mut_mask[mut_start:mut_end, mut_columns] = True
                wt_mask[wt_start:wt_end, wt_columns] = True
            seed = self._seed(
                pair_batch,
                pair_index,
                run_seed=run_seed,
                epoch=epoch,
                view_id=view_id,
                transform_id=self.transform_id,
            )
            adjusted = bool(diversity_adjusted[pair_index])
            adjustment_seed = diversity_adjustment_seeds[pair_index]
            if adjusted != (adjustment_seed is not None):
                raise ModelAPairAugmentationError(
                    "diversity_adjusted and diversity_adjustment_seed are inconsistent."
                )
            metadata.append(
                ModelAAugmentationExampleMetadata(
                    variant_id=str(pair_batch.variant_ids[pair_index]),
                    pair_key=pair_batch.pair_keys[pair_index],
                    view_id=view_id,
                    effective_seed=seed,
                    transform_id=self.transform_id,
                    applied=bool(selected),
                    status=self._status(selected, diversity_adjusted=adjusted),
                    selected_feature_names=selected,
                    masked_column_indices_MUT=mut_columns,
                    masked_column_indices_WT=wt_columns,
                    masked_value=self.config.masked_value,
                    diversity_adjusted=adjusted,
                    diversity_adjustment_seed=adjustment_seed,
                )
            )

        cloned.graph_mut.augmentation_feature_mask = mut_mask
        cloned.graph_wt.augmentation_feature_mask = wt_mask
        cloned.graph_mut.augmentation_view_id = view_id
        cloned.graph_wt.augmentation_view_id = view_id
        return ModelAPairView(
            pair_batch=cloned,
            view_id=view_id,
            augmentation_metadata=tuple(metadata),
        )

    def create_view(
        self,
        pair_batch: MutWtPairBatch,
        *,
        run_seed: int,
        epoch: int,
        view_id: int,
    ) -> ModelAPairView:
        """Create one normally sampled view without cross-view adjustment."""

        if view_id not in (1, 2):
            raise ModelAPairAugmentationError("view_id must be 1 or 2.")
        mut_layout, wt_layout = self._validate_and_resolve(pair_batch)
        decisions = self._sample_decisions(
            pair_batch, run_seed=run_seed, epoch=epoch, view_id=view_id
        )
        return self._build_view(
            pair_batch,
            run_seed=run_seed,
            epoch=epoch,
            view_id=view_id,
            decisions=decisions,
            diversity_adjusted=(False,) * pair_batch.batch_size,
            diversity_adjustment_seeds=(None,) * pair_batch.batch_size,
            mut_layout=mut_layout,
            wt_layout=wt_layout,
        )

    def create_two_views(
        self,
        pair_batch: MutWtPairBatch,
        *,
        run_seed: int,
        epoch: int,
    ) -> tuple[ModelAPairView, ModelAPairView]:
        """Create two views, deterministically flipping view 2 on collisions.

        For ``0 < p < 1``, one group selected by a separate stable seed is
        inverted in view 2 whenever both normally sampled masks are identical.
        This bounded adjustment preserves every other Bernoulli decision.
        """

        mut_layout, wt_layout = self._validate_and_resolve(pair_batch)
        decisions1 = self._sample_decisions(
            pair_batch, run_seed=run_seed, epoch=epoch, view_id=1
        )
        decisions2 = [
            list(items)
            for items in self._sample_decisions(
                pair_batch, run_seed=run_seed, epoch=epoch, view_id=2
            )
        ]
        adjusted = [False] * pair_batch.batch_size
        adjustment_seeds: list[int | None] = [None] * pair_batch.batch_size
        if (
            self.config.enabled
            and 0.0 < self.config.feature_mask_probability < 1.0
        ):
            group_count = len(self.config.allowed_feature_names)
            for pair_index, first in enumerate(decisions1):
                if tuple(decisions2[pair_index]) == tuple(first):
                    adjustment_seed = self._seed(
                        pair_batch,
                        pair_index,
                        run_seed=run_seed,
                        epoch=epoch,
                        view_id=2,
                        transform_id=self.diversity_transform_id,
                    )
                    group_index = adjustment_seed % group_count
                    decisions2[pair_index][group_index] = not decisions2[pair_index][
                        group_index
                    ]
                    adjusted[pair_index] = True
                    adjustment_seeds[pair_index] = adjustment_seed

        view1 = self._build_view(
            pair_batch,
            run_seed=run_seed,
            epoch=epoch,
            view_id=1,
            decisions=decisions1,
            diversity_adjusted=(False,) * pair_batch.batch_size,
            diversity_adjustment_seeds=(None,) * pair_batch.batch_size,
            mut_layout=mut_layout,
            wt_layout=wt_layout,
        )
        view2 = self._build_view(
            pair_batch,
            run_seed=run_seed,
            epoch=epoch,
            view_id=2,
            decisions=tuple(tuple(items) for items in decisions2),
            diversity_adjusted=tuple(adjusted),
            diversity_adjustment_seeds=tuple(adjustment_seeds),
            mut_layout=mut_layout,
            wt_layout=wt_layout,
        )
        return view1, view2
