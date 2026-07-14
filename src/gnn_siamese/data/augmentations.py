"""Controlled graph augmentations for two mutant views in Model B training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch


class AugmentationConfigError(ValueError):
    """Raised when graph augmentation settings are invalid or unsupported."""


def clone_graph_batch(graph_batch: object) -> object:
    """Clone a PyG `Data` or `Batch` object without mutating the original tensors."""

    return deepcopy(graph_batch)


@dataclass(frozen=True)
class GraphAugmentationConfig:
    """Validated augmentation config for mutant-view generation."""

    enabled: bool
    feature_dropout_enabled: bool
    feature_dropout_probability: float
    feature_dropout_allowed_feature_names: tuple[str, ...]
    feature_dropout_protect: tuple[str, ...]
    feature_jitter_enabled: bool
    feature_jitter_std: float
    feature_jitter_allowed_feature_names: tuple[str, ...]
    feature_jitter_protect: tuple[str, ...]
    edge_dropout_enabled: bool
    edge_dropout_probability: float
    preserve_mutation_node: bool
    num_views: int
    seed: int


def resolve_graph_augmentation_config(
    config: Mapping[str, Any],
    *,
    seed: int,
) -> GraphAugmentationConfig:
    augmentation_cfg = config.get("augmentation")
    if not isinstance(augmentation_cfg, Mapping):
        raise AugmentationConfigError("config.augmentation must be a mapping.")

    num_views = int(augmentation_cfg.get("num_views", 2))
    if num_views != 2:
        raise AugmentationConfigError("Model B baseline requires augmentation.num_views == 2.")

    feature_dropout_cfg = augmentation_cfg.get("feature_dropout", {})
    feature_jitter_cfg = augmentation_cfg.get("feature_jitter", {})
    edge_dropout_cfg = augmentation_cfg.get("edge_dropout", {})
    if not isinstance(feature_dropout_cfg, Mapping):
        raise AugmentationConfigError("augmentation.feature_dropout must be a mapping.")
    if not isinstance(feature_jitter_cfg, Mapping):
        raise AugmentationConfigError("augmentation.feature_jitter must be a mapping.")
    if not isinstance(edge_dropout_cfg, Mapping):
        raise AugmentationConfigError("augmentation.edge_dropout must be a mapping.")

    feature_dropout_probability = float(feature_dropout_cfg.get("probability", 0.0))
    feature_jitter_std = float(feature_jitter_cfg.get("std", 0.0))
    edge_dropout_probability = float(edge_dropout_cfg.get("probability", 0.0))
    if not 0.0 <= feature_dropout_probability < 1.0:
        raise AugmentationConfigError(
            "augmentation.feature_dropout.probability must be in [0, 1)."
        )
    if feature_jitter_std < 0.0:
        raise AugmentationConfigError("augmentation.feature_jitter.std must be non-negative.")
    if not 0.0 <= edge_dropout_probability < 1.0:
        raise AugmentationConfigError("augmentation.edge_dropout.probability must be in [0, 1).")

    return GraphAugmentationConfig(
        enabled=bool(augmentation_cfg.get("enabled", False)),
        feature_dropout_enabled=bool(feature_dropout_cfg.get("enabled", False)),
        feature_dropout_probability=feature_dropout_probability,
        feature_dropout_allowed_feature_names=tuple(
            str(name) for name in feature_dropout_cfg.get("allowed_feature_names", ())
        ),
        feature_dropout_protect=tuple(str(name) for name in feature_dropout_cfg.get("protect", ())),
        feature_jitter_enabled=bool(feature_jitter_cfg.get("enabled", False)),
        feature_jitter_std=feature_jitter_std,
        feature_jitter_allowed_feature_names=tuple(
            str(name) for name in feature_jitter_cfg.get("allowed_feature_names", ())
        ),
        feature_jitter_protect=tuple(str(name) for name in feature_jitter_cfg.get("protect", ())),
        edge_dropout_enabled=bool(edge_dropout_cfg.get("enabled", False)),
        edge_dropout_probability=edge_dropout_probability,
        preserve_mutation_node=bool(augmentation_cfg.get("preserve_mutation_node", True)),
        num_views=num_views,
        seed=int(seed),
    )


class GraphViewAugmenter:
    """Create two deterministic-yet-distinct copies of the same mutant graph batch."""

    def __init__(
        self,
        *,
        config: GraphAugmentationConfig,
        node_feature_names: Sequence[str],
    ) -> None:
        self.config = config
        self.node_feature_names = tuple(str(name) for name in node_feature_names)
        self._call_index = 0
        self._dropout_indices = self._resolve_allowed_indices(
            allowed_feature_names=config.feature_dropout_allowed_feature_names,
            protected_patterns=config.feature_dropout_protect,
        )
        self._jitter_indices = self._resolve_allowed_indices(
            allowed_feature_names=config.feature_jitter_allowed_feature_names,
            protected_patterns=config.feature_jitter_protect,
        )

    def _resolve_allowed_indices(
        self,
        *,
        allowed_feature_names: Sequence[str],
        protected_patterns: Sequence[str],
    ) -> tuple[int, ...]:
        allowed = set(str(name) for name in allowed_feature_names)
        unknown = sorted(allowed.difference(self.node_feature_names))
        if unknown:
            raise AugmentationConfigError(
                f"Augmentation references unknown node features: {unknown!r}."
            )
        indices: list[int] = []
        for index, feature_name in enumerate(self.node_feature_names):
            if allowed and feature_name not in allowed:
                continue
            if self._is_protected(feature_name, protected_patterns):
                continue
            indices.append(index)
        return tuple(indices)

    @staticmethod
    def _is_protected(feature_name: str, protected_patterns: Sequence[str]) -> bool:
        for pattern in protected_patterns:
            if pattern.endswith("_"):
                if feature_name.startswith(pattern):
                    return True
                continue
            if pattern and feature_name == pattern:
                return True
        return False

    def _make_generator(self, view_offset: int, *, device: torch.device) -> torch.Generator:
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(self.config.seed + 2 * self._call_index + view_offset)
        return generator

    def _apply_feature_dropout(
        self,
        graph_batch: object,
        *,
        generator: torch.Generator,
    ) -> None:
        if not self.config.feature_dropout_enabled or not self._dropout_indices:
            return
        dropout_mask = torch.rand(
            (graph_batch.x.shape[0], len(self._dropout_indices)),
            generator=generator,
            device=graph_batch.x.device,
        ) >= self.config.feature_dropout_probability
        graph_batch.x[:, self._dropout_indices] = (
            graph_batch.x[:, self._dropout_indices] * dropout_mask.to(graph_batch.x.dtype)
        )

    def _apply_feature_jitter(
        self,
        graph_batch: object,
        *,
        generator: torch.Generator,
    ) -> None:
        if not self.config.feature_jitter_enabled or not self._jitter_indices:
            return
        noise = torch.randn(
            (graph_batch.x.shape[0], len(self._jitter_indices)),
            generator=generator,
            device=graph_batch.x.device,
            dtype=graph_batch.x.dtype,
        )
        graph_batch.x[:, self._jitter_indices] = (
            graph_batch.x[:, self._jitter_indices] + self.config.feature_jitter_std * noise
        )

    def _mutation_protected_edge_mask(self, graph_batch: object) -> torch.Tensor:
        if not self.config.preserve_mutation_node:
            return torch.zeros(graph_batch.edge_index.shape[1], dtype=torch.bool, device=graph_batch.x.device)
        mutation_nodes = torch.nonzero(graph_batch.is_mutation > 0.5, as_tuple=False).flatten()
        if mutation_nodes.numel() == 0:
            return torch.zeros(graph_batch.edge_index.shape[1], dtype=torch.bool, device=graph_batch.x.device)
        src = graph_batch.edge_index[0]
        dst = graph_batch.edge_index[1]
        protected = torch.zeros(src.shape[0], dtype=torch.bool, device=graph_batch.x.device)
        for node_index in mutation_nodes:
            protected |= (src == node_index) | (dst == node_index)
        return protected

    def _apply_edge_dropout(
        self,
        graph_batch: object,
        *,
        generator: torch.Generator,
    ) -> None:
        if not self.config.edge_dropout_enabled or graph_batch.edge_index.numel() == 0:
            return

        edge_count = graph_batch.edge_index.shape[1]
        keep_mask = torch.rand(edge_count, generator=generator, device=graph_batch.x.device)
        keep_mask = keep_mask >= self.config.edge_dropout_probability
        keep_mask |= self._mutation_protected_edge_mask(graph_batch)

        graph_ids = graph_batch.batch[graph_batch.edge_index[0]]
        for graph_id in torch.unique(graph_ids):
            graph_edge_indices = torch.nonzero(graph_ids == graph_id, as_tuple=False).flatten()
            if graph_edge_indices.numel() == 0:
                continue
            if not bool(keep_mask[graph_edge_indices].any()):
                keep_mask[graph_edge_indices[0]] = True

        if not bool(keep_mask.any()):
            keep_mask[0] = True

        graph_batch.edge_index = graph_batch.edge_index[:, keep_mask]
        graph_batch.edge_attr = graph_batch.edge_attr[keep_mask]

    def augment_graph_batch(self, graph_batch: object, *, view_offset: int) -> object:
        cloned = clone_graph_batch(graph_batch)
        if not self.config.enabled:
            return cloned

        generator = self._make_generator(view_offset, device=graph_batch.x.device)
        self._apply_feature_dropout(cloned, generator=generator)
        self._apply_feature_jitter(cloned, generator=generator)
        self._apply_edge_dropout(cloned, generator=generator)
        return cloned

    def create_two_views(self, graph_batch: object) -> tuple[object, object]:
        view1 = self.augment_graph_batch(graph_batch, view_offset=0)
        view2 = self.augment_graph_batch(graph_batch, view_offset=1)
        self._call_index += 1
        return view1, view2
