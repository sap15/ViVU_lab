"""Pure operational node alignment for paired Mut--WT residue graphs.

The module consumes already-loaded sequences.  Coordinates are used only for
distance-to-anchor metadata and radial membership; residue identity is always
``(normalized(chain_id), int(res_id))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

ResidueKey = tuple[str, int]


class NodePairAlignmentError(ValueError):
    """Base class for invalid operational alignment inputs."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class InvalidResidueKeyError(NodePairAlignmentError):
    """Raised when a chain or residue number cannot form an identity key."""


class DuplicateResidueKeyError(NodePairAlignmentError):
    """Raised when one branch contains an ambiguous residue key."""


class InvalidCoordinateError(NodePairAlignmentError):
    """Raised when coordinates violate the observed ``[N, 3]`` contract."""


@dataclass(frozen=True)
class NodePresence:
    key: ResidueKey
    exists_MUT: bool
    exists_WT: bool
    support: str
    distance_MUT: float | None
    distance_WT: float | None


@dataclass(frozen=True)
class GlobalMetrics:
    n_union: int
    n_aligned: int
    n_graph_mut_only: int
    n_graph_wt_only: int
    coverage_union: float | None
    coverage_mut: float | None
    coverage_wt: float | None
    graph_mut_only_fraction_union: float | None
    graph_wt_only_fraction_union: float | None


@dataclass(frozen=True)
class AlignedRadialState:
    key: ResidueKey
    inside_radius_MUT: bool
    inside_radius_WT: bool
    distance_MUT: float
    distance_WT: float
    delta_distance: float


@dataclass(frozen=True)
class LocalAlignmentView:
    radius_angstrom: float
    K_MUT: tuple[ResidueKey, ...]
    K_WT: tuple[ResidueKey, ...]
    K_local_union: tuple[ResidueKey, ...]
    K_local_aligned: tuple[ResidueKey, ...]
    local_mut_aligned_index: tuple[int, ...]
    local_wt_aligned_index: tuple[int, ...]
    local_graph_mut_only_keys: tuple[ResidueKey, ...]
    local_graph_wt_only_keys: tuple[ResidueKey, ...]
    radial_mut_only: tuple[ResidueKey, ...]
    radial_wt_only: tuple[ResidueKey, ...]
    aligned_radial_states: tuple[AlignedRadialState, ...]


@dataclass(frozen=True)
class NodePairAlignment:
    key_policy: str
    anchor_key: ResidueKey
    anchor_aligned: bool
    union_keys: tuple[ResidueKey, ...]
    aligned_keys: tuple[ResidueKey, ...]
    graph_mut_only_keys: tuple[ResidueKey, ...]
    graph_wt_only_keys: tuple[ResidueKey, ...]
    presence: tuple[NodePresence, ...]
    exists_MUT: tuple[bool, ...]
    exists_WT: tuple[bool, ...]
    mut_aligned_index: tuple[int, ...]
    wt_aligned_index: tuple[int, ...]
    mut_only_index: tuple[int, ...]
    wt_only_index: tuple[int, ...]
    metrics: GlobalMetrics
    local_views: tuple[LocalAlignmentView, ...]
    technical_radius_angstrom: float
    alignment_quality_group: str
    quality_reason_codes: tuple[str, ...]
    baseline_clean_candidate: bool
    include_in_full_dataset: bool
    training_eligibility: str

    def local_view(self, radius_angstrom: float) -> LocalAlignmentView:
        """Return the requested immutable local view by its numeric radius."""

        radius = float(radius_angstrom)
        for view in self.local_views:
            if view.radius_angstrom == radius:
                return view
        raise KeyError(f"No local view was constructed for radius {radius:g} Å.")


def _normalize_chain(value: Any, *, role: str, index: int) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidResidueKeyError(
                "undecodable_utf8", f"{role} chain at index {index} is not valid UTF-8."
            ) from exc
    if not isinstance(value, (str, np.str_)):
        raise InvalidResidueKeyError(
            "invalid_chain", f"{role} chain at index {index} is not a string."
        )
    chain = str(value).strip().upper()
    if not chain:
        raise InvalidResidueKeyError(
            "empty_chain", f"{role} chain at index {index} is empty."
        )
    return chain


def _normalize_res_id(value: Any, *, role: str, index: int) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidResidueKeyError(
            "invalid_res_id", f"{role} res_id at index {index} is invalid: {value!r}."
        ) from exc
    if not math.isfinite(number):
        raise InvalidResidueKeyError(
            "nonfinite_res_id", f"{role} res_id at index {index} is not finite."
        )
    if not number.is_integer():
        raise InvalidResidueKeyError(
            "nonintegral_res_id",
            f"{role} res_id at index {index} is not an exact integer: {value!r}.",
        )
    return int(number)


def build_residue_keys(
    chain_ids: Sequence[Any], res_ids: Sequence[Any], *, role: str
) -> tuple[ResidueKey, ...]:
    """Build unique audited fallback keys without consulting other fields."""

    if len(chain_ids) != len(res_ids):
        raise InvalidResidueKeyError(
            "identity_length_mismatch",
            f"{role} chain_ids ({len(chain_ids)}) and res_ids ({len(res_ids)}) differ.",
        )
    keys = tuple(
        (
            _normalize_chain(chain, role=role, index=index),
            _normalize_res_id(residue, role=role, index=index),
        )
        for index, (chain, residue) in enumerate(zip(chain_ids, res_ids))
    )
    seen: dict[ResidueKey, int] = {}
    for index, key in enumerate(keys):
        if key in seen:
            raise DuplicateResidueKeyError(
                "duplicate_key",
                f"{role} key {key!r} occurs at indices {seen[key]} and {index}.",
            )
        seen[key] = index
    return keys


def _normalize_anchor(anchor_key: Sequence[Any]) -> ResidueKey:
    if len(anchor_key) != 2:
        raise InvalidResidueKeyError("invalid_anchor_key", "anchor_key must contain chain and res_id.")
    return (
        _normalize_chain(anchor_key[0], role="anchor", index=0),
        _normalize_res_id(anchor_key[1], role="anchor", index=0),
    )


def _coordinates(values: Any, expected_length: int, *, role: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number):
        raise InvalidCoordinateError("nonnumeric_coordinates", f"{role} coordinates are not numeric.")
    if array.ndim != 2 or array.shape[1] != 3:
        raise InvalidCoordinateError(
            "invalid_coordinate_shape", f"{role} coordinates must have shape [N, 3], got {array.shape}."
        )
    if array.shape[0] != expected_length:
        raise InvalidCoordinateError(
            "coordinate_length_mismatch",
            f"{role} has {array.shape[0]} coordinate rows for {expected_length} nodes.",
        )
    numeric = array.astype(float, copy=True)
    invalid = np.flatnonzero(~np.isfinite(numeric).all(axis=1))
    if invalid.size:
        raise InvalidCoordinateError(
            "nonfinite_coordinate",
            f"{role} coordinate rows are non-finite at indices {tuple(int(i) for i in invalid)}.",
        )
    numeric.setflags(write=False)
    return numeric


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def align_node_pair(
    mut_chain_ids: Sequence[Any],
    mut_res_ids: Sequence[Any],
    mut_positions: Any,
    wt_chain_ids: Sequence[Any],
    wt_res_ids: Sequence[Any],
    wt_positions: Any,
    *,
    anchor_key: Sequence[Any],
    radii: Sequence[float] = (8.0,),
    technical_radius_angstrom: float = 8.0,
) -> NodePairAlignment:
    """Construct a deterministic, immutable Mut--WT correspondence contract."""

    mut_keys = build_residue_keys(mut_chain_ids, mut_res_ids, role="MUT")
    wt_keys = build_residue_keys(wt_chain_ids, wt_res_ids, role="WT")
    anchor = _normalize_anchor(anchor_key)
    mut_xyz = _coordinates(mut_positions, len(mut_keys), role="MUT")
    wt_xyz = _coordinates(wt_positions, len(wt_keys), role="WT")

    requested_radii = tuple(float(radius) for radius in radii)
    technical_radius = float(technical_radius_angstrom)
    if any(not math.isfinite(radius) or radius < 0 for radius in (*requested_radii, technical_radius)):
        raise NodePairAlignmentError("invalid_radius", "Radii must be finite and non-negative.")
    all_radii = tuple(sorted(set((*requested_radii, technical_radius))))

    mut_index = {key: index for index, key in enumerate(mut_keys)}
    wt_index = {key: index for index, key in enumerate(wt_keys)}
    mut_set, wt_set = set(mut_keys), set(wt_keys)
    aligned = tuple(sorted(mut_set & wt_set))
    mut_only = tuple(sorted(mut_set - wt_set))
    wt_only = tuple(sorted(wt_set - mut_set))
    union = tuple(sorted(mut_set | wt_set))
    metrics = GlobalMetrics(
        n_union=len(union),
        n_aligned=len(aligned),
        n_graph_mut_only=len(mut_only),
        n_graph_wt_only=len(wt_only),
        coverage_union=_ratio(len(aligned), len(union)),
        coverage_mut=_ratio(len(aligned), len(mut_keys)),
        coverage_wt=_ratio(len(aligned), len(wt_keys)),
        graph_mut_only_fraction_union=_ratio(len(mut_only), len(union)),
        graph_wt_only_fraction_union=_ratio(len(wt_only), len(union)),
    )

    anchor_aligned = anchor in mut_set and anchor in wt_set
    reasons: list[str] = []
    if anchor not in mut_set:
        reasons.append("anchor_missing_mut")
    if anchor not in wt_set:
        reasons.append("anchor_missing_wt")
    if not aligned:
        reasons.append("empty_alignment")

    views: list[LocalAlignmentView] = []
    mut_dist: dict[ResidueKey, float] = {}
    wt_dist: dict[ResidueKey, float] = {}
    if anchor_aligned:
        mut_anchor = mut_xyz[mut_index[anchor]]
        wt_anchor = wt_xyz[wt_index[anchor]]
        mut_dist = {
            key: float(np.linalg.norm(mut_xyz[index] - mut_anchor))
            for key, index in mut_index.items()
        }
        wt_dist = {
            key: float(np.linalg.norm(wt_xyz[index] - wt_anchor))
            for key, index in wt_index.items()
        }
        aligned_set = set(aligned)
        for radius in all_radii:
            km = {key for key, distance in mut_dist.items() if distance <= radius}
            kw = {key for key, distance in wt_dist.items() if distance <= radius}
            local_union = km | kw
            local_aligned = tuple(sorted(local_union & aligned_set))
            states = tuple(
                AlignedRadialState(
                    key=key,
                    inside_radius_MUT=key in km,
                    inside_radius_WT=key in kw,
                    distance_MUT=mut_dist[key],
                    distance_WT=wt_dist[key],
                    delta_distance=mut_dist[key] - wt_dist[key],
                )
                for key in aligned
            )
            views.append(
                LocalAlignmentView(
                    radius_angstrom=radius,
                    K_MUT=tuple(sorted(km)),
                    K_WT=tuple(sorted(kw)),
                    K_local_union=tuple(sorted(local_union)),
                    K_local_aligned=local_aligned,
                    local_mut_aligned_index=tuple(mut_index[key] for key in local_aligned),
                    local_wt_aligned_index=tuple(wt_index[key] for key in local_aligned),
                    local_graph_mut_only_keys=tuple(sorted(km & set(mut_only))),
                    local_graph_wt_only_keys=tuple(sorted(kw & set(wt_only))),
                    radial_mut_only=tuple(sorted((km - kw) & aligned_set)),
                    radial_wt_only=tuple(sorted((kw - km) & aligned_set)),
                    aligned_radial_states=states,
                )
            )

    presence = tuple(
        NodePresence(
            key=key,
            exists_MUT=key in mut_set,
            exists_WT=key in wt_set,
            support=(
                "aligned"
                if key in mut_set and key in wt_set
                else "graph_mut_only"
                if key in mut_set
                else "graph_wt_only"
            ),
            distance_MUT=mut_dist.get(key),
            distance_WT=wt_dist.get(key),
        )
        for key in union
    )

    if reasons:
        quality_group = "invalid_identity"
        baseline_clean = False
    else:
        technical_view = next(view for view in views if view.radius_angstrom == technical_radius)
        has_local_global_exclusive = bool(
            technical_view.local_graph_mut_only_keys
            or technical_view.local_graph_wt_only_keys
        )
        quality_group = (
            "limited_local_comparability"
            if has_local_global_exclusive
            else "high_comparability"
        )
        baseline_clean = not has_local_global_exclusive
        if has_local_global_exclusive:
            reasons.append("global_exclusive_within_technical_radius")

    return NodePairAlignment(
        key_policy="audited_chain_residue_fallback",
        anchor_key=anchor,
        anchor_aligned=anchor_aligned,
        union_keys=union,
        aligned_keys=aligned,
        graph_mut_only_keys=mut_only,
        graph_wt_only_keys=wt_only,
        presence=presence,
        exists_MUT=tuple(item.exists_MUT for item in presence),
        exists_WT=tuple(item.exists_WT for item in presence),
        mut_aligned_index=tuple(mut_index[key] for key in aligned),
        wt_aligned_index=tuple(wt_index[key] for key in aligned),
        mut_only_index=tuple(mut_index[key] for key in mut_only),
        wt_only_index=tuple(wt_index[key] for key in wt_only),
        metrics=metrics,
        local_views=tuple(views),
        technical_radius_angstrom=technical_radius,
        alignment_quality_group=quality_group,
        quality_reason_codes=tuple(reasons),
        baseline_clean_candidate=baseline_clean,
        include_in_full_dataset=True,
        training_eligibility="pending",
    )
