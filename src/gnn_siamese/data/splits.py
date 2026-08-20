"""Operational split helpers for paired Mutant-WT datasets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from gnn_siamese.data.dataset import MutWtPairDataset, MutWtPairRecord


class SplitError(ValueError):
    """Base error for dataset split construction and validation."""


class UnsupportedSplitTypeError(SplitError):
    """Raised when configuration requests a split type not implemented here."""


class SplitSerializationError(SplitError):
    """Raised when persisted split assignments are invalid or incompatible."""


@dataclass(frozen=True)
class LeavePositionOutSplitConfig:
    """Resolved configuration required to build a leave-position-out split."""

    split_type: str = "leave_position_out"
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    shuffle_groups: bool = True
    group_key: str = "position"
    enforce_no_position_overlap: bool = True
    enforce_no_variant_overlap: bool = True


@dataclass(frozen=True)
class SplitRecord:
    """Minimal normalized record used by the split builder."""

    variant_id: str
    position: int
    mutant_key: str | None = None
    wt_key: str | None = None
    chain_id: str | None = None
    wt_aa: str | None = None
    mut_aa: str | None = None
    mutant_source_h5: str | None = None
    wt_source_h5: str | None = None


@dataclass(frozen=True)
class SplitAssignment:
    """One exact partition assignment for one variant."""

    variant_id: str
    position: int
    partition: str


@dataclass(frozen=True)
class LeavePositionOutSplit:
    """Serializable leave-position-out split with exact assignments."""

    split_type: str
    dataset_fingerprint: str
    config: LeavePositionOutSplitConfig
    assignments: tuple[SplitAssignment, ...]
    _validated_relocated_fingerprint: str | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def assignments_by_partition(self) -> dict[str, tuple[SplitAssignment, ...]]:
        grouped: dict[str, list[SplitAssignment]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for assignment in self.assignments:
            grouped[assignment.partition].append(assignment)
        return {
            partition: tuple(grouped[partition])
            for partition in ("train", "validation", "test")
        }

    def variant_ids_by_partition(self) -> dict[str, tuple[str, ...]]:
        grouped = self.assignments_by_partition()
        return {
            partition: tuple(item.variant_id for item in grouped[partition])
            for partition in grouped
        }

    def positions_by_partition(self) -> dict[str, tuple[int, ...]]:
        grouped = self.assignments_by_partition()
        return {
            partition: tuple(sorted({item.position for item in grouped[partition]}))
            for partition in grouped
        }

    def dataset_indices_by_partition(
        self,
        dataset_or_records: MutWtPairDataset | Sequence[MutWtPairRecord] | Sequence[SplitRecord] | Sequence[Mapping[str, Any]],
    ) -> dict[str, list[int]]:
        records = _coerce_split_records(dataset_or_records)
        self.validate_against_records(records)
        partition_by_variant = {
            assignment.variant_id: assignment.partition
            for assignment in self.assignments
        }
        indices: dict[str, list[int]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for index, record in enumerate(records):
            indices[partition_by_variant[record.variant_id]].append(index)
        return indices

    def validate_against_records(self, records: Sequence[SplitRecord]) -> None:
        expected_fingerprint = fingerprint_split_records(records)
        if (
            self.dataset_fingerprint != expected_fingerprint
            and self._validated_relocated_fingerprint != expected_fingerprint
        ):
            raise SplitSerializationError(
                "Persisted split dataset_fingerprint does not match the provided dataset."
            )
        self._validate_inventory_and_assignments(records)

    def _validate_inventory_and_assignments(self, records: Sequence[SplitRecord]) -> None:
        record_by_variant = {record.variant_id: record for record in records}
        if len(record_by_variant) != len(records):
            raise SplitSerializationError("Dataset records contain duplicated variant_id values.")

        assignment_by_variant = {assignment.variant_id: assignment for assignment in self.assignments}
        if len(assignment_by_variant) != len(self.assignments):
            raise SplitSerializationError("Persisted split contains duplicated variant assignments.")
        if set(assignment_by_variant) != set(record_by_variant):
            raise SplitSerializationError(
                "Persisted split assignments do not match the dataset variant_id set."
            )

        for variant_id, assignment in assignment_by_variant.items():
            record = record_by_variant[variant_id]
            if assignment.position != record.position:
                raise SplitSerializationError(
                    f"Persisted split position mismatch for variant {variant_id!r}."
                )

        _validate_assignments(
            self.assignments,
            config=self.config,
            dataset_size=len(records),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_type": self.split_type,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config": asdict(self.config),
            "assignments": [asdict(item) for item in self.assignments],
            "partitions": {
                partition: {
                    "variant_ids": list(self.variant_ids_by_partition()[partition]),
                    "positions": list(self.positions_by_partition()[partition]),
                }
                for partition in ("train", "validation", "test")
            },
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LeavePositionOutSplit":
        config_payload = payload.get("config")
        assignments_payload = payload.get("assignments")
        if not isinstance(config_payload, Mapping):
            raise SplitSerializationError("Persisted split is missing a valid config mapping.")
        if not isinstance(assignments_payload, Sequence) or isinstance(assignments_payload, (str, bytes)):
            raise SplitSerializationError("Persisted split is missing a valid assignments sequence.")

        assignments: list[SplitAssignment] = []
        for item in assignments_payload:
            if not isinstance(item, Mapping):
                raise SplitSerializationError("Persisted split assignment entries must be mappings.")
            assignments.append(
                SplitAssignment(
                    variant_id=_require_non_empty_string(item.get("variant_id"), field_name="variant_id"),
                    position=_require_int(item.get("position"), field_name="position"),
                    partition=_require_partition(item.get("partition")),
                )
            )

        return cls(
            split_type=_require_non_empty_string(payload.get("split_type"), field_name="split_type"),
            dataset_fingerprint=_require_non_empty_string(
                payload.get("dataset_fingerprint"),
                field_name="dataset_fingerprint",
            ),
            config=LeavePositionOutSplitConfig(
                split_type=_require_non_empty_string(
                    config_payload.get("split_type", "leave_position_out"),
                    field_name="config.split_type",
                ),
                validation_fraction=_require_fraction(
                    config_payload.get("validation_fraction", 0.15),
                    field_name="config.validation_fraction",
                ),
                test_fraction=_require_fraction(
                    config_payload.get("test_fraction", 0.15),
                    field_name="config.test_fraction",
                ),
                seed=_require_int(config_payload.get("seed", 42), field_name="config.seed"),
                shuffle_groups=bool(config_payload.get("shuffle_groups", True)),
                group_key=_require_non_empty_string(
                    config_payload.get("group_key", "position"),
                    field_name="config.group_key",
                ),
                enforce_no_position_overlap=bool(
                    config_payload.get("enforce_no_position_overlap", True)
                ),
                enforce_no_variant_overlap=bool(
                    config_payload.get("enforce_no_variant_overlap", True)
                ),
            ),
            assignments=tuple(assignments),
        )

    @classmethod
    def load_json(
        cls,
        path: str | Path,
        *,
        dataset_or_records: MutWtPairDataset | Sequence[MutWtPairRecord] | Sequence[SplitRecord] | Sequence[Mapping[str, Any]] | None = None,
        hdf5_content_fingerprint: Mapping[str, Any] | None = None,
    ) -> "LeavePositionOutSplit":
        target = Path(path)
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise SplitSerializationError("Persisted split JSON must contain a top-level object.")
        split = cls.from_dict(payload)
        if dataset_or_records is not None:
            records = _coerce_split_records(dataset_or_records)
            current_fingerprint = fingerprint_split_records(records)
            if split.dataset_fingerprint == current_fingerprint:
                split.validate_against_records(records)
            else:
                if hdf5_content_fingerprint is None:
                    raise SplitSerializationError(
                        "Persisted split dataset_fingerprint does not match the provided dataset."
                    )
                _validate_relocated_split(
                    split,
                    records,
                    payload=payload,
                    hdf5_content_fingerprint=hdf5_content_fingerprint,
                )
                split = replace(
                    split,
                    _validated_relocated_fingerprint=current_fingerprint,
                )
        return split


def resolve_leave_position_out_config(config: Mapping[str, Any]) -> LeavePositionOutSplitConfig:
    """Resolve only the split.* YAML fields needed for leave-position-out."""

    split_cfg = config.get("split")
    if not isinstance(split_cfg, Mapping):
        raise SplitError("config.split must be a mapping.")

    split_type = _require_non_empty_string(
        split_cfg.get("type", "leave_position_out"),
        field_name="split.type",
    )
    if split_type != "leave_position_out":
        raise UnsupportedSplitTypeError(
            f"Unsupported split.type {split_type!r}; only 'leave_position_out' is implemented."
        )

    group_key = _require_non_empty_string(
        split_cfg.get("group_key", "position"),
        field_name="split.group_key",
    )
    if group_key != "position":
        raise SplitError(
            f"Unsupported split.group_key {group_key!r}; leave_position_out requires 'position'."
        )

    return LeavePositionOutSplitConfig(
        split_type=split_type,
        validation_fraction=_require_fraction(
            split_cfg.get("validation_fraction", 0.15),
            field_name="split.validation_fraction",
        ),
        test_fraction=_require_fraction(
            split_cfg.get("test_fraction", 0.15),
            field_name="split.test_fraction",
        ),
        seed=_require_int(split_cfg.get("seed", 42), field_name="split.seed"),
        shuffle_groups=bool(split_cfg.get("shuffle_groups", True)),
        group_key=group_key,
        enforce_no_position_overlap=bool(split_cfg.get("enforce_no_position_overlap", True)),
        enforce_no_variant_overlap=bool(split_cfg.get("enforce_no_variant_overlap", True)),
    )


def fingerprint_split_records(
    dataset_or_records: MutWtPairDataset | Sequence[MutWtPairRecord] | Sequence[SplitRecord] | Sequence[Mapping[str, Any]],
) -> str:
    """Return a stable dataset fingerprint for split persistence and reload."""

    records = _coerce_split_records(dataset_or_records)
    payload = [
        {
            "variant_id": record.variant_id,
            "position": record.position,
            "mutant_key": record.mutant_key,
            "wt_key": record.wt_key,
            "chain_id": record.chain_id,
            "wt_aa": record.wt_aa,
            "mut_aa": record.mut_aa,
            "mutant_source_h5": record.mutant_source_h5,
            "wt_source_h5": record.wt_source_h5,
        }
        for record in sorted(
            records,
            key=lambda item: (
                item.position,
                item.variant_id,
                "" if item.mutant_key is None else item.mutant_key,
                "" if item.wt_key is None else item.wt_key,
            ),
        )
    ]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_leave_position_out_split(
    dataset_or_records: MutWtPairDataset | Sequence[MutWtPairRecord] | Sequence[SplitRecord] | Sequence[Mapping[str, Any]],
    config: LeavePositionOutSplitConfig | Mapping[str, Any],
) -> LeavePositionOutSplit:
    """Build a reproducible split where one position belongs to exactly one partition."""

    resolved_config = (
        resolve_leave_position_out_config(config)
        if isinstance(config, Mapping)
        else config
    )
    if resolved_config.split_type != "leave_position_out":
        raise UnsupportedSplitTypeError(
            f"Unsupported split_type {resolved_config.split_type!r}; only leave_position_out is implemented."
        )

    records = _coerce_split_records(dataset_or_records)
    if not records:
        raise SplitError("leave_position_out requires at least one dataset record.")

    positions = sorted({record.position for record in records})
    ordered_positions = list(positions)
    if resolved_config.shuffle_groups:
        random.Random(resolved_config.seed).shuffle(ordered_positions)

    validation_count, test_count = _resolve_partition_group_counts(
        total_groups=len(ordered_positions),
        validation_fraction=resolved_config.validation_fraction,
        test_fraction=resolved_config.test_fraction,
    )
    test_positions = set(ordered_positions[:test_count])
    validation_positions = set(ordered_positions[test_count : test_count + validation_count])
    assignments: list[SplitAssignment] = []
    for record in records:
        if record.position in test_positions:
            partition = "test"
        elif record.position in validation_positions:
            partition = "validation"
        else:
            partition = "train"
        assignments.append(
            SplitAssignment(
                variant_id=record.variant_id,
                position=record.position,
                partition=partition,
            )
        )

    split = LeavePositionOutSplit(
        split_type=resolved_config.split_type,
        dataset_fingerprint=fingerprint_split_records(records),
        config=resolved_config,
        assignments=tuple(assignments),
    )
    _validate_assignments(split.assignments, config=resolved_config, dataset_size=len(records))
    return split


def load_leave_position_out_split(
    path: str | Path,
    *,
    dataset_or_records: MutWtPairDataset | Sequence[MutWtPairRecord] | Sequence[SplitRecord] | Sequence[Mapping[str, Any]] | None = None,
    hdf5_content_fingerprint: Mapping[str, Any] | None = None,
) -> LeavePositionOutSplit:
    """Load a persisted leave-position-out split and optionally validate it."""

    return LeavePositionOutSplit.load_json(
        path,
        dataset_or_records=dataset_or_records,
        hdf5_content_fingerprint=hdf5_content_fingerprint,
    )


def _validate_relocated_split(
    split: LeavePositionOutSplit,
    records: Sequence[SplitRecord],
    *,
    payload: Mapping[str, Any],
    hdf5_content_fingerprint: Mapping[str, Any] | None,
) -> None:
    """Accept a legacy path-sensitive split only after proving pure relocation."""

    from gnn_siamese.utils.fingerprints import (
        combine_content_fingerprints,
        fingerprint_pairing_inventory,
        fingerprint_split_definition,
    )

    metadata = payload.get("audit_metadata")
    if not isinstance(metadata, Mapping):
        raise SplitSerializationError(
            "Persisted split dataset_fingerprint mismatch requires valid portable audit_metadata."
        )
    if metadata.get("legacy_fingerprint_limitation") != (
        "depends_on_physical_mutant_source_h5_and_wt_source_h5_paths"
    ):
        raise SplitSerializationError(
            "Persisted split does not document the supported legacy path-fingerprint limitation."
        )
    if metadata.get("legacy_dataset_fingerprint") != split.dataset_fingerprint:
        raise SplitSerializationError("Portable metadata has an inconsistent legacy dataset fingerprint.")

    canonical_paths = metadata.get("canonical_hdf5_paths")
    if not isinstance(canonical_paths, Mapping):
        raise SplitSerializationError("Portable metadata is missing canonical_hdf5_paths.")
    mutant_path = canonical_paths.get("mutants")
    wt_path = canonical_paths.get("wt_companion")
    if not isinstance(mutant_path, str) or not mutant_path or not isinstance(wt_path, str) or not wt_path:
        raise SplitSerializationError("Portable metadata contains invalid canonical HDF5 paths.")
    historical_records = tuple(
        replace(record, mutant_source_h5=mutant_path, wt_source_h5=wt_path)
        for record in records
    )
    if fingerprint_split_records(historical_records) != split.dataset_fingerprint:
        raise SplitSerializationError(
            "Legacy dataset fingerprint cannot be reproduced by changing only physical HDF5 paths."
        )

    expected_content = metadata.get("hdf5_content_fingerprint")
    if not isinstance(expected_content, Mapping) or not isinstance(hdf5_content_fingerprint, Mapping):
        raise SplitSerializationError(
            "Dataset relocation requires expected and current HDF5 content fingerprints."
        )
    try:
        for content in (expected_content, hdf5_content_fingerprint):
            if (
                content.get("algorithm") != "sha256"
                or content.get("version") != 1
                or content.get("scope") != "raw_file_bytes"
            ):
                raise SplitSerializationError("Portable HDF5 content fingerprint metadata is invalid.")
        expected_file_list = expected_content["files"]
        actual_file_list = hdf5_content_fingerprint["files"]
        expected_files = {str(item["role"]): item for item in expected_file_list}
        actual_files = {str(item["role"]): item for item in actual_file_list}
        if (
            len(expected_file_list) != 2
            or len(actual_file_list) != 2
            or set(expected_files) != {"mutants", "wt_companion"}
            or set(actual_files) != set(expected_files)
        ):
            raise ValueError("unexpected HDF5 roles")
        identity_fields = (
            "algorithm",
            "version",
            "scope",
            "digest",
            "size_bytes",
            "logical_identity",
        )
        for role in sorted(expected_files):
            if any(expected_files[role].get(key) != actual_files[role].get(key) for key in identity_fields):
                raise SplitSerializationError(f"HDF5 content fingerprint mismatch for {role}.")
        recomputed = combine_content_fingerprints(expected_files.values())
        recomputed_actual = combine_content_fingerprints(actual_files.values())
        expected_combined = expected_content["combined"]
        actual_combined = hdf5_content_fingerprint["combined"]
        for combined in (expected_combined, actual_combined):
            if (
                combined.get("algorithm") != "sha256"
                or combined.get("version") != 1
                or combined.get("scope") != "hdf5_content_set"
            ):
                raise SplitSerializationError("Portable metadata has invalid combined fingerprint metadata.")
        if recomputed["digest"] != expected_combined.get("digest"):
            raise SplitSerializationError("Portable metadata has an inconsistent combined fingerprint.")
        if recomputed_actual["digest"] != actual_combined.get("digest"):
            raise SplitSerializationError("Current combined HDF5 content fingerprint is inconsistent.")
        if expected_combined.get("digest") != actual_combined.get("digest"):
            raise SplitSerializationError("Combined HDF5 content fingerprint mismatch.")
    except SplitSerializationError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SplitSerializationError("Portable HDF5 content fingerprint metadata is invalid.") from exc

    if metadata.get("pairing_inventory_fingerprint") != fingerprint_pairing_inventory(records):
        raise SplitSerializationError("Pairing inventory fingerprint does not match the current dataset.")
    if metadata.get("split_fingerprint") != fingerprint_split_definition(split):
        raise SplitSerializationError("Persisted split assignments/config do not match split_fingerprint.")
    expected_count = metadata.get("biological_variant_count", metadata.get("created_from_variant_count"))
    if expected_count != len(records):
        raise SplitSerializationError("Portable metadata biological variant count does not match the dataset.")

    split._validate_inventory_and_assignments(records)


def _coerce_split_records(
    dataset_or_records: MutWtPairDataset | Sequence[MutWtPairRecord] | Sequence[SplitRecord] | Sequence[Mapping[str, Any]],
) -> tuple[SplitRecord, ...]:
    if isinstance(dataset_or_records, MutWtPairDataset):
        raw_records: Sequence[MutWtPairRecord | SplitRecord | Mapping[str, Any]] = dataset_or_records.pairs
    else:
        raw_records = dataset_or_records

    records: list[SplitRecord] = []
    for item in raw_records:
        if isinstance(item, SplitRecord):
            records.append(item)
            continue

        if isinstance(item, MutWtPairRecord):
            records.append(
                SplitRecord(
                    variant_id=item.variant_id,
                    position=item.position,
                    mutant_key=item.mutant_key,
                    wt_key=item.wt_key,
                    chain_id=item.chain_id,
                    wt_aa=item.wt_aa,
                    mut_aa=item.mut_aa,
                    mutant_source_h5=item.mutant_source_h5,
                    wt_source_h5=item.wt_source_h5,
                )
            )
            continue

        if not isinstance(item, Mapping):
            raise SplitError(
                "Split records must be SplitRecord or MutWtPairRecord instances, or metadata mappings."
            )
        records.append(
            SplitRecord(
                variant_id=_require_non_empty_string(item.get("variant_id"), field_name="variant_id"),
                position=_require_int(item.get("position"), field_name="position"),
                mutant_key=_optional_string(item.get("mutant_key")),
                wt_key=_optional_string(item.get("wt_key")),
                chain_id=_optional_string(item.get("chain_id")),
                wt_aa=_optional_string(item.get("wt_aa")),
                mut_aa=_optional_string(item.get("mut_aa")),
                mutant_source_h5=_optional_string(item.get("mutant_source_h5")),
                wt_source_h5=_optional_string(item.get("wt_source_h5")),
            )
        )

    variant_ids = [record.variant_id for record in records]
    if len(set(variant_ids)) != len(variant_ids):
        raise SplitError("Split records must contain unique variant_id values.")
    return tuple(records)


def _resolve_partition_group_counts(
    *,
    total_groups: int,
    validation_fraction: float,
    test_fraction: float,
) -> tuple[int, int]:
    validation_count = int(round(total_groups * validation_fraction))
    test_count = int(round(total_groups * test_fraction))

    if total_groups == 0:
        return 0, 0

    max_held_out = max(total_groups - 1, 0)
    while validation_count + test_count > max_held_out:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    return validation_count, test_count


def _validate_assignments(
    assignments: Sequence[SplitAssignment],
    *,
    config: LeavePositionOutSplitConfig,
    dataset_size: int,
) -> None:
    if len(assignments) != dataset_size:
        raise SplitSerializationError(
            "Split assignments must cover the dataset exactly once."
        )

    variant_partitions: dict[str, str] = {}
    positions_by_partition: dict[str, set[int]] = defaultdict(set)
    for assignment in assignments:
        if assignment.variant_id in variant_partitions:
            raise SplitSerializationError(
                f"Variant {assignment.variant_id!r} appears more than once in split assignments."
            )
        variant_partitions[assignment.variant_id] = assignment.partition
        positions_by_partition[assignment.partition].add(assignment.position)

    if config.enforce_no_position_overlap:
        train = positions_by_partition["train"]
        validation = positions_by_partition["validation"]
        test = positions_by_partition["test"]
        if train & validation or train & test or validation & test:
            raise SplitSerializationError("A position appears in more than one partition.")

    if config.enforce_no_variant_overlap and len(variant_partitions) != dataset_size:
        raise SplitSerializationError("Variants overlap across partitions.")


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SplitSerializationError(f"{field_name} must be a non-empty string.")
    return value


def _optional_string(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _require_int(value: Any, *, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SplitSerializationError(f"{field_name} must be an integer.") from exc


def _require_fraction(value: Any, *, field_name: str) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise SplitError(f"{field_name} must be a float.") from exc
    if not 0.0 <= fraction < 1.0:
        raise SplitError(f"{field_name} must be in the range [0, 1).")
    return fraction


def _require_partition(value: Any) -> str:
    partition = _require_non_empty_string(value, field_name="partition")
    if partition not in {"train", "validation", "test"}:
        raise SplitSerializationError(
            f"Unsupported partition {partition!r}; expected train, validation or test."
        )
    return partition
