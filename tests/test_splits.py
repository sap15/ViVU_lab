from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import pytest

from gnn_siamese.data import (
    LeavePositionOutSplit,
    SplitSerializationError,
    UnsupportedSplitTypeError,
    build_leave_position_out_split,
    fingerprint_split_records,
    load_leave_position_out_split,
    resolve_leave_position_out_config,
)
from gnn_siamese.data.splits import SplitRecord
from gnn_siamese.utils.fingerprints import (
    fingerprint_hdf5_inputs,
    fingerprint_pairing_inventory,
    fingerprint_split_definition,
)


def _records() -> list[dict[str, object]]:
    return [
        {"variant_id": "pos_100_G_D", "position": 100, "mutant_key": "mut_100_d", "wt_key": "wt_100"},
        {"variant_id": "pos_100_G_S", "position": 100, "mutant_key": "mut_100_s", "wt_key": "wt_100"},
        {"variant_id": "pos_101_R_C", "position": 101, "mutant_key": "mut_101_c", "wt_key": "wt_101"},
        {"variant_id": "pos_200_A_V", "position": 200, "mutant_key": "mut_200_v", "wt_key": "wt_200"},
        {"variant_id": "pos_250_A_L", "position": 250, "mutant_key": "mut_250_l", "wt_key": "wt_250"},
        {"variant_id": "pos_563_C_W", "position": 563, "mutant_key": "mut_563_w", "wt_key": "wt_563"},
    ]


def _config(*, seed: int = 42, validation_fraction: float = 0.2, test_fraction: float = 0.2) -> dict:
    return {
        "split": {
            "type": "leave_position_out",
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "group_key": "position",
            "shuffle_groups": True,
            "seed": seed,
            "enforce_no_position_overlap": True,
            "enforce_no_variant_overlap": True,
            "leave_neighborhood_out": {"enabled": False},
        }
    }


def test_leave_position_out_groups_all_substitutions_of_same_position_together() -> None:
    split = build_leave_position_out_split(_records(), _config(seed=7))
    assignment_by_variant = {item.variant_id: item.partition for item in split.assignments}

    assert assignment_by_variant["pos_100_G_D"] == assignment_by_variant["pos_100_G_S"]


def test_leave_position_out_has_no_position_leakage_between_partitions() -> None:
    split = build_leave_position_out_split(_records(), _config(seed=11))
    positions = split.positions_by_partition()

    assert set(positions["train"]).isdisjoint(positions["validation"])
    assert set(positions["train"]).isdisjoint(positions["test"])
    assert set(positions["validation"]).isdisjoint(positions["test"])
    assert sorted(
        list(positions["train"]) + list(positions["validation"]) + list(positions["test"])
    ) == [100, 101, 200, 250, 563]


def test_leave_position_out_is_reproducible_with_same_seed() -> None:
    split_a = build_leave_position_out_split(_records(), _config(seed=123))
    split_b = build_leave_position_out_split(_records(), _config(seed=123))

    assert split_a.assignments == split_b.assignments
    assert split_a.dataset_fingerprint == split_b.dataset_fingerprint


def test_fingerprint_split_records_accepts_pre_normalized_split_records() -> None:
    normalized_records = [
        SplitRecord(
            variant_id="pos_100_G_D",
            position=100,
            mutant_key="mut_100_d",
            wt_key="wt_100",
        ),
        SplitRecord(
            variant_id="pos_101_R_C",
            position=101,
            mutant_key="mut_101_c",
            wt_key="wt_101",
        ),
    ]

    assert fingerprint_split_records(normalized_records) == fingerprint_split_records(
        [
            {
                "variant_id": "pos_100_G_D",
                "position": 100,
                "mutant_key": "mut_100_d",
                "wt_key": "wt_100",
            },
            {
                "variant_id": "pos_101_R_C",
                "position": 101,
                "mutant_key": "mut_101_c",
                "wt_key": "wt_101",
            },
        ]
    )


def test_leave_position_out_changes_when_seed_changes() -> None:
    split_a = build_leave_position_out_split(_records(), _config(seed=123))
    split_b = build_leave_position_out_split(_records(), _config(seed=456))

    assert split_a.positions_by_partition() != split_b.positions_by_partition()


def test_leave_position_out_can_be_serialized_and_reloaded_exactly(tmp_path: Path) -> None:
    split = build_leave_position_out_split(_records(), _config(seed=5))
    path = tmp_path / "split.json"

    split.save_json(path)
    reloaded = load_leave_position_out_split(path, dataset_or_records=_records())

    assert reloaded == split
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_fingerprint"] == fingerprint_split_records(_records())
    assert payload["partitions"]["train"]["variant_ids"] == list(split.variant_ids_by_partition()["train"])


def test_leave_position_out_reload_fails_for_dataset_fingerprint_mismatch(tmp_path: Path) -> None:
    split = build_leave_position_out_split(_records(), _config(seed=5))
    path = tmp_path / "split.json"
    split.save_json(path)

    changed_records = _records()[:-1]
    with pytest.raises(SplitSerializationError, match="dataset_fingerprint"):
        load_leave_position_out_split(path, dataset_or_records=changed_records)


def test_leave_position_out_dataset_indices_follow_dataset_order() -> None:
    reversed_records = list(reversed(_records()))
    split = build_leave_position_out_split(reversed_records, _config(seed=9))

    indices = split.dataset_indices_by_partition(reversed_records)
    variants_in_order = {
        partition: [reversed_records[index]["variant_id"] for index in partition_indices]
        for partition, partition_indices in indices.items()
    }

    assert variants_in_order["train"] == list(split.variant_ids_by_partition()["train"])
    assert variants_in_order["validation"] == list(split.variant_ids_by_partition()["validation"])
    assert variants_in_order["test"] == list(split.variant_ids_by_partition()["test"])


def test_resolve_leave_position_out_config_consumes_only_supported_split_fields() -> None:
    resolved = resolve_leave_position_out_config(_config(seed=99, validation_fraction=0.25, test_fraction=0.4))

    assert resolved.split_type == "leave_position_out"
    assert resolved.seed == 99
    assert resolved.validation_fraction == pytest.approx(0.25)
    assert resolved.test_fraction == pytest.approx(0.4)
    assert resolved.group_key == "position"


def test_resolve_leave_position_out_config_rejects_unimplemented_split_types() -> None:
    with pytest.raises(UnsupportedSplitTypeError, match="leave_position_out"):
        resolve_leave_position_out_config(
            {
                "split": {
                    "type": "leave_neighborhood_out",
                    "validation_fraction": 0.15,
                    "test_fraction": 0.15,
                    "group_key": "position",
                    "shuffle_groups": True,
                    "seed": 42,
                }
            }
        )


def test_leave_position_out_round_trip_from_dict_preserves_assignments() -> None:
    split = build_leave_position_out_split(_records(), _config(seed=2))

    restored = LeavePositionOutSplit.from_dict(split.to_dict())

    assert restored == split


def _portable_split_fixture(tmp_path: Path) -> tuple[Path, list[SplitRecord], dict]:
    original_mutants = tmp_path / "original" / "mutants.hdf5"
    original_wt = tmp_path / "original" / "wt.hdf5"
    staged_mutants = tmp_path / "staged" / "mutants.hdf5"
    staged_wt = tmp_path / "staged" / "wt.hdf5"
    for path, content in (
        (original_mutants, b"mutants-byte-identical"),
        (original_wt, b"wt-byte-identical"),
        (staged_mutants, b"mutants-byte-identical"),
        (staged_wt, b"wt-byte-identical"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    original_records = [
        SplitRecord(
            variant_id=str(item["variant_id"]),
            position=int(item["position"]),
            mutant_key=str(item["mutant_key"]),
            wt_key=str(item["wt_key"]),
            chain_id="A",
            wt_aa="G",
            mut_aa="D",
            mutant_source_h5=str(original_mutants),
            wt_source_h5=str(original_wt),
        )
        for item in _records()
    ]
    staged_records = [
        replace(
            record,
            mutant_source_h5=str(staged_mutants),
            wt_source_h5=str(staged_wt),
        )
        for record in original_records
    ]
    split = build_leave_position_out_split(original_records, _config(seed=5))
    content = fingerprint_hdf5_inputs(
        mutants_path=original_mutants,
        wt_companion_path=original_wt,
        dataset_id="portable-test",
    )
    payload = split.to_dict()
    payload["audit_metadata"] = {
        "legacy_fingerprint_limitation": "depends_on_physical_mutant_source_h5_and_wt_source_h5_paths",
        "legacy_dataset_fingerprint": split.dataset_fingerprint,
        "canonical_hdf5_paths": {
            "mutants": str(original_mutants),
            "wt_companion": str(original_wt),
        },
        "hdf5_content_fingerprint": content,
        "pairing_inventory_fingerprint": fingerprint_pairing_inventory(original_records),
        "split_fingerprint": fingerprint_split_definition(split),
        "biological_variant_count": len(original_records),
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    staged_content = fingerprint_hdf5_inputs(
        mutants_path=staged_mutants,
        wt_companion_path=staged_wt,
        dataset_id="portable-test",
    )
    return path, staged_records, staged_content


def test_relocated_byte_identical_hdf5_split_passes_strong_portable_validation(tmp_path: Path) -> None:
    path, records, content = _portable_split_fixture(tmp_path)

    split = load_leave_position_out_split(
        path,
        dataset_or_records=records,
        hdf5_content_fingerprint=content,
    )

    assert sum(map(len, split.dataset_indices_by_partition(records).values())) == len(records)


@pytest.mark.parametrize("role", ["mutants", "wt_companion"])
def test_relocated_split_rejects_changed_individual_hdf5_digest(tmp_path: Path, role: str) -> None:
    path, records, content = _portable_split_fixture(tmp_path)
    next(item for item in content["files"] if item["role"] == role)["digest"] = "0" * 64

    with pytest.raises(SplitSerializationError, match=role):
        load_leave_position_out_split(path, dataset_or_records=records, hdf5_content_fingerprint=content)


def test_relocated_split_rejects_inconsistent_combined_fingerprint(tmp_path: Path) -> None:
    path, records, content = _portable_split_fixture(tmp_path)
    content["combined"]["digest"] = "0" * 64
    with pytest.raises(SplitSerializationError, match="combined"):
        load_leave_position_out_split(path, dataset_or_records=records, hdf5_content_fingerprint=content)


@pytest.mark.parametrize("change", ["variant_id", "position"])
def test_relocated_split_rejects_changed_logical_inventory(tmp_path: Path, change: str) -> None:
    path, records, content = _portable_split_fixture(tmp_path)
    if change == "variant_id":
        records[0] = replace(records[0], variant_id="unexpected_variant")
    else:
        records[0] = replace(records[0], position=999)
    with pytest.raises(SplitSerializationError):
        load_leave_position_out_split(path, dataset_or_records=records, hdf5_content_fingerprint=content)


def test_relocated_split_rejects_modified_assignments(tmp_path: Path) -> None:
    path, records, content = _portable_split_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["assignments"][0]["partition"] = "validation"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitSerializationError, match="split_fingerprint"):
        load_leave_position_out_split(path, dataset_or_records=records, hdf5_content_fingerprint=content)


def test_relocated_split_rejects_missing_portable_metadata_but_strict_load_still_works(tmp_path: Path) -> None:
    path, relocated_records, content = _portable_split_fixture(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("audit_metadata")
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SplitSerializationError, match="portable audit_metadata"):
        load_leave_position_out_split(
            path,
            dataset_or_records=relocated_records,
            hdf5_content_fingerprint=content,
        )
