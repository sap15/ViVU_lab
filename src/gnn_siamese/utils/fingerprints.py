"""Versioned, location-independent fingerprints for external HDF5 inputs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import os
from pathlib import Path
from typing import Any


CONTENT_FINGERPRINT_VERSION = 1
CONTENT_FINGERPRINT_SCOPE = "raw_file_bytes"


def resolve_hdf5_dataset_id(config: Mapping[str, Any]) -> str:
    """Return the versioned logical identity used for HDF5 fingerprints."""

    project = config.get("project")
    if not isinstance(project, Mapping):
        raise ValueError("config.project must be a mapping.")
    return str(project.get("name", "dataset"))


def fingerprint_file(
    path: str | Path,
    *,
    role: str,
    logical_identity: str | None = None,
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Hash a file using bounded-memory streaming.

    ``path`` is deliberately provenance only and is not included in the digest.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    source = Path(path)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while computing SHA-256 fingerprint: {source}")
    return {
        "role": str(role),
        "algorithm": "sha256",
        "version": CONTENT_FINGERPRINT_VERSION,
        "scope": CONTENT_FINGERPRINT_SCOPE,
        "size_bytes": size,
        "digest": digest.hexdigest(),
        "logical_identity": str(logical_identity or role),
    }


def combine_content_fingerprints(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a deterministic digest independent of input order and locators."""

    normalized = [
        {
            "role": str(record["role"]),
            "logical_identity": str(record.get("logical_identity", record["role"])),
            "digest": str(record["digest"]),
            "size_bytes": int(record["size_bytes"]),
        }
        for record in records
    ]
    normalized.sort(key=lambda item: (item["role"], item["logical_identity"], item["digest"], item["size_bytes"]))
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {
        "algorithm": "sha256",
        "version": CONTENT_FINGERPRINT_VERSION,
        "scope": "hdf5_content_set",
        "digest": hashlib.sha256(serialized).hexdigest(),
        "files": normalized,
    }


def fingerprint_hdf5_inputs(
    *,
    mutants_path: str | Path,
    wt_companion_path: str | Path,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    identity_prefix = str(dataset_id or "dataset")
    files = [
        fingerprint_file(mutants_path, role="mutants", logical_identity=f"{identity_prefix}:mutants"),
        fingerprint_file(wt_companion_path, role="wt_companion", logical_identity=f"{identity_prefix}:wt_companion"),
    ]
    return {
        "algorithm": "sha256",
        "version": CONTENT_FINGERPRINT_VERSION,
        "scope": "raw_file_bytes",
        "files": files,
        "combined": combine_content_fingerprints(files),
    }


def fingerprint_pairing_inventory(records: Iterable[Any]) -> str:
    """Fingerprint logical Mutant-WT pairing without physical source paths."""

    def field(record: Any, name: str) -> Any:
        return record.get(name) if isinstance(record, Mapping) else getattr(record, name)

    payload = []
    for record in records:
        chain_id = field(record, "chain_id")
        payload.append(
            {
                "variant_id": str(field(record, "variant_id")),
                "position": int(field(record, "position")),
                "mutant_key": str(field(record, "mutant_key")),
                "wt_key": str(field(record, "wt_key")),
                "chain_id": None if chain_id is None else str(chain_id),
                "wt_aa": str(field(record, "wt_aa")),
                "mut_aa": str(field(record, "mut_aa")),
            }
        )
    payload.sort(key=lambda item: (item["position"], item["variant_id"], item["mutant_key"], item["wt_key"]))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def fingerprint_split_definition(split: Any) -> str:
    """Fingerprint split assignments/config independently from dataset locators."""

    if not hasattr(split, "to_dict"):
        return str(split.dataset_fingerprint)
    payload = dict(split.to_dict())
    payload.pop("dataset_fingerprint", None)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
