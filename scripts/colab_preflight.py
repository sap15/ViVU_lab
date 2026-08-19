"""Reusable, offline-testable preflight helpers for Model A/B Colab workflows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (str(REPO_ROOT), str(SRC_ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


class ColabPreflightError(RuntimeError):
    """Raised for actionable Colab bootstrap/preflight failures."""


_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _git(repo_root: str | Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(Path(repo_root).resolve()), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def checkout_git_revision(
    repo_root: str | Path,
    revision: str,
    *,
    expected_remote_url: str,
    fresh_clone: bool = False,
) -> dict[str, str]:
    """Fetch and detach at an unambiguous SHA, tag, or remote branch.

    A fresh ``git clone --no-checkout`` has index deletions until its first
    checkout, so only reused clones are checked for dirtiness before fetch.
    Every clone is required to be clean after the detached checkout.
    """

    root = Path(repo_root).resolve()
    origin = _git(root, "remote", "get-url", "origin").stdout.strip()
    if origin != expected_remote_url:
        raise ColabPreflightError(
            f"Unexpected origin URL for controlled clone: expected {expected_remote_url!r}, got {origin!r}."
        )
    if not fresh_clone and _git(root, "status", "--porcelain").stdout.strip():
        raise ColabPreflightError(f"Controlled Colab clone is dirty: {root}")

    _git(root, "fetch", "--tags", "--prune", "origin")
    requested = str(revision).strip()
    if not requested:
        raise ColabPreflightError("Git revision must not be empty.")

    revision_type: str
    resolved_ref: str
    if _COMMIT_SHA.fullmatch(requested):
        revision_type = "commit"
        resolved_ref = requested
        probe = _git(root, "rev-parse", "--verify", f"{resolved_ref}^{{commit}}", check=False)
        if probe.returncode != 0:
            raise ColabPreflightError(f"Requested Git commit does not exist: {requested}")
    else:
        branch_name = requested.removeprefix("origin/")
        remote_ref = f"refs/remotes/origin/{branch_name}"
        tag_ref = f"refs/tags/{requested}"
        branch_exists = _git(root, "show-ref", "--verify", "--quiet", remote_ref, check=False).returncode == 0
        tag_exists = _git(root, "show-ref", "--verify", "--quiet", tag_ref, check=False).returncode == 0
        if branch_exists and tag_exists:
            raise ColabPreflightError(
                f"Ambiguous Git revision {requested!r}: both an origin branch and a tag exist."
            )
        if branch_exists:
            revision_type = "remote_branch"
            resolved_ref = remote_ref
        elif tag_exists:
            revision_type = "tag"
            resolved_ref = tag_ref
        else:
            raise ColabPreflightError(
                f"Requested Git revision is neither a fetched origin branch, tag, nor commit SHA: {requested}"
            )
        probe = _git(root, "rev-parse", "--verify", f"{resolved_ref}^{{commit}}", check=False)
        if probe.returncode != 0:
            raise ColabPreflightError(f"Requested Git revision cannot be resolved: {requested}")

    expected_commit = probe.stdout.strip()
    _git(root, "checkout", "--detach", expected_commit)
    commit = _git(root, "rev-parse", "HEAD").stdout.strip()
    if commit != expected_commit:
        raise ColabPreflightError(
            f"Checked-out commit {commit} does not match resolved revision {expected_commit}."
        )
    if _git(root, "status", "--porcelain").stdout.strip():
        raise ColabPreflightError(f"Controlled Colab clone is dirty after checkout: {root}")
    return {
        "requested_revision": requested,
        "revision_type": revision_type,
        "resolved_ref": resolved_ref,
        "commit": commit,
        "remote_url": origin,
        "working_tree": "clean",
    }


def confined_path(path: str | Path, root: str | Path, *, label: str) -> Path:
    """Resolve ``path`` and require it to remain below an explicit root."""

    candidate = Path(path).expanduser().resolve()
    allowed_root = Path(root).expanduser().resolve()
    if candidate == allowed_root or not candidate.is_relative_to(allowed_root):
        raise ColabPreflightError(f"{label} must be a child of {allowed_root}, got {candidate}.")
    return candidate


def require_hdf5(path: str | Path, *, label: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        raise ColabPreflightError(f"Missing {label} HDF5 file: {candidate}")
    if candidate.stat().st_size <= 0:
        raise ColabPreflightError(f"Empty {label} HDF5 file: {candidate}")
    return candidate


def stage_file(
    source: str | Path,
    destination: str | Path,
    *,
    staging_root: str | Path,
    role: str,
    chunk_size: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Stream an immutable input to local storage and verify its SHA-256."""

    from gnn_siamese.utils.fingerprints import fingerprint_file

    source_path = require_hdf5(source, label=role)
    destination_path = confined_path(destination, staging_root, label="staging destination")
    if source_path == destination_path:
        raise ColabPreflightError("Drive source and local staging destination must be different files.")
    source_fp = fingerprint_file(source_path, role=role)
    reused = False
    if destination_path.is_file():
        local_fp = fingerprint_file(destination_path, role=role)
        reused = (
            local_fp["digest"] == source_fp["digest"]
            and local_fp["size_bytes"] == source_fp["size_bytes"]
        )
    if not reused:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
        try:
            _stream_copy(source_path, temporary, chunk_size=chunk_size)
            os.replace(temporary, destination_path)
        finally:
            temporary.unlink(missing_ok=True)
    local_fp = fingerprint_file(destination_path, role=role)
    if local_fp["digest"] != source_fp["digest"] or local_fp["size_bytes"] != source_fp["size_bytes"]:
        raise ColabPreflightError(f"Staged {role} file does not match its Drive source.")
    return {
        "role": role,
        "drive_locator": str(source_path),
        "local_locator": str(destination_path),
        "sha256": source_fp["digest"],
        "size_bytes": source_fp["size_bytes"],
        "reused": reused,
    }


def _stream_copy(source: Path, destination: Path, *, chunk_size: int) -> None:
    with source.open("rb") as reader, destination.open("wb") as writer:
        while chunk := reader.read(chunk_size):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())


def require_free_space(path: str | Path, required_bytes: int) -> int:
    candidate = Path(path).expanduser().resolve()
    probe = candidate if candidate.exists() else candidate.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = int(shutil.disk_usage(probe).free)
    if free < int(required_bytes):
        raise ColabPreflightError(
            f"Insufficient free space below {candidate}: need {required_bytes} bytes, have {free}."
        )
    return free


def preflight_output_root(path: str | Path, *, allowed_root: str | Path) -> dict[str, Any]:
    """Exercise create/replace/read/delete; this is not a strong atomicity proof."""

    output = confined_path(path, allowed_root, label="output root")
    output.mkdir(parents=True, exist_ok=True)
    first = output / f".colab-preflight-{os.getpid()}.tmp"
    second = output / f".colab-preflight-{os.getpid()}.replaced"
    payload = b"model-b-colab-preflight\n"
    try:
        with first.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(first, second)
        if second.read_bytes() != payload:
            raise ColabPreflightError(f"Output read-after-replace failed below {output}.")
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)
    return {"output_root": str(output), "free_bytes": require_free_space(output, 1), "operational": True}


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ColabPreflightError(f"Cannot override non-mapping configuration field {part!r}.")
        current = child
    current[parts[-1]] = value


ALLOWED_RUNTIME_OVERRIDES = {
    "paths.mutants_hdf5",
    "paths.wt_companion_hdf5",
    "outputs.root_dir",
    "split.persist_path",
    "training.device",
    "training.epochs",
    "training.batch_size",
    "project.seed",
}


def generate_runtime_config(
    base_config: str | Path,
    destination: str | Path,
    *,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve inheritance and change only the declared operational allowlist."""

    from gnn_siamese.config import load_config, resolve_training_device, save_config

    unknown = sorted(set(overrides) - ALLOWED_RUNTIME_OVERRIDES)
    if unknown:
        raise ColabPreflightError(f"Scientific or unsupported configuration overrides requested: {unknown}")
    resolved = deepcopy(load_config(base_config))
    before = deepcopy(resolved)
    for key, value in overrides.items():
        _set_nested(resolved, key, value)
    resolve_training_device(str(resolved["training"]["device"]))
    if int(resolved["training"]["epochs"]) <= 0 or int(resolved["training"]["batch_size"]) <= 0:
        raise ColabPreflightError("training.epochs and training.batch_size must be positive.")
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    save_config(resolved, target)
    changes = {key: {"before": _get_nested(before, key), "after": value} for key, value in overrides.items()}
    return {"config_path": str(target), "changes": changes, "config": resolved}


def validate_runtime_hdf5(
    config_path: str | Path,
    *,
    expected_schema_name: str = "GNN_siamese_PKP2_HDF5_sample_schema",
    expected_schema_version: str = "1.0.0",
) -> dict[str, Any]:
    """Validate schema metadata and paired HDF5 inputs through production builders."""

    from gnn_siamese.builders import build_dataset_bundle
    from gnn_siamese.config import load_config, load_schema

    config = load_config(config_path)
    config["__config_path__"] = str(Path(config_path).resolve())
    schema_path = _get_nested(config, "paths.sample_schema")
    schema_candidate = Path(str(schema_path))
    if not schema_candidate.is_absolute():
        config_relative = Path(config_path).resolve().parent / schema_candidate
        schema_candidate = config_relative if config_relative.exists() else Path(schema_path).resolve()
    schema = load_schema(schema_candidate)
    if schema.get("schema_name") != expected_schema_name:
        raise ColabPreflightError(
            f"Incompatible schema_name: expected {expected_schema_name!r}, got {schema.get('schema_name')!r}."
        )
    if schema.get("schema_version") != expected_schema_version:
        raise ColabPreflightError(
            f"Incompatible schema_version: expected {expected_schema_version!r}, got {schema.get('schema_version')!r}."
        )
    try:
        bundle = build_dataset_bundle(config)
        if not bundle.dataset.pairs:
            raise ColabPreflightError("Productive HDF5 validation found no resolvable mutant-WT pairs.")
        invalid_mutant_records = [
            pair.variant_id for pair in bundle.dataset.pairs if pair.mut_aa == pair.wt_aa
        ]
        if invalid_mutant_records:
            raise ColabPreflightError(
                "Mutants HDF5 contains WT-companion records in the mutant role: "
                f"{invalid_mutant_records[:3]}"
            )
        bundle.dataset[0]
    except ColabPreflightError:
        raise
    except Exception as exc:
        raise ColabPreflightError(f"Productive HDF5 validation failed: {exc}") from exc
    return {
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "pair_count": len(bundle.dataset.pairs),
        "mutants_hdf5": str(Path(bundle.dataset.mutant_h5_path).resolve()),
        "wt_companion_hdf5": str(Path(bundle.dataset.wt_h5_path).resolve()),
    }


MODEL_A_ARCHITECTURE = "model_a_nodal_multiscale_pair"
MODEL_A_PILOT_SPLIT = "splits/leave_position_out_seed_42.json"
MODEL_A_EXPECTED_PARTITIONS = {"train": 342, "validation": 78, "test": 63}


def validate_model_a_dataset_identity(
    actual: Mapping[str, Any],
    *,
    frozen_split_path: str | Path,
) -> dict[str, Any]:
    """Match HDF5 bytes against the identity frozen with the A8.4 split."""

    split_path = Path(frozen_split_path).resolve()
    try:
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        expected = split_payload["audit_metadata"]["hdf5_content_fingerprint"]
        expected_files = {record["role"]: record for record in expected["files"]}
        actual_files = {record["role"]: record for record in actual["files"]}
        expected_combined = expected["combined"]
        actual_combined = actual["combined"]
        identity = {
            "expected_mutant_sha256": expected_files["mutants"]["digest"],
            "actual_mutant_sha256": actual_files["mutants"]["digest"],
            "expected_wt_sha256": expected_files["wt_companion"]["digest"],
            "actual_wt_sha256": actual_files["wt_companion"]["digest"],
            "expected_combined_fingerprint": expected_combined["digest"],
            "actual_combined_fingerprint": actual_combined["digest"],
        }
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ColabPreflightError(
            f"Frozen Model A split lacks a usable A8.4 HDF5 content identity: {split_path}"
        ) from exc

    mismatches = {
        field: value
        for field, value in (
            ("mutants", (identity["expected_mutant_sha256"], identity["actual_mutant_sha256"])),
            ("wt_companion", (identity["expected_wt_sha256"], identity["actual_wt_sha256"])),
            (
                "combined",
                (
                    identity["expected_combined_fingerprint"],
                    identity["actual_combined_fingerprint"],
                ),
            ),
        )
        if value[0] != value[1]
    }
    if mismatches:
        raise ColabPreflightError(
            f"Model A dataset identity mismatch against frozen A8.4 split: {mismatches}"
        )
    return {
        **identity,
        "dataset_identity_status": "PASS",
        "identity_source": str(split_path),
    }


def validate_model_a_preflight(
    config_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    expected_commit: str | None = None,
    mode: str = "fresh",
    resume_from: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless a resolved config preserves the frozen A8.5 contract."""

    from gnn_siamese.builders import build_dataset_bundle, build_split_bundle
    from gnn_siamese.config import load_config
    from gnn_siamese.utils.fingerprints import fingerprint_hdf5_inputs

    if mode not in {"fresh", "resume"}:
        raise ColabPreflightError(f"Model A mode must be 'fresh' or 'resume', got {mode!r}.")
    if mode == "fresh" and resume_from is not None:
        raise ColabPreflightError("Fresh Model A runs must not receive a resume checkpoint.")
    if mode == "resume" and resume_from is None:
        raise ColabPreflightError("Model A resume requires an explicit last.pt checkpoint.")

    root = Path(repo_root).resolve()
    config = load_config(config_path)
    config["__config_path__"] = str(Path(config_path).resolve())
    required = {
        "model.architecture": MODEL_A_ARCHITECTURE,
        "training.batch_size": 4,
        "split.type": "leave_position_out",
        "split.seed": 42,
        "split.persist_path": MODEL_A_PILOT_SPLIT,
        "split.allow_create": False,
        "loss.main": "nt_xent",
        "loss.lambda_wt": 0.0,
        "loss.lambda_delta": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": _get_nested(config, key)}
        for key, expected in required.items()
        if _get_nested(config, key) != expected
    }
    if mismatches:
        raise ColabPreflightError(f"Model A A8.5 scientific contract mismatch: {mismatches}")
    epochs = int(_get_nested(config, "training.epochs"))
    if (mode == "fresh" and epochs != 5) or (mode == "resume" and epochs < 6):
        raise ColabPreflightError(
            "Model A requires 5 epochs for the fresh A8.5 pilot and at least 6 total epochs for resume."
        )

    scales = list(_get_nested(config, "model.active_scales"))
    if scales != ["mutation", "local", "global"] or "domain" in scales:
        raise ColabPreflightError(
            "Model A A8.5 requires active_scales=[mutation, local, global] with domain disabled."
        )
    frozen_split = root / MODEL_A_PILOT_SPLIT
    if not frozen_split.is_file():
        raise ColabPreflightError(f"Frozen Model A split does not exist: {frozen_split}")

    output_root = Path(str(_get_nested(config, "outputs.root_dir"))).expanduser().resolve()
    if any(part.lower().startswith("model_b") for part in output_root.parts):
        raise ColabPreflightError(f"Model A output root overlaps a Model B namespace: {output_root}")

    try:
        dataset_bundle = build_dataset_bundle(config)
        split_load_config = deepcopy(config)
        # Resolve only for the builder call: the serialized runtime config keeps
        # the repository-relative frozen path as its scientific source of truth.
        split_load_config["split"]["persist_path"] = str(frozen_split)
        split_bundle = build_split_bundle(split_load_config, dataset_bundle.dataset)
    except Exception as exc:
        raise ColabPreflightError(f"Model A dataset/split validation failed: {exc}") from exc
    dataset = dataset_bundle.dataset
    if split_bundle.created:
        raise ColabPreflightError("Model A preflight unexpectedly created a split.")
    if Path(split_bundle.split_path).resolve() != frozen_split.resolve():
        raise ColabPreflightError(
            f"Model A did not load the frozen repository split: {split_bundle.split_path}"
        )

    counts = {
        "train": len(split_bundle.train_indices),
        "validation": len(split_bundle.validation_indices),
        "test": len(split_bundle.test_indices),
    }
    if counts != MODEL_A_EXPECTED_PARTITIONS:
        raise ColabPreflightError(
            f"Model A partition counts mismatch: expected {MODEL_A_EXPECTED_PARTITIONS}, got {counts}."
        )
    if len(dataset.pairs) != 483 or len(dataset.native_wt_controls) != 1:
        raise ColabPreflightError(
            "Model A requires 483 trainable biological variants and one non-trainable native WT control."
        )
    trainable_ids = {pair.variant_id for pair in dataset.pairs}
    control_ids = {control.variant_id for control in dataset.native_wt_controls}
    if trainable_ids & control_ids:
        raise ColabPreflightError("Native WT control leaked into the trainable Model A inventory.")

    assignments = split_bundle.split.assignments_by_partition()
    variant_sets = {
        partition: {item.variant_id for item in values}
        for partition, values in assignments.items()
    }
    position_sets = {
        partition: {int(item.position) for item in values}
        for partition, values in assignments.items()
    }
    partitions = ("train", "validation", "test")
    for index, left in enumerate(partitions):
        for right in partitions[index + 1 :]:
            if variant_sets[left] & variant_sets[right] or position_sets[left] & position_sets[right]:
                raise ColabPreflightError(f"Model A split leakage detected between {left} and {right}.")
    if control_ids & set().union(*variant_sets.values()):
        raise ColabPreflightError("Native WT control appears in a Model A split partition.")

    checkpoint = None
    if resume_from is not None:
        checkpoint = Path(resume_from).expanduser().resolve()
        if checkpoint.name != "last.pt" or not checkpoint.is_file():
            raise ColabPreflightError(f"Model A resume requires an existing explicit last.pt: {checkpoint}")

    git = git_revision(root, expected_commit) if expected_commit is not None else None
    fingerprints = fingerprint_hdf5_inputs(
        mutants_path=dataset.mutant_h5_path,
        wt_companion_path=dataset.wt_h5_path,
        dataset_id=str(_get_nested(config, "data.protein_id")),
    )
    dataset_identity = validate_model_a_dataset_identity(
        fingerprints,
        frozen_split_path=frozen_split,
    )
    return {
        "architecture": MODEL_A_ARCHITECTURE,
        "config": str(Path(config_path).resolve()),
        "commit": None if git is None else git["commit"],
        "device": str(_get_nested(config, "training.device")),
        "dataset_paths": {
            "mutants": str(Path(dataset.mutant_h5_path).resolve()),
            "wt_companion": str(Path(dataset.wt_h5_path).resolve()),
        },
        "dataset_fingerprint": fingerprints["combined"],
        **dataset_identity,
        "biological_variants": len(dataset.pairs),
        "native_wt_controls": len(dataset.native_wt_controls),
        "native_wt_trainable": False,
        "partitions": counts,
        "split_seed": 42,
        "split_allow_create": False,
        "split_path": str(frozen_split),
        "split_created": False,
        "variant_overlaps": 0,
        "position_overlaps": 0,
        "active_scales": scales,
        "domain_enabled": False,
        "loss": {"main": "nt_xent", "lambda_wt": 0.0, "lambda_delta": 0.0},
        "run_root": str(output_root),
        "mode": mode,
        "epochs": epochs,
        "resume_checkpoint": None if checkpoint is None else str(checkpoint),
    }


def _get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        value = value[part]
    return value


def build_train_command(
    repo_root: str | Path,
    config_path: str | Path,
    *,
    device: str,
    smoke_test: bool = False,
    resume_from: str | Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(repo_root).resolve() / "scripts" / "train.py"),
        "--config",
        str(Path(config_path).resolve()),
        "--device",
        str(device),
    ]
    if smoke_test:
        command.append("--smoke-test")
    if resume_from is not None:
        checkpoint = Path(resume_from).expanduser().resolve()
        if not checkpoint.is_file():
            raise ColabPreflightError(f"Explicit resume checkpoint does not exist: {checkpoint}")
        command.extend(["--resume-from", str(checkpoint)])
    return command


def git_revision(repo_root: str | Path, expected_commit: str | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if status:
        raise ColabPreflightError(f"Controlled Colab clone is dirty: {root}")
    if expected_commit and commit != str(expected_commit):
        raise ColabPreflightError(
            f"Checked-out commit {commit} does not match expected exact commit {expected_commit}."
        )
    return {"commit": commit, "expected_commit": expected_commit, "working_tree": "clean"}


def runtime_summary(device: str) -> dict[str, Any]:
    import torch
    import torch_geometric
    from gnn_siamese.config import resolve_training_device

    selected = resolve_training_device(device)
    tensor = torch.tensor([1.0], device=selected)
    if float((tensor + 1).cpu().item()) != 2.0:
        raise ColabPreflightError(f"Tensor execution failed on {selected}.")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_visible": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "selected_device": str(selected),
        "device_tensor_test": "passed",
        "local_disk": shutil.disk_usage("/")._asdict(),
    }


def environment_identity(
    *,
    commit: str,
    requirements_path: str | Path,
    python_version: str,
    torch_version: str,
) -> dict[str, str]:
    requirements_sha256 = hashlib.sha256(Path(requirements_path).read_bytes()).hexdigest()
    fields = {
        "commit": str(commit),
        "requirements_sha256": requirements_sha256,
        "python": str(python_version),
        "torch": str(torch_version),
    }
    fields["identity_sha256"] = hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return fields


def _validate_installed_environment(device: str) -> dict[str, Any]:
    for module_name in ("yaml", "h5py", "numpy", "torch", "torch_geometric", "gnn_siamese"):
        importlib.import_module(module_name)
    summary = runtime_summary(device)
    from packaging.version import Version

    if not (Version("2.5") <= Version(str(summary["torch_geometric"])) < Version("3")):
        raise ColabPreflightError(
            f"PyG outside supported range: {summary['torch_geometric']}"
        )
    return summary


def prepare_colab_environment(
    *,
    repo_root: str | Path,
    marker_root: str | Path,
    commit: str,
    requirements_path: str | Path,
    device: str,
    python_version: str,
    torch_version: str,
    installer: Any | None = None,
    validator: Any | None = None,
) -> dict[str, Any]:
    """Install if needed, always validate, then atomically publish the marker."""

    identity = environment_identity(
        commit=commit,
        requirements_path=requirements_path,
        python_version=python_version,
        torch_version=torch_version,
    )
    marker = Path(marker_root).resolve() / f".environment-{identity['identity_sha256']}.json"
    validate = validator or _validate_installed_environment
    install = installer or (
        lambda: (
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(Path(requirements_path).resolve())],
                check=True,
            ),
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(Path(repo_root).resolve())],
                check=True,
            ),
        )
    )

    marker_payload: dict[str, Any] | None = None
    if marker.is_file():
        try:
            candidate = json.loads(marker.read_text(encoding="utf-8"))
            if all(candidate.get(key) == value for key, value in identity.items()):
                marker_payload = candidate
        except (OSError, json.JSONDecodeError):
            marker_payload = None

    reused = False
    if marker_payload is not None:
        try:
            runtime = validate(device)
            reused = marker_payload.get("torch_geometric") == runtime.get("torch_geometric")
        except Exception:
            marker.unlink(missing_ok=True)
            marker_payload = None
        else:
            if not reused:
                marker.unlink(missing_ok=True)
                marker_payload = None

    if marker_payload is None:
        marker.unlink(missing_ok=True)
        install()
        runtime = validate(device)
        payload = {**identity, **runtime}
        from gnn_siamese.utils.atomic_io import atomic_write_text

        atomic_write_text(marker, json.dumps(payload, indent=2, sort_keys=True, default=str))
    return {"marker": str(marker), "reused": reused, "identity": identity, "runtime": runtime}


def parse_cli_contract(
    stdout: str,
    *,
    required_keys: Sequence[str],
    allowed_keys: Sequence[str],
) -> dict[str, str]:
    """Parse only named CLI contract keys and reject duplicate or missing fields."""

    required = set(required_keys)
    allowed = set(allowed_keys)
    if not required <= allowed:
        raise ColabPreflightError("CLI parser required keys must be included in allowed keys.")
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key not in allowed:
            continue
        if key in parsed:
            raise ColabPreflightError(f"Ambiguous duplicate CLI field: {key}")
        parsed[key] = value
    missing = sorted(required - set(parsed))
    if missing:
        raise ColabPreflightError(f"CLI output is missing required contract fields: {missing}")
    return parsed


def build_rsync_command(source: str | Path, destination: str | Path) -> list[str]:
    executable = shutil.which("rsync")
    if executable is None:
        raise ColabPreflightError(
            "rsync is unavailable; local outputs have not been persisted to Drive."
        )
    return [executable, "-a", "--partial", f"{Path(source).resolve()}/", f"{Path(destination).resolve()}/"]


def sync_local_outputs(
    source: str | Path,
    destination: str | Path,
    *,
    runner: Any = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    command = build_rsync_command(source, destination)
    result = runner(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise ColabPreflightError(
            "rsync failed; local outputs have not been persisted to Drive: " + result.stderr
        )
    return result


def run_command(command: Sequence[str], *, cwd: str | Path) -> subprocess.CompletedProcess[str]:
    if isinstance(command, (str, bytes)):
        raise ColabPreflightError("Commands must be argument lists, never shell strings.")
    return subprocess.run(list(command), cwd=Path(cwd), capture_output=True, text=True, check=False)


def write_session_record(
    path: str | Path,
    *,
    allowed_root: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Persist notebook provenance without allowing a path escape."""

    from gnn_siamese.utils.atomic_io import atomic_write_text

    destination = confined_path(path, allowed_root, label="session record")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
    return destination


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mutants-hdf5", required=True)
    parser.add_argument("--wt-companion-hdf5", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--allowed-output-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = generate_runtime_config(
        args.config,
        args.resolved_config,
        overrides={
            "paths.mutants_hdf5": str(require_hdf5(args.mutants_hdf5, label="mutants")),
            "paths.wt_companion_hdf5": str(require_hdf5(args.wt_companion_hdf5, label="WT companion")),
            "outputs.root_dir": str(args.output_root),
            "training.device": args.device,
            "training.epochs": args.epochs,
            "training.batch_size": args.batch_size,
            "project.seed": args.seed,
        },
    )
    result["output_preflight"] = preflight_output_root(args.output_root, allowed_root=args.allowed_output_root)
    result["runtime"] = runtime_summary(args.device)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
