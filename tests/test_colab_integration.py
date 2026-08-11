from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.colab_preflight import (
    ColabPreflightError,
    build_rsync_command,
    build_train_command,
    checkout_git_revision,
    confined_path,
    environment_identity,
    generate_runtime_config,
    parse_cli_contract,
    preflight_output_root,
    prepare_colab_environment,
    require_free_space,
    require_hdf5,
    stage_file,
    sync_local_outputs,
    validate_runtime_hdf5,
    write_session_record,
)
from gnn_siamese.config import load_config, save_config
from gnn_siamese.data import prepare_smoke_data


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "model_b_colab_master.ipynb"
BASE_CONFIG = REPO_ROOT / "configs" / "model_b_baseline.yaml"


def _notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def test_notebook_is_valid_clean_json_with_required_sections() -> None:
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"] if cell["cell_type"] == "code")
    headings = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    for required in (
        "Montaje", "Checkout reproducible", "Instalación idempotente", "Staging local",
        "configuración resuelta", "Preflight", "Smoke", "Piloto", "Resume", "resumen final",
    ):
        assert required.lower() in headings.lower()


def test_parameters_are_centralized_and_dangerous_defaults_are_off() -> None:
    notebook = _notebook()
    parameter_cells = [cell for cell in notebook["cells"] if "parameters" in cell.get("metadata", {}).get("tags", [])]
    assert len(parameter_cells) == 1
    source = "".join(parameter_cells[0]["source"])
    for name in (
        "REPO_URL", "GIT_REVISION", "DRIVE_BASE", "DRIVE_MUTANTS_HDF5", "DRIVE_WT_HDF5",
        "DRIVE_RUNS_ROOT", "EXECUTION_MODE", "REQUESTED_DEVICE", "SEED", "PILOT_EPOCHS",
        "PILOT_BATCH_SIZE", "RESUME_CHECKPOINT", "OUTPUT_MODE", "SYNC_LOCAL_OUTPUTS",
    ):
        assert name in source
    assert "RUN_PILOT = False" in source
    assert "RUN_RESUME = False" in source

    editable_names = {
        "GIT_REVISION", "DRIVE_BASE", "DRIVE_MUTANTS_HDF5", "DRIVE_WT_HDF5",
        "DRIVE_RUNS_ROOT", "OUTPUT_MODE", "EXECUTION_MODE", "RUN_PILOT", "RUN_RESUME",
        "SYNC_LOCAL_OUTPUTS", "REQUESTED_DEVICE", "SEED", "PILOT_EPOCHS",
        "PILOT_BATCH_SIZE", "RESUME_CHECKPOINT",
    }
    later_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"][2:]
        if cell["cell_type"] == "code"
    )
    later_assignments = {
        node.targets[0].id
        for node in ast.walk(ast.parse(later_source))
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert not editable_names & later_assignments


def test_execution_mode_allowlist_and_guards_fail_immediately() -> None:
    parameter_source = "".join(
        next(
            cell for cell in _notebook()["cells"]
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        )["source"]
    )
    namespace: dict[str, object] = {}
    exec(parameter_source, namespace)
    assert namespace["VALID_EXECUTION_MODES"] == {"preflight", "smoke", "pilot", "resume"}
    invalid = parameter_source.replace("EXECUTION_MODE = 'preflight'", "EXECUTION_MODE = 'invalid'")
    with pytest.raises(ValueError, match="EXECUTION_MODE inválido"):
        exec(invalid, {})
    contradictory = parameter_source.replace("RUN_PILOT = False", "RUN_PILOT = True")
    with pytest.raises(ValueError, match="RUN_PILOT"):
        exec(contradictory, {})


def test_notebook_has_no_outputs_secrets_personal_paths_or_shell_execution() -> None:
    source = _source()
    forbidden = ("/home/", "C:\\Users\\", "api_key", "github_token", "shell=True", "rm -rf")
    assert not any(value.lower() in source.lower() for value in forbidden)
    assert "from google.colab import drive" in source
    assert "git-rev-parse" not in source
    assert "refs/remotes/origin/" in source
    assert "git_revision(repo, GIT_COMMIT)" in source


def test_notebook_code_is_a_structural_dry_run() -> None:
    for cell in _notebook()["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_google_colab_is_not_imported_by_local_module() -> None:
    assert "google.colab" not in (REPO_ROOT / "scripts" / "colab_preflight.py").read_text(encoding="utf-8")
    assert "google.colab" not in sys.modules


def _run_git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_fixture(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    clone = tmp_path / "clone"
    _run_git("init", "--bare", str(remote))
    _run_git("init", str(seed))
    _run_git("checkout", "-b", "main", cwd=seed)
    _run_git("config", "user.email", "test@example.invalid", cwd=seed)
    _run_git("config", "user.name", "Test", cwd=seed)
    (seed / "tracked.txt").write_text("one\n", encoding="utf-8")
    _run_git("add", "tracked.txt", cwd=seed)
    _run_git("commit", "-m", "initial", cwd=seed)
    first = _run_git("rev-parse", "HEAD", cwd=seed)
    _run_git("remote", "add", "origin", str(remote), cwd=seed)
    _run_git("push", "-u", "origin", "main", cwd=seed)
    _run_git("symbolic-ref", "HEAD", "refs/heads/main", cwd=remote)
    _run_git("clone", str(remote), str(clone))
    return remote, seed, clone, first


def test_fresh_no_checkout_clone_reaches_clean_expected_commit(tmp_path: Path) -> None:
    remote, _seed, _clone, expected = _git_fixture(tmp_path)
    fresh = tmp_path / "fresh-no-checkout"
    _run_git("clone", "--no-checkout", str(remote), str(fresh))
    assert _run_git("status", "--porcelain", cwd=fresh).startswith("D ")

    result = checkout_git_revision(
        fresh,
        "main",
        expected_remote_url=str(remote),
        fresh_clone=True,
    )

    assert result["commit"] == expected
    assert _run_git("rev-parse", "HEAD", cwd=fresh) == expected
    assert _run_git("status", "--porcelain", cwd=fresh) == ""


def test_reused_clean_clone_reaches_expected_commit(tmp_path: Path) -> None:
    remote, _seed, clone, expected = _git_fixture(tmp_path)

    result = checkout_git_revision(clone, "main", expected_remote_url=str(remote))

    assert result["commit"] == expected
    assert _run_git("rev-parse", "HEAD", cwd=clone) == expected
    assert _run_git("status", "--porcelain", cwd=clone) == ""


def test_reused_dirty_clone_is_rejected_without_replacing_modification(tmp_path: Path) -> None:
    remote, seed, clone, expected = _git_fixture(tmp_path)
    changed = clone / "tracked.txt"
    changed.write_text("user modification\n", encoding="utf-8")
    (seed / "tracked.txt").write_text("remote advance\n", encoding="utf-8")
    _run_git("add", "tracked.txt", cwd=seed)
    _run_git("commit", "-m", "remote advances", cwd=seed)
    _run_git("push", "origin", "main", cwd=seed)

    with pytest.raises(ColabPreflightError, match="dirty"):
        checkout_git_revision(clone, "main", expected_remote_url=str(remote))

    assert _run_git("rev-parse", "HEAD", cwd=clone) == expected
    assert _run_git("rev-parse", "refs/remotes/origin/main", cwd=clone) == expected
    assert changed.read_text(encoding="utf-8") == "user modification\n"
    assert _run_git("status", "--porcelain", cwd=clone) == "M tracked.txt"


def test_remote_branch_checkout_ignores_stale_local_branch(tmp_path: Path) -> None:
    remote, seed, clone, first = _git_fixture(tmp_path)
    checkout_git_revision(clone, "main", expected_remote_url=str(remote))
    _run_git("checkout", "main", cwd=clone)
    assert _run_git("rev-parse", "main", cwd=clone) == first

    (seed / "tracked.txt").write_text("two\n", encoding="utf-8")
    _run_git("add", "tracked.txt", cwd=seed)
    _run_git("commit", "-m", "remote advances", cwd=seed)
    second = _run_git("rev-parse", "HEAD", cwd=seed)
    _run_git("push", "origin", "main", cwd=seed)

    result = checkout_git_revision(clone, "main", expected_remote_url=str(remote))
    assert result["revision_type"] == "remote_branch"
    assert result["resolved_ref"] == "refs/remotes/origin/main"
    assert result["commit"] == second
    assert _run_git("rev-parse", "main", cwd=clone) == first
    assert _run_git("rev-parse", "HEAD", cwd=clone) == second


def test_checkout_supports_exact_sha_and_rejects_invalid_dirty_and_remote(tmp_path: Path) -> None:
    remote, _seed, clone, first = _git_fixture(tmp_path)
    assert checkout_git_revision(clone, first, expected_remote_url=str(remote))["commit"] == first
    with pytest.raises(ColabPreflightError, match="neither a fetched"):
        checkout_git_revision(clone, "missing-revision", expected_remote_url=str(remote))
    (clone / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ColabPreflightError, match="dirty"):
        checkout_git_revision(clone, "main", expected_remote_url=str(remote))
    _run_git("restore", "tracked.txt", cwd=clone)
    with pytest.raises(ColabPreflightError, match="Unexpected origin"):
        checkout_git_revision(clone, "main", expected_remote_url=str(tmp_path / "other.git"))


def test_checkout_rejects_ambiguous_tag_and_remote_branch(tmp_path: Path) -> None:
    remote, seed, clone, _first = _git_fixture(tmp_path)
    _run_git("tag", "main", cwd=seed)
    _run_git("push", "origin", "refs/tags/main", cwd=seed)
    with pytest.raises(ColabPreflightError, match="Ambiguous"):
        checkout_git_revision(clone, "main", expected_remote_url=str(remote))


def test_safe_command_is_list_and_resume_is_explicit(tmp_path: Path) -> None:
    config = tmp_path / "resolved.yaml"
    config.write_text("training: {}\n", encoding="utf-8")
    command = build_train_command(REPO_ROOT, config, device="cpu", smoke_test=True)
    assert isinstance(command, list)
    assert "--smoke-test" in command
    assert "--resume-from" not in command
    with pytest.raises(ColabPreflightError, match="does not exist"):
        build_train_command(REPO_ROOT, config, device="cpu", resume_from=tmp_path / "missing.pt")
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    resumed = build_train_command(REPO_ROOT, config, device="cpu", resume_from=checkpoint)
    assert resumed[-2:] == ["--resume-from", str(checkpoint.resolve())]


def test_staging_is_confined_and_preserves_locator_and_identity(tmp_path: Path) -> None:
    drive = tmp_path / "drive" / "mutants.hdf5"
    drive.parent.mkdir()
    drive.write_bytes(b"hdf5-like-content")
    staging = tmp_path / "local" / "staging"
    record = stage_file(drive, staging / "mutants.hdf5", staging_root=staging, role="mutants", chunk_size=3)
    assert record["drive_locator"] == str(drive.resolve())
    assert record["local_locator"] != record["drive_locator"]
    assert record["role"] == "mutants"
    assert record["sha256"]
    assert Path(record["local_locator"]).read_bytes() == drive.read_bytes()
    with pytest.raises(ColabPreflightError, match="must be a child"):
        confined_path(tmp_path / "escape.hdf5", staging, label="staging")


def test_staging_is_idempotent_and_corruption_is_replaced(tmp_path: Path) -> None:
    source = tmp_path / "drive" / "wt.hdf5"
    source.parent.mkdir()
    source.write_bytes(b"original-content")
    root = tmp_path / "staging"
    destination = root / "wt.hdf5"
    first = stage_file(source, destination, staging_root=root, role="wt")
    second = stage_file(source, destination, staging_root=root, role="wt")
    assert first["reused"] is False
    assert second["reused"] is True
    destination.write_bytes(b"corrupt")
    repaired = stage_file(source, destination, staging_root=root, role="wt")
    assert repaired["reused"] is False
    assert destination.read_bytes() == source.read_bytes()
    assert repaired["sha256"] == first["sha256"]


def test_staging_rejects_same_file_symlink_escape_and_supports_spaces(tmp_path: Path) -> None:
    source = tmp_path / "Drive data" / "mutants source.hdf5"
    source.parent.mkdir()
    source.write_bytes(b"content")
    staging = tmp_path / "local staging"
    with pytest.raises(ColabPreflightError, match="different files"):
        stage_file(source, source, staging_root=source.parent, role="mutants")
    staging.mkdir()
    escape = staging / "escape"
    escape.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ColabPreflightError, match="must be a child"):
        stage_file(source, escape / "escaped.hdf5", staging_root=staging, role="mutants")
    record = stage_file(
        source, staging / "directory with spaces" / "mutants copy.hdf5",
        staging_root=staging, role="mutants",
    )
    assert Path(record["local_locator"]).read_bytes() == b"content"


def test_staging_copy_failure_cleans_own_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.hdf5"
    source.write_bytes(b"source")
    staging = tmp_path / "staging"
    destination = staging / "destination.hdf5"

    def partial_copy(_source: Path, temporary: Path, *, chunk_size: int) -> None:
        assert chunk_size > 0
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(b"partial")
        raise OSError("copy failed")

    monkeypatch.setattr("scripts.colab_preflight._stream_copy", partial_copy)
    with pytest.raises(OSError, match="copy failed"):
        stage_file(source, destination, staging_root=staging, role="mutants")
    assert not destination.exists()
    assert not list(staging.glob(".*.tmp"))


def test_missing_hdf5_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ColabPreflightError, match="Missing mutants HDF5"):
        require_hdf5(tmp_path / "missing.hdf5", label="mutants")


def test_output_preflight_and_space_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "allowed"
    result = preflight_output_root(root / "runs", allowed_root=root)
    assert result["operational"] is True
    assert not list((root / "runs").glob(".colab-preflight-*"))

    class Usage:
        total = 100
        used = 99
        free = 1

    monkeypatch.setattr("scripts.colab_preflight.shutil.disk_usage", lambda _path: Usage())
    with pytest.raises(ColabPreflightError, match="Insufficient free space"):
        require_free_space(root, 2)


def _runtime(torch_geometric: str = "2.6.1") -> dict[str, object]:
    return {
        "python": "3.11.0", "torch": "2.5.0", "torch_geometric": torch_geometric,
        "selected_device": "cpu", "device_tensor_test": "passed",
    }


def test_environment_marker_is_published_only_after_successful_validation(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example>=1\n", encoding="utf-8")
    calls: list[str] = []
    result = prepare_colab_environment(
        repo_root=tmp_path, marker_root=tmp_path, commit="a" * 40,
        requirements_path=requirements, device="cpu", python_version="3.11.0",
        torch_version="2.5.0", installer=lambda: calls.append("install"),
        validator=lambda _device: calls.append("validate") or _runtime(),
    )
    assert calls == ["install", "validate"]
    marker = Path(result["marker"])
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["torch_geometric"] == "2.6.1"
    assert payload["requirements_sha256"]

    calls.clear()
    reused = prepare_colab_environment(
        repo_root=tmp_path, marker_root=tmp_path, commit="a" * 40,
        requirements_path=requirements, device="cpu", python_version="3.11.0",
        torch_version="2.5.0", installer=lambda: calls.append("install"),
        validator=lambda _device: calls.append("validate") or _runtime(),
    )
    assert reused["reused"] is True
    assert calls == ["validate"]


@pytest.mark.parametrize("failure", ["import", "tensor"])
def test_environment_validation_failure_never_creates_marker(tmp_path: Path, failure: str) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example>=1\n", encoding="utf-8")

    def fail(_device: str) -> dict[str, object]:
        raise RuntimeError(failure)

    with pytest.raises(RuntimeError, match=failure):
        prepare_colab_environment(
            repo_root=tmp_path, marker_root=tmp_path, commit="b" * 40,
            requirements_path=requirements, device="cpu", python_version="3.11.0",
            torch_version="2.5.0", installer=lambda: None, validator=fail,
        )
    assert not list(tmp_path.glob(".environment-*.json"))


@pytest.mark.parametrize(
    ("field", "changed"),
    [("commit", "c" * 40), ("python", "3.12.0"), ("torch", "2.6.0")],
)
def test_environment_identity_changes_invalidate_marker(
    tmp_path: Path, field: str, changed: str
) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("one\n", encoding="utf-8")
    baseline = {"commit": "a" * 40, "python_version": "3.11.0", "torch_version": "2.5.0"}
    first = environment_identity(requirements_path=requirements, **baseline)
    if field == "commit": baseline["commit"] = changed
    elif field == "python": baseline["python_version"] = changed
    else: baseline["torch_version"] = changed
    second = environment_identity(requirements_path=requirements, **baseline)
    assert first["identity_sha256"] != second["identity_sha256"]


def test_requirements_change_invalidates_environment_identity(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("one\n", encoding="utf-8")
    kwargs = {"commit": "a" * 40, "python_version": "3.11", "torch_version": "2.5"}
    first = environment_identity(requirements_path=requirements, **kwargs)
    requirements.write_text("two\n", encoding="utf-8")
    assert first["identity_sha256"] != environment_identity(
        requirements_path=requirements, **kwargs
    )["identity_sha256"]


def test_environment_marker_with_different_resolved_pyg_forces_install(tmp_path: Path) -> None:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("one\n", encoding="utf-8")
    identity = environment_identity(
        commit="a" * 40, requirements_path=requirements,
        python_version="3.11", torch_version="2.5",
    )
    marker = tmp_path / f".environment-{identity['identity_sha256']}.json"
    marker.write_text(json.dumps({**identity, **_runtime("2.5.0")}), encoding="utf-8")
    calls: list[str] = []
    result = prepare_colab_environment(
        repo_root=tmp_path, marker_root=tmp_path, commit="a" * 40,
        requirements_path=requirements, device="cpu", python_version="3.11",
        torch_version="2.5", installer=lambda: calls.append("install"),
        validator=lambda _device: _runtime("2.6.1"),
    )
    assert calls == ["install"]
    assert result["reused"] is False
    assert json.loads(marker.read_text(encoding="utf-8"))["torch_geometric"] == "2.6.1"


def test_session_record_is_confined_and_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    destination = write_session_record(
        root / "sessions" / "one.json",
        allowed_root=root,
        payload={"commit": "abc", "command": ["python", "scripts/train.py"]},
    )
    assert json.loads(destination.read_text(encoding="utf-8"))["commit"] == "abc"
    with pytest.raises(ColabPreflightError, match="must be a child"):
        write_session_record(tmp_path / "escape.json", allowed_root=root, payload={})


def _runtime_overrides(tmp_path: Path) -> dict[str, object]:
    return {
        "paths.mutants_hdf5": str(tmp_path / "local" / "mutants.hdf5"),
        "paths.wt_companion_hdf5": str(tmp_path / "local" / "wt.hdf5"),
        "outputs.root_dir": str(tmp_path / "runs"),
        "split.persist_path": str(tmp_path / "runs" / "splits" / "positions.json"),
        "training.device": "cpu",
        "training.epochs": 2,
        "training.batch_size": 3,
        "project.seed": 7,
    }


def test_runtime_config_generation_changes_only_operational_allowlist(tmp_path: Path) -> None:
    from gnn_siamese.config import load_config

    original = load_config(BASE_CONFIG)
    result = generate_runtime_config(BASE_CONFIG, tmp_path / "resolved.yaml", overrides=_runtime_overrides(tmp_path))
    resolved = result["config"]
    scrubbed = deepcopy(resolved)
    baseline = deepcopy(original)
    for dotted in _runtime_overrides(tmp_path):
        parts = dotted.split(".")
        left, right = scrubbed, baseline
        for part in parts[:-1]:
            left, right = left[part], right[part]
        left[parts[-1]] = right[parts[-1]]
    assert scrubbed == baseline
    assert Path(result["config_path"]).is_file()
    assert set(result["changes"]) == set(_runtime_overrides(tmp_path))


def test_scientific_override_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ColabPreflightError, match="Scientific or unsupported"):
        generate_runtime_config(
            BASE_CONFIG,
            tmp_path / "bad.yaml",
            overrides={"loss.lambda_wt": 1.0},
        )


def _synthetic_runtime_config(tmp_path: Path) -> tuple[Path, object]:
    config = load_config(REPO_ROOT / "configs" / "model_b_end_to_end_smoke.yaml")
    artifacts = prepare_smoke_data(config)
    config["paths"]["mutants_hdf5"] = artifacts.mutants_hdf5
    config["paths"]["wt_companion_hdf5"] = artifacts.wt_companion_hdf5
    config["paths"]["sample_schema"] = artifacts.schema_json
    config["split"]["persist_path"] = str(tmp_path / "split.json")
    path = tmp_path / "runtime.yaml"
    save_config(config, path)
    return path, artifacts


def test_productive_hdf5_preflight_accepts_valid_roles(tmp_path: Path) -> None:
    config_path, artifacts = _synthetic_runtime_config(tmp_path)
    try:
        result = validate_runtime_hdf5(config_path)
        assert result["pair_count"] == 8
        assert result["mutants_hdf5"] != result["wt_companion_hdf5"]
    finally:
        artifacts.cleanup()


@pytest.mark.parametrize("field", ["schema_name", "schema_version"])
def test_productive_hdf5_preflight_rejects_incompatible_schema(
    tmp_path: Path, field: str
) -> None:
    config_path, artifacts = _synthetic_runtime_config(tmp_path)
    try:
        schema_path = Path(artifacts.schema_json)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        schema[field] = "wrong"
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        with pytest.raises(ColabPreflightError, match=field):
            validate_runtime_hdf5(config_path)
    finally:
        artifacts.cleanup()


def test_productive_hdf5_preflight_rejects_missing_schema_corrupt_file_and_swapped_roles(
    tmp_path: Path,
) -> None:
    config_path, artifacts = _synthetic_runtime_config(tmp_path)
    try:
        config = load_config(config_path)
        config["paths"]["sample_schema"] = str(tmp_path / "missing-schema.json")
        save_config(config, config_path)
        with pytest.raises(Exception, match="Schema file does not exist"):
            validate_runtime_hdf5(config_path)

        config["paths"]["sample_schema"] = artifacts.schema_json
        corrupt = tmp_path / "corrupt.hdf5"
        corrupt.write_text("not hdf5", encoding="utf-8")
        config["paths"]["mutants_hdf5"] = str(corrupt)
        save_config(config, config_path)
        with pytest.raises(ColabPreflightError, match="Productive HDF5 validation failed"):
            validate_runtime_hdf5(config_path)

        config["paths"]["mutants_hdf5"] = artifacts.wt_companion_hdf5
        config["paths"]["wt_companion_hdf5"] = artifacts.mutants_hdf5
        save_config(config, config_path)
        with pytest.raises(ColabPreflightError, match="mutant role"):
            validate_runtime_hdf5(config_path)
    finally:
        artifacts.cleanup()


def test_notebook_contract_mentions_artifact_validation_and_modes() -> None:
    source = _source()
    for token in (
        "format_version", "compatibility_metadata", "manifest_status", "resume_manifest_status",
        "global_step", "resume_global_step", "OUTPUT_MODE", "local_sync", "SYNC_LOCAL_OUTPUTS",
        "drive_locator", "local_locator", "sha256", "RESUME_CHECKPOINT",
    ):
        assert token in source


def test_rsync_command_is_list_and_missing_binary_fails_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.colab_preflight.shutil.which", lambda _name: "/usr/bin/rsync")
    command = build_rsync_command(tmp_path / "local output", tmp_path / "Drive output")
    assert command == [
        "/usr/bin/rsync", "-a", "--partial",
        f"{(tmp_path / 'local output').resolve()}/",
        f"{(tmp_path / 'Drive output').resolve()}/",
    ]
    monkeypatch.setattr("scripts.colab_preflight.shutil.which", lambda _name: None)
    with pytest.raises(ColabPreflightError, match="unavailable"):
        build_rsync_command(tmp_path / "local", tmp_path / "drive")


def test_rsync_failure_is_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.colab_preflight.shutil.which", lambda _name: "/usr/bin/rsync")
    seen: list[list[str]] = []

    def runner(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 23, "", "disk error")

    with pytest.raises(ColabPreflightError, match="not been persisted.*disk error"):
        sync_local_outputs(tmp_path / "local", tmp_path / "drive", runner=runner)
    assert seen and isinstance(seen[0], list)


def test_cli_contract_parser_accepts_only_known_unique_fields() -> None:
    stdout = "noise=value\nrun_dir=/runs/one\nmanifest_status=completed\n"
    assert parse_cli_contract(
        stdout, required_keys=("run_dir", "manifest_status"),
        allowed_keys=("run_dir", "manifest_status"),
    ) == {"run_dir": "/runs/one", "manifest_status": "completed"}
    with pytest.raises(ColabPreflightError, match="duplicate"):
        parse_cli_contract(
            "run_dir=/one\nrun_dir=/two\n", required_keys=("run_dir",),
            allowed_keys=("run_dir",),
        )
    with pytest.raises(ColabPreflightError, match="missing"):
        parse_cli_contract("unrelated=x\n", required_keys=("run_dir",), allowed_keys=("run_dir",))
