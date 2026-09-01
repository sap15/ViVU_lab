from __future__ import annotations

from pathlib import Path

import pytest

from gnn_siamese.training.a9_contract import A9ContractError
from scripts.a9_preflight import (
    _parse_args,
    prospective_a9_run_dir,
    resolve_a9_runtime_configs,
)
from scripts.colab_preflight import ColabPreflightError


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_A = REPO_ROOT / "configs" / "model_a_a9.yaml"


def _runtime(tmp_path: Path, *, seed: int = 11, a_root: Path | None = None, b_root: Path | None = None):
    mutants = tmp_path / "mutants.hdf5"
    wt = tmp_path / "wt.hdf5"
    return resolve_a9_runtime_configs(
        CONFIG_A, architecture="model_a_nodal_multiscale_pair", run_seed=seed,
        mutants_hdf5=mutants, wt_hdf5=wt,
        output_root=a_root or tmp_path / "model_a_a9",
        peer_output_root=b_root or tmp_path / "model_b_a9", repo_root=REPO_ROOT,
    )


def test_a9_preflight_cli_requires_runtime_overrides() -> None:
    args = _parse_args([
        "--config", "a.yaml", "--mutants-hdf5", "mut.h5", "--wt-hdf5", "wt.h5",
        "--output-root", "/drive/model_a_a9", "--peer-output-root", "/drive/model_b_a9",
        "--allowed-output-root", "/drive", "--expected-commit", "abc",
        "--architecture", "model_a_nodal_multiscale_pair", "--run-seed", "23",
    ])
    assert args.run_seed == 23 and args.mutants_hdf5 == "mut.h5"


def test_runtime_config_overrides_paths_root_and_all_run_rng_without_editing_yaml(tmp_path: Path) -> None:
    original = CONFIG_A.read_text(encoding="utf-8")
    config_a, config_b = _runtime(tmp_path, seed=23)
    assert CONFIG_A.read_text(encoding="utf-8") == original
    for config in (config_a, config_b):
        assert config["project"]["seed"] == 23
        assert set(config["reproducibility"][key] for key in (
            "seed_python", "seed_numpy", "seed_torch", "seed_cuda", "seed_dataloader"
        )) == {23}
        assert config["split"]["seed"] == 42 and config["split"]["allow_create"] is False
        assert config["paths"]["mutants_hdf5"] == str((tmp_path / "mutants.hdf5").resolve())
    assert config_a["outputs"]["root_dir"] == str((tmp_path / "model_a_a9").resolve())
    assert config_b["outputs"]["root_dir"] == str((tmp_path / "model_b_a9").resolve())


def test_runtime_config_rejects_invalid_seed_and_architecture(tmp_path: Path) -> None:
    with pytest.raises(A9ContractError):
        _runtime(tmp_path, seed=42)
    with pytest.raises(A9ContractError, match="does not match"):
        resolve_a9_runtime_configs(
            CONFIG_A, architecture="model_b_graph_level_relational", run_seed=11,
            mutants_hdf5=tmp_path / "m", wt_hdf5=tmp_path / "w",
            output_root=tmp_path / "model_b_a9", peer_output_root=tmp_path / "model_a_a9",
            repo_root=REPO_ROOT,
        )


def test_runtime_config_rejects_nested_and_symlink_roots(tmp_path: Path) -> None:
    real = tmp_path / "model_a_a9"
    real.mkdir()
    alias = tmp_path / "model_b_a9"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(A9ContractError, match="outputs.namespaces|physically separate"):
        _runtime(tmp_path, a_root=real, b_root=alias)


def test_prospective_fresh_path_uses_real_layout_and_never_overwrites(tmp_path: Path) -> None:
    config_a, _ = _runtime(tmp_path)
    candidate = prospective_a9_run_dir(config_a, run_id="fixed")
    assert candidate == (
        tmp_path / "model_a_a9" / "model_a_nodal_multiscale_pair" / "run_fixed"
    ).resolve()
    assert not candidate.exists()
    candidate.mkdir(parents=True)
    with pytest.raises(ColabPreflightError, match="already exists"):
        prospective_a9_run_dir(config_a, run_id="fixed")
