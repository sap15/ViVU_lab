from __future__ import annotations

from copy import deepcopy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from gnn_siamese.builders import _apply_reproducibility_seeds, build_training_pipeline
from gnn_siamese.data.model_a_pair_augmentations import stable_seed
from gnn_siamese.data.pairing import PairingKey
from gnn_siamese.training.a9_contract import A9ContractError, apply_a9_run_seed
from gnn_siamese.training.checkpointing import (
    build_resume_compatibility_payload,
    validate_resume_compatibility,
)
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


def _base(tmp_path: Path) -> dict:
    mutants = tmp_path / "mutants.hdf5"
    wt = tmp_path / "wt.hdf5"
    schema = tmp_path / "schema.json"
    split = tmp_path / "split.json"
    create_multi_pair_hdf5(mutants, wt)
    write_schema_json(schema)
    base = build_model_b_config(mutants, wt, schema, split)
    base["__config_path__"] = str(tmp_path / "config.yaml")
    # Materialize one split with its independent seed, then freeze it.
    build_training_pipeline(deepcopy(base))
    base["split"].update({"seed": 42, "allow_create": False})
    base["a9"] = {"run_seeds": [11, 23, 37, 41, 53]}
    return base


def _first_parameter(pipeline: object) -> torch.Tensor:
    return next(pipeline.model.parameters()).detach().clone()


def test_a9_run_seed_controls_initialization_dataloader_and_b_augmentation(tmp_path: Path) -> None:
    base = _base(tmp_path)
    eleven_a = build_training_pipeline(apply_a9_run_seed(base, 11))
    eleven_b = build_training_pipeline(apply_a9_run_seed(base, 11))
    twenty_three = build_training_pipeline(apply_a9_run_seed(base, 23))
    assert torch.equal(_first_parameter(eleven_a), _first_parameter(eleven_b))
    assert not torch.equal(_first_parameter(eleven_a), _first_parameter(twenty_three))
    assert eleven_a.dataloaders.train_generator.initial_seed() == 11
    assert twenty_three.dataloaders.train_generator.initial_seed() == 23
    assert eleven_a.augmenter.config.seed == 11
    assert twenty_three.augmenter.config.seed == 23
    ids_11 = [item.variant_id for item in eleven_a.split_bundle.split.assignments]
    ids_23 = [item.variant_id for item in twenty_three.split_bundle.split.assignments]
    assert ids_11 == ids_23
    assert eleven_a.config["split"]["seed"] == twenty_three.config["split"]["seed"] == 42


def test_a9_run_seed_controls_python_numpy_and_torch_rng() -> None:
    config = apply_a9_run_seed(
        {"project": {}, "reproducibility": {}, "split": {"seed": 42, "allow_create": False}}, 11
    )
    draws = []
    for _ in range(2):
        _apply_reproducibility_seeds(config)
        draws.append((random.random(), float(np.random.random()), float(torch.rand(()))))
    assert draws[0] == draws[1]
    assert config["reproducibility"]["seed_cuda"] == 11


def test_model_a_pair_augmentation_seed_is_derived_from_run_seed() -> None:
    kwargs = dict(
        epoch=1, variant_id="v", pair_key=PairingKey("A", 1, "M"), mutant_key="m", wt_key="w",
        view_id=1, transform_id="paired_feature_mask",
    )
    assert stable_seed(run_seed=11, **kwargs) == stable_seed(run_seed=11, **kwargs)
    assert stable_seed(run_seed=11, **kwargs) != stable_seed(run_seed=23, **kwargs)


def test_a9_resume_compatibility_rejects_different_run_seed(tmp_path: Path) -> None:
    base = _base(tmp_path)
    pipeline_11 = build_training_pipeline(apply_a9_run_seed(base, 11))
    pipeline_23 = build_training_pipeline(apply_a9_run_seed(base, 23))
    def compatibility(pipeline: object) -> dict:
        return build_resume_compatibility_payload(
            config=pipeline.config, dataset=pipeline.dataset, split_bundle=pipeline.split_bundle,
            optimizer=pipeline.optimizer, scheduler=pipeline.scheduler, schema=pipeline.schema,
        )
    source = compatibility(pipeline_11)
    target = compatibility(pipeline_23)
    with pytest.raises(ValueError, match="a9_run_seed"):
        validate_resume_compatibility({"compatibility": source}, target)


def test_a9_rejects_undeclared_seed_and_never_changes_split_seed() -> None:
    base = {"project": {}, "reproducibility": {}, "split": {"seed": 42, "allow_create": False}}
    with pytest.raises(A9ContractError, match="not predeclared"):
        apply_a9_run_seed(base, 42)
    resolved = apply_a9_run_seed(base, 53)
    assert resolved["project"]["seed"] == 53
    assert all(value == 53 for value in resolved["reproducibility"].values())
    assert resolved["split"]["seed"] == 42
