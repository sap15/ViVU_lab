from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pytest
import torch

from gnn_siamese.builders import BuilderError, build_dataloaders, build_model
from gnn_siamese.config import load_config
from gnn_siamese.data.position_batch_sampler import (
    PositionDiverseBatchError,
    PositionDiverseBatchSampler,
)
from gnn_siamese.losses import build_false_negative_mask


def _batches(positions: list[int], batch_size: int, seed: int | None = None) -> list[list[int]]:
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    sampler = PositionDiverseBatchSampler(
        positions, batch_size=batch_size, generator=generator, partition="test"
    )
    return list(sampler)


def _assert_valid_and_exact(positions: list[int], batches: list[list[int]]) -> None:
    flat = [index for batch in batches for index in batch]
    assert Counter(flat) == Counter(range(len(positions)))
    for batch in batches:
        batch_positions = [positions[index] for index in batch]
        assert len(set(batch_positions)) >= 2
        mask = build_false_negative_mask(
            batch_size=len(batch), mode="same_position", positions=batch_positions,
            min_valid_negatives=1, min_valid_fraction=0.0, strict=True,
        )
        assert min(item.valid_negatives for item in mask.per_anchor_stats) >= 1


def test_real_shape_78_rebalances_terminal_same_position_pair() -> None:
    positions = list(range(76)) + [827, 827]
    batches = _batches(positions, 4)
    assert [len(batch) for batch in batches] == [4] * 19 + [2]
    _assert_valid_and_exact(positions, batches)
    assert [positions[index] for index in batches[-1]] != [827, 827]


@pytest.mark.parametrize("seed", [42, 123])
def test_train_seed_is_reproducible_and_preserves_every_variant(seed: int) -> None:
    positions = [786, 786, 797, 800, 827, 827]
    first = _batches(positions, 4, seed)
    second = _batches(positions, 4, seed)
    assert first == second
    assert [len(batch) for batch in first] == [4, 2]
    _assert_valid_and_exact(positions, first)


def test_validation_and_test_without_generator_are_deterministic() -> None:
    positions = [10, 10, 20, 30, 40, 40, 50]
    assert _batches(positions, 4) == _batches(positions, 4)
    _assert_valid_and_exact(positions, _batches(positions, 4))


@pytest.mark.parametrize("positions", [[827, 827], [5, 5, 5], [7]])
def test_impossible_partition_fails_during_loader_planning(positions: list[int]) -> None:
    with pytest.raises(PositionDiverseBatchError, match="partition|at least two"):
        _batches(positions, 4)


def test_batch_size_larger_than_valid_small_dataset_uses_one_batch() -> None:
    positions = [10, 20, 20]
    batches = _batches(positions, 8)
    assert [len(batch) for batch in batches] == [3]
    _assert_valid_and_exact(positions, batches)


def test_model_a_policy_is_explicit_and_builder_rejects_drift() -> None:
    config = load_config("configs/model_a_pilot.yaml")
    policy = config["loss"]["false_negative_mask"]
    assert policy == {
        **policy,
        "enabled": True,
        "mode": "same_position",
        "same_position": True,
        "strict": True,
        "min_valid_negatives": 1,
        "min_valid_negative_fraction": 0.0,
    }

    class Dataset:
        node_input_dim = 2
        edge_input_dim = 1

    changed = deepcopy(config)
    changed["loss"]["false_negative_mask"]["mode"] = "none"
    with pytest.raises(BuilderError, match="policy mismatch"):
        build_model(changed, Dataset())


def test_model_b_keeps_standard_dataloader_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    class Pair:
        def __init__(self, position: int) -> None:
            self.position = position

    class Dataset(torch.utils.data.Dataset):
        pairs = [Pair(1), Pair(1), Pair(2), Pair(3)]

        def __len__(self) -> int:
            return len(self.pairs)

        def __getitem__(self, index: int) -> int:
            return index

    class Split:
        train_indices = [0, 1]
        validation_indices = [2, 3]
        test_indices = [0, 3]

    monkeypatch.setattr("gnn_siamese.builders.collate_mut_wt_pairs", lambda values: values)
    config = {
        "project": {"seed": 42},
        "model": {"architecture": "model_b_graph_level_relational"},
        "training": {"batch_size": 2, "num_workers": 0},
    }
    loaders = build_dataloaders(config, Dataset(), Split())
    assert not isinstance(loaders.train_loader.batch_sampler, PositionDiverseBatchSampler)
    assert list(loaders.validation_loader) == [[2, 3]]


def test_generator_state_reproduces_next_train_epoch_for_resume() -> None:
    positions = [1, 1, 2, 2, 3, 3, 4, 5]
    generator = torch.Generator().manual_seed(123)
    sampler = PositionDiverseBatchSampler(
        positions, batch_size=4, generator=generator, partition="train"
    )
    list(sampler)
    checkpoint_state = generator.get_state()
    expected_next_epoch = list(sampler)

    resumed_generator = torch.Generator()
    resumed_generator.set_state(checkpoint_state)
    resumed = PositionDiverseBatchSampler(
        positions, batch_size=4, generator=resumed_generator, partition="train"
    )
    assert list(resumed) == expected_next_epoch
