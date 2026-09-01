from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gnn_siamese.training.checkpointing import load_validated_historical_best
from gnn_siamese.training.checkpointing import (
    _scheduler_config_from_instance,
    _validate_scheduler_resume,
)
from gnn_siamese.training.loop import _update_early_stopping


def test_early_stopping_patience_min_delta_and_no_off_by_one() -> None:
    best, bad = None, 0
    best, bad, improved = _update_early_stopping(1.0, best, bad, mode="min", min_delta=0.0)
    assert (best, bad, improved) == (1.0, 0, True)
    for expected_bad in range(1, 16):
        best, bad, improved = _update_early_stopping(1.0, best, bad, mode="min", min_delta=0.0)
        assert not improved and bad == expected_bad
        assert (bad >= 15) is (expected_bad == 15)


def test_early_stopping_uses_independent_best_for_min_delta() -> None:
    best, bad = 1.0, 0
    best, bad, improved = _update_early_stopping(0.95, best, bad, mode="min", min_delta=0.1)
    assert (best, bad, improved) == (1.0, 1, False)
    best, bad, improved = _update_early_stopping(0.89, best, bad, mode="min", min_delta=0.1)
    assert (best, bad, improved) == (0.89, 0, True)
    best, bad, improved = _update_early_stopping(0.90, best, bad, mode="min", min_delta=0.1)
    assert (best, bad, improved) == (0.89, 1, False)


def _payload(*, epoch: int, best_metric: float, compatibility: dict) -> dict:
    return {
        "run_id": "source", "architecture": "model_b_graph_level_relational", "seed": 11,
        "split_fingerprint": "split", "dataset_fingerprint": "dataset",
        "epoch_completed": epoch, "best_metric": best_metric, "compatibility": compatibility,
    }


def test_resume_from_last_preserves_real_historical_best(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    compatibility = {"compatibility_metadata": {"version": 2}, "architecture": {"name": "model_b_graph_level_relational"}}
    last = _payload(epoch=5, best_metric=0.3, compatibility=compatibility)
    best = _payload(epoch=3, best_metric=0.3, compatibility=compatibility)
    torch.save(last, checkpoints / "last.pt")
    torch.save(best, checkpoints / "best.pt")
    restored = load_validated_historical_best(
        checkpoints / "last.pt", last_payload=last, expected_compatibility=compatibility
    )
    assert restored["epoch_completed"] == 3
    assert restored["best_metric"] == 0.3


def test_resume_fails_closed_without_demonstrable_historical_best(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    compatibility = {"compatibility_metadata": {"version": 2}, "architecture": {"name": "model_b_graph_level_relational"}}
    last = _payload(epoch=5, best_metric=0.3, compatibility=compatibility)
    torch.save(last, checkpoints / "last.pt")
    with pytest.raises(ValueError, match="historical best"):
        load_validated_historical_best(
            checkpoints / "last.pt", last_payload=last, expected_compatibility=compatibility
        )


def test_cosine_tmax_100_continuous_equals_state_dict_resume() -> None:
    continuous_parameter = torch.nn.Parameter(torch.tensor(1.0))
    continuous_optimizer = torch.optim.AdamW([continuous_parameter], lr=0.001)
    continuous = torch.optim.lr_scheduler.CosineAnnealingLR(continuous_optimizer, T_max=100)
    continuous_lrs = []
    saved_optimizer = saved_scheduler = None
    for epoch in range(1, 11):
        continuous_optimizer.step()
        continuous.step()
        continuous_lrs.append(continuous_optimizer.param_groups[0]["lr"])
        if epoch == 4:
            saved_optimizer = continuous_optimizer.state_dict()
            saved_scheduler = continuous.state_dict()

    resumed_parameter = torch.nn.Parameter(torch.tensor(1.0))
    resumed_optimizer = torch.optim.AdamW([resumed_parameter], lr=0.001)
    resumed = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=100)
    resumed_optimizer.load_state_dict(saved_optimizer)
    resumed.load_state_dict(saved_scheduler)
    resumed_lrs = []
    for _epoch in range(5, 11):
        resumed_optimizer.step()
        resumed.step()
        resumed_lrs.append(resumed_optimizer.param_groups[0]["lr"])
    assert resumed_lrs == continuous_lrs[4:]
    assert resumed.state_dict()["T_max"] == 100


def test_scheduler_compatibility_records_and_rejects_tmax() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.001)
    scheduler_100 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scheduler_99 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=99)
    config_100 = _scheduler_config_from_instance(scheduler_100, scheduler_name="cosine")
    config_99 = _scheduler_config_from_instance(scheduler_99, scheduler_name="cosine")
    assert config_100 == {
        "name": "cosine", "step_semantics": "once_per_epoch", "eta_min": 0.0, "T_max": 100
    }
    checkpoint = {
        "scheduler_state_dict": scheduler_100.state_dict(),
        "compatibility": {"scheduler": {"class": "CosineAnnealingLR", "config": config_100}},
    }
    with pytest.raises(ValueError, match="scheduler config"):
        _validate_scheduler_resume(
            scheduler=scheduler_99,
            checkpoint_payload=checkpoint,
            expected_compatibility={
                "a9_run_seed": 11,
                "scheduler": {"class": "CosineAnnealingLR", "config": config_99}
            },
        )
