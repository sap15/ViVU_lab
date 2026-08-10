from __future__ import annotations

import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.training import checkpointing
from gnn_siamese.training.checkpointing import (
    _json_safe,
    build_resume_compatibility_payload,
    load_checkpoint,
    resume_from_checkpoint,
    save_checkpoint,
    save_checkpoint_payload_atomic,
)
from gnn_siamese.utils import atomic_io


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class DummyDataset:
    node_feature_names = ["a", "b"]
    edge_feature_names = ["distance"]
    graph_feature_names: list[str] = []
    pairs = [
        {
            "variant_id": "pkp2_A10V",
            "position": 10,
            "wt_aa": "A",
            "mut_aa": "V",
            "mutant_key": "mut_pos_10_A_V",
            "wt_key": "PKP2_WT",
            "chain_id": "A",
            "mutant_source_h5": "mutants_a.h5",
            "wt_source_h5": "wt_a.h5",
        },
        {
            "variant_id": "pkp2_G25D",
            "position": 25,
            "wt_aa": "G",
            "mut_aa": "D",
            "mutant_key": "mut_pos_25_G_D",
            "wt_key": "PKP2_WT",
            "chain_id": "A",
            "mutant_source_h5": "mutants_a.h5",
            "wt_source_h5": "wt_a.h5",
        },
    ]


class DummySplit:
    dataset_fingerprint = "split-1"
    split_type = "leave_position_out"


class DummySplitBundle:
    split = DummySplit()


def _own_temporaries(destination: Path) -> list[Path]:
    return list(destination.parent.glob(f".{destination.name}.*.tmp"))


def _write_checkpoint_fixture(path: Path) -> dict:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(4321)
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch_completed=2,
        global_step=7,
        best_metric=0.25,
        train_metrics={"loss": 1.0},
        validation_metrics={"loss": 0.5},
        resolved_config={"training": {"epochs": 4}},
        seed=123,
        split_id="split.json",
        split_fingerprint="split-1",
        dataset_fingerprint="dataset-1",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility=_compatibility(scheduler_name="none", scheduler=None),
        run_id="run_test",
        augmenter_state={"generator_state": generator.get_state()},
        data_loader_state={"train_generator_state": generator.get_state()},
    )
    return load_checkpoint(path)


def _compatibility(*, temperature: float = 0.2, scheduler_name: str = "cosine", scheduler: object | None = None) -> dict:
    optimizer = torch.optim.AdamW(TinyModel().parameters(), lr=0.01, weight_decay=0.1)
    config = {
        "model": {
            "architecture": "model_b",
            "graph_dim": 12,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "pooling": {"fusion": {"output_dim": 16}},
            "projection_instance": {"enabled": True, "hidden_dim": 10, "output_dim": 8, "num_layers": 2, "normalize_output": True},
            "projection_pair": {"enabled": False, "input": "r_delta", "hidden_dim": 12, "output_dim": 6, "normalize_output": True},
            "mlp_delta": {"enabled": False, "hidden_dim": 24, "output_dim": 12, "num_layers": 2, "dropout": 0.0},
        },
        "loss": {
            "main": "nt_xent",
            "temperature": temperature,
            "false_negative_mask": {"enabled": False},
            "lambda_wt": 0.0,
            "relative_wt": {"mode": "none"},
            "lambda_delta": 0.0,
            "delta": {"mode": "none"},
        },
        "augmentation": {"enabled": True, "feature_jitter": {"enabled": True, "std": 0.05}},
        "training": {
            "scheduler": scheduler_name,
            "gradient_clip_norm": None,
        },
    }
    return build_resume_compatibility_payload(
        config=config,
        dataset=DummyDataset(),
        split_bundle=DummySplitBundle(),
        optimizer=optimizer,
        scheduler=scheduler,
    )


def test_save_and_load_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    batch = torch.randn(4, 3)
    target = torch.randn(4, 2)
    loss = torch.nn.functional.mse_loss(model(batch), target)
    loss.backward()
    optimizer.step()
    scheduler.step()

    compatibility = _compatibility(scheduler=scheduler)
    compatibility["dataset_fingerprint"] = "dataset-1"
    checkpoint_path = tmp_path / "last.pt"
    random.seed(123)
    torch.manual_seed(123)

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_completed=2,
        global_step=7,
        best_metric=0.25,
        train_metrics={"loss": 1.0},
        validation_metrics={"loss": 0.5},
        resolved_config={"training": {"epochs": 4}},
        seed=123,
        split_id="split.json",
        split_fingerprint="split-1",
        dataset_fingerprint="dataset-1",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility=compatibility,
        run_id="run_test",
    )

    payload = load_checkpoint(checkpoint_path)
    assert payload["epoch_completed"] == 2
    assert payload["global_step"] == 7
    assert payload["best_metric"] == 0.25
    assert payload["scheduler_state_dict"] is not None

    fresh_model = TinyModel()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=4)
    resume_state = resume_from_checkpoint(
        checkpoint_path,
        model=fresh_model,
        optimizer=fresh_optimizer,
        scheduler=fresh_scheduler,
        expected_compatibility=compatibility,
    )

    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, fresh_model.state_dict()[key])
    original_optimizer_state = optimizer.state_dict()
    restored_optimizer_state = fresh_optimizer.state_dict()
    assert original_optimizer_state["param_groups"] == restored_optimizer_state["param_groups"]
    assert original_optimizer_state["state"].keys() == restored_optimizer_state["state"].keys()
    for param_id, state in original_optimizer_state["state"].items():
        restored_state = restored_optimizer_state["state"][param_id]
        assert state.keys() == restored_state.keys()
        for state_key, state_value in state.items():
            restored_value = restored_state[state_key]
            if isinstance(state_value, torch.Tensor):
                assert torch.equal(state_value, restored_value)
            else:
                assert state_value == restored_value
    assert fresh_scheduler.state_dict() == scheduler.state_dict()
    assert resume_state.epoch_completed == 2
    assert resume_state.next_epoch == 3
    assert resume_state.global_step == 7
    assert resume_state.best_metric == 0.25


def test_resume_rejects_incompatible_checkpoint(tmp_path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch_completed=1,
        global_step=1,
        best_metric=1.0,
        train_metrics={},
        validation_metrics={},
        resolved_config={"training": {"epochs": 1}},
        seed=1,
        split_id="split.json",
        split_fingerprint="split-a",
        dataset_fingerprint="dataset-a",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility={
            **_compatibility(scheduler_name="none", scheduler=None),
            "features": {
                "node_feature_names": ["a"],
                "edge_feature_names": ["distance"],
                "graph_feature_names": [],
            },
            "dataset_fingerprint": "dataset-a",
            "split_fingerprint": "split-a",
            "split_type": "leave_position_out",
        },
        run_id="run_test",
    )

    fresh_model = TinyModel()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="resume incompatibility"):
        resume_from_checkpoint(
            checkpoint_path,
            model=fresh_model,
            optimizer=fresh_optimizer,
            scheduler=None,
            expected_compatibility={
                **_compatibility(scheduler_name="none", scheduler=None),
                "architecture": {
                    **_compatibility(scheduler_name="none", scheduler=None)["architecture"],
                    "dimensions": {
                        **_compatibility(scheduler_name="none", scheduler=None)["architecture"]["dimensions"],
                        "hidden_dim": 32,
                    },
                },
                "features": {
                    "node_feature_names": ["a"],
                    "edge_feature_names": ["distance"],
                    "graph_feature_names": [],
                },
                "dataset_fingerprint": "dataset-a",
                "split_fingerprint": "split-a",
                "split_type": "leave_position_out",
            },
        )


def test_resume_rejects_scheduler_presence_class_and_config_mismatches(tmp_path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    compatibility = _compatibility(scheduler_name="step", scheduler=scheduler)
    checkpoint_path = tmp_path / "scheduler.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_completed=1,
        global_step=2,
        best_metric=0.5,
        train_metrics={},
        validation_metrics={},
        resolved_config={"training": {"epochs": 1}},
        seed=1,
        split_id="split.json",
        split_fingerprint="split-1",
        dataset_fingerprint="dataset-1",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility=compatibility,
        run_id="run_test",
    )

    fresh_model = TinyModel()
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="scheduler presence"):
        resume_from_checkpoint(
            checkpoint_path,
            model=fresh_model,
            optimizer=fresh_optimizer,
            scheduler=None,
            expected_compatibility=compatibility,
        )

    no_scheduler_path = tmp_path / "no_scheduler.pt"
    save_checkpoint(
        no_scheduler_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch_completed=1,
        global_step=2,
        best_metric=0.5,
        train_metrics={},
        validation_metrics={},
        resolved_config={"training": {"epochs": 1}},
        seed=1,
        split_id="split.json",
        split_fingerprint="split-1",
        dataset_fingerprint="dataset-1",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility=_compatibility(scheduler_name="none", scheduler=None),
        run_id="run_test",
    )
    with pytest.raises(ValueError, match="scheduler presence"):
        resume_from_checkpoint(
            no_scheduler_path,
            model=TinyModel(),
            optimizer=torch.optim.AdamW(TinyModel().parameters(), lr=0.01),
            scheduler=torch.optim.lr_scheduler.StepLR(torch.optim.AdamW(TinyModel().parameters(), lr=0.01), step_size=1),
            expected_compatibility=compatibility,
        )

    with pytest.raises(ValueError, match="scheduler class"):
        resume_from_checkpoint(
            checkpoint_path,
            model=TinyModel(),
            optimizer=torch.optim.AdamW(TinyModel().parameters(), lr=0.01),
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(torch.optim.AdamW(TinyModel().parameters(), lr=0.01), T_max=4),
            expected_compatibility=_compatibility(
                scheduler_name="cosine",
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(torch.optim.AdamW(TinyModel().parameters(), lr=0.01), T_max=4),
            ),
        )

    with pytest.raises(ValueError, match="scheduler config"):
        resume_from_checkpoint(
            checkpoint_path,
            model=TinyModel(),
            optimizer=torch.optim.AdamW(TinyModel().parameters(), lr=0.01),
            scheduler=torch.optim.lr_scheduler.StepLR(torch.optim.AdamW(TinyModel().parameters(), lr=0.01), step_size=2, gamma=0.1),
            expected_compatibility=_compatibility(
                scheduler_name="step",
                scheduler=torch.optim.lr_scheduler.StepLR(torch.optim.AdamW(TinyModel().parameters(), lr=0.01), step_size=2, gamma=0.1),
            ),
        )


def test_json_safe_normalizes_metadata_and_rejects_tensors() -> None:
    payload = {"b": Path("a.txt"), "a": (1, {"z": True, "y": [2, 3]})}
    assert _json_safe(payload) == {"a": [1, {"y": [2, 3], "z": True}], "b": "a.txt"}
    with pytest.raises(TypeError, match="Torch tensors"):
        _json_safe({"tensor": torch.ones(1)})


def test_resume_rejects_scientific_signature_changes_with_explicit_message(tmp_path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    compatibility = _compatibility(scheduler_name="none", scheduler=None, temperature=0.2)
    checkpoint_path = tmp_path / "signature.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch_completed=1,
        global_step=1,
        best_metric=1.0,
        train_metrics={},
        validation_metrics={},
        resolved_config={"training": {"epochs": 1}},
        seed=1,
        split_id="split.json",
        split_fingerprint="split-1",
        dataset_fingerprint="dataset-1",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility=compatibility,
        run_id="run_test",
    )

    with pytest.raises(ValueError, match=r"compatibility\.losses\.temperature"):
        resume_from_checkpoint(
            checkpoint_path,
            model=TinyModel(),
            optimizer=torch.optim.AdamW(TinyModel().parameters(), lr=0.01),
            scheduler=None,
            expected_compatibility=_compatibility(scheduler_name="none", scheduler=None, temperature=0.35),
        )


def test_dataset_fingerprint_changes_when_dataset_identity_or_position_changes() -> None:
    base = _compatibility(scheduler_name="none", scheduler=None)["dataset_fingerprint"]

    class PositionChangedDataset(DummyDataset):
        pairs = [dict(item) for item in DummyDataset.pairs]
        pairs[0]["position"] = 11

    class IdentityChangedDataset(DummyDataset):
        pairs = [dict(item) for item in DummyDataset.pairs]
        pairs[0]["mutant_source_h5"] = "mutants_b.h5"

    optimizer = torch.optim.AdamW(TinyModel().parameters(), lr=0.01)
    config = {
        "model": {
            "architecture": "model_b",
            "graph_dim": 12,
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "pooling": {"fusion": {"output_dim": 16}},
            "projection_instance": {"enabled": True, "hidden_dim": 10, "output_dim": 8, "num_layers": 2, "normalize_output": True},
            "projection_pair": {"enabled": False, "input": "r_delta", "hidden_dim": 12, "output_dim": 6, "normalize_output": True},
            "mlp_delta": {"enabled": False, "hidden_dim": 24, "output_dim": 12, "num_layers": 2, "dropout": 0.0},
        },
        "loss": {
            "main": "nt_xent",
            "temperature": 0.2,
            "false_negative_mask": {"enabled": False},
            "lambda_wt": 0.0,
            "relative_wt": {"mode": "none"},
            "lambda_delta": 0.0,
            "delta": {"mode": "none"},
        },
        "augmentation": {"enabled": True, "feature_jitter": {"enabled": True, "std": 0.05}},
        "training": {"scheduler": "none", "gradient_clip_norm": None},
    }

    position_changed = build_resume_compatibility_payload(
        config=config,
        dataset=PositionChangedDataset(),
        split_bundle=DummySplitBundle(),
        optimizer=optimizer,
        scheduler=None,
    )["dataset_fingerprint"]
    identity_changed = build_resume_compatibility_payload(
        config=config,
        dataset=IdentityChangedDataset(),
        split_bundle=DummySplitBundle(),
        optimizer=optimizer,
        scheduler=None,
    )["dataset_fingerprint"]

    assert position_changed != base
    assert identity_changed != base


def test_checkpoint_preserves_binary_torch_generator_state_exactly(tmp_path: Path) -> None:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    generator_state = generator.get_state()
    checkpoint_path = tmp_path / "generator.pt"

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch_completed=0,
        global_step=0,
        best_metric=None,
        train_metrics={},
        validation_metrics={},
        resolved_config={"training": {"epochs": 1}},
        seed=1234,
        split_id="split.json",
        split_fingerprint="split-1",
        dataset_fingerprint="dataset-1",
        dataset_id={"mutants_hdf5": "mutants.h5", "wt_companion_hdf5": "wt.h5"},
        compatibility=_compatibility(scheduler_name="none", scheduler=None),
        run_id="run_test",
        data_loader_state={"train_generator_state": generator_state},
    )

    payload = load_checkpoint(checkpoint_path)
    restored_state = payload["data_loader_state"]["train_generator_state"]
    assert isinstance(restored_state, torch.Tensor)
    assert torch.equal(restored_state, generator_state)


@pytest.mark.parametrize("checkpoint_name", ["last.pt", "best.pt"])
def test_atomic_checkpoint_save_failure_preserves_existing_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_name: str,
) -> None:
    checkpoint_path = tmp_path / checkpoint_name
    payload = _write_checkpoint_fixture(checkpoint_path)
    original_bytes = checkpoint_path.read_bytes()

    def fail_before_completion(payload_to_save, handle) -> None:
        raise RuntimeError("simulated torch.save failure")

    monkeypatch.setattr(checkpointing.torch, "save", fail_before_completion)

    with pytest.raises(RuntimeError, match="simulated torch.save failure"):
        save_checkpoint_payload_atomic(payload, checkpoint_path)

    assert checkpoint_path.read_bytes() == original_bytes
    assert _own_temporaries(checkpoint_path) == []


def test_partial_torch_save_failure_preserves_checkpoint_and_foreign_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "last.pt"
    payload = _write_checkpoint_fixture(checkpoint_path)
    original_bytes = checkpoint_path.read_bytes()
    foreign = tmp_path / ".last.pt.foreign.tmp"
    foreign.write_bytes(b"foreign")

    def write_partially_then_fail(payload_to_save, handle) -> None:
        handle.write(b"partial torch checkpoint")
        raise RuntimeError("simulated partial torch.save failure")

    monkeypatch.setattr(checkpointing.torch, "save", write_partially_then_fail)

    with pytest.raises(RuntimeError, match="simulated partial torch.save failure"):
        save_checkpoint_payload_atomic(payload, checkpoint_path)

    assert checkpoint_path.read_bytes() == original_bytes
    assert _own_temporaries(checkpoint_path) == [foreign]
    assert foreign.read_bytes() == b"foreign"


def test_checkpoint_validator_failure_preserves_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "last.pt"
    payload = _write_checkpoint_fixture(checkpoint_path)
    original_bytes = checkpoint_path.read_bytes()

    def fail_validation(*args, **kwargs):
        raise ValueError("simulated checkpoint validation failure")

    monkeypatch.setattr(checkpointing.torch, "load", fail_validation)

    with pytest.raises(ValueError, match="simulated checkpoint validation failure"):
        save_checkpoint_payload_atomic(payload, checkpoint_path)

    assert checkpoint_path.read_bytes() == original_bytes
    assert _own_temporaries(checkpoint_path) == []


def test_checkpoint_replace_failure_preserves_existing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "best.pt"
    payload = _write_checkpoint_fixture(checkpoint_path)
    original_bytes = checkpoint_path.read_bytes()

    def fail_replace(source: str | Path, target: str | Path) -> None:
        raise OSError("simulated checkpoint replace failure")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated checkpoint replace failure"):
        save_checkpoint_payload_atomic(payload, checkpoint_path)

    assert checkpoint_path.read_bytes() == original_bytes
    assert _own_temporaries(checkpoint_path) == []


def test_checkpoint_is_loaded_from_closed_temporary_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "last.pt"
    source_path = tmp_path / "source.pt"
    payload = _write_checkpoint_fixture(source_path)
    events: list[tuple[str, Path]] = []
    real_load = checkpointing.torch.load
    real_replace = atomic_io.os.replace

    def record_load(path, *args, **kwargs):
        temporary_path = Path(path)
        events.append(("validate", temporary_path))
        assert temporary_path.parent == checkpoint_path.parent
        assert temporary_path.name.startswith(f".{checkpoint_path.name}.")
        return real_load(path, *args, **kwargs)

    def record_replace(source: str | Path, target: str | Path) -> None:
        source_path_seen = Path(source)
        assert events == [("validate", source_path_seen)]
        events.append(("replace", source_path_seen))
        real_replace(source, target)

    monkeypatch.setattr(checkpointing.torch, "load", record_load)
    monkeypatch.setattr(atomic_io.os, "replace", record_replace)

    save_checkpoint_payload_atomic(payload, checkpoint_path)

    assert [event for event, _ in events] == ["validate", "replace"]
    loaded = real_load(checkpoint_path, map_location="cpu", weights_only=False)
    assert loaded.keys() == payload.keys()
    assert loaded["format_version"] == 1
    assert loaded["epoch_completed"] == payload["epoch_completed"]
    assert loaded["global_step"] == payload["global_step"]
    assert loaded["best_metric"] == payload["best_metric"]
    assert torch.equal(
        loaded["model_state_dict"]["linear.weight"],
        payload["model_state_dict"]["linear.weight"],
    )
    assert loaded["optimizer_state_dict"] == payload["optimizer_state_dict"]
    assert loaded["scheduler_state_dict"] == payload["scheduler_state_dict"]
    assert torch.equal(
        loaded["rng_state"]["torch_cpu_rng_state"],
        payload["rng_state"]["torch_cpu_rng_state"],
    )
    assert torch.equal(
        loaded["augmenter_state"]["generator_state"],
        payload["augmenter_state"]["generator_state"],
    )
    assert torch.equal(
        loaded["data_loader_state"]["train_generator_state"],
        payload["data_loader_state"]["train_generator_state"],
    )
    assert _own_temporaries(checkpoint_path) == []
