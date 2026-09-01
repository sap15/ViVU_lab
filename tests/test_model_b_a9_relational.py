from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from gnn_siamese.builders import BuilderError, build_training_pipeline
from gnn_siamese.config import load_config
from gnn_siamese.models import (
    ModelBContrastiveBaseline,
    ModelBGraphLevelRelationalContrastive,
)
from gnn_siamese.training import forward_contrastive_batch, run_model_b_epoch
from gnn_siamese.training.checkpointing import (
    build_resume_compatibility_payload,
    validate_resume_compatibility,
)
from gnn_siamese.training.gradient_audit import build_module_registry
from tests.model_b_test_utils import build_model_b_config, create_multi_pair_hdf5, write_schema_json


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, *, relational: bool = True) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mutant_path = tmp_path / "mutants.hdf5"
    wt_path = tmp_path / "wt.hdf5"
    schema_path = tmp_path / "schema.json"
    split_path = tmp_path / "split.json"
    create_multi_pair_hdf5(mutant_path, wt_path)
    write_schema_json(schema_path)
    config = build_model_b_config(mutant_path, wt_path, schema_path, split_path)
    config["__config_path__"] = str(tmp_path / "config.yaml")
    config["loss"]["false_negative_mask"] = {
        "enabled": True,
        "mode": "same_position",
        "same_position": True,
        "strict": True,
        "min_valid_negatives": 1,
        "min_valid_negative_fraction": 0.0,
    }
    if relational:
        config["model"]["architecture"] = "model_b_graph_level_relational"
        config["model"]["projection_instance"]["enabled"] = False
        config["model"]["mlp_delta"]["enabled"] = True
        config["model"]["projection_pair"].update({"enabled": True, "input": "z_delta"})
    return config


def _changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
    return any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(before, module.parameters(), strict=True)
    )


def test_productive_b_a9_config_derives_expected_relational_dimensions() -> None:
    config = load_config(REPO_ROOT / "configs" / "model_b_a9.yaml")
    graph_dim = config["model"]["graph_dim"]
    assert config["model"]["architecture"] == "model_b_graph_level_relational"
    assert config["model"]["projection_instance"]["enabled"] is False
    assert 5 * graph_dim == 640
    assert config["model"]["mlp_delta"]["output_dim"] == 128
    assert config["model"]["projection_pair"]["input"] == "z_delta"
    assert config["model"]["projection_pair"]["output_dim"] == 64
    assert config["loss"]["lambda_wt"] == config["loss"]["lambda_delta"] == 0.0


def test_relational_b_builds_expected_route_shapes_and_outputs(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(_config(tmp_path))
    assert isinstance(pipeline.model, ModelBGraphLevelRelationalContrastive)
    assert pipeline.model.architecture_name == "model_b_graph_level_relational"
    assert pipeline.model.siamese_model.projection_instance is None

    batch = next(iter(pipeline.dataloaders.train_loader))
    dispatched = forward_contrastive_batch(
        pipeline.model,
        batch,
        augmenter=pipeline.augmenter,
        loss_fn=pipeline.loss_fn,
        run_seed=123,
        epoch=1,
    )
    output = dispatched.model_output
    graph_dim = pipeline.config["model"]["graph_dim"]
    delta_dim = pipeline.config["model"]["mlp_delta"]["output_dim"]
    projection_dim = pipeline.config["model"]["projection_pair"]["output_dim"]
    assert output.view1.r_delta.shape == (batch.batch_size, 5 * graph_dim)
    assert output.view1.z_delta.shape == (batch.batch_size, delta_dim)
    assert output.view1.z_instance_pair.shape == (batch.batch_size, projection_dim)
    assert output.z1 is output.view1.z_instance_pair
    assert output.z2 is output.view2.z_instance_pair
    assert output.view1.z_instance is None and output.view2.z_instance is None
    payload = output.to_dict()
    for name in ("h_mut", "h_wt", "r_delta", "z_delta", "z_instance_pair", "z1", "z2"):
        assert name in payload
    assert output.view1.z_delta_status == "unvalidated"
    loss = pipeline.loss_fn(output.z1, output.z2).loss
    assert torch.isfinite(loss)


def test_relational_b_nt_xent_updates_encoder_delta_and_pair_projection(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(_config(tmp_path))
    siamese = pipeline.model.siamese_model
    modules = [siamese.shared_encoder, siamese.relational_module.mlp_delta, siamese.projection_pair]
    assert all(module is not None for module in modules)
    snapshots = [[parameter.detach().clone() for parameter in module.parameters()] for module in modules]

    epoch = run_model_b_epoch(
        pipeline.model,
        pipeline.dataloaders.train_loader,
        pipeline.total_loss_assembler,
        optimizer=pipeline.optimizer,
        device="cpu",
        augmenter=pipeline.augmenter,
        run_seed=123,
        epoch=1,
        contrastive_loss_fn=pipeline.loss_fn,
    )

    assert torch.isfinite(torch.tensor(epoch.mean_loss))
    for before, module in zip(snapshots, modules, strict=True):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
        assert gradients
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum().item() > 0.0 for gradient in gradients)
        assert _changed(before, module)


def test_relational_b_gradient_registry_connects_pair_modules_to_nt_xent(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(_config(tmp_path))
    registry = build_module_registry(
        pipeline.model,
        {"nt_xent": 1.0, "relative_wt": 0.0, "delta": 0.0},
    )
    for name in ("encoder", "mlp_delta", "projection_pair"):
        assert registry[name]["module"] is not None
        assert registry[name]["status_hint"] == "active"
        assert registry[name]["connected_losses"] == ["nt_xent"]
    assert registry["projection_instance"]["status_hint"] == "not_applicable"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("model", "mlp_delta", "enabled"), False, "mlp_delta"),
        (("model", "projection_pair", "enabled"), False, "projection_pair"),
        (("model", "projection_pair", "input"), "r_delta", "input=z_delta"),
        (("model", "projection_instance", "enabled"), True, "projection_instance"),
    ],
)
def test_relational_b_builder_rejects_incomplete_route(
    tmp_path: Path, path: tuple[str, ...], value: object, match: str
) -> None:
    config = _config(tmp_path)
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    with pytest.raises(BuilderError, match=match):
        build_training_pipeline(config)


def test_historical_b_baseline_remains_instance_contrastive(tmp_path: Path) -> None:
    pipeline = build_training_pipeline(_config(tmp_path, relational=False))
    assert isinstance(pipeline.model, ModelBContrastiveBaseline)
    batch = next(iter(pipeline.dataloaders.train_loader))
    output = forward_contrastive_batch(
        pipeline.model, batch, augmenter=pipeline.augmenter, loss_fn=pipeline.loss_fn,
        run_seed=123, epoch=1,
    ).model_output
    assert output.z1 is output.view1.z_instance
    assert output.z2 is output.view2.z_instance


def test_resume_compatibility_rejects_baseline_vs_relational_architecture(tmp_path: Path) -> None:
    baseline = build_training_pipeline(_config(tmp_path / "baseline", relational=False))
    relational = build_training_pipeline(_config(tmp_path / "relational"))
    baseline_payload = build_resume_compatibility_payload(
        config=baseline.config, dataset=baseline.dataset, split_bundle=baseline.split_bundle,
        optimizer=baseline.optimizer, scheduler=baseline.scheduler, schema=baseline.schema,
    )
    relational_payload = build_resume_compatibility_payload(
        config=relational.config, dataset=relational.dataset, split_bundle=relational.split_bundle,
        optimizer=relational.optimizer, scheduler=relational.scheduler, schema=relational.schema,
    )
    with pytest.raises(ValueError, match="compatibility.architecture"):
        validate_resume_compatibility(
            {"compatibility": baseline_payload}, relational_payload
        )
