"""Operational builders for the end-to-end Model B baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from gnn_siamese.config import load_schema
from gnn_siamese.data import (
    MutWtPairDataset,
    SmokeDataArtifacts,
    build_leave_position_out_split,
    collate_mut_wt_pairs,
    load_leave_position_out_split,
    prepare_smoke_data,
)
from gnn_siamese.data.augmentations import (
    AugmentationConfigError,
    GraphViewAugmenter,
    resolve_graph_augmentation_config,
)
from gnn_siamese.models import (
    EdgeAwareGraphEncoder,
    InstanceProjectionHead,
    ModelBContrastiveBaseline,
    PairProjectionHead,
    ProjectionHeadConfig,
    RelationalRepresentation,
    SharedSiameseEncoderModel,
)
from gnn_siamese.losses import NTXentLoss
from gnn_siamese.training import TotalLossAssembler


class BuilderError(ValueError):
    """Raised when Model B operational components cannot be constructed."""


@dataclass(frozen=True)
class DatasetBundle:
    dataset: MutWtPairDataset
    schema: dict[str, Any]
    smoke_data: SmokeDataArtifacts | None = None


@dataclass(frozen=True)
class SplitBundle:
    split: Any
    split_path: str
    created: bool
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


@dataclass(frozen=True)
class DataLoadersBundle:
    train_dataset: Subset
    validation_dataset: Subset
    test_dataset: Subset
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader


@dataclass(frozen=True)
class TrainingPipeline:
    config: dict[str, Any]
    dataset: MutWtPairDataset
    schema: dict[str, Any]
    split_bundle: SplitBundle
    dataloaders: DataLoadersBundle
    model: ModelBContrastiveBaseline
    loss_fn: NTXentLoss
    total_loss_assembler: TotalLossAssembler
    optimizer: torch.optim.Optimizer
    augmenter: GraphViewAugmenter
    device: torch.device
    smoke_data: SmokeDataArtifacts | None = None


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BuilderError(f"{field_name} must be a mapping.")
    return value


def _resolve_device(requested: str | None) -> torch.device:
    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(requested))


def _resolve_path(config: Mapping[str, Any], raw_path: str | None) -> Path:
    if raw_path is None:
        raise BuilderError("Required path is missing from config.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    config_path = config.get("__config_path__")
    if config_path is None:
        return candidate.resolve()
    return (Path(str(config_path)).resolve().parent / candidate).resolve()


def build_dataset_bundle(config: Mapping[str, Any]) -> DatasetBundle:
    paths_cfg = _require_mapping(config.get("paths"), field_name="config.paths")
    smoke_cfg = _require_mapping(
        _require_mapping(config.get("training"), field_name="config.training").get("smoke_test", {}),
        field_name="config.training.smoke_test",
    )
    smoke_enabled = bool(smoke_cfg.get("enabled", False))
    smoke_data: SmokeDataArtifacts | None = None

    mutants_path_raw = paths_cfg.get("mutants_hdf5")
    wt_path_raw = paths_cfg.get("wt_companion_hdf5")
    if smoke_enabled and (mutants_path_raw is None or wt_path_raw is None):
        smoke_data = prepare_smoke_data(config)
        mutants_hdf5 = smoke_data.mutants_hdf5
        wt_companion_hdf5 = smoke_data.wt_companion_hdf5
        schema_path = Path(smoke_data.schema_json)
    else:
        mutants_hdf5 = str(_resolve_path(config, str(mutants_path_raw) if mutants_path_raw else None))
        wt_companion_hdf5 = str(_resolve_path(config, str(wt_path_raw) if wt_path_raw else None))
        schema_path = _resolve_path(config, str(paths_cfg.get("sample_schema", "sample_data/sample_schema.json")))
    schema = load_schema(schema_path)
    dataset = MutWtPairDataset(
        mutant_h5_path=mutants_hdf5,
        wt_h5_path=wt_companion_hdf5,
        config=config,
        schema=schema,
    )
    return DatasetBundle(dataset=dataset, schema=schema, smoke_data=smoke_data)


def build_split_bundle(
    config: Mapping[str, Any],
    dataset: MutWtPairDataset,
    *,
    smoke_data: SmokeDataArtifacts | None = None,
) -> SplitBundle:
    split_cfg = _require_mapping(config.get("split"), field_name="config.split")
    smoke_cfg = _require_mapping(
        _require_mapping(config.get("training"), field_name="config.training").get("smoke_test", {}),
        field_name="config.training.smoke_test",
    )
    smoke_enabled = bool(smoke_cfg.get("enabled", False))
    split_path = (
        Path(smoke_data.split_json)
        if smoke_enabled and smoke_data is not None
        else _resolve_path(config, str(split_cfg.get("persist_path", "runs/model_b_split.json")))
    )
    allow_create = bool(split_cfg.get("allow_create", True))
    if split_path.exists():
        split = load_leave_position_out_split(split_path, dataset_or_records=dataset)
        created = False
    else:
        if not allow_create:
            raise BuilderError(f"Persisted split does not exist and split.allow_create=false: {split_path}")
        split = build_leave_position_out_split(dataset, config)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split.save_json(split_path)
        created = True

    indices = split.dataset_indices_by_partition(dataset)
    return SplitBundle(
        split=split,
        split_path=str(split_path),
        created=created,
        train_indices=indices["train"],
        validation_indices=indices["validation"],
        test_indices=indices["test"],
    )


def _subset_dataset(dataset: MutWtPairDataset, indices: list[int]) -> Subset:
    return Subset(dataset, indices)


def build_dataloaders(config: Mapping[str, Any], dataset: MutWtPairDataset, split_bundle: SplitBundle) -> DataLoadersBundle:
    training_cfg = _require_mapping(config.get("training"), field_name="config.training")
    batch_size = int(training_cfg.get("batch_size", 4))
    if batch_size <= 0:
        raise BuilderError("training.batch_size must be positive.")
    num_workers = int(training_cfg.get("num_workers", 0))
    train_dataset = _subset_dataset(dataset, split_bundle.train_indices)
    validation_dataset = _subset_dataset(dataset, split_bundle.validation_indices)
    test_dataset = _subset_dataset(dataset, split_bundle.test_indices)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_mut_wt_pairs,
    }
    return DataLoadersBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        validation_loader=DataLoader(validation_dataset, shuffle=False, **loader_kwargs),
        test_loader=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


def build_model(config: Mapping[str, Any], dataset: MutWtPairDataset) -> ModelBContrastiveBaseline:
    model_cfg = _require_mapping(config.get("model"), field_name="config.model")
    architecture = str(model_cfg.get("architecture", "model_b"))
    if architecture not in {"model_b", "siamese_shared_encoder"}:
        raise BuilderError(f"Unsupported model.architecture {architecture!r}.")

    graph_dim = int(model_cfg.get("graph_dim", 128))
    encoder = EdgeAwareGraphEncoder(
        node_input_dim=int(dataset.node_input_dim),
        edge_input_dim=int(dataset.edge_input_dim),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        num_layers=int(model_cfg.get("num_layers", 3)),
        fusion_hidden_dim=int(
            _require_mapping(
                _require_mapping(model_cfg.get("pooling"), field_name="config.model.pooling").get("fusion", {}),
                field_name="config.model.pooling.fusion",
            ).get("output_dim", graph_dim)
        ),
        graph_output_dim=graph_dim,
        dropout=float(model_cfg.get("dropout", 0.1)),
    )

    projection_instance_cfg = _require_mapping(
        model_cfg.get("projection_instance"),
        field_name="config.model.projection_instance",
    )
    if not bool(projection_instance_cfg.get("enabled", True)):
        raise BuilderError("Model B baseline requires model.projection_instance.enabled=true.")
    if int(projection_instance_cfg.get("num_layers", 2)) != 2:
        raise BuilderError("Current projection_instance implementation only supports num_layers=2.")

    relational_cfg = _require_mapping(model_cfg.get("relational"), field_name="config.model.relational")
    severity_cfg = _require_mapping(relational_cfg.get("severity"), field_name="config.model.relational.severity")
    mlp_delta_cfg = _require_mapping(model_cfg.get("mlp_delta"), field_name="config.model.mlp_delta")
    projection_pair_cfg = _require_mapping(model_cfg.get("projection_pair"), field_name="config.model.projection_pair")

    siamese = SharedSiameseEncoderModel(
        shared_encoder=encoder,
        relational_module=RelationalRepresentation(
            embedding_dim=graph_dim,
            severity_eps=float(severity_cfg.get("epsilon", 1.0e-8)),
            mlp_delta_enabled=bool(mlp_delta_cfg.get("enabled", False)),
            mlp_delta_hidden_dim=int(mlp_delta_cfg.get("hidden_dim", 2 * graph_dim)),
            mlp_delta_output_dim=int(mlp_delta_cfg.get("output_dim", graph_dim)),
            mlp_delta_num_layers=int(mlp_delta_cfg.get("num_layers", 2)),
            mlp_delta_dropout=float(mlp_delta_cfg.get("dropout", 0.0)),
        ),
        projection_instance=InstanceProjectionHead(
            config=ProjectionHeadConfig(
                input_dim=graph_dim,
                hidden_dim=int(projection_instance_cfg.get("hidden_dim", graph_dim)),
                output_dim=int(projection_instance_cfg.get("output_dim", 64)),
                dropout=float(model_cfg.get("dropout", 0.0)),
                normalize_output=bool(projection_instance_cfg.get("normalize_output", True)),
            )
        ),
        projection_pair=(
            PairProjectionHead(
                config=ProjectionHeadConfig(
                    input_dim=5 * graph_dim
                    if str(projection_pair_cfg.get("input", "r_delta")) == "r_delta"
                    else int(mlp_delta_cfg.get("output_dim", graph_dim)),
                    hidden_dim=int(projection_pair_cfg.get("hidden_dim", graph_dim)),
                    output_dim=int(projection_pair_cfg.get("output_dim", 64)),
                    dropout=float(model_cfg.get("dropout", 0.0)),
                    normalize_output=bool(projection_pair_cfg.get("normalize_output", True)),
                )
            )
            if bool(projection_pair_cfg.get("enabled", False))
            else None
        ),
        pair_projection_source=str(projection_pair_cfg.get("input", "r_delta")),
    )
    return ModelBContrastiveBaseline(siamese)


def _validate_model_input_dim(dataset: MutWtPairDataset, model: ModelBContrastiveBaseline) -> None:
    if not dataset.pairs:
        raise BuilderError("Cannot validate model input dimensions because the dataset has no pairs.")

    graph_mut, graph_wt = dataset._load_pair_graphs(dataset.pairs[0])
    mut_dim = int(graph_mut.x.shape[1])
    wt_dim = int(graph_wt.x.shape[1])
    configured_dim = int(model.siamese_model.shared_encoder.input_projection.in_features)
    if mut_dim != wt_dim:
        raise BuilderError(
            "Mutant and WT node feature dimensions differ during Model B pipeline construction: "
            f"mutant_dim={mut_dim}, wt_dim={wt_dim}, configured_dim={configured_dim}. "
            f"Configured node features: {dataset.input_spec.configured_node_feature_names!r}. "
            f"Mutant final node features: {graph_mut.node_feature_names!r}. "
            f"WT final node features: {graph_wt.node_feature_names!r}."
        )
    if configured_dim != mut_dim:
        raise BuilderError(
            "Encoder input dimension does not match the final graph.x width during Model B pipeline construction: "
            f"mutant_dim={mut_dim}, wt_dim={wt_dim}, configured_dim={configured_dim}. "
            f"Configured node features: {dataset.input_spec.configured_node_feature_names!r}. "
            f"Mutant final node features: {graph_mut.node_feature_names!r}. "
            f"WT final node features: {graph_wt.node_feature_names!r}."
        )


def build_nt_xent_loss(config: Mapping[str, Any]) -> NTXentLoss:
    loss_cfg = _require_mapping(config.get("loss"), field_name="config.loss")
    if str(loss_cfg.get("main", "nt_xent")) != "nt_xent":
        raise BuilderError("Model B baseline currently supports only loss.main == 'nt_xent'.")
    if str(loss_cfg.get("positive_pair", "same_mutant_augmentations")) != "same_mutant_augmentations":
        raise BuilderError("Model B baseline requires loss.positive_pair == 'same_mutant_augmentations'.")
    if bool(loss_cfg.get("use_wt_as_strong_positive", False)):
        raise BuilderError("Model B baseline forbids WT as a strong positive in NT-Xent.")
    return NTXentLoss(temperature=float(loss_cfg.get("temperature", 0.2)))


def build_total_loss_assembler(config: Mapping[str, Any]) -> TotalLossAssembler:
    loss_cfg = _require_mapping(config.get("loss"), field_name="config.loss")
    false_negative_mask_cfg = _require_mapping(
        loss_cfg.get("false_negative_mask", {}),
        field_name="config.loss.false_negative_mask",
    )
    relative_wt_cfg = _require_mapping(
        loss_cfg.get("relative_wt", {}),
        field_name="config.loss.relative_wt",
    )
    delta_cfg = _require_mapping(
        loss_cfg.get("delta", {}),
        field_name="config.loss.delta",
    )

    mask_mode = "none"
    if bool(false_negative_mask_cfg.get("enabled", False)):
        mask_mode = str(false_negative_mask_cfg.get("mode", "none"))

    stop_gradient_wt = relative_wt_cfg.get("stop_gradient")
    if stop_gradient_wt is None:
        stop_gradient_wt = relative_wt_cfg.get("stop_gradient_wt", False)

    return TotalLossAssembler.from_config(
        {
            "nt_xent_weight": 1.0,
            "relative_wt_weight": float(loss_cfg.get("lambda_wt", 0.0)),
            "delta_weight": float(loss_cfg.get("lambda_delta", 0.0)),
            "nt_xent_kwargs": {
                "temperature": float(loss_cfg.get("temperature", 0.2)),
            },
            "false_negative_mask_kwargs": {
                "mode": mask_mode,
                "alpha": false_negative_mask_cfg.get("structural_soft", {}).get("alpha"),
                "min_valid_negatives": float(false_negative_mask_cfg.get("min_valid_negatives", 8.0)),
                "min_valid_fraction": float(false_negative_mask_cfg.get("min_valid_negative_fraction", 0.25)),
                "strict": bool(false_negative_mask_cfg.get("strict", False)),
                "combine_same_position": bool(false_negative_mask_cfg.get("same_position", False)),
            },
            "relative_wt_kwargs": {
                "mode": str(relative_wt_cfg.get("mode", "none")),
                "distance": str(relative_wt_cfg.get("distance", "euclidean")),
                "margin": float(relative_wt_cfg.get("margin", 0.0)),
                "direction": str(relative_wt_cfg.get("direction", "min")),
                "stop_gradient_wt": bool(stop_gradient_wt),
                "predictive_loss": str(relative_wt_cfg.get("predictive_loss", "mse")),
                "allow_energy_target": bool(relative_wt_cfg.get("allow_energy_target", False)),
            },
            "relative_wt_target_name": relative_wt_cfg.get("target_name"),
            "delta_kwargs": {
                "mode": str(delta_cfg.get("mode", "none")),
                "consistency_loss": str(delta_cfg.get("consistency_loss", "mse")),
                "gamma": float(delta_cfg.get("gamma", 1.0)),
                "descriptor_loss": str(delta_cfg.get("descriptor_loss", "mse")),
                "allow_energy_target": bool(delta_cfg.get("allow_energy_target", False)),
            },
            "delta_target_name": delta_cfg.get("target_name"),
        }
    )


def build_optimizer(config: Mapping[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    training_cfg = _require_mapping(config.get("training"), field_name="config.training")
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    learning_rate = float(training_cfg.get("learning_rate", 1.0e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    if learning_rate <= 0.0:
        raise BuilderError("training.learning_rate must be strictly positive.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise BuilderError(f"Unsupported training.optimizer {optimizer_name!r}.")


def build_augmenter(config: Mapping[str, Any], dataset: MutWtPairDataset) -> GraphViewAugmenter:
    project_cfg = _require_mapping(config.get("project"), field_name="config.project")
    try:
        augmentation_config = resolve_graph_augmentation_config(
            config,
            seed=int(project_cfg.get("seed", 42)),
        )
        return GraphViewAugmenter(
            config=augmentation_config,
            node_feature_names=dataset.node_feature_names,
        )
    except AugmentationConfigError as exc:
        raise BuilderError(str(exc)) from exc


def build_training_pipeline(config: Mapping[str, Any]) -> TrainingPipeline:
    dataset_bundle = build_dataset_bundle(config)
    try:
        split_bundle = build_split_bundle(config, dataset_bundle.dataset, smoke_data=dataset_bundle.smoke_data)
        dataloaders = build_dataloaders(config, dataset_bundle.dataset, split_bundle)
        model = build_model(config, dataset_bundle.dataset)
        _validate_model_input_dim(dataset_bundle.dataset, model)
        loss_fn = build_nt_xent_loss(config)
        total_loss_assembler = build_total_loss_assembler(config)
        optimizer = build_optimizer(config, model)
        augmenter = build_augmenter(config, dataset_bundle.dataset)
        device = _resolve_device(str(_require_mapping(config.get("training"), field_name="config.training").get("device", "auto")))
        return TrainingPipeline(
            config=dict(config),
            dataset=dataset_bundle.dataset,
            schema=dataset_bundle.schema,
            split_bundle=split_bundle,
            dataloaders=dataloaders,
            model=model,
            loss_fn=loss_fn,
            total_loss_assembler=total_loss_assembler,
            optimizer=optimizer,
            augmenter=augmenter,
            device=device,
            smoke_data=dataset_bundle.smoke_data,
        )
    except Exception:
        if dataset_bundle.smoke_data is not None:
            dataset_bundle.smoke_data.cleanup()
        raise
