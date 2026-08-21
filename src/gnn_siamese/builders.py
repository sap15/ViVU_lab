"""Operational builders for the end-to-end Model B baseline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Callable

import h5py
import torch
from torch.utils.data import DataLoader, Subset

from gnn_siamese.config import load_schema, resolve_training_device, validate_c1_config
from gnn_siamese.data import (
    MutWtPairDataset,
    SmokeDataArtifacts,
    SplitSerializationError,
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
from gnn_siamese.data.model_a_pair_augmentations import (
    ModelAPairAugmentationConfig,
    ModelAPairAugmenter,
)
from gnn_siamese.data.position_batch_sampler import PositionDiverseBatchSampler
from gnn_siamese.models import (
    EdgeAwareGraphEncoder,
    InstanceProjectionHead,
    ModelBContrastiveBaseline,
    PairProjectionHead,
    ProjectionHeadConfig,
    RelationalRepresentation,
    SharedSiameseEncoderModel,
    ModelANodalMultiscalePair,
    ModelAOneView,
    ModelATwoView,
    ModelAMultiscalePooling,
    ModelAMultiscaleRelational,
    ModelAProjectionHead,
    NodeDeltaBlock,
)
from gnn_siamese.losses import NTXentLoss
from gnn_siamese.training import ModelAContrastive, TotalLossAssembler
from gnn_siamese.utils.fingerprints import fingerprint_hdf5_inputs, resolve_hdf5_dataset_id


class BuilderError(ValueError):
    """Raised when Model B operational components cannot be constructed."""


@dataclass(frozen=True)
class DatasetBundle:
    dataset: MutWtPairDataset
    schema: dict[str, Any]
    smoke_data: SmokeDataArtifacts | None = None
    smoke_selection: dict[str, Any] | None = None


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
    train_generator: torch.Generator | None = None


@dataclass(frozen=True)
class TrainingPipeline:
    config: dict[str, Any]
    dataset: MutWtPairDataset
    schema: dict[str, Any]
    split_bundle: SplitBundle
    dataloaders: DataLoadersBundle
    model: torch.nn.Module
    loss_fn: NTXentLoss
    total_loss_assembler: TotalLossAssembler
    optimizer: torch.optim.Optimizer
    scheduler: Any | None
    augmenter: GraphViewAugmenter | ModelAPairAugmenter
    device: torch.device
    smoke_data: SmokeDataArtifacts | None = None
    smoke_selection: dict[str, Any] | None = None


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BuilderError(f"{field_name} must be a mapping.")
    return value


def _apply_reproducibility_seeds(config: Mapping[str, Any]) -> None:
    reproducibility_cfg = _require_mapping(config.get("reproducibility", {}), field_name="config.reproducibility")
    project_cfg = _require_mapping(config.get("project", {}), field_name="config.project")
    seed = int(
        reproducibility_cfg.get(
            "seed_torch",
            reproducibility_cfg.get("seed_python", project_cfg.get("seed", 42)),
        )
    )
    random.seed(int(reproducibility_cfg.get("seed_python", seed)))
    try:
        import numpy as np

        np.random.seed(int(reproducibility_cfg.get("seed_numpy", seed)))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(reproducibility_cfg.get("seed_cuda", seed)))


def _resolve_path(config: Mapping[str, Any], raw_path: str | None) -> Path:
    if raw_path is None:
        raise BuilderError("Required path is missing from config.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    config_path = config.get("__config_path__")
    if config_path is None:
        return candidate.resolve()
    config_relative = (Path(str(config_path)).resolve().parent / candidate).resolve()
    if config_relative.exists():
        return config_relative
    working_tree_relative = candidate.resolve()
    if working_tree_relative.exists():
        return working_tree_relative
    return config_relative


def build_dataset_bundle(
    config: Mapping[str, Any],
    *,
    stage_callback: Callable[[str], None] | None = None,
) -> DatasetBundle:
    paths_cfg = _require_mapping(config.get("paths"), field_name="config.paths")
    smoke_cfg = _require_mapping(
        _require_mapping(config.get("training"), field_name="config.training").get("smoke_test", {}),
        field_name="config.training.smoke_test",
    )
    smoke_enabled = bool(smoke_cfg.get("enabled", False))
    smoke_data: SmokeDataArtifacts | None = None
    smoke_selection: dict[str, Any] | None = None

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
    if stage_callback is not None:
        stage_callback("opening_mutants_hdf5")
        with h5py.File(mutants_hdf5, "r"):
            pass
        stage_callback("opening_wt_hdf5")
        with h5py.File(wt_companion_hdf5, "r"):
            pass
        stage_callback("building_dataset")
    dataset = MutWtPairDataset(
        mutant_h5_path=mutants_hdf5,
        wt_h5_path=wt_companion_hdf5,
        config=config,
        schema=schema,
    )
    if smoke_enabled and smoke_data is None:
        dataset, smoke_selection = _apply_real_data_smoke_subset(config, dataset)
    return DatasetBundle(
        dataset=dataset,
        schema=schema,
        smoke_data=smoke_data,
        smoke_selection=smoke_selection,
    )


def _apply_real_data_smoke_subset(
    config: Mapping[str, Any],
    dataset: MutWtPairDataset,
) -> tuple[MutWtPairDataset, dict[str, Any] | None]:
    training_cfg = _require_mapping(config.get("training"), field_name="config.training")
    smoke_cfg = _require_mapping(training_cfg.get("smoke_test", {}), field_name="config.training.smoke_test")
    max_pairs = smoke_cfg.get("max_pairs")
    if max_pairs is None:
        return dataset, None

    max_pairs_int = int(max_pairs)
    if max_pairs_int <= 0:
        raise BuilderError("training.smoke_test.max_pairs must be positive when provided.")
    if len(dataset.pairs) <= max_pairs_int:
        return dataset, {
            "enabled": True,
            "source": "configured_hdf5_full_dataset",
            "requested_max_pairs": max_pairs_int,
            "selected_pair_count": len(dataset.pairs),
            "selected_variant_ids": [pair.variant_id for pair in dataset.pairs],
            "selected_positions": sorted({int(pair.position) for pair in dataset.pairs}),
        }

    split_cfg = _require_mapping(config.get("split"), field_name="config.split")
    seed = int(split_cfg.get("seed", _require_mapping(config.get("project"), field_name="config.project").get("seed", 42)))
    rng = random.Random(seed)

    by_position: dict[int, list[tuple[int, Any]]] = {}
    for index, pair in enumerate(dataset.pairs):
        by_position.setdefault(int(pair.position), []).append((index, pair))

    ordered_positions = sorted(by_position)
    rng.shuffle(ordered_positions)
    selected_indices: list[int] = []

    # Prefer one pair per position first so leave-position-out keeps train/validation non-empty.
    for position in ordered_positions:
        if len(selected_indices) >= max_pairs_int:
            break
        selected_indices.append(by_position[position][0][0])

    if len(selected_indices) < max_pairs_int:
        for position in ordered_positions:
            for index, _pair in by_position[position][1:]:
                if len(selected_indices) >= max_pairs_int:
                    break
                selected_indices.append(index)
            if len(selected_indices) >= max_pairs_int:
                break

    selected_indices = sorted(selected_indices)
    selected_pairs = [dataset.pairs[index] for index in selected_indices]
    selected_positions = sorted({int(pair.position) for pair in selected_pairs})
    if len(selected_positions) < 3:
        raise BuilderError(
            "Smoke subset resolved fewer than three unique positions; "
            "cannot build non-empty train/validation/test splits deterministically."
        )

    return dataset.subset_with_pairs(selected_pairs), {
        "enabled": True,
        "source": "configured_hdf5_subset",
        "requested_max_pairs": max_pairs_int,
        "selected_pair_count": len(selected_pairs),
        "selected_variant_ids": [pair.variant_id for pair in selected_pairs],
        "selected_positions": selected_positions,
        "selection_seed": seed,
    }


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
        try:
            split = load_leave_position_out_split(split_path, dataset_or_records=dataset)
        except SplitSerializationError as strict_error:
            if "dataset_fingerprint" not in str(strict_error):
                raise
            content_fingerprint = fingerprint_hdf5_inputs(
                mutants_path=dataset.mutant_h5_path,
                wt_companion_path=dataset.wt_h5_path,
                dataset_id=resolve_hdf5_dataset_id(config),
            )
            split = load_leave_position_out_split(
                split_path,
                dataset_or_records=dataset,
                hdf5_content_fingerprint=content_fingerprint,
            )
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
    project_cfg = _require_mapping(config.get("project"), field_name="config.project")
    batch_size = int(training_cfg.get("batch_size", 4))
    if batch_size <= 0:
        raise BuilderError("training.batch_size must be positive.")
    num_workers = int(training_cfg.get("num_workers", 0))
    train_dataset = _subset_dataset(dataset, split_bundle.train_indices)
    validation_dataset = _subset_dataset(dataset, split_bundle.validation_indices)
    test_dataset = _subset_dataset(dataset, split_bundle.test_indices)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
        "persistent_workers": bool(training_cfg.get("persistent_workers", False)),
        "collate_fn": collate_mut_wt_pairs,
    }
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(project_cfg.get("seed", 42)))
    architecture = str(
        _require_mapping(config.get("model"), field_name="config.model").get(
            "architecture", "model_b"
        )
    )
    if architecture == "model_a_nodal_multiscale_pair":
        def positions(indices: list[int]) -> list[int]:
            return [int(dataset.pairs[index].position) for index in indices]

        train_sampler = PositionDiverseBatchSampler(
            positions(split_bundle.train_indices), batch_size=batch_size,
            generator=train_generator, partition="train",
        )
        validation_sampler = PositionDiverseBatchSampler(
            positions(split_bundle.validation_indices), batch_size=batch_size,
            partition="validation",
        )
        test_sampler = PositionDiverseBatchSampler(
            positions(split_bundle.test_indices), batch_size=batch_size,
            partition="test",
        )
        return DataLoadersBundle(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            train_loader=DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs),
            validation_loader=DataLoader(validation_dataset, batch_sampler=validation_sampler, **loader_kwargs),
            test_loader=DataLoader(test_dataset, batch_sampler=test_sampler, **loader_kwargs),
            train_generator=train_generator,
        )
    loader_kwargs["batch_size"] = batch_size
    return DataLoadersBundle(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=DataLoader(train_dataset, shuffle=True, generator=train_generator, **loader_kwargs),
        validation_loader=DataLoader(validation_dataset, shuffle=False, **loader_kwargs),
        test_loader=DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        train_generator=train_generator,
    )


def _build_model_b(config: Mapping[str, Any], dataset: MutWtPairDataset) -> ModelBContrastiveBaseline:
    model_cfg = _require_mapping(config.get("model"), field_name="config.model")

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


def _build_model_a(config: Mapping[str, Any], dataset: MutWtPairDataset) -> ModelANodalMultiscalePair:
    model_cfg = _require_mapping(config.get("model"), field_name="config.model")
    loss_cfg = _require_mapping(config.get("loss", {}), field_name="config.loss")
    mask_cfg = _require_mapping(
        loss_cfg.get("false_negative_mask", {}), field_name="config.loss.false_negative_mask"
    )
    expected_mask = {
        "enabled": True,
        "mode": "same_position",
        "same_position": True,
        "strict": True,
        "min_valid_negatives": 1,
        "min_valid_negative_fraction": 0.0,
    }
    mismatches = {
        key: {"expected": expected, "actual": mask_cfg.get(key)}
        for key, expected in expected_mask.items()
        if mask_cfg.get(key) != expected
    }
    if mismatches:
        raise BuilderError(
            "Model A false-negative masking policy mismatch; the effective scientific "
            f"contract must be explicit in YAML: {mismatches}."
        )
    if float(loss_cfg.get("lambda_wt", 0.0)) > 0.0:
        raise BuilderError(
            "Model A currently supports only its contrastive loss; "
            "loss.lambda_wt must be 0 until an explicit Model A Relative-WT route is implemented."
        )
    if float(loss_cfg.get("lambda_delta", 0.0)) > 0.0:
        raise BuilderError(
            "Model A currently supports only its contrastive loss; "
            "loss.lambda_delta must be 0 until an explicit Model A Delta-loss route is implemented."
        )
    hidden_dim = int(model_cfg.get("hidden_dim", 128))
    encoder_cfg = _require_mapping(model_cfg.get("encoder_a", {}), field_name="config.model.encoder_a")
    delta_cfg = _require_mapping(model_cfg.get("node_delta", {}), field_name="config.model.node_delta")
    scales = tuple(
        str(value)
        for value in model_cfg.get("active_scales", ("mutation", "local", "global"))
    )
    unsupported_masks = set(scales).difference({"mutation", "local", "global"})
    if unsupported_masks:
        raise BuilderError(
            "Model A cannot enable scales without real batch annotations; "
            f"no compatible annotation/mask is available for {sorted(unsupported_masks)!r}. "
            "In particular, domain must remain disabled until a real domain mask is provided."
        )
    delta_dim = int(delta_cfg.get("output_dim", hidden_dim))
    pair_cfg = _require_mapping(model_cfg.get("pair_fusion"), field_name="config.model.pair_fusion")
    projection_cfg = _require_mapping(
        model_cfg.get("projection_pair_a"), field_name="config.model.projection_pair_a"
    )
    if not bool(pair_cfg.get("enabled", True)):
        raise BuilderError("Model A final architecture requires model.pair_fusion.enabled=true.")
    if not bool(projection_cfg.get("enabled", True)):
        raise BuilderError("Model A contrastive route requires model.projection_pair_a.enabled=true.")
    if str(pair_cfg.get("input", "h_pair_delta")) != "h_pair_delta":
        raise BuilderError("Model A pair_fusion input must be 'h_pair_delta'.")
    if str(projection_cfg.get("input", "z_delta_pair")) != "z_delta_pair":
        raise BuilderError("Model A projection_pair_a input must be 'z_delta_pair'.")
    if str(projection_cfg.get("class_name", "ModelAProjectionHead")) != "ModelAProjectionHead":
        raise BuilderError("Model A projection_pair_a class_name must be 'ModelAProjectionHead'.")
    pair_output_dim = int(pair_cfg.get("output_dim", 128))
    shared_encoder = EdgeAwareGraphEncoder(
        node_input_dim=int(dataset.node_input_dim),
        edge_input_dim=int(dataset.edge_input_dim),
        hidden_dim=hidden_dim,
        num_layers=int(model_cfg.get("num_layers", 3)),
        edge_mlp_hidden_dim=int(encoder_cfg.get("edge_mlp_hidden_dim", hidden_dim)),
        fusion_hidden_dim=int(encoder_cfg.get("fusion_hidden_dim", hidden_dim)),
        graph_output_dim=int(model_cfg.get("graph_dim", hidden_dim)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    # Model A consumes the shared encoder's nodal output and performs its own
    # multiscale pooling/fusion.  The graph-level fusion submodules are not on
    # its forward path, so they must not remain nominally trainable.
    for module in (
        shared_encoder.pre_fusion_norm,
        shared_encoder.mlp_fusion,
        shared_encoder.post_fusion_norm,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    one_view = ModelAOneView(
        shared_encoder=shared_encoder,
        node_delta_block=NodeDeltaBlock(
            input_dim=hidden_dim,
            hidden_dim=int(delta_cfg.get("hidden_dim", 2 * hidden_dim)),
            output_dim=delta_dim,
            activation=str(delta_cfg.get("activation", "relu")),
            dropout=float(delta_cfg.get("dropout", 0.0)),
        ),
        multiscale_pooling=ModelAMultiscalePooling(enabled_scales=scales),
        multiscale_relational=ModelAMultiscaleRelational(
            embedding_dim=delta_dim,
            active_scales=scales,
            pair_fusion_enabled=True,
            pair_fusion_hidden_dim=int(pair_cfg.get("hidden_dim", 256)),
            pair_fusion_output_dim=pair_output_dim,
            pair_fusion_activation=str(pair_cfg.get("activation", "relu")),
            pair_fusion_dropout=float(pair_cfg.get("dropout", 0.1)),
        ),
    )
    augmentation_cfg = _require_mapping(
        config.get("augmentation_pair_a", {}), field_name="config.augmentation_pair_a"
    )
    augmenter = ModelAPairAugmenter(ModelAPairAugmentationConfig(
        enabled=bool(augmentation_cfg.get("enabled", True)),
        feature_mask_probability=float(augmentation_cfg.get("feature_mask_probability", 0.1)),
        allowed_feature_names=tuple(augmentation_cfg.get("allowed_feature_names", ("bsa",))),
        masked_value=float(augmentation_cfg.get("masked_value", 0.0)),
    ))
    loss = build_nt_xent_loss(config)
    return ModelANodalMultiscalePair(
        two_view_model=ModelATwoView(one_view_model=one_view, pair_augmenter=augmenter),
        contrastive=ModelAContrastive(
            projection_head=ModelAProjectionHead(
                z_delta_pair_dim=pair_output_dim,
                hidden_dim=int(projection_cfg.get("hidden_dim", pair_output_dim)),
                projection_dim=int(projection_cfg.get("output_dim", 64)),
            ),
            nt_xent=loss,
            mask_mode=str(mask_cfg["mode"]),
            min_valid_negatives=int(mask_cfg["min_valid_negatives"]),
            min_valid_fraction=float(mask_cfg["min_valid_negative_fraction"]),
            strict=bool(mask_cfg["strict"]),
        ),
    )


def build_model(config: Mapping[str, Any], dataset: MutWtPairDataset) -> torch.nn.Module:
    model_cfg = _require_mapping(config.get("model"), field_name="config.model")
    architecture = str(model_cfg.get("architecture", "model_b"))
    if architecture in {"model_b", "siamese_shared_encoder", "model_b_graph_level_relational"}:
        return _build_model_b(config, dataset)
    if architecture == "model_a_nodal_multiscale_pair":
        return _build_model_a(config, dataset)
    supported = "model_a_nodal_multiscale_pair, model_b_graph_level_relational"
    raise BuilderError(
        f"Unsupported model.architecture {architecture!r}; supported architectures: {supported}."
    )


def _validate_model_input_dim(dataset: MutWtPairDataset, model: torch.nn.Module) -> None:
    if not dataset.pairs:
        raise BuilderError("Cannot validate model input dimensions because the dataset has no pairs.")

    graph_mut, graph_wt = dataset._load_pair_graphs(dataset.pairs[0])
    mut_dim = int(graph_mut.x.shape[1])
    wt_dim = int(graph_wt.x.shape[1])
    encoder = (
        model.siamese_model.shared_encoder
        if isinstance(model, ModelBContrastiveBaseline)
        else model.two_view_model.one_view_model.shared_encoder
    )
    configured_dim = int(encoder.input_projection.in_features)
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
        raise BuilderError("The integrated Model A/Model B pipeline supports only loss.main == 'nt_xent'.")
    if str(loss_cfg.get("positive_pair", "same_mutant_augmentations")) != "same_mutant_augmentations":
        raise BuilderError("The integrated pipeline requires same-pair augmentations as positives.")
    if bool(loss_cfg.get("use_wt_as_strong_positive", False)):
        raise BuilderError("The integrated pipeline forbids WT as a strong positive in NT-Xent.")
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


def build_scheduler(
    config: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
) -> Any | None:
    training_cfg = _require_mapping(config.get("training"), field_name="config.training")
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    epochs = int(training_cfg.get("epochs", 1))
    if scheduler_name in {"none", "", "null"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
    raise BuilderError(f"Unsupported training.scheduler {scheduler_name!r}.")


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


def build_training_augmenter(config: Mapping[str, Any], dataset: MutWtPairDataset, model: torch.nn.Module):
    if isinstance(model, ModelANodalMultiscalePair):
        return model.two_view_model.pair_augmenter
    return build_augmenter(config, dataset)


def build_training_pipeline(
    config: Mapping[str, Any],
    *,
    stage_callback: Callable[[str], None] | None = None,
) -> TrainingPipeline:
    def stage(name: str) -> None:
        if stage_callback is not None:
            stage_callback(name)

    config = validate_c1_config(config)
    stage("resolving_paths")
    device = resolve_training_device(
        str(_require_mapping(config.get("training"), field_name="config.training").get("device", "auto"))
    )
    _apply_reproducibility_seeds(config)
    dataset_bundle = build_dataset_bundle(config, stage_callback=stage_callback)
    try:
        # Dataset construction is eager for inventory/pairing and lazy for graph reads.
        len(dataset_bundle.dataset)
        stage("building_split")
        split_bundle = build_split_bundle(config, dataset_bundle.dataset, smoke_data=dataset_bundle.smoke_data)
        stage("building_dataloaders")
        dataloaders = build_dataloaders(config, dataset_bundle.dataset, split_bundle)
        stage("building_model")
        model = build_model(config, dataset_bundle.dataset)
        _validate_model_input_dim(dataset_bundle.dataset, model)
        model.to(device)
        stage("building_loss")
        loss_fn = build_nt_xent_loss(config)
        total_loss_assembler = build_total_loss_assembler(config)
        stage("building_optimizer")
        optimizer = build_optimizer(config, model)
        stage("building_scheduler")
        scheduler = build_scheduler(config, optimizer)
        stage("building_augmenter")
        augmenter = build_training_augmenter(config, dataset_bundle.dataset, model)
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
            scheduler=scheduler,
            augmenter=augmenter,
            device=device,
            smoke_data=dataset_bundle.smoke_data,
            smoke_selection=dataset_bundle.smoke_selection,
        )
    except Exception:
        if dataset_bundle.smoke_data is not None:
            dataset_bundle.smoke_data.cleanup()
        raise
