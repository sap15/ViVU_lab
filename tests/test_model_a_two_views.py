from __future__ import annotations

import pytest
import torch

from gnn_siamese.data.model_a_pair_augmentations import (
    ModelAPairAugmentationConfig,
    ModelAPairAugmenter,
)
from gnn_siamese.models.delta_block import NodeDeltaBlock
from gnn_siamese.models.encoder import EdgeAwareGraphEncoder
from gnn_siamese.models.model_a import ModelAOneView, ModelATwoView
from gnn_siamese.models.multiscale_pooling_a import ModelAMultiscalePooling
from gnn_siamese.models.multiscale_relational_a import ModelAMultiscaleRelational
from test_model_a_pair_augmentations import make_pair_batch


def _model(device: torch.device) -> ModelATwoView:
    one_view = ModelAOneView(
        shared_encoder=EdgeAwareGraphEncoder(
            node_input_dim=4,
            edge_input_dim=1,
            hidden_dim=8,
            num_layers=1,
            edge_mlp_hidden_dim=6,
            fusion_hidden_dim=10,
            graph_output_dim=6,
            dropout=0.0,
        ),
        node_delta_block=NodeDeltaBlock(
            input_dim=8, hidden_dim=12, output_dim=8, dropout=0.0
        ),
        multiscale_pooling=ModelAMultiscalePooling(
            enabled_scales=("mutation", "local", "global")
        ),
        multiscale_relational=ModelAMultiscaleRelational(
            embedding_dim=8,
            active_scales=("mutation", "local", "global"),
            pair_fusion_hidden_dim=20,
            pair_fusion_output_dim=16,
            pair_fusion_dropout=0.0,
        ),
    )
    augmenter = ModelAPairAugmenter(
        ModelAPairAugmentationConfig(
            feature_mask_probability=0.5,
            allowed_feature_names=("bsa", "hydrophobicity"),
            masked_value=-99.0,
        )
    )
    return ModelATwoView(
        one_view_model=one_view, pair_augmenter=augmenter
    ).to(device)


def _forward(
    device: torch.device,
    *,
    one_view_hook_calls: list[int] | None = None,
) -> tuple[ModelATwoView, object]:
    torch.manual_seed(33)
    pair_batch = make_pair_batch().to(device)
    model = _model(device)
    model.eval()
    hook = None
    if one_view_hook_calls is not None:
        hook = model.one_view_model.register_forward_hook(
            lambda module, _inputs, _output: one_view_hook_calls.append(id(module))
        )
    mutation_mut = pair_batch.graph_mut.is_mutation.bool()
    mutation_wt = torch.zeros_like(pair_batch.graph_wt.is_mutation, dtype=torch.bool)
    mutation_wt[pair_batch.wt_aligned_index[mutation_mut[pair_batch.mut_aligned_index]]] = True
    mutation_delta = mutation_mut[pair_batch.mut_aligned_index]
    local_mut = torch.ones(pair_batch.graph_mut.num_nodes, dtype=torch.bool, device=device)
    local_wt = torch.ones(pair_batch.graph_wt.num_nodes, dtype=torch.bool, device=device)
    local_delta = torch.ones(
        pair_batch.mut_aligned_index.numel(), dtype=torch.bool, device=device
    )
    output = model(
        pair_batch,
        run_seed=91,
        epoch=4,
        mutation_mask_MUT=mutation_mut,
        mutation_mask_WT=mutation_wt,
        mutation_mask_delta=mutation_delta,
        local_mask_MUT=local_mut,
        local_mask_WT=local_wt,
        local_mask_delta=local_delta,
    )
    if hook is not None:
        hook.remove()
    return model, output


def test_two_view_real_a1_to_a5_forward_eval_shapes_diversity_and_gradients() -> None:
    model, output = _forward(torch.device("cpu"))
    assert output.h_pair_delta_view1.shape == (2, 120)
    assert output.h_pair_delta_view2.shape == (2, 120)
    assert output.z_delta_pair_view1.shape == (2, 16)
    assert output.z_delta_pair_view2.shape == (2, 16)
    assert torch.isfinite(output.z_delta_pair_view1).all()
    assert torch.isfinite(output.z_delta_pair_view2).all()
    assert not torch.equal(output.z_delta_pair_view1, output.z_delta_pair_view2)
    assert output.view1.variant_id == ("v100", "v200")
    assert output.view2.variant_id == ("v100", "v200")

    loss = (
        output.z_delta_pair_view1.square().mean()
        + output.z_delta_pair_view2.square().mean()
    )
    loss.backward()
    for module in (
        model.one_view_model.shared_encoder,
        model.one_view_model.node_delta_block,
        model.one_view_model.multiscale_relational.pair_fusion,
    ):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


def test_two_view_has_one_shared_one_view_model_and_no_pair_mixing() -> None:
    calls: list[int] = []
    model, output = _forward(
        torch.device("cpu"), one_view_hook_calls=calls
    )
    assert calls == [id(model.one_view_model), id(model.one_view_model)]
    one_view_modules = [
        module
        for module in model.modules()
        if isinstance(module, ModelAOneView)
    ]
    assert one_view_modules == [model.one_view_model]
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert any(name.startswith("one_view_model.shared_encoder.") for name in parameter_names)
    assert not any("view1_encoder" in name or "view2_encoder" in name for name in parameter_names)
    encoder_parameter_ids = {
        id(parameter) for parameter in model.one_view_model.shared_encoder.parameters()
    }
    registered_encoder_parameter_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if name.startswith("one_view_model.shared_encoder.")
    }
    assert registered_encoder_parameter_ids == encoder_parameter_ids
    assert output.view1.alignment_metadata["alignment_ptr"].tolist() == [0, 3, 5]
    assert output.view2.alignment_metadata["aligned_pair_batch"].tolist() == [0, 0, 0, 1, 1]
    assert [item.variant_id for item in output.augmentation_metadata_view1] == [
        "v100",
        "v200",
    ]

    loss1 = output.z_delta_pair_view1.square().mean()
    loss2 = output.z_delta_pair_view2.square().mean()
    model.zero_grad(set_to_none=True)
    loss1.backward(retain_graph=True)
    gradients_after_view1 = {
        id(parameter): parameter.grad.detach().clone()
        for parameter in model.one_view_model.shared_encoder.parameters()
        if parameter.grad is not None
    }
    loss2.backward()
    assert gradients_after_view1
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.one_view_model.shared_encoder.parameters()
        if id(parameter) in gradients_after_view1
    )
    assert any(
        not torch.equal(
            parameter.grad,
            gradients_after_view1[id(parameter)],
        )
        for parameter in model.one_view_model.shared_encoder.parameters()
        if id(parameter) in gradients_after_view1
    )


def test_a1_to_a4_one_view_still_accepts_synthetic_batch_without_layout() -> None:
    device = torch.device("cpu")
    pair_batch = make_pair_batch().to(device)
    del pair_batch.graph_mut.node_feature_slices
    del pair_batch.graph_wt.node_feature_slices
    model = _model(device).one_view_model.eval()
    mutation_mut = pair_batch.graph_mut.is_mutation.bool()
    mutation_wt = torch.zeros_like(pair_batch.graph_wt.is_mutation, dtype=torch.bool)
    mutation_wt[
        pair_batch.wt_aligned_index[
            mutation_mut[pair_batch.mut_aligned_index]
        ]
    ] = True
    output = model(
        graph_mut=pair_batch.graph_mut,
        graph_wt=pair_batch.graph_wt,
        mut_aligned_index=pair_batch.mut_aligned_index,
        wt_aligned_index=pair_batch.wt_aligned_index,
        aligned_pair_batch=pair_batch.aligned_pair_batch,
        alignment_ptr=pair_batch.alignment_ptr,
        num_pairs=pair_batch.batch_size,
        mutation_mask_MUT=mutation_mut,
        mutation_mask_WT=mutation_wt,
        mutation_mask_delta=mutation_mut[pair_batch.mut_aligned_index],
        local_mask_MUT=torch.ones(pair_batch.graph_mut.num_nodes, dtype=torch.bool),
        local_mask_WT=torch.ones(pair_batch.graph_wt.num_nodes, dtype=torch.bool),
        local_mask_delta=torch.ones(
            pair_batch.mut_aligned_index.numel(), dtype=torch.bool
        ),
        variant_id=tuple(pair_batch.variant_ids),
    )
    assert output.z_delta_pair.shape == (2, 16)
    assert torch.isfinite(output.z_delta_pair).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_two_view_creation_and_forward_on_cuda() -> None:
    pair_batch = make_pair_batch().to(torch.device("cuda"))
    cuda_model = _model(torch.device("cuda"))
    view1, view2 = cuda_model.pair_augmenter.create_two_views(
        pair_batch, run_seed=91, epoch=4
    )
    assert view1.graph_mut.x.is_cuda
    assert view1.graph_wt.x.is_cuda
    assert view2.graph_mut.x.is_cuda
    assert view2.graph_wt.x.is_cuda
    assert view1.graph_mut.augmentation_feature_mask.is_cuda
    assert view2.graph_wt.augmentation_feature_mask.is_cuda
    model, output = _forward(torch.device("cuda"))
    assert output.z_delta_pair_view1.is_cuda
    assert output.z_delta_pair_view2.is_cuda
    assert torch.isfinite(output.z_delta_pair_view1).all()
    loss = output.z_delta_pair_view1.square().mean() + output.z_delta_pair_view2.square().mean()
    loss.backward()
    for module in (
        model.one_view_model.shared_encoder,
        model.one_view_model.node_delta_block,
        model.one_view_model.multiscale_relational.pair_fusion,
    ):
        gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
