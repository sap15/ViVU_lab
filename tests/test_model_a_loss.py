from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gnn_siamese.losses import NTXentLoss, build_false_negative_mask
from gnn_siamese.losses.false_negative_mask import FalseNegativeMaskDegenerateError
from gnn_siamese.models import ModelAProjectionHead, ModelATwoViewOutput
from gnn_siamese.training.model_a_contrastive import ModelAContrastive


def _two_view(z1: torch.Tensor, z2: torch.Tensor) -> ModelATwoViewOutput:
    return ModelATwoViewOutput(
        view1=SimpleNamespace(z_delta_pair=z1),
        view2=SimpleNamespace(z_delta_pair=z2),
        h_pair_delta_view1=torch.full((z1.shape[0], 11), -101.0),
        h_pair_delta_view2=torch.full((z2.shape[0], 11), -102.0),
        z_delta_pair_view1=torch.full_like(z1, -201.0),
        z_delta_pair_view2=torch.full_like(z2, -202.0),
        augmentation_metadata_view1=(),
        augmentation_metadata_view2=(),
    )


class _SpyNTXent(NTXentLoss):
    received: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, **kwargs: object):  # type: ignore[no-untyped-def]
        self.received = (z1, z2)
        return super().forward(z1, z2, **kwargs)


def _route(loss: NTXentLoss | None = None) -> ModelAContrastive:
    return ModelAContrastive(
        projection_head=ModelAProjectionHead(
            z_delta_pair_dim=5, hidden_dim=8, projection_dim=3
        ),
        nt_xent=loss or NTXentLoss(temperature=0.2),
    )


def test_finite_symmetric_loss_and_exact_valid_negative_counts() -> None:
    torch.manual_seed(4)
    z1, z2 = torch.randn(3, 5), torch.randn(3, 5)
    route = _route().eval()
    forward = route(_two_view(z1, z2), positions=[100, 100, 220])
    reverse = route(_two_view(z2, z1), positions=[100, 100, 220])
    assert torch.isfinite(forward.loss)
    assert forward.loss.item() == pytest.approx(reverse.loss.item(), abs=1.0e-6)
    assert forward.valid_negative_counts.tolist() == [2, 2, 4, 2, 2, 4]


def test_same_position_mask_excludes_self_and_variants_but_never_positive() -> None:
    mask = build_false_negative_mask(
        3, mode="same_position", positions=[100, 100, 220]
    ).negative_weights
    positive = torch.tensor([3, 4, 5, 0, 1, 2])
    rows = torch.arange(6)
    assert torch.equal(mask[rows, rows], torch.zeros(6))
    assert torch.equal(mask[rows, positive], torch.zeros(6))
    assert mask[0, 1].item() == 0.0 and mask[0, 4].item() == 0.0
    assert mask[0, 2].item() == 1.0 and mask[0, 5].item() == 1.0


def test_zero_valid_negatives_fail_informatively() -> None:
    with pytest.raises(FalseNegativeMaskDegenerateError, match="degenerate anchors"):
        _route()(_two_view(torch.randn(2, 5), torch.randn(2, 5)), positions=[7, 7])


def test_missing_z_delta_pair_fails_informatively() -> None:
    output = _two_view(torch.randn(2, 5), torch.randn(2, 5))
    object.__setattr__(output, "view1", SimpleNamespace(h_pair_delta=torch.randn(2, 5)))
    with pytest.raises(ValueError, match="provide z_delta_pair"):
        _route()(output, positions=[1, 2])


def test_nt_xent_receives_only_projected_instance_pair_embeddings() -> None:
    spy = _SpyNTXent(temperature=0.2)
    route = _route(spy).eval()
    source1 = torch.randn(3, 5)
    source2 = torch.randn(3, 5)
    output = route(_two_view(source1, source2), positions=[1, 2, 3])
    assert spy.received is not None
    assert spy.received[0] is output.z_instance_pair_view1
    assert spy.received[1] is output.z_instance_pair_view2
    assert spy.received[0].shape[-1] == 3
    assert source1.shape[-1] == 5


def test_matching_rows_are_the_positive_pairs() -> None:
    criterion = NTXentLoss(temperature=0.1)
    embeddings = torch.eye(4)
    aligned = criterion(embeddings, embeddings).loss
    permuted = criterion(embeddings, embeddings[[1, 2, 3, 0]]).loss
    assert aligned < permuted
