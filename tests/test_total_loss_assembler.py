from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gnn_siamese.losses import DeltaLoss, NTXentLoss, RelativeWTLoss
from gnn_siamese.training import TotalLossAssembler, TotalLossConfig, TotalLossOutput


def _sample_inputs() -> dict[str, torch.Tensor]:
    return {
        "z1": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True),
        "z2": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32, requires_grad=True),
        "h_mut": torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32, requires_grad=True),
        "h_wt": torch.zeros((2, 2), dtype=torch.float32, requires_grad=True),
        "z_delta": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, requires_grad=True),
    }


def test_total_loss_baseline_matches_nt_xent_and_tracks_inactive_components() -> None:
    inputs = _sample_inputs()
    nt_xent = NTXentLoss(temperature=0.2)
    baseline = nt_xent(inputs["z1"], inputs["z2"]).loss
    assembler = TotalLossAssembler(
        nt_xent=nt_xent,
        relative_wt=RelativeWTLoss(mode="none"),
        delta=DeltaLoss(mode="none"),
        nt_xent_weight=1.0,
        relative_wt_weight=1.0,
        delta_weight=1.0,
    )

    output = assembler(**inputs)

    assert isinstance(output, TotalLossOutput)
    assert output.loss.item() == pytest.approx(baseline.item(), abs=1.0e-6)
    assert output.active_components == ["nt_xent"]
    assert output.inactive_components == ["relative_wt", "delta"]
    assert output.skipped_components == []
    assert output.metrics["loss_nt_xent"].item() == pytest.approx(baseline.item(), abs=1.0e-6)
    assert output.metrics["loss_relative_wt"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.metrics["loss_delta"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.metrics["weighted_loss_nt_xent"].item() == pytest.approx(baseline.item(), abs=1.0e-6)
    assert output.metrics["weighted_loss_relative_wt"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.metrics["weighted_loss_delta"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.metrics["loss_total"].item() == pytest.approx(output.metrics["weighted_loss_nt_xent"].item(), abs=1.0e-6)
    assert output.audit_flags["component_status"]["relative_wt"] == "inactive_mode_none"
    assert output.audit_flags["component_status"]["delta"] == "inactive_mode_none"


def test_total_loss_combines_weighted_components_exactly() -> None:
    inputs = _sample_inputs()
    w1, w2, w3 = 1.25, 0.5, 2.0
    assembler = TotalLossAssembler(
        nt_xent=NTXentLoss(temperature=0.2),
        relative_wt=RelativeWTLoss(mode="margin", margin=0.5),
        delta=DeltaLoss(mode="variance", gamma=1.0),
        nt_xent_weight=w1,
        relative_wt_weight=w2,
        delta_weight=w3,
    )

    output = assembler(**inputs)
    expected = (
        w1 * output.components["nt_xent"]
        + w2 * output.components["relative_wt"]
        + w3 * output.components["delta"]
    )

    assert output.loss.item() == pytest.approx(expected.item(), abs=1.0e-6)
    assert output.active_components == ["nt_xent", "relative_wt", "delta"]
    assert output.metrics["weighted_loss_nt_xent"].item() == pytest.approx((w1 * output.components["nt_xent"]).item(), abs=1.0e-6)
    assert output.metrics["weighted_loss_relative_wt"].item() == pytest.approx((w2 * output.components["relative_wt"]).item(), abs=1.0e-6)
    assert output.metrics["weighted_loss_delta"].item() == pytest.approx((w3 * output.components["delta"]).item(), abs=1.0e-6)
    assert output.metrics["loss_total"].item() == pytest.approx(expected.item(), abs=1.0e-6)


def test_zero_weight_component_is_skipped_and_does_not_contribute() -> None:
    inputs = _sample_inputs()
    assembler = TotalLossAssembler(
        nt_xent=NTXentLoss(temperature=0.2),
        relative_wt=RelativeWTLoss(mode="margin", margin=0.5),
        delta=DeltaLoss(mode="variance", gamma=1.0),
        nt_xent_weight=1.0,
        relative_wt_weight=0.0,
        delta_weight=1.0,
    )

    output = assembler(**inputs)

    assert "relative_wt" in output.skipped_components
    assert output.audit_flags["component_status"]["relative_wt"] == "skipped_weight_zero"
    assert output.metrics["weighted_components"]["relative_wt"] == pytest.approx(0.0, abs=1.0e-8)
    assert output.metrics["weighted_loss_relative_wt"].item() == pytest.approx(0.0, abs=1.0e-8)


def test_mode_none_component_is_inactive_and_does_not_contribute() -> None:
    inputs = _sample_inputs()
    assembler = TotalLossAssembler(
        nt_xent=NTXentLoss(temperature=0.2),
        relative_wt=RelativeWTLoss(mode="none"),
        delta=DeltaLoss(mode="variance", gamma=1.0),
        nt_xent_weight=1.0,
        relative_wt_weight=1.0,
        delta_weight=1.0,
    )

    output = assembler(**inputs)

    assert "relative_wt" in output.inactive_components
    assert output.audit_flags["component_status"]["relative_wt"] == "inactive_mode_none"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"z2": None}, "nt_xent_weight > 0 requires both z1 and z2"),
        ({"h_mut": None}, "relative_wt_weight > 0 requires both h_mut and h_wt"),
        ({"z_delta": None}, "delta_weight > 0 requires z_delta"),
    ],
)
def test_total_loss_errors_explicitly_when_required_inputs_are_missing(
    kwargs: dict[str, torch.Tensor | None],
    message: str,
) -> None:
    inputs = _sample_inputs()
    inputs.update(kwargs)
    assembler = TotalLossAssembler(
        nt_xent=NTXentLoss(temperature=0.2),
        relative_wt=RelativeWTLoss(mode="margin", margin=0.5),
        delta=DeltaLoss(mode="variance", gamma=1.0),
        nt_xent_weight=1.0,
        relative_wt_weight=1.0,
        delta_weight=1.0,
    )

    with pytest.raises(ValueError, match=message):
        assembler(**inputs)


def test_total_loss_returns_zero_with_explicit_audit_when_all_components_are_inactive() -> None:
    assembler = TotalLossAssembler.from_config(
        TotalLossConfig(
            nt_xent_weight=0.0,
            relative_wt_weight=0.0,
            delta_weight=0.0,
        )
    )

    output = assembler()

    assert output.loss.item() == pytest.approx(0.0, abs=1.0e-8)
    assert output.active_components == []
    assert output.audit_flags["all_components_inactive"] is True
    assert set(output.skipped_components) == {"nt_xent", "relative_wt", "delta"}


def test_total_loss_public_api_exports_training_symbols() -> None:
    from gnn_siamese.training import __all__ as training_public_api

    assert "TotalLossAssembler" in training_public_api
    assert "TotalLossConfig" in training_public_api
    assert "TotalLossOutput" in training_public_api
