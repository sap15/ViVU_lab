from __future__ import annotations

import pytest

from gnn_siamese.data.pairing import (
    AmbiguousWTCompanionError,
    MissingWTCompanionError,
    PairingKey,
    SignatureParseError,
    build_wt_index,
    pair_mutants_with_wt,
    parse_variant_signature,
    resolve_wt_companion,
)


def test_pairing_resolves_by_signature_not_file_order() -> None:
    mutants = [
        {"variant_id": "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W"},
        {"variant_id": "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"},
    ]
    wt_companions = [
        {"variant_id": "residue-srv:A:100:Glycine->Glycine:PKP2_WT"},
        {"variant_id": "residue-srv:A:563:Cysteine->Cysteine:PKP2_WT"},
    ]

    paired = pair_mutants_with_wt(mutants, wt_companions)

    assert [item["wt_companion_id"] for item in paired] == [
        "residue-srv:A:563:Cysteine->Cysteine:PKP2_WT",
        "residue-srv:A:100:Glycine->Glycine:PKP2_WT",
    ]
    assert paired[0]["pairing_signature"] == PairingKey(chain_id="A", position=563, wt_aa="C")
    assert paired[1]["pairing_signature"] == PairingKey(chain_id="A", position=100, wt_aa="G")


def test_pairing_rejects_missing_wt() -> None:
    mutants = [{"variant_id": "residue-srv:A:250:Alanine->Valine:pos_250_A_V"}]
    wt_companions = [{"variant_id": "residue-srv:A:251:Alanine->Alanine:PKP2_WT"}]

    with pytest.raises(MissingWTCompanionError, match="No WT companion found for mutant"):
        pair_mutants_with_wt(mutants, wt_companions)


def test_pairing_rejects_ambiguous_wt() -> None:
    wt_companions = [
        {"variant_id": "residue-srv:A:100:Glycine->Glycine:PKP2_WT"},
        {
            "variant_id": "duplicate-wt-record",
            "chain_id": "A",
            "position": 100,
            "wt_aa": "G",
            "mut_aa": "G",
        },
    ]

    with pytest.raises(AmbiguousWTCompanionError, match="Ambiguous WT companion signature"):
        build_wt_index(wt_companions)


def test_pairing_preserves_variant_metadata() -> None:
    mutants = [
        {
            "variant_id": "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
            "source_path": "proc_483p.hdf5",
        }
    ]
    wt_companions = [
        {
            "variant_id": "residue-srv:A:100:Glycine->Glycine:PKP2_WT",
            "source_path": "wt_companion.hdf5",
        }
    ]

    paired = pair_mutants_with_wt(mutants, wt_companions)

    assert paired == [
        {
            "variant_id": "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
            "graph_id": "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D",
            "chain_id": "A",
            "position": 100,
            "wt_aa": "G",
            "mut_aa": "D",
            "wt_companion_id": "residue-srv:A:100:Glycine->Glycine:PKP2_WT",
            "pairing_signature": PairingKey(chain_id="A", position=100, wt_aa="G"),
            "mutant_signature": parse_variant_signature(mutants[0]),
            "wt_signature": parse_variant_signature(wt_companions[0]),
        }
    ]


def test_pairing_rejects_unparseable_position_or_wt_aa() -> None:
    wt_companions = [{"variant_id": "residue-srv:A:100:Glycine->Glycine:PKP2_WT"}]

    with pytest.raises(SignatureParseError, match="Cannot parse position"):
        resolve_wt_companion(
            {
                "variant_id": "mutant-without-parseable-key",
                "chain_id": "A",
                "position": "not-an-int",
                "wt_aa": "G",
                "mut_aa": "D",
            },
            build_wt_index(wt_companions),
        )


def test_parse_variant_signature_accepts_full_amino_acid_names() -> None:
    signature = parse_variant_signature(
        {"variant_id": "residue-srv:A:563:Cysteine->Tryptophan:pos_563_C_W"}
    )
    assert signature.wt_aa == "C"
    assert signature.mut_aa == "W"

    assert parse_variant_signature(
        {"variant_id": "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"}
    ).wt_aa == "G"
    assert parse_variant_signature(
        {"variant_id": "residue-srv:A:100:Glycine->Aspartate:pos_100_G_D"}
    ).mut_aa == "D"

    manual = parse_variant_signature(
        {
            "variant_id": "manual-variant",
            "chain_id": "A",
            "position": 321,
            "wt_aa": "Phenylalanine",
            "mut_aa": "Cysteine",
        }
    )
    assert manual.wt_aa == "F"
    assert manual.mut_aa == "C"

    with pytest.raises(SignatureParseError, match="Cannot parse wt_aa"):
        parse_variant_signature(
            {
                "variant_id": "mutant-without-parseable-key",
                "chain_id": "A",
                "position": 100,
                "wt_aa": "",
                "mut_aa": "D",
            }
        )
