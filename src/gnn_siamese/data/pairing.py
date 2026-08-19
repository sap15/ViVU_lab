"""Mutant-WT pairing helpers based on stable variant signatures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gnn_siamese.data.validation import parse_case_key

BIOLOGICAL_VARIANT = "biological_variant"
NATIVE_WT_CONTROL = "native_wt_control"

_AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "STOP": "*",
    "TER": "*",
}

_AA_FULL_TO_AA1 = {
    "ALANINE": "A",
    "ARGININE": "R",
    "ASPARAGINE": "N",
    "ASPARTATE": "D",
    "ASPARTIC ACID": "D",
    "CYSTEINE": "C",
    "GLUTAMATE": "E",
    "GLUTAMIC ACID": "E",
    "GLUTAMINE": "Q",
    "GLYCINE": "G",
    "HISTIDINE": "H",
    "ISOLEUCINE": "I",
    "LEUCINE": "L",
    "LYSINE": "K",
    "METHIONINE": "M",
    "PHENYLALANINE": "F",
    "PROLINE": "P",
    "SERINE": "S",
    "THREONINE": "T",
    "TRYPTOPHAN": "W",
    "TYROSINE": "Y",
    "VALINE": "V",
    "STOP": "*",
    "TER": "*",
}


class PairingError(ValueError):
    """Base error for mutant-WT pairing failures."""


class SignatureParseError(PairingError):
    """Raised when a variant signature cannot be parsed."""


class IncompleteSignatureError(PairingError):
    """Raised when a parsed signature lacks required pairing fields."""


class MissingWTCompanionError(PairingError):
    """Raised when no WT companion exists for a mutant signature."""


class AmbiguousWTCompanionError(PairingError):
    """Raised when more than one WT companion matches the same signature."""


@dataclass(frozen=True)
class PairingKey:
    """Stable key used to resolve a mutant against its WT companion."""

    chain_id: str | None
    position: int
    wt_aa: str


@dataclass(frozen=True)
class VariantSignature:
    """Parsed variant metadata used for pairing and traceability."""

    variant_id: str
    graph_id: str
    chain_id: str | None
    position: int
    wt_aa: str
    mut_aa: str
    wt_aa_full: str | None
    mut_aa_full: str | None
    source_path: str | None = None

    @property
    def pairing_key(self) -> PairingKey:
        return PairingKey(
            chain_id=self.chain_id,
            position=self.position,
            wt_aa=self.wt_aa,
        )


def classify_variant_record(record: str | dict[str, Any]) -> str:
    """Classify a mutant-input record without conflating WT-like biology with controls.

    The native WT control is created explicitly by the dataset generator's
    ``--include-wt-graph`` route.  Its stable identity is the final
    ``PKP2_WT`` token together with a WT->WT amino-acid signature.  Amino-acid
    equality alone is deliberately insufficient because valid truncation
    records may use WT-like query amino acids during graph construction.
    """

    signature = parse_variant_signature(record)
    identity = signature.graph_id.rsplit(":", 1)[-1]
    variant_identity = signature.variant_id.rsplit(":", 1)[-1]
    if (
        identity == "PKP2_WT"
        and variant_identity == "PKP2_WT"
        and signature.wt_aa == signature.mut_aa
    ):
        return NATIVE_WT_CONTROL
    return BIOLOGICAL_VARIANT


def _coerce_amino_acid_code(value: Any, *, field_name: str) -> str:
    if value is None:
        raise SignatureParseError(f"Cannot parse {field_name}: value is missing.")

    text = str(value).strip()
    if not text:
        raise SignatureParseError(f"Cannot parse {field_name}: value is empty.")

    upper = text.upper()
    if len(upper) == 1 and upper.isalpha():
        return upper
    if upper in {"*", "X"}:
        return upper
    if upper in _AA_FULL_TO_AA1:
        return _AA_FULL_TO_AA1[upper]
    if upper in _AA3_TO_AA1:
        return _AA3_TO_AA1[upper]

    raise SignatureParseError(f"Cannot parse {field_name}: unsupported amino acid value {value!r}.")


def _normalize_variant_record(record: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, str):
        return {"variant_id": record}
    return dict(record)


def parse_variant_signature(record: str | dict[str, Any]) -> VariantSignature:
    """Parse a variant signature from a case key or metadata mapping."""

    payload = _normalize_variant_record(record)
    variant_id = payload.get("variant_id") or payload.get("graph_id") or payload.get("case_key")
    if not variant_id:
        raise SignatureParseError("Cannot parse variant signature: missing variant_id/graph_id/case_key.")

    parsed = parse_case_key(str(variant_id))
    if parsed.get("valid"):
        chain_id = payload.get("chain_id", parsed.get("chain_id"))
        position = payload.get("position", parsed.get("position"))
        wt_aa_full = payload.get("wt_aa_full", parsed.get("wt_aa_full"))
        mut_aa_full = payload.get("mut_aa_full", parsed.get("mut_aa_full"))
        wt_aa = payload.get("wt_aa") or _coerce_amino_acid_code(wt_aa_full, field_name="wt_aa")
        mut_aa = payload.get("mut_aa") or _coerce_amino_acid_code(mut_aa_full, field_name="mut_aa")
    else:
        chain_id = payload.get("chain_id")
        position = payload.get("position")
        wt_aa_full = payload.get("wt_aa_full")
        mut_aa_full = payload.get("mut_aa_full")
        wt_aa = _coerce_amino_acid_code(payload.get("wt_aa"), field_name="wt_aa")
        mut_aa = _coerce_amino_acid_code(payload.get("mut_aa"), field_name="mut_aa")

    try:
        parsed_position = int(position)
    except (TypeError, ValueError) as exc:
        raise SignatureParseError(
            f"Cannot parse position for variant {variant_id!r}: {position!r}."
        ) from exc

    return VariantSignature(
        variant_id=str(variant_id),
        graph_id=str(payload.get("graph_id") or variant_id),
        chain_id=None if chain_id in (None, "") else str(chain_id),
        position=parsed_position,
        wt_aa=str(wt_aa),
        mut_aa=str(mut_aa),
        wt_aa_full=None if wt_aa_full in (None, "") else str(wt_aa_full),
        mut_aa_full=None if mut_aa_full in (None, "") else str(mut_aa_full),
        source_path=None if payload.get("source_path") in (None, "") else str(payload.get("source_path")),
    )


def _require_complete_pairing_signature(signature: VariantSignature, *, role: str) -> None:
    missing: list[str] = []
    if signature.position is None:
        missing.append("position")
    if not signature.wt_aa:
        missing.append("wt_aa")
    if missing:
        raise IncompleteSignatureError(
            f"{role} signature is incomplete for {signature.variant_id!r}: missing {', '.join(missing)}."
        )


def build_wt_index(wt_records: list[str | dict[str, Any]]) -> dict[PairingKey, VariantSignature]:
    """Build a stable WT index keyed by chain, position and WT amino acid."""

    wt_index: dict[PairingKey, VariantSignature] = {}
    for record in wt_records:
        signature = parse_variant_signature(record)
        _require_complete_pairing_signature(signature, role="WT companion")
        key = signature.pairing_key
        existing = wt_index.get(key)
        if existing is not None:
            raise AmbiguousWTCompanionError(
                "Ambiguous WT companion signature "
                f"{key!r}: both {existing.variant_id!r} and {signature.variant_id!r} match."
            )
        wt_index[key] = signature
    return wt_index


def resolve_wt_companion(
    mutant_record: str | dict[str, Any],
    wt_index: dict[PairingKey, VariantSignature],
) -> VariantSignature:
    """Resolve the WT companion for a mutant record using a stable signature."""

    mutant_signature = parse_variant_signature(mutant_record)
    _require_complete_pairing_signature(mutant_signature, role="Mutant")
    key = mutant_signature.pairing_key
    wt_signature = wt_index.get(key)
    if wt_signature is None:
        raise MissingWTCompanionError(
            "No WT companion found for mutant "
            f"{mutant_signature.variant_id!r} with signature {key!r}."
        )
    return wt_signature


def pair_mutants_with_wt(
    mutant_records: list[str | dict[str, Any]],
    wt_records: list[str | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pair each mutant with exactly one WT companion and preserve parsed metadata."""

    wt_index = build_wt_index(wt_records)
    paired: list[dict[str, Any]] = []
    for record in mutant_records:
        mutant_signature = parse_variant_signature(record)
        _require_complete_pairing_signature(mutant_signature, role="Mutant")
        wt_signature = resolve_wt_companion(record, wt_index)
        paired.append(
            {
                "variant_id": mutant_signature.variant_id,
                "graph_id": mutant_signature.graph_id,
                "chain_id": mutant_signature.chain_id,
                "position": mutant_signature.position,
                "wt_aa": mutant_signature.wt_aa,
                "mut_aa": mutant_signature.mut_aa,
                "wt_companion_id": wt_signature.variant_id,
                "pairing_signature": mutant_signature.pairing_key,
                "mutant_signature": mutant_signature,
                "wt_signature": wt_signature,
            }
        )
    return paired
