"""Registered-office attribution, never generic top-of-page company guessing."""

import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import Token, _registered_office_supplier, from_tokens
from invoice_grounding import RECOVERED, reconcile
from ocr_consensus import build_consensus


def tokens():
    rows = [
        (0, [(0, "Actual"), (50, "Studio"), (500, "FACTURE")]),
        (20, [(0, "22"), (25, "Avenue"), (85, "Voltaire")]),
        (40, [(0, "SIRET"), (60, "73282932000074")]),
        (60, [(0, "Client"), (45, ":")]),
        (80, [(0, "Customer"), (80, "Company")]),
        (110, [(0, "Description"), (180, "Quantity"), (270, "Unit"), (310, "price"), (420, "Amount")]),
        (130, [(0, "Service"), (180, "2"), (270, "10"), (420, "20")]),
        (150, [(0, "Subtotal:"), (90, "20")]),
        (170, [(0, "Invoice"), (50, "total:"), (90, "20")]),
        (200, [(0, "Si\u00e8ge"), (55, "social"), (300, "Coordonn\u00e9es"), (500, "Banque")]),
        (220, [(0, "Actual"), (50, "Studio"), (300, "Contact"), (500, "Bank")]),
        (240, [(0, "22"), (25, "Avenue"), (85, "Voltaire"), (300, "Telephone")]),
    ]
    return [
        Token(text, 0, (x, y, x + len(text) * 5, y + 10), 95)
        for y, cells in rows for x, text in cells
    ]


def office_from(source):
    return _registered_office_supplier(from_tokens(source, locale="fr-FR").lines)


def test_registered_office_needs_repeated_issuer_name_registration_and_column_attribution():
    evidence = from_tokens(tokens(), locale="fr-FR")
    assert evidence.supplier == "Actual Studio"
    assert evidence.supplier_source.text == "Actual Studio"
    assert [token.text for token in evidence.supplier_source.tokens] == ["Actual", "Studio"]
    assert not evidence.supplier_is_explicit
    result = build_consensus({
        "gray": [evidence], "contrast": [copy.deepcopy(evidence)],
    })
    grounded = reconcile(result, [{"supplier_name": "Customer Company", "line_items": []}])
    assert grounded.payload["supplier_name"] == "Actual Studio"
    assert grounded.statuses["supplier_name"] == RECOVERED


@pytest.mark.parametrize("mutation", [
    "no-registration", "no-client-boundary", "different-header-name",
    "customer-name-in-office", "no-address", "low-name-confidence",
    "low-label-confidence", "distant-office-name", "other-page",
])
def test_registered_office_unsupported_attribution_is_refused(mutation):
    source = tokens()
    if mutation == "no-registration":
        source = [token for token in source if token.box[1] != 40]
    elif mutation == "no-client-boundary":
        source = [token for token in source if token.box[1] != 60]
    elif mutation == "different-header-name":
        source = [
            replace(token, text="Other") if token.text == "Actual" and token.box[1] == 0 else token
            for token in source
        ]
    elif mutation == "customer-name-in-office":
        source = [
            replace(token, text="Customer" if token.text == "Actual" else "Company")
            if token.box[1] == 220 and token.text in {"Actual", "Studio"} else token
            for token in source
        ]
    elif mutation == "no-address":
        source = [token for token in source if token.box[1] != 240]
    elif mutation == "low-name-confidence":
        source = [
            replace(token, confidence=59) if token.box[1] == 220 and token.text == "Studio" else token
            for token in source
        ]
    elif mutation == "low-label-confidence":
        source = [
            replace(token, confidence=39) if token.box[1] == 200 and token.text == "social" else token
            for token in source
        ]
    elif mutation == "distant-office-name":
        source = [
            replace(token, box=(token.box[0], token.box[1] + 100, token.box[2], token.box[3] + 100))
            if token.box[1] in {220, 240} else token for token in source
        ]
    else:
        source = [replace(token, page=1) if token.box[1] >= 220 else token for token in source]
    assert office_from(source) is None


def test_registered_office_one_family_cannot_confirm_supplier():
    evidence = from_tokens(tokens(), locale="fr-FR")
    unknown = from_tokens([
        token for token in tokens() if token.box[1] < 200
    ], locale="fr-FR")
    unknown.supplier = None
    result = build_consensus({
        "gray": [evidence, copy.deepcopy(evidence)], "contrast": [unknown],
    })
    assert result.supplier is None
    assert reconcile(result, [{"supplier_name": "Actual Studio"}]).payload["supplier_name"] is None


def test_conflicting_office_names_are_not_resolved_by_model_preference():
    first = from_tokens(tokens(), locale="fr-FR")
    second = from_tokens([
        replace(token, text="Different") if token.text == "Actual" else token for token in tokens()
    ], locale="fr-FR")
    result = build_consensus({"gray": [first], "contrast": [second]})
    assert result.supplier is None
    assert reconcile(result, [{"supplier_name": "Actual Studio"}]).payload["supplier_name"] is None


@pytest.mark.parametrize("role, confidence", [
    ("heading", 30), ("name", 57), ("registration", 32),
])
def test_observed_staging_confidence_rejections_remain_fail_closed(role, confidence):
    """Synthetic geometry with exact observed confidence values, not a new image fixture."""
    source = []
    for token in tokens():
        affected = (
            role == "heading" and token.box[1] == 200 and token.text == "Si\u00e8ge"
            or role == "name" and token.box[1] == 220 and token.text == "Studio"
            or role == "registration" and token.box[1] == 40 and token.text.isdigit()
        )
        source.append(replace(token, confidence=confidence) if affected else token)
    assert office_from(source) is None


def test_single_scale_family_segmentation_cannot_rescue_rejected_primary_families():
    def reading(role=None, confidence=95):
        result = []
        for token in tokens():
            affected = (
                role == "heading" and token.box[1] == 200 and token.text == "Si\u00e8ge"
                or role == "name" and token.box[1] == 220 and token.text == "Studio"
                or role == "registration" and token.box[1] == 40 and token.text.isdigit()
            )
            result.append(replace(token, confidence=confidence) if affected else token)
        return from_tokens(result, locale="fr-FR")

    groups = {
        "deskew": [reading("heading", 30)],
        "clahe": [reading("name", 57)],
        "scale3": [reading(), reading("registration", 32)],
    }
    assert groups["scale3"][0].supplier == "Actual Studio"
    assert groups["scale3"][1].supplier is None
    consensus = build_consensus(groups)
    assert consensus.supplier is None
    assert reconcile(consensus, [{"supplier_name": "Actual Studio"}]).payload["supplier_name"] is None
