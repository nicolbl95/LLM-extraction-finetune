"""Compact totals must share documentary currency validation, not numeric locale."""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from app import main
from invoice_evidence import from_text
from invoice_grounding import reconcile


@pytest.mark.parametrize(("printed", "locale", "expected"), [
    ("", "fr-FR", None),
    ("$", "fr-FR", None),
    ("$", "en-US", None),
    ("EUR", "en-US", "EUR"),
    ("GBP", "fr-FR", "GBP"),
    ("USD", "fr-FR", "USD"),
    ("CAD", "fr-FR", "CAD"),
    ("\u20ac", "fr-FR", "EUR"),
    ("\u00a3", "fr-FR", "GBP"),
    ("EUR USD", "fr-FR", None),
    ("\u20ac \u00a3", "fr-FR", None),
    ("\u00a3 $", "en-GB", None),
])
def test_compact_total_uses_only_the_documented_currency(printed, locale, expected):
    evidence = from_text(
        "Facture\nDescription | Quantit\u00e9 | Prix unitaire | Montant\n"
        f"Service | 2 | 10 | 20\nTotal g\u00e9n\u00e9ral: 20.00 {printed}",
        locale=locale,
    )
    result = reconcile(evidence, [{}])
    assert result.payload["total_ttc"] == 20
    assert result.payload["total_ht"] is None
    diagnostics = {
        "evidence": evidence, "evidence_available": True,
        "field_statuses": result.statuses, "validation": result.validation.export(),
    }
    original = copy.deepcopy((result.payload, diagnostics["validation"], result.statuses))
    summary = main._compact_validation_summary(result.payload, diagnostics, "")
    suffix = f" {expected}" if expected else ""
    assert (
        f"{main._summary_code(result.payload['total_ttc'])}{suffix} sans pr\u00e9ciser"
    ) in summary
    if expected:
        assert any(
            message.startswith(f"Devise lue : {expected}")
            for message in diagnostics["validation"]["passed"]
        )
    else:
        assert any(
            "aucune devise unique" in message
            for message in diagnostics["validation"]["not_verified"]
        )
    assert (result.payload, diagnostics["validation"], result.statuses) == original


def test_canadian_french_total_does_not_inherit_eur_or_invent_cad():
    evidence = from_text(
        "Facture\nLongueuil (Qu\u00e9bec)\n"
        "Description | Quantit\u00e9 | Prix unitaire | Montant\n"
        "Livre | 2 | 10,00 $ | 20,00 $\n"
        "Frais de livraison: 12,50 $\nTotal g\u00e9n\u00e9ral: 32,50 $",
        locale="fr-FR",
    )
    result = reconcile(evidence, [{}])
    assert len(result.payload["line_items"]) == 1
    assert result.payload["total_ttc"] == 32.5
    summary = main._compact_validation_summary(result.payload, {
        "evidence": evidence, "evidence_available": True,
        "field_statuses": result.statuses, "validation": result.validation.export(),
    }, "")
    assert "`32.5` sans pr\u00e9ciser" in summary
    assert not any(code in summary for code in ("EUR", "GBP", "USD", "CAD"))
    assert all(item["quantity"] != 1 for item in result.payload["line_items"])
