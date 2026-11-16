"""PR176 — an unqualified printed grand total read as the tax-inclusive amount by a rate-only gross-up.

Three of the frozen layouts print a grand total with no HT/TTC qualifier next to a percentage and no tax
*amount*, so every existing document-total route abstains: the additive routes need a `Tax` (a rate plus an
amount), and the equality route needs the line sum to be the candidate itself. The comparison added here uses
the printed percentage as corroboration only — the exported value is always the printed total, never a
computed number — and it is measured to leave the six negative shapes below abstaining.

Fixtures are synthetic and use no OCR, so a fresh clone runs them.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text  # noqa: E402
from invoice_grounding import reconcile  # noqa: E402

TABLE = "Description | Quantity | Unit Price | Amount\n"
FRENCH_TABLE = "Description | Quantité | Prix unitaire HT | Montant HT\n"


def ground(text: str, locale: str):
    evidence = from_text(text, locale=locale)
    return evidence, reconcile(evidence, [{}])


# Shape of `layout-c-retail-anon`: a labelled net total, a rate and one unqualified grand total.
RETAIL = (
    "INVOICE RC-1\nVendor: Acme\n" + TABLE +
    "Design work | 1 | 61.97 | 61.97\n"
    "Sub Total 61.97\nTax Rate 5.00%\nTotal $65.06\n"
)
# Shape of `layout-d-service-anon`: no labelled net total at all, only rows, a rate and a grand total.
SERVICE = (
    "INVOICE SV-1\nVendor: Acme\n" + TABLE +
    "Consulting | 1 | 2150.00 | 2150.00\n"
    "Tax 3 %\nTotal USD 2,214.50\n"
)
# Shape of `layout-b-consulting-anon`: the unqualified total is an unrelated printed amount.
LABOUR = (
    "FACTURE LB-1\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Prestation | 1 | 1 005,00 | 1 005,00\n"
    "TOTAL HORS TAXE : 1 005,00 EUR\nTAUX FISCAL/TVA : 10%\nTOTAL TRAVAIL : 380,00 EUR\n"
)
MULTI_RATE = (
    "FACTURE MR-1\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 EUR\nTaux TVA 5 %\nTaux TVA 10 %\nTotal 210,00 EUR\n"
)
NO_TOTAL_PRINTED = (
    "FACTURE NT-1\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Étude | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 EUR\nTaux TVA 5 %\n"
)
PERCENTAGE_ELSEWHERE = (
    "FACTURE PE-1\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Ouvrage | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 EUR\nRemise 10 %\nTotal 220,00 EUR\n"
)
TAX_RATE_WITH_AMOUNT = (
    "FACTURE TA-1\nFournisseur: Acme SARL\n" + FRENCH_TABLE +
    "Prestation | 1 | 200,00 | 200,00\n"
    "Total HT : 200,00 EUR\nTVA 40,00 EUR (20 %)\nTotal 240,00 EUR\n"
)
# Same print, minus the table: the additive route has no operands, so only the gross-up could answer, and it
# deliberately stays out of documents that already show a tax amount.
TAX_AMOUNT_WITHOUT_ROWS = (
    "FACTURE TB-1\nFournisseur: Acme SARL\n"
    "Total HT : 200,00 EUR\nTVA 40,00 EUR (20 %)\nTotal 240,00 EUR\n"
)


@pytest.mark.parametrize(
    "text,locale,expected",
    [
        (RETAIL, "en-US", "65.06"),
        (SERVICE, "en-US", "2214.5"),
        (LABOUR, "fr-FR", None),
        (MULTI_RATE, "fr-FR", None),
        (NO_TOTAL_PRINTED, "fr-FR", None),
        (PERCENTAGE_ELSEWHERE, "fr-FR", None),
    ],
)
def test_the_printed_total_only_survives_when_the_rate_reproduces_it(text, locale, expected):
    _, result = ground(text, locale)
    assert result.payload["total_ttc"] == (float(expected) if expected is not None else None)


def test_a_rounded_gross_up_exports_the_printed_cent_and_not_the_computed_one():
    """61.97 × 1.05 = 65.0685, the document prints 65.06; the print is exported, the arithmetic is not."""
    _, result = ground(RETAIL, "en-US")
    assert result.payload["total_ht"] == 61.97
    assert Decimal(str(result.payload["total_ttc"])) == Decimal("65.06")


def test_the_gross_up_never_invents_the_missing_net_total():
    evidence, result = ground(SERVICE, "en-US")
    assert evidence.totals.get("total_ht", []) == []
    assert result.payload["total_ht"] is None
    assert result.payload["total_ttc"] == 2214.5


def test_a_grand_total_missing_from_the_document_is_never_computed_in():
    evidence, result = ground(NO_TOTAL_PRINTED, "fr-FR")
    assert evidence.totals.get("total_candidate", []) == []
    assert result.payload["total_ttc"] is None


def test_a_percentage_on_a_line_that_does_not_mention_tax_is_not_a_rate():
    """`Remise 10 %` grossed up would reach the printed 220,00 — a discount is not a tax rate."""
    _, result = ground(PERCENTAGE_ELSEWHERE, "fr-FR")
    assert result.payload["total_ttc"] is None


def test_a_printed_tax_amount_keeps_the_existing_additive_route_in_charge():
    _, result = ground(TAX_RATE_WITH_AMOUNT, "fr-FR")
    assert result.payload["total_ht"] == 200.0
    assert result.payload["total_ttc"] == 240.0


def test_the_gross_up_does_not_reach_documents_that_already_show_a_tax_amount():
    evidence, result = ground(TAX_AMOUNT_WITHOUT_ROWS, "fr-FR")
    assert [tax.amount.value for tax in evidence.taxes] == [Decimal("40.00")]
    assert evidence.rows == []
    assert result.payload["total_ttc"] is None


def test_rows_whose_print_order_is_unproven_do_not_supply_the_base():
    evidence, result = ground(SERVICE, "en-US")
    assert result.payload["total_ttc"] == 2214.5
    evidence.rows_ordinal_proof = False
    blocked = reconcile(evidence, [{}])
    assert blocked.payload["total_ttc"] is None
