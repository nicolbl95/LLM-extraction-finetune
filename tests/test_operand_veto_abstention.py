"""PR87 operand-veto abstention: the guard signal and the exported amount, pinned deterministically.

Offline and synthetic — no Tesseract, no model, no private fixture. The seven monetary situations
come from PR86; each is asserted twice, once on the explicit `TableRow.amount_operand_veto` signal
the consensus product guard sets, and once on the value the production export actually produces.
The signal must never be inferred from empty cells: a row whose operands were merely absent keeps
its printed amount unless printed document totals settle the interpretation, which is PR123's
absent-operand abstention covered at the end of this module.
"""

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

HEADER_Y = 100
COLUMN_X = (60, 200, 380, 520, 660)


def token(text, x, y, width=None):
    return Token(text, 0, (x, y, x + (width or 12 + 5 * len(text)), y + 14), 95.0)


def table(labels, rows, xs=COLUMN_X):
    tokens = [token(label, x, HEADER_Y) for label, x in zip(labels, xs) if label]
    for index, row in enumerate(rows):
        y = HEADER_Y + 30 + index * 22
        tokens += [token(value, x, y, width) for value, x, width in
                   zip(row, xs, (None, 14, 44, 24, 48)) if value]
    return tokens


def run(labels, rows, xs=COLUMN_X):
    evidence = from_tokens(table(labels, rows, xs), literal_columns=False)
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    grounded = reconcile(consensus, [{}])
    return evidence, consensus, grounded


UNQUALIFIED = ["Description", "Quantité", "Prix unitaire", "", "Montant"]
WITH_TAX = ["Description", "Quantité", "Prix unitaire", "TVA", "Montant"]
RECOGNIZED_TAX = ["Description", "Quantité", "Prix unitaire", "% TVA", "Montant"]


def exported(grounded, index=0):
    return grounded.payload["line_items"][index]


def test_explicit_ht_amount_is_preserved_and_not_vetoed():
    evidence, consensus, grounded = run(
        ["Description", "Quantité", "Prix unitaire", "", "Montant HT"],
        [["Prestation", "2", "35,00", "", "70,00"]])
    assert len(evidence.rows) == 1
    assert consensus.rows[0].amount_operand_veto is False
    assert exported(grounded)["total_ht"] == 70.0


def test_explicit_ttc_column_still_builds_no_row():
    """The pre-existing safe behaviour: a TTC label collides with the HT role, so no header opens."""
    evidence, consensus, grounded = run(
        ["Description", "Quantité", "Prix unitaire", "", "Montant TTC"],
        [["Prestation", "2", "35,00", "", "84,00"]])
    assert evidence.rows == []
    assert consensus.rows == []
    assert all(item["total_ht"] is None for item in grounded.payload["line_items"])


def test_unqualified_amount_reconciling_as_ht_is_preserved():
    _, consensus, grounded = run(UNQUALIFIED, [["Prestation", "2", "35,00", "", "70,00"]])
    assert consensus.rows[0].amount_operand_veto is False
    assert exported(grounded)["total_ht"] == 70.0
    assert exported(grounded)["quantity"] == 2.0
    assert exported(grounded)["unit_price"] == 35.0


def test_unqualified_amount_reconciling_only_as_ttc_is_abstained():
    """3 x 25,00 = 75,00 but 90,00 is printed: the guard vetoes, so the amount must not export as HT."""
    _, consensus, grounded = run(WITH_TAX, [["Prestation", "3", "25,00", "20%", "90,00"]])
    row = consensus.rows[0]
    assert row.amount_operand_veto is True
    assert not row.cells.get("quantity") and not row.cells.get("unit_price")
    assert exported(grounded)["total_ht"] is None
    assert exported(grounded)["quantity"] is None
    assert exported(grounded)["unit_price"] is None


def test_valid_amount_with_absent_operands_is_never_vetoed():
    """No price was read, so the guard never ran: the printed amount stays, unlike PR86's strict rules."""
    _, consensus, grounded = run(UNQUALIFIED, [["Prestation", "2", "", "", "70,00"]])
    row = consensus.rows[0]
    assert row.amount_operand_veto is False
    assert not row.cells.get("unit_price")
    assert exported(grounded)["total_ht"] == 70.0


def test_vetoed_operands_with_a_recognized_tax_column_abstain():
    _, consensus, grounded = run(RECOGNIZED_TAX, [["Prestation", "3", "25,00", "20%", "90,00"]])
    assert consensus.rows[0].amount_operand_veto is True
    assert exported(grounded)["total_ht"] is None


def test_abstention_is_traceable_to_the_guard():
    _, _, grounded = run(WITH_TAX, [["Prestation", "3", "25,00", "20%", "90,00"]])
    assert any("line_items.0.total_ht" in warning and "contrôle du produit" in warning
               for warning in grounded.warnings)
    assert grounded.provenance.get("line_items.0.total_ht")


def test_flag_defaults_false_before_consensus():
    evidence = from_tokens(table(UNQUALIFIED, [["Prestation", "3", "25,00", "", "90,00"]]),
                           literal_columns=False)
    assert evidence.rows[0].amount_operand_veto is False


def test_an_unrelated_valid_row_is_not_suppressed():
    """One vetoed row must not blank a neighbouring row whose own arithmetic reconciles."""
    _, consensus, grounded = run(
        UNQUALIFIED,
        [["Prestation", "2", "35,00", "", "70,00"],
         ["Atelier", "3", "25,00", "", "90,00"]])
    assert [row.amount_operand_veto for row in consensus.rows] == [False, True]
    assert exported(grounded, 0)["total_ht"] == 70.0
    assert exported(grounded, 0)["description"] == "Prestation"
    assert exported(grounded, 1)["total_ht"] is None


def test_taux_de_tva_column_is_not_a_price_header():
    """PR86's negative fixture, with no rate vocabulary added: the tax column must not open a table."""
    evidence = from_tokens(table(
        ["Description", "Quantité", "Taux de TVA", "Montant"],
        [["Prestation", "2", "20%", "70,00"]], xs=(60, 200, 380, 520)), literal_columns=False)
    assert evidence.rows == []


def test_repeated_reconcile_keeps_the_abstention():
    """The targeted second pass re-runs reconcile; the abstention must survive a model proposal."""
    evidence = from_tokens(table(WITH_TAX, [["Prestation", "3", "25,00", "20%", "90,00"]]),
                           literal_columns=False)
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    proposal = {"line_items": [{"description": "Prestation", "quantity": 3, "unit_price": 25.0,
                                "total_ht": 90.0}]}
    first = reconcile(consensus, [{}])
    second = reconcile(consensus, [proposal])
    assert exported(first)["total_ht"] is None
    assert exported(second)["total_ht"] is None


UK_VETOED = (
    "Supplier: Oak Studio Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Logo design | {quantity} | 757.97 | 5912.17\n"
    "{totals}"
)
SUBTOTAL = "Subtotal: 5912.17\nVAT (0%): 0.00\nInvoice total: 5912.17"


def _uk_consensus(totals):
    from invoice_evidence import from_text
    readings = {
        family: [from_text(UK_VETOED.format(quantity="7.8" if family == "scale3" else "78",
                                            totals=totals), locale="en-GB")]
        for family in ("deskew", "clahe", "crop", "denoise", "scale3")
    }
    return build_consensus(readings, confidence=40)


def test_vetoed_amount_corroborated_by_the_document_ht_is_preserved():
    """Four families misread 7.8 as 78, so the guard vetoes; the printed Net amount still sums to the
    labelled subtotal, which is positive HT support. PR86's offline rule never exercised this case
    and would have suppressed a valid amount — two existing tests pin it."""
    consensus = _uk_consensus(SUBTOTAL)
    assert consensus.rows[0].amount_operand_veto is True
    grounded = reconcile(consensus, [{"line_items": []}])
    item = exported(grounded)
    assert item["total_ht"] == 5912.17
    assert item["quantity"] is None
    assert item["unit_price"] is None
    assert item["description"] == "Logo design"


def test_the_same_vetoed_amount_without_document_ht_support_abstains():
    """The exemption is evidence-driven, not label-driven: remove the labelled subtotal and the
    amount's HT interpretation is no longer established, so the abstention applies."""
    consensus = _uk_consensus("VAT (0%): 0.00")
    assert consensus.rows[0].amount_operand_veto is True
    grounded = reconcile(consensus, [{"line_items": []}])
    assert exported(grounded)["total_ht"] is None


def test_corroborated_printing_error_is_warned_about_not_suppressed():
    """The established policy for a printed amount that mismatches qty x price but explains the
    document subtotal delta: keep it and warn. Three existing tests pin this on a real fixture;
    this pins the same signal synthetically so the abstention cannot silently swallow it."""
    from invoice_evidence import from_text
    evidence = from_text(
        "Description | Quantité | Prix unitaire | Montant\n"
        "Prestation | 2 | 35,00 | 70,00\n"
        "Atelier | 3 | 25,00 | 90,00\n"
        "Total HT: 145,00\n", locale="fr-FR")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    vetoed = consensus.rows[1]
    assert vetoed.amount_operand_veto is True
    assert vetoed.corroborated_amount_mismatch is not None
    grounded = reconcile(consensus, [{}])
    assert exported(grounded, 1)["total_ht"] == 90.0
    assert exported(grounded, 1)["quantity"] is None
    assert exported(grounded, 0)["total_ht"] == 70.0
    assert any(flag.code == "corroborated_line_amount_mismatch" for flag in grounded.validation.flags)


def test_mislabeled_ht_subtotal_does_not_exempt_a_vat_inclusive_amount():
    """PR89: the document says Total HT, but the printed amount is the TTC of the corroborated
    product (3 x 25,00 x 1,20 = 90,00). The sum_supported exemption must not preserve it."""
    from invoice_evidence import from_text
    evidence = from_text(
        "Description | Quantité | Prix unitaire | Montant\n"
        "Prestation | 3 | 25,00 | 90,00\n"
        "Total HT: 90,00\n"
        "TVA 20% : 15,00\n", locale="fr-FR")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert consensus.rows[0].amount_operand_veto is True
    grounded = reconcile(consensus, [{}])
    assert exported(grounded)["total_ht"] is None


def test_mismatch_that_is_a_vat_gross_up_is_not_preserved():
    """PR89: the 'anomaly' that explains the subtotal delta is exactly the VAT gross-up of the
    corroborated product, so it is a TTC amount, not a printing error to keep."""
    from invoice_evidence import from_text
    evidence = from_text(
        "Description | Quantité | Prix unitaire | % TVA | Montant\n"
        "Prestation | 2 | 35,00 | 20% | 70,00\n"
        "Atelier | 3 | 25,00 | 20% | 90,00\n"
        "Total HT: 145,00\n", locale="fr-FR")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert consensus.rows[1].amount_operand_veto is True
    assert consensus.rows[1].corroborated_amount_mismatch is not None
    grounded = reconcile(consensus, [{}])
    assert exported(grounded, 1)["total_ht"] is None
    assert exported(grounded, 0)["total_ht"] == 70.0


def test_ocr_error_versus_printing_error_remains_undecidable():
    """Documented residual: without a tax rate, a systematic OCR misread and a genuine printing
    error produce identical evidence. The amount is kept with a warning; this is NOT a fix."""
    from invoice_evidence import from_text
    evidence = from_text(
        "Description | Quantité | Prix unitaire | Montant\n"
        "Prestation | 2 | 35,00 | 70,00\n"
        "Atelier | 3 | 25,00 | 90,00\n"
        "Total HT: 145,00\n", locale="fr-FR")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    grounded = reconcile(consensus, [{}])
    assert exported(grounded, 1)["total_ht"] == 90.0
    assert any(flag.code == "corroborated_line_amount_mismatch" for flag in grounded.validation.flags)


def test_a_vat_rate_never_suppresses_an_amount_that_reconciles_as_ht():
    """Guard against over-reach: a printed rate alone must not abstain when qty x price equals the
    amount. The gross-up test requires the product NOT to reconcile."""
    from invoice_evidence import from_text
    evidence = from_text(
        "Description | Quantité | Prix unitaire | % TVA | Montant\n"
        "Prestation | 2 | 35,00 | 20% | 70,00\n"
        "Total HT: 70,00\n", locale="fr-FR")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert consensus.rows[0].amount_operand_veto is False
    grounded = reconcile(consensus, [{}])
    assert exported(grounded)["total_ht"] == 70.0


# ------------------------------------------------------------------ PR123 absent operands
# PR89's product check has nothing to compare when a quantity or price cell is absent, and the
# operand-veto branch is unreachable because the guard never ran. Printed document totals can still
# settle the interpretation; the value only abstains — no replacement HT amount is ever derived.


def _fr(totals, amount="90,00", price="", tax_column=False):
    from invoice_evidence import from_text
    header = "Description | Quantité | Prix unitaire | % TVA | Montant" if tax_column \
        else "Description | Quantité | Prix unitaire | Montant"
    row = f"Prestation | 3 | {price} | {'20% | ' if tax_column else ''}{amount}"
    return from_text("\n".join([header, row, *totals]) + "\n", locale="fr-FR")


def _export(consensus):
    return reconcile(consensus, [{}]).payload["line_items"][0]


def test_absent_operands_with_labelled_ht_and_ttc_abstain():
    """PR88's `tax_rate_recognized`: no price was read, so nothing reconciles, but the document prints
    HT 75,00 / TTC 90,00 and the line amount is the TTC one."""
    evidence = _fr(["Total HT : 75,00", "TVA 20% : 15,00", "Total TTC : 90,00"])
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert consensus.rows[0].amount_operand_veto is False
    assert not consensus.rows[0].family_votes.get("unit_price")
    assert _export(consensus)["total_ht"] is None


def test_absent_operands_abstain_on_the_printed_ttc_total_alone():
    """PR88's `operands_missing_price`: no HT total is printed, but a printed VAT amount means HT and
    TTC differ, so an amount equal to the TTC total cannot be an HT amount."""
    evidence = _fr(["TVA 20% : 15,00", "Total TTC : 90,00"])
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert _export(consensus)["total_ht"] is None


def test_a_misrecognized_rate_still_abstains_on_the_printed_total_pair():
    """PR88's `tax_rate_misrecognized` (`TVA 2O%`): the rate is unreadable, yet two distinct printed
    totals already establish which one the line amount matches."""
    evidence = _fr(["Total HT : 75,00", "TVA 2O% : 15,00", "Total TTC : 90,00"])
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert _export(consensus)["total_ht"] is None


def test_absent_operands_without_any_tax_evidence_keep_the_amount():
    """The documented residual: with only an unqualified total there is no evidence that the amount
    includes tax, so the printed amount must stay (PR88's `tax_rate_missing`)."""
    evidence = _fr(["Total : 90,00"])
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert _export(consensus)["total_ht"] == 90.0


def test_an_amount_matching_the_labelled_ht_total_is_preserved_with_two_rates():
    """Multiple VAT rates make the rate unusable; the amount must still export when it matches the
    printed HT total, and abstain when it matches the TTC one instead."""
    ht_side = _fr(["Total HT : 75,00", "TVA 20% : 12,00", "TVA 5,5% : 3,30", "Total TTC : 90,30"],
                  amount="75,00")
    consensus = build_consensus({"gray": [ht_side], "contrast": [copy.deepcopy(ht_side)]})
    assert _export(consensus)["total_ht"] == 75.0
    ttc_side = _fr(["Total HT : 75,00", "TVA 20% : 12,00", "TVA 5,5% : 3,30", "Total TTC : 90,30"],
                   amount="90,30")
    consensus = build_consensus({"gray": [ttc_side], "contrast": [copy.deepcopy(ttc_side)]})
    assert _export(consensus)["total_ht"] is None


def test_a_zero_vat_document_keeping_ht_equal_to_ttc_is_preserved():
    """The 0 % case PR88's `document_ttc_sum` lost: the amount equals the labelled HT total, so it is
    HT and must not abstain even though it also equals the TTC total."""
    evidence = _fr(["Total HT : 90,00", "TVA 0% : 0,00", "Total TTC : 90,00"])
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert _export(consensus)["total_ht"] == 90.0


def test_only_the_row_equal_to_the_ttc_total_abstains():
    """Selectivity: a neighbour whose own amount equals the printed HT total keeps its value; only the
    row equal to the TTC total abstains, so one tax-inclusive line cannot blank the whole table."""
    from invoice_evidence import from_text
    evidence = from_text(
        "Description | Quantité | Prix unitaire | Montant\n"
        "Prestation | 3 |  | 75,00\n"
        "Atelier | 1 |  | 90,00\n"
        "Total HT : 75,00\nTVA 20% : 15,00\nTotal TTC : 90,00\n", locale="fr-FR")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    items = reconcile(consensus, [{}]).payload["line_items"]
    assert [row.amount_operand_veto for row in consensus.rows] == [False, False]
    assert items[0]["total_ht"] == 75.0
    assert items[1]["total_ht"] is None


def test_contradictory_printed_tax_information_abstains_rather_than_chooses():
    """HT 75,00 + VAT 30,00 does not reach the printed TTC 90,00: the tax block is contradictory, so
    no amount may be asserted as HT."""
    evidence = _fr(["Total HT : 75,00", "TVA 20% : 30,00", "Total TTC : 90,00"])
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert _export(consensus)["total_ht"] is None


def test_absent_operand_abstention_never_derives_a_replacement_amount():
    """Phase 2.3/2.4: the corrected export is null, the printed value stays only as provenance and no
    computed 75,00 ever appears in the payload."""
    evidence = _fr(["Total HT : 75,00", "TVA 20% : 15,00", "Total TTC : 90,00"], tax_column=True,
                   price="")
    consensus = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    grounded = reconcile(consensus, [{}])
    item = grounded.payload["line_items"][0]
    assert item["total_ht"] is None and item["unit_price"] is None
    assert 75.0 not in set(item.values()) | {item.get("total_ht")}
    assert "75.0" not in json.dumps(item)
    assert any("total TTC" in warning for warning in grounded.warnings)
