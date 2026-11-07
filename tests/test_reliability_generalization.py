"""Synthetic structural regressions; these are not independent real-model evaluations."""

import copy
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import Number, Token, from_text, from_tokens
from invoice_grounding import RECOVERED, reconcile, run_grounded_inference
from invoice_validation import document_validation
from ocr_consensus import build_consensus
from ocr_budget import ProcessingBudget
from ocr_recovery import _clarify_decimal_quantity


def positioned(rows):
    """Explicit synthetic token geometry, independent of Tesseract."""
    return from_tokens([
        Token(text, 0, (x, y, x + max(8, len(text) * 5), y + 10), 95)
        for y, cells in rows for x, text in cells
    ])


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize("currency", ["USD", "GBP"])
def test_due_date_disambiguates_issue_date_independently_of_currency(separator, currency):
    evidence = from_text(
        f"Currency: {currency}\n"
        f"Invoice date: 09{separator}01{separator}2025 Due date: 09{separator}14{separator}2025\n"
    )
    assert evidence.invoice_date == "2025-09-01"
    validation = document_validation(evidence)
    assert not any(flag.code in {"due_before_invoice", "date_invalid"} for flag in validation.flags)


def test_ambiguous_date_without_locale_or_other_evidence_remains_null():
    assert from_text("Invoice date: 09/01/2025").invoice_date is None


def test_european_due_date_overrides_dollar_currency_date_assumption():
    evidence = from_text("Currency: USD\nIssued: 01.09.2025\nDue: 14.09.2025")
    assert evidence.invoice_date == "2025-09-01"


def test_conflicting_date_orders_do_not_resolve_an_ambiguous_issue_date():
    evidence = from_text(
        "Currency: USD\nInvoice date: 01/02/2025\nDue date: 09/14/2025\n"
        "Date de livraison: 24/02/2025"
    )
    assert evidence.invoice_date is None


@pytest.mark.parametrize("digits, expected", [
    ("2422021", "2021-02-24"),
    ("24022021", "2021-02-24"),
    ("1112025", None),
    ("01092025", None),
    ("3122025", None),
    ("99002025", None),
])
def test_separator_loss_needs_one_calendar_interpretation_not_a_locale_guess(digits, expected):
    evidence = from_text(f"Currency: EUR\nDate: {digits}")
    assert evidence.invoice_date == expected


def test_identifier_digits_without_date_label_cannot_become_invoice_date():
    assert from_text("Invoice number: 2422021\nCurrency: EUR").invoice_date is None


@pytest.mark.parametrize("price_header", ["Hourly rate", "Unit price", "Net unit price"])
def test_structural_header_aliases_reconstruct_real_rows_not_model_summary(price_header):
    evidence = from_text(
        "Supplier: Studio Ltd\nCurrency: USD\n"
        f"Product | {price_header} | Quantity | Tax rate | Amount\n"
        "Design | 200 | 5 | 10% | 1000\n"
        "Subtotal: 1000\nTax: 100\nTotal: 1100"
    )
    assert len(evidence.rows) == 1
    assert evidence.rows[0].cells["tax_rate"][0].value == 10
    result = reconcile(evidence, [{
        "line_items": [{"description": "Services", "quantity": 1, "unit_price": 1000, "total_ht": 1000}],
    }])
    assert result.payload["line_items"] == [{
        "description": "Design", "quantity": 5.0, "unit_price": 200.0, "total_ht": 1000.0,
    }]
    assert result.payload["total_ttc"] == 1100


def test_minor_header_ocr_error_is_not_a_license_to_correct_body_values():
    evidence = positioned([
        (0, [(0, "Product"), (140, "Hourly"), (180, "rote"), (300, "Quontity"), (420, "Amount")]),
        (25, [(0, "Portraits"), (140, "$250.00"), (300, "2"), (420, "$500.00")]),
    ])
    assert len(evidence.rows) == 1
    assert evidence.rows[0].cells["quantity"][0].value == 2
    assert evidence.rows[0].cells["total_ht"][0].value == 500


def test_gross_column_never_becomes_net_amount_or_calculated_net_fact():
    evidence = from_text(
        "Fournisseur: Atelier SARL\nCurrency: EUR\n"
        "Description | Quantité | Unité | Prix unitaire HT | % TVA | Total TVA | Total TTC\n"
        "Travail | 5 | h | 60 | 20% | 60 | 360\n"
        "Produit | 10 | pcs | 105 | 20% | 210 | 1260\n"
        "Total HT: 1350\nTotal TVA: 270\nTotal TTC: 1620"
    )
    assert len(evidence.rows) == 2
    result = reconcile(evidence, [{
        "line_items": [
            {"description": "Travail", "quantity": 5, "unit_price": 60, "total_ht": 360},
            {"description": "Produit", "quantity": 10, "unit_price": 105, "total_ht": 1260},
        ],
    }])
    assert [row["total_ht"] for row in result.payload["line_items"]] == [None, None]
    assert [row["quantity"] for row in result.payload["line_items"]] == [5, 10]
    assert result.payload["total_ht"] == 1350
    assert result.payload["total_ttc"] == 1620


def test_billed_from_and_billed_to_columns_do_not_merge_supplier_and_customer():
    evidence = positioned([
        (0, [(0, "Billed from"), (300, "Billed to")]),
        (20, [(0, "Actual Studio"), (300, "Customer Company")]),
        (40, [(0, "2132 Duncan Street"), (300, "3332 Stevens Creek Blvd")]),
        (60, [(0, "Currency: USD")]),
        (80, [(0, "Description"), (140, "Quantity"), (260, "Unit price"), (400, "Amount")]),
        (100, [(0, "Service"), (140, "2"), (260, "10"), (400, "20")]),
    ])
    assert evidence.supplier == "Actual Studio"
    assert evidence.supplier_is_explicit
    assert "Customer" not in evidence.supplier


def test_missing_model_rows_do_not_hide_documentary_rows():
    evidence = from_text(
        "Supplier: Supplies Ltd\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Product A | 15 | 10 | 150\nProduct B | 10 | 5 | 50\nProduct C | 3 | 25 | 75\n"
        "Subtotal: 275\nVAT (20%): 55\nTotal due: 330"
    )
    evidence.source_kind = "ocr"
    evidence.ocr_cross_checked = True
    predict = MagicMock(return_value={
        "raw": json.dumps({"line_items": [{"description": "Services", "quantity": 1, "unit_price": 275, "total_ht": 275}]}),
        "finish_reason": "stop",
    })
    result = run_grounded_inference(evidence, {"input": evidence.text}, [], predict)
    assert len(result["payload"]["line_items"]) == 3
    assert [r["description"] for r in result["payload"]["line_items"]] == ["Product A", "Product B", "Product C"]
    assert result["payload"]["total_ttc"] == 330


def test_two_conflicting_printed_amounts_cannot_be_chosen_by_arithmetic():
    evidence = from_text(
        "Supplier: Supplies Ltd\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Service | 5.1 | 77.66 | 369.07\nSubtotal: 396.07\nTotal due: 396.07"
    )
    row = evidence.rows[0]
    row.cells["total_ht"].append(Number("396.07", Decimal("396.07"), 0, row.line.box))
    result = reconcile(evidence, [{
        "line_items": [{"description": "Service", "quantity": 5.1, "unit_price": 77.66, "total_ht": 396.07}],
    }])
    assert result.payload["line_items"][0]["total_ht"] is None


def test_multiline_description_gap_scales_with_font_height_and_does_not_cross_rows():
    evidence = positioned([
        (0, [(0, "Description"), (240, "Quantity"), (340, "Unit price"), (480, "Amount")]),
        (30, [(0, "Service A"), (240, "1"), (340, "400"), (480, "400")]),
        (65, [(0, "Continuation A")]),
        (100, [(0, "Service B"), (240, "1"), (340, "200"), (480, "200")]),
        (135, [(0, "Continuation B")]),
    ])
    consensus = build_consensus({"gray": [copy.deepcopy(evidence)], "contrast": [copy.deepcopy(evidence)]})
    assert [row.description for row in consensus.rows] == [
        "Service A Continuation A", "Service B Continuation B",
    ]


def test_continuation_suffix_requires_two_families_even_with_segmentation_truncation():
    base = from_text(
        "Description | Quantity | Unit price | Amount\nService | 1 | 200 | 200"
    )
    long = copy.deepcopy(base)
    long.rows[0].secondary_description = "Detailed service 1"
    short = copy.deepcopy(base)
    short.rows[0].secondary_description = "Detailed service"
    consensus = build_consensus({"gray": [long], "contrast": [copy.deepcopy(long), short]})
    assert consensus.rows[0].description == "Service Detailed service 1"
    one_family = build_consensus({"gray": [long, short], "contrast": [base]})
    assert one_family.rows[0].description == "Service"
    conflicting = copy.deepcopy(long)
    conflicting.rows[0].secondary_description = "Detailed service 2"
    conflict = build_consensus({"gray": [long], "contrast": [copy.deepcopy(long), conflicting]})
    assert conflict.rows[0].description == "Service"


@pytest.mark.parametrize("second_read, accepted", [("7.8", True), ("78", False), ("", False)])
def test_decimal_refinement_requires_two_agreeing_region_reads(monkeypatch, second_read, accepted):
    from PIL import Image
    import ocr_recovery

    text = (
        "Supplier: Studio Ltd\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Design | 78 | 757.97 | 5912.17\nSubtotal: 5912.17\nInvoice total: 5912.17"
    )
    wrong = from_text(text)
    clear = from_text(text.replace("Design | 78 |", "Design | 7.8 |"))
    readings = {"deskew": [wrong], "clahe": [copy.deepcopy(wrong)], "scale3": [clear]}
    evidence = build_consensus(readings)
    assert not evidence.rows[0].cells["quantity"]
    parsed = iter([from_text("7.8"), from_text(second_read)])
    monkeypatch.setattr(ocr_recovery, "from_tsv", lambda *args, **kwargs: next(parsed))
    reader = MagicMock(return_value={})
    _clarify_decimal_quantity(
        evidence, readings, {key: Image.new("L", (200, 200)) for key in ("deskew", "clahe")},
        reader, ProcessingBudget(),
    )
    assert reader.call_count == 2
    assert [number.value for number in evidence.rows[0].cells["quantity"]] == (
        [Decimal("7.8"), Decimal("7.8")] if accepted else []
    )
    if accepted:
        assert evidence.rows[0].corroborated_amount_mismatch is None


def test_decimal_refinement_never_creates_a_value_when_the_crop_contains_no_number(monkeypatch):
    from PIL import Image
    import ocr_recovery

    wrong = from_text(
        "Description | Quantity | Unit price | Amount\nDesign | 78 | 757.97 | 5912.17"
    )
    readings = {"deskew": [wrong], "clahe": [copy.deepcopy(wrong)]}
    evidence = build_consensus(readings)
    monkeypatch.setattr(ocr_recovery, "from_tsv", lambda *args, **kwargs: from_text(""))
    reader = MagicMock(return_value={})
    _clarify_decimal_quantity(
        evidence, readings, {key: Image.new("L", (200, 200)) for key in readings},
        reader, ProcessingBudget(),
    )
    reader.assert_called_once()
    assert not evidence.rows[0].cells["quantity"]


def test_corroborated_printed_amount_mismatch_does_not_trigger_decimal_repair():
    from PIL import Image

    evidence = from_text(
        "Description | Quantity | Unit price | Amount\nDesign | 5.1 | 77.66 | 369.07"
    )
    readings = {"deskew": [evidence], "clahe": [copy.deepcopy(evidence)]}
    consensus = build_consensus(readings)
    consensus.rows[0].corroborated_amount_mismatch = (Decimal("369.07"), Decimal("396.07"))
    reader = MagicMock(side_effect=AssertionError("Do not repair a printed discrepancy"))
    _clarify_decimal_quantity(
        consensus, readings, {key: Image.new("L", (200, 200)) for key in readings},
        reader, ProcessingBudget(),
    )
    reader.assert_not_called()
    assert consensus.rows[0].cells["total_ht"][0].value == Decimal("369.07")


def test_consensus_refuses_zero_price_with_corroborated_positive_gross_amount():
    evidence = from_text(
        "Currency: EUR\nDescription | Quantité | Prix unitaire HT | Total TTC\n"
        "Travail | 5 | 0 | 360\n"
    )
    result = build_consensus({"gray": [evidence], "contrast": [copy.deepcopy(evidence)]})
    assert result.rows[0].cells["unit_price"] == []
    assert result.rows[0].cells["gross_amount"][0].value == 360
    strict = reconcile(result, [{"line_items": [{"description": "Travail", "quantity": 5, "unit_price": 0}]}])
    assert strict.payload["line_items"][0]["unit_price"] is None


def test_partial_net_row_reports_corroborated_printed_tax_discrepancy_without_exporting_net():
    text = (
        "Currency: EUR\n"
        "Description | Quantité | Prix unitaire HT | % TVA | Total TVA | Total TTC\n"
        "Produit | 10 | 105 | 20% | 270 | 1260\n"
        "Total HT: 1050\nTotal TTC: 1260"
    )
    evidence = build_consensus({
        "gray": [from_text(text)], "contrast": [from_text(text)],
    })
    result = reconcile(evidence, [{}])
    assert result.payload["line_items"][0]["total_ht"] is None
    flag = next(flag for flag in result.validation.flags if flag.code == "corroborated_line_tax_mismatch")
    assert Decimal(flag.printed) == 270
    assert Decimal(flag.expected) == 210
    assert Decimal(flag.difference) == 60
    one_family = build_consensus({"gray": [from_text(text), from_text(text)]})
    assert not any(
        flag.code == "corroborated_line_tax_mismatch"
        for flag in document_validation(one_family).flags
    )
    single_tax_vote = copy.deepcopy(evidence)
    single_tax_vote.rows[0].family_votes["tax_amount"] = {Decimal("270"): ("gray",)}
    assert not any(
        flag.code == "corroborated_line_tax_mismatch"
        for flag in document_validation(single_tax_vote).flags
    )


def test_ligature_is_typographic_equivalence_not_supplier_fuzzy_matching():
    evidence = from_text(
        "Fournisseur: Atelier SARL\nCurrency: EUR\n"
        "Description | Quantité | Prix unitaire HT | Total TTC\n"
        "Main-d'œuvre | 5 | 60 | 360"
    )
    result = reconcile(evidence, [{
        "line_items": [{"description": "Main-d'oeuvre", "quantity": 5, "unit_price": 60}],
    }])
    assert result.payload["line_items"][0]["description"] == "Main-d'œuvre"
    assert result.payload["line_items"][0]["total_ht"] is None


def test_compact_summary_groups_only_the_same_missing_tax_classification():
    from app import main

    evidence = from_text(
        "Supplier: Studio Ltd\nInvoice date: 2022-04-30\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Service | 1 | 600 | 600\nTotal due: 600"
    )
    result = run_grounded_inference(evidence, {"input": evidence.text}, [], lambda *args: {
        "raw": '{"line_items":[]}', "finish_reason": "stop",
    })
    diagnostic = dict(result, evidence=evidence, evidence_available=True)
    original = copy.deepcopy(result["validation"])
    summary = main._compact_validation_summary(result["payload"], diagnostic, "")
    assert summary.count("Total HT non confirmé") == 1
    assert "sans préciser le montant HT ni la TVA" in summary
    assert "TVA par taux" not in summary
    assert "HT + TVA = TTC" not in summary
    assert "Échéance" in summary
    assert result["validation"] == original


def test_siret_warning_is_separate_from_complete_json_fields():
    from app import main

    rows = [json.loads(line) for line in (ROOT / "data/test.jsonl").read_text(encoding="utf-8").splitlines()]
    row = next(item for item in rows if item["id"] == "invoice-0406")
    evidence = from_text(row["input"], locale=row["locale"])
    result = run_grounded_inference(evidence, {"input": evidence.text}, [], lambda *args: {
        "raw": json.dumps(row["output"]), "finish_reason": "stop",
    })
    summary = main._compact_validation_summary(result["payload"], dict(result, evidence=evidence), "")
    assert "20/20 champs recoupés" in summary
    assert "SIRET invalide" in summary
    assert "40023297980955" in summary
    assert "hors schéma" not in summary
    assert result["payload"] == row["output"]


def test_documentary_row_coverage_cannot_hide_a_truncated_output():
    from app import main

    summary = main._compact_validation_summary(
        {"line_items": [{"description": "First", "quantity": 1, "unit_price": 1, "total_ht": 1}]},
        {
            "field_statuses": dict.fromkeys(
                ["supplier_name", "invoice_date", "total_ht", "total_ttc",
                 "line_items.0.description", "line_items.0.quantity",
                 "line_items.0.unit_price", "line_items.0.total_ht"], "valide",
            ),
            "document_coverage": {"documented_rows": 3},
        },
        "",
    )
    assert "Vérification incomplète — 8/16" in summary
    assert "1/3 lignes documentées" in summary
    assert "Facture cohérente" not in summary


def test_examples_hide_heading_with_supported_dataset_configuration():
    from app import main

    app = main.build_app()
    dataset = next(component for component in app.config["components"] if component["type"] == "dataset")
    assert dataset["props"]["show_label"] is False
    assert dataset["props"]["container"] is False
    assert len(dataset["props"]["samples"]) == 3


def test_missing_documentary_net_amount_never_triggers_futile_correction():
    evidence = from_text(
        "Supplier: Studio Ltd\nCurrency: GBP\n"
        "Description | Quantity | Unit price | Amount\n"
        "Design | 1 | 600 | 600\nTotal due: 600"
    )
    predict = MagicMock(return_value={
        "raw": '{"line_items":[]}', "finish_reason": "stop",
    })
    result = run_grounded_inference(evidence, {"input": evidence.text}, [], predict)
    predict.assert_called_once()
    assert result["payload"]["total_ht"] is None
    assert not result["correction_attempted"]
    assert result["document_coverage"]["documented_rows"] == 1
    assert result["document_coverage"]["exhaustive"] is None
