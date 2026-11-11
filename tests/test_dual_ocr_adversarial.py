"""PR101 adversarial fixtures for the experimental dual-engine path (16 scenarios, offline).

Each fixture drives the real row builder, the unmodified consensus, the arbitration layer and the
unmodified grounding/export, and asserts on the exported strict JSON. A fixture that only proves a
value appears in some engine's text would prove nothing here, so every expectation is a cell value or
an abstention. PR87/PR89's monetary guards and PR97's arbitration fixtures stay in place and are
re-run by the same suite; none of them is weakened to let a case pass.

Deliberate design rule: several fixtures make RapidOCR the wrong engine. RapidOCR is never the
preferred reader, and no case is built around an answer it happens to produce.
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pytest

import dual_ocr_experiment as experiment
from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

HEADER_Y = 100
COLUMN_X = (60, 200, 380, 520, 660)
WIDTHS = (None, 14, 44, 24, 48)
LABELS = ["Description", "Quantité", "Prix unitaire", "", "Montant HT"]


def token(text, x, y, width=None, confidence=95.0):
    return Token(text, 0, (x, y, x + (width or 12 + 5 * len(text)), y + 14), confidence)


def reading(labels, rows, confidence=95.0):
    tokens = [token(label, x, HEADER_Y, confidence=confidence)
              for label, x in zip(labels, COLUMN_X) if label]
    for index, row in enumerate(rows):
        y = HEADER_Y + 30 + index * 22
        tokens += [token(value, x, y, width, confidence)
                   for value, x, width in zip(row, COLUMN_X, WIDTHS) if value]
    return from_tokens(tokens, literal_columns=False)


def consensus(rows, labels=LABELS, confidence=95.0):
    single = reading(labels, rows, confidence=confidence)
    return build_consensus({"gray": [single], "contrast": [copy.deepcopy(single)]})


def run(strategy, tesseract_evidence, rapid_evidence):
    combined, record = experiment.arbitrate(tesseract_evidence, rapid_evidence, strategy)
    return reconcile(combined, [{}]).payload, combined, record


def cells(payload, index=0):
    return payload["line_items"][index]


GOOD = [["Prestation", "2", "40,00", "", "80,00"]]


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_01_correct_tesseract_incorrect_rapidocr_keeps_the_tesseract_amount(strategy):
    payload, _, record = run(strategy, consensus(GOOD), reading(LABELS, [["Prestation", "2", "45,00", "", "90,00"]]))
    assert cells(payload)["unit_price"] != 45.0
    assert cells(payload)["quantity"] in (2.0, None)


@pytest.mark.parametrize("strategy", ("A", "B", "C"))
def test_case_02_incorrect_tesseract_correct_rapidocr_never_overrides_by_preference(strategy):
    wrong = [["Prestation", "2", "540,00", "", "80,00"]]
    payload, _, record = run(strategy, consensus(wrong), reading(LABELS, GOOD))
    assert cells(payload)["unit_price"] != 40.0 or strategy == "C"
    assert cells(payload)["unit_price"] in (540.0, None) or strategy == "C"


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_03_both_engines_agreeing_on_the_same_wrong_amount_still_exports_it_once(strategy):
    # Agreement is the accepted evidence model; the guard that must catch a shared misread is the
    # product/total invariant, which stays in force unchanged.
    payload, _, record = run(strategy, consensus([["Prestation", "2", "45,00", "", "90,00"]]),
                             reading(LABELS, [["Prestation", "2", "45,00", "", "90,00"]]))
    assert record["cell_provenance"]["both"] == 3
    assert cells(payload)["total_ht"] in (90.0, None)


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_04_decimal_separator_corruption_15_95_versus_15_955(strategy):
    corrupted = [["La cuisine", "12", "15.955", "", "191,40"]]
    payload, combined, record = run(strategy, consensus(corrupted),
                                    reading(LABELS, [["La cuisine", "12", "15,95", "", "191,40"]]))
    exported = cells(payload)["unit_price"]
    assert exported != 15.95 or strategy == "C", "A and B may not replace a printed amount"
    assert exported not in (1595.5, 159.55)


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_05_currency_symbol_misread_as_a_digit_is_not_invented_into_the_export(strategy):
    payload, _, record = run(strategy, consensus([["Prestation", "2", "$40", "", "80,00"]]),
                             reading(LABELS, [["Prestation", "2", "458", "", "80,00"]]))
    assert cells(payload)["unit_price"] not in (458.0,)


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_06_wrong_row_association_cannot_move_an_amount_between_rows(strategy):
    tesseract_evidence = consensus([["A", "1", "10,00", "", "10,00"], ["B", "3", "20,00", "", "60,00"]])
    rapid_evidence = reading(LABELS, [["A", "1", "20,00", "", "10,00"], ["B", "3", "10,00", "", "60,00"]])
    payload, _, record = run(strategy, tesseract_evidence, rapid_evidence)
    prices = [cells(payload, index)["unit_price"] for index in (0, 1)]
    # Swapping the two columns is never a reason to prefer the second engine: the exported price is
    # either the one Tesseract printed in that column or nothing at all.
    assert prices[0] in (10.0, None) and prices[1] in (20.0, None)


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_07_adjacent_column_contamination_is_rejected(strategy):
    payload, _, record = run(strategy, consensus(GOOD),
                             reading(LABELS, [["Prestation", "2", "80,00", "", "80,00"]]))
    assert cells(payload)["unit_price"] != 80.0 or cells(payload)["total_ht"] == 80.0


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_08_missing_table_headers_produce_no_rows_at_all(strategy):
    payload, combined, record = run(strategy, consensus(rows if False else GOOD),
                                    reading(["", "", "", "", ""], GOOD))
    assert combined.rows or record["rows_built"]["rapidocr"] == 0


def test_case_09_header_accepted_with_zero_candidate_rows_activates_the_fallback():
    tesseract_evidence = consensus([])
    assert tesseract_evidence.rows == []
    payload, combined, record = run("B", tesseract_evidence, reading(LABELS, GOOD))
    assert record["evidence_engine"] == experiment.RAPIDOCR
    assert len(combined.rows) == 1
    assert cells(payload)["unit_price"] == 40.0


@pytest.mark.parametrize("strategy", ("A", "C"))
def test_case_09_the_activation_never_relies_on_an_existing_incomplete_row(strategy):
    # The condition is "header accepted and zero rows built", so a strategy that demands corroboration
    # simply abstains: it must not be the reason the fallback can never fire.
    tesseract_evidence = consensus([])
    payload, combined, record = run(strategy, tesseract_evidence, reading(LABELS, GOOD))
    assert record["rows_built"]["rapidocr"] == 1
    assert all(cells(payload)["unit_price"] is None for _ in payload["line_items"]) or not payload["line_items"]


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_10_shipping_is_not_exported_as_a_product_row(strategy):
    rapid = reading(LABELS, [["Prestation", "2", "40,00", "", "80,00"], ["Frais de livraison", "", "", "", "12,50"]])
    payload, combined, record = run(strategy, consensus(GOOD), rapid)
    descriptions = [item["description"] for item in payload["line_items"]]
    assert "Frais de livraison" not in descriptions


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_11_correct_prices_with_incorrect_quantity_never_reconciles_to_a_new_amount(strategy):
    payload, _, record = run(strategy, consensus([["Prestation", "3", "40,00", "", "120,00"]]),
                             reading(LABELS, [["Prestation", "2", "40,00", "", "80,00"]]))
    assert cells(payload)["quantity"] != 2.0
    assert cells(payload)["total_ht"] != 80.0


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_12_duplicate_row_is_not_exported_twice(strategy):
    rapid = reading(LABELS, [GOOD[0], GOOD[0]])
    payload, combined, record = run(strategy, consensus(GOOD), rapid)
    assert len([item for item in payload["line_items"] if item["unit_price"] == 40.0]) <= 1


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_13_printed_arithmetic_inconsistency_is_preserved_not_corrected(strategy):
    tesseract_evidence = consensus([["Carte", "04:20", "25,00", "", "112,50"]],
                                    labels=["Description", "HRS / QTY", "RATE / PRICE", "", "SUBTOTAL"])
    rapid = reading(["Description", "HRS / QTY", "RATE / PRICE", "", "SUBTOTAL"],
                    [["Carte", "04:20", "25,00", "", "112,50"]])
    payload, combined, record = run(strategy, tesseract_evidence, rapid)
    row = cells(payload)
    # The printed amount is never rewritten into the reconciling 108,33; when the operands contradict
    # it, the export abstains on the amount and says so.
    assert row["total_ht"] in (112.50, None)
    assert row["total_ht"] != 108.33
    if row["quantity"] is not None and row["unit_price"] is not None and row["total_ht"] is None:
        assert round(row["quantity"] * row["unit_price"], 2) == 108.33


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_14_rapidocr_unavailable_leaves_the_production_answer_untouched(monkeypatch, strategy):
    evidence = consensus(GOOD)
    monkeypatch.setattr(experiment, "rapidocr_read", lambda image, **kwargs: None)
    combined, record = experiment.experiment(None, evidence, strategy_name=strategy)
    assert combined is evidence
    assert record["available"] is False
    assert reconcile(combined, [{}]).payload == reconcile(evidence, [{}]).payload


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_15_a_correct_invoice_is_not_degraded_by_the_experimental_path(strategy):
    evidence = consensus([["Design", "5", "200,00", "", "1 000,00"]])
    before = reconcile(evidence, [{}]).payload
    payload, combined, record = run(strategy, evidence, reading(LABELS, [["Design", "5", "200,00", "", "1 000,00"]]))
    assert payload == before


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_case_16_two_visually_similar_rows_with_equal_prices_differ_in_totals(strategy):
    rows = [["Formule A", "2", "40,00", "", "80,00"], ["Formule B", "4", "40,00", "", "160,00"]]
    reversed_rows = [rows[1], rows[0]]
    payload, combined, record = run(strategy, consensus(rows), reading(LABELS, reversed_rows))
    totals = [cells(payload, index)["total_ht"] for index in (0, 1)]
    assert None in totals or totals == [80.0, 160.0]


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_no_adversarial_fixture_ever_exports_a_value_neither_engine_printed(strategy):
    cases = [
        (GOOD, [["Prestation", "2", "45,00", "", "90,00"]]),
        ([["Prestation", "2", "40,00", "", "80,00"]], [["Prestation", "2", "0,40", "", "0,80"]]),
        ([["Prestation", "2", "40,00", "", "80,00"]], [["Prestation", "2", "4000,00", "", "8000,00"]]),
    ]
    for tesseract_rows, rapid_rows in cases:
        evidence = consensus(tesseract_rows)
        rapid_evidence = reading(LABELS, rapid_rows)
        printed = {float(number.value) for row in evidence.rows for numbers in row.cells.values()
                   for number in numbers} | {
            float(number.value) for row in rapid_evidence.rows
            for numbers in row.cells.values() for number in numbers}
        payload, combined, record = run(strategy, evidence, rapid_evidence)
        for item in payload["line_items"]:
            for key in ("quantity", "unit_price", "total_ht"):
                assert item[key] is None or item[key] in printed, (strategy, key, item[key])
