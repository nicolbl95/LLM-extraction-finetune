"""PR102 Priority 6: the negative fixtures PR101 did not cover, asserted on exported strict JSON.

Covers a missing row whose amount appears elsewhere, an engine timeout, a low-quality read, header and
footer contamination, and the two failure modes of the new whitespace-insensitive description rule.
Nothing here weakens a consensus, confidence or grounding safeguard: every expectation is an exported
value or an abstention produced by the unchanged production stages.
"""

from __future__ import annotations

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


def reading(labels, rows, confidence=95.0, extra=()):
    tokens = [token(label, x, HEADER_Y, confidence=confidence)
              for label, x in zip(labels, COLUMN_X) if label]
    for index, row in enumerate(rows):
        y = HEADER_Y + 30 + index * 22
        tokens += [token(value, x, y, width, confidence)
                   for value, x, width in zip(row, COLUMN_X, WIDTHS) if value]
    tokens += [token(text, x, y, confidence=confidence) for text, x, y in extra]
    return from_tokens(tokens, literal_columns=False)


def consensus(rows, labels=LABELS, confidence=95.0, extra=()):
    single = reading(labels, rows, confidence=confidence, extra=extra)
    return build_consensus({"gray": [single], "contrast": [copy.deepcopy(single)]})


def run(strategy, tesseract_evidence, rapid_evidence):
    combined, record = experiment.arbitrate(tesseract_evidence, rapid_evidence, strategy)
    return reconcile(combined, [{}]).payload, combined, record


TWO_ROWS = [["Alpha", "2", "40,00", "", "80,00"], ["Beta", "3", "10,00", "", "30,00"]]


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_missing_row_whose_amount_appears_elsewhere_is_not_reconstructed(strategy):
    # Tesseract reads only the first row. The second row's total is printed in the document footer, so a
    # naive recoverer could attach 30,00 to a row it never saw. No arm may invent that row.
    primary = consensus([TWO_ROWS[0]])
    other = reading(LABELS, [TWO_ROWS[0]], extra=[("Total 30,00", 520, 400)])
    payload, _, _ = run(strategy, primary, other)
    assert len(payload["line_items"]) == 1
    assert payload["line_items"][0]["total_ht"] == 80.0
    assert all(row.get("total_ht") != 30.0 for row in payload["line_items"])


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_an_engine_timeout_leaves_the_production_answer_untouched(monkeypatch, strategy):
    def timeout(image, **kwargs):
        raise TimeoutError("RapidOCR exceeded its budget")

    monkeypatch.setattr(experiment, "rapidocr_read", timeout)
    primary = consensus(TWO_ROWS)
    expected = reconcile(primary, [{}]).payload
    combined, record = experiment.experiment(None, primary, strategy_name=strategy)
    assert reconcile(combined, [{}]).payload == expected
    assert record["available"] is False
    assert "TimeoutError" in record["failure"]


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_low_quality_read_does_not_export_low_confidence_amounts(strategy):
    # Every token is below the production confidence floor. The row builder must not produce a usable
    # row, and no arm may export an amount from it.
    primary = consensus(TWO_ROWS)
    other = reading(LABELS, [["Gamma", "7", "99,00", "", "693,00"]], confidence=20.0)
    payload, _, _ = run(strategy, primary, other)
    exported = [row.get("total_ht") for row in payload["line_items"]]
    assert 693.0 not in exported
    assert all(value in (80.0, 30.0, None) for value in exported)


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_header_and_footer_numbers_are_not_exported_as_product_rows(strategy):
    contamination = [("Invoice 2024 12 34", 60, 20), ("Page 1 of 2 50,00", 60, 500),
                     ("TVA 20% 16,00", 60, 520)]
    primary = consensus(TWO_ROWS, extra=contamination)
    other = reading(LABELS, TWO_ROWS, extra=contamination)
    payload, _, _ = run(strategy, primary, other)
    totals = [row.get("total_ht") for row in payload["line_items"]]
    assert totals == [80.0, 30.0]
    assert 50.0 not in totals and 16.0 not in totals and 34.0 not in totals


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_genuinely_different_description_is_still_blanked(strategy):
    # The new whitespace-insensitive rule must not let a different product name through.
    primary = consensus([["Prestation", "2", "40,00", "", "80,00"]])
    other = reading(LABELS, [["Autre produit", "2", "40,00", "", "80,00"]])
    payload, combined, _ = run(strategy, primary, other)
    descriptions = [row.description for row in combined.rows]
    assert "Autre produit" not in descriptions
    assert payload["line_items"][0]["description"] != "Autre produit"


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_a_short_description_cannot_match_by_chance_inside_an_unrelated_line(strategy):
    # Tesseract never reads "Ab" as a word; the letters only occur inside "abonnement". The second engine
    # must not be able to borrow that as corroboration for a two-letter description.
    primary = consensus([["Prestation", "2", "40,00", "", "80,00"]],
                        extra=[("Laboratoire abonnement", 60, 20)])
    other = reading(LABELS, [["Ab", "2", "40,00", "", "80,00"]])
    _, combined, _ = run(strategy, primary, other)
    assert experiment.MIN_COMPACT_DESCRIPTION > len("ab")
    assert not experiment._text_agrees("Ab", primary)
    assert "Ab" not in [row.description for row in combined.rows]


def test_word_merge_agreement_is_accepted_only_for_the_same_letters():
    evidence = reading(LABELS, TWO_ROWS, extra=[("5498721685 3 Ismène etla ville 8.75", 60, 20)])
    assert experiment._text_agrees("Ismène et la ville", evidence)
    assert not experiment._text_agrees("Ismène et la campagne", evidence)
    assert not experiment._text_agrees("Nov 05, z00g - Website Updates", evidence)
