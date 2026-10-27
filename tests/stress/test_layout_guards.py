"""Counter-tests for the table header guard rails.

Two layouts are deliberately hostile: one declares the same column twice, the other has
no description column at all. A third case keeps a correct header but a broken data row
order, so the description cell holds a number. In every case the answer must stay
unverified: no fabricated row, no false value, no phantom anomaly and no crash.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress"), str(ROOT / "scripts")]

from cases import LAYOUT_HEADER, LAYOUT_TOTALS, layout_guard_cases  # noqa: E402
from invoice_evidence import description_is_usable, from_text  # noqa: E402
from invoice_grounding import NULL, UNVERIFIED, read_model, reconcile  # noqa: E402
from run_stress_check import run_case  # noqa: E402

JARGON = (
    "Traceback", "None", "Pydantic", "non ancré", "ancrage",
    "description_non_verifiee", "Mise en page non reconnue",
)


@pytest.mark.parametrize("case", list(layout_guard_cases()), ids=lambda case: case.name)
def test_guarded_layouts_never_fabricate_a_row_or_a_value(case):
    evidence = from_text(case.text)
    assert evidence.rows == [], case.name

    record = run_case(case)
    assert record["crash"] is None, record
    assert record["wrong_values"] == [], record
    assert record["phantom_anomalies"] == [], record
    assert record["false_nulls"], record
    for path in record["false_nulls"]:
        assert record["statuses"].get(path) in {UNVERIFIED, NULL}, (case.name, path, record)
    assert all(
        value is None for item in record["final"]["line_items"] for value in item.values()
    ), record
    for jargon in JARGON:
        assert jargon not in record["report"], (case.name, jargon)


BROKEN_ROW_ORDER = (
    "Item | Qty | P.U. | Amount\n"
    "2 | Installation | 617.28 | 1,234.56"
)


def test_a_number_in_the_description_column_is_never_accepted_as_a_description():
    assert not description_is_usable("2")
    assert not description_is_usable("1 234,56 €")
    assert description_is_usable("Installation")

    evidence = from_text(LAYOUT_HEADER + BROKEN_ROW_ORDER + LAYOUT_TOTALS)
    assert evidence.rows == []
    assert any("contenu numérique" in issue for issue in evidence.issues)


INVALID_MONETARY_DESCRIPTIONS = (
    "1 234,56 EUR",
    "USD 50",
    "GBP 757.97",
    "£1,234.56",
    "50 %",
)

VALID_DESCRIPTIONS = (
    "Manuel",
    "Installation",
    "Web development",
    "Article 2",
)

MONETARY_LINE = (
    '{"line_items":[{"description":"1 234,56 EUR","quantity":2,'
    '"unit_price":617.28,"total_ht":1234.56}]}'
)


@pytest.mark.parametrize("description", INVALID_MONETARY_DESCRIPTIONS)
def test_monetary_values_are_never_accepted_as_descriptions(description):
    assert not description_is_usable(description)


@pytest.mark.parametrize("description", VALID_DESCRIPTIONS)
def test_legitimate_text_descriptions_remain_accepted(description):
    assert description_is_usable(description)


def test_shifted_monetary_description_is_rejected_and_never_exported():
    shifted = (
        "Item | Qty | P.U. | Amount\n"
        "1 234,56 EUR | 2 | 617.28 | Installation"
    )
    evidence = from_text(LAYOUT_HEADER + shifted + LAYOUT_TOTALS)
    assert evidence.rows == []
    assert any("contenu numérique" in issue for issue in evidence.issues)

    shifted_result = reconcile(evidence, [read_model(MONETARY_LINE)])
    assert shifted_result.payload["line_items"] == [{
        "description": None, "quantity": None, "unit_price": None, "total_ht": None,
    }]
    assert shifted_result.statuses["line_items.0.description"] == NULL

    grounded = from_text(
        LAYOUT_HEADER
        + "Item | Qty | P.U. | Amount\nInstallation | 2 | 617.28 | 1,234.56"
        + LAYOUT_TOTALS
    )
    assert len(grounded.rows) == 1
    grounded.rows[0].description = "1 234,56 EUR"
    result = reconcile(grounded, [read_model(MONETARY_LINE)])
    line = result.payload["line_items"][0]
    assert line["description"] is None
    assert line["quantity"] == 2
    assert line["unit_price"] == 617.28
    assert line["total_ht"] == 1234.56
    assert result.statuses["line_items.0.description"] in {NULL, UNVERIFIED}


def test_grounding_never_exports_a_purely_numeric_description():
    # Defence in depth: even if a foreign snapshot still carries a numeric description,
    # grounding refuses it while keeping the numbers corroborated by the document.
    evidence = from_text(
        LAYOUT_HEADER + "Item | Qty | P.U. | Amount\nInstallation | 2 | 617.28 | 1,234.56"
        + LAYOUT_TOTALS
    )
    assert len(evidence.rows) == 1
    evidence.rows[0].description = "2"
    model = read_model(
        '{"line_items":[{"description":"2","quantity":2,"unit_price":617.28,"total_ht":1234.56}]}'
    )
    result = reconcile(evidence, [model])
    line = result.payload["line_items"][0]
    assert line["description"] is None
    assert line["quantity"] == 2
    assert line["unit_price"] == 617.28
    assert line["total_ht"] == 1234.56
    assert result.statuses["line_items.0.description"] in {NULL, UNVERIFIED}
