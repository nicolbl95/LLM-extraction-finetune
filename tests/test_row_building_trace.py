"""PR84 row-building diagnosis: focused deterministic tests.

Every test here is offline — no Tesseract, no model, no frozen private fixture. They pin the
three claims PR84 rests on: the header gate is what stops a French hourly table from becoming
rows, the research probe changes exactly that and nothing else, the recorded-input replay used
by the trace really is the production parser, and the PR84 outcome metric counts a value that
contradicts a null reference as unsupported rather than as a gain.
"""

import contextlib
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import invoice_evidence
from invoice_evidence import Token, from_tokens, from_tsv

_spec = importlib.util.spec_from_file_location("pr84_trace", ROOT / "scripts" / "trace_row_building.py")
trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trace)

HEADER_Y, ROW_Y = 100, 122
COLUMN_X = {"quantity": 60, "description": 120, "unit_price": 250, "total_ht": 420}


def token(text, x, y, width=None, confidence=95.0):
    return Token(text, 0, (x, y, x + (width or 12 + 5 * len(text)), y + 14), confidence)


def hourly_tokens():
    """A time-billed French table: HEURES | DESCRIPTION | €/HEURE | MONTANT, two items."""
    header = [
        token("HEURES", COLUMN_X["quantity"], HEADER_Y),
        token("DESCRIPTION", COLUMN_X["description"], HEADER_Y),
        token("€/HEURE", COLUMN_X["unit_price"], HEADER_Y),
        token("MONTANT", COLUMN_X["total_ht"], HEADER_Y),
    ]
    rows = [
        [token("2", COLUMN_X["quantity"], ROW_Y, 12), token("Prestation conseil", COLUMN_X["description"], ROW_Y),
         token("35,00", COLUMN_X["unit_price"], ROW_Y, 40), token("70,00", COLUMN_X["total_ht"], ROW_Y, 40)],
        [token("5", COLUMN_X["quantity"], ROW_Y + 20, 12), token("Atelier equipe", COLUMN_X["description"], ROW_Y + 20),
         token("25,00", COLUMN_X["unit_price"], ROW_Y + 20, 40), token("125,00", COLUMN_X["total_ht"], ROW_Y + 20, 48)],
    ]
    return header + [item for row in rows for item in row]


def test_hourly_table_builds_no_rows_under_the_pre_pr85_vocabulary():
    """The measured PR84 cause, pinned against the vocabulary that produced it.

    PR85 integrated the hourly labels into production, so this assertion is scoped to the
    pre-PR85 `HEADERS` (the values at 8598a8f) instead of being dropped: the header gate really
    did never open for these labels, and `tests/test_hourly_header_integration.py` pins that it
    now does. The assertion itself is unchanged.
    """
    pre_pr85 = {
        **invoice_evidence.HEADERS,
        "quantity": r"\b(?:quantite|qte|quantity|units|qty)\b",
        "unit_price": (
            r"(?:\bp\s*\.?\s*u\s*\.?(?=\s|$)"
            r"|\bpu\b(?:\s+(?:net|ht))*(?=\s|$)"
            r"|\bp\.u\.(?:\s+(?:net|ht))*(?=\s|$)"
            r"|\bprix\s+unit(?:aire)?\.?(?:\s+(?:net|ht))*(?=\s|$)"
            r"|\bnet\s+unit(?:\s+price)?\b|\bunit\s+price\b|\bhourly\s+rate\b)"
        ),
    }
    with patch.object(invoice_evidence, "HEADERS", pre_pr85):
        evidence = from_tokens(hourly_tokens(), literal_columns=False)
        assert evidence.rows == []
        header_line = next(line for line in evidence.lines if "MONTANT" in line.text.upper())
        facts = trace.header_predicates(header_line)
    assert facts["accepted_as_header"] is False
    assert facts["missing"] == ["quantity", "unit_price"]
    assert facts["contextual_rate_header"] is False


def test_pr84_hourly_probe_recovers_exactly_the_printed_cells():
    stack = contextlib.ExitStack()
    try:
        for entered in trace.VARIANTS["hourly_vocabulary"]["patches"][0]:
            stack.enter_context(entered)
        evidence = from_tokens(hourly_tokens(), literal_columns=False)
    finally:
        stack.close()
    assert len(evidence.rows) == 2
    assert [row.cells["quantity"][0].value for row in evidence.rows] == [Decimal(2), Decimal(5)]
    assert [row.cells["unit_price"][0].value for row in evidence.rows] == [Decimal("35.00"), Decimal("25.00")]
    assert [row.cells["total_ht"][0].value for row in evidence.rows] == [Decimal("70.00"), Decimal("125.00")]
    assert [row.description for row in evidence.rows] == ["Prestation conseil", "Atelier equipe"]


def test_probe_patches_are_confined_to_the_research_process():
    headers_before = dict(invoice_evidence.HEADERS)
    optional_before = dict(invoice_evidence.OPTIONAL_HEADERS)
    for name in ("unit_price_vocabulary", "hourly_vocabulary", "all_printed_labels", "tax_aware_labels"):
        with contextlib.ExitStack() as stack:
            for entered in trace.VARIANTS[name]["patches"][0]:
                stack.enter_context(entered)
            assert (invoice_evidence.HEADERS != headers_before
                    or invoice_evidence.OPTIONAL_HEADERS != optional_before)
        assert invoice_evidence.HEADERS == headers_before
        assert invoice_evidence.OPTIONAL_HEADERS == optional_before


def test_recorded_input_replay_is_the_production_parser():
    data = {
        "text": ["Description", "Quantité", "Prix", "Montant", "Café", "2", "3,50", "7,00", ""],
        "conf": ["95", "95", "95", "95", "95", "95", "95", "95", "-1"],
        "left": ["10", "60", "120", "200", "10", "60", "120", "200", "0"],
        "top": ["10"] * 4 + ["30"] * 4 + ["50"],
        "width": ["40"] * 4 + ["30"] * 4 + ["5"],
        "height": ["14"] * 8 + ["1"],
    }
    record = {"kind": "tsv", "tsv": data, "kwargs": {}}
    replayed = trace.replay_from_capture(record)
    production = from_tsv({key: list(values) for key, values in data.items()})
    assert trace.candidate_signature(replayed.rows) == trace.candidate_signature(production.rows)
    assert replayed.text == production.text
    assert trace.recorded_candidate_signature(
        [trace.serialize_row_candidate(row, confidence=40, index=index)
         for index, row in enumerate(production.rows)]
    ) == trace.candidate_signature(production.rows)


def test_value_contradicting_a_null_reference_is_never_counted_as_a_gain():
    reference = {"line_items": [
        {"description": "Prestation", "quantity": 2, "unit_price": 35.0, "total_ht": None},
        {"description": "Atelier", "quantity": 5, "unit_price": 25.0, "total_ht": None},
    ]}
    payload = {"line_items": [
        {"description": "Prestation", "quantity": 2.0, "unit_price": 35.0, "total_ht": 70.0},
        {"description": "Autre ligne", "quantity": None, "unit_price": None, "total_ht": 99.0},
    ]}
    outcomes = trace.cell_outcomes(payload, reference)
    assert len(outcomes["cells"]["total_ht"]["contradicts_null_reference"]) == 1
    assert outcomes["cells"]["total_ht"]["correct"] == []
    assert outcomes["cells"]["quantity"]["correct"][0]["gold_index"] == 0
    assert [row["export_index"] for row in outcomes["unsupported_exported_rows"]] == [1]
    totals = trace.variant_totals(trace.score(payload, reference), {
        field: {outcome: len(items) for outcome, items in lists.items()}
        for field, lists in outcomes["cells"].items()})
    assert totals["cells_contradicting_null_reference"] == 1
    assert totals["cells_incorrect"] == 0
