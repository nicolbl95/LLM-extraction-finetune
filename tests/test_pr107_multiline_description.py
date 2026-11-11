"""Multiline description reconstruction regressions (PR107 Phase 7).

Every test here pins the behaviour of unmodified production: the stage-5 continuation branch of
``invoice_evidence.from_tokens`` and the stage-7 continuation path of ``ocr_consensus.build_consensus``.
The twelve cases are the ones PR107 required before any reconstruction candidate could be retained.  Four
of them are the refusals that the two candidate relaxations measured in
``scripts/pr107_measure.py`` would have removed, so they double as the evidence for the KEEP-BASELINE
decision rather than as aspirational specifications.

Synthetic token geometry only; no Tesseract, no model, no invoice text.
"""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

HEADER = (0, [(0, "Description"), (240, "Quantity"), (340, "Unit price"), (480, "Amount")])
FIRST_ROW = (30, [(0, "Service A"), (240, "1"), (340, "400"), (480, "400")])
SECOND_ROW = (100, [(0, "Service B"), (240, "2"), (340, "30"), (480, "60")])


def positioned(rows, confidence=95):
    """Explicit synthetic geometry: one token per printed cell, 10 px line boxes."""
    return from_tokens([
        Token(text, 0, (x, y, x + max(8, len(text) * 5), y + 10), confidence)
        for y, cells in [HEADER] + rows for x, text in cells
    ])


def secondary(rows, confidence=95):
    return [(row.description, row.secondary_description) for row in positioned(rows, confidence).rows]


# --------------------------------------------------------------------- legitimate printed segments

def test_two_line_cell_merges_with_no_invented_punctuation():
    assert secondary([FIRST_ROW, (65, [(0, "Continuation A")])]) == [
        ("Service A", "Continuation A"),
    ]


def test_three_line_cell_merges_every_line_inside_the_spacing_ceiling():
    """Tight leading: the limit is geometric, not a depth of one."""
    assert secondary([FIRST_ROW, (50, [(0, "ligne deux")]), (70, [(0, "ligne trois")])]) == [
        ("Service A", "ligne deux ligne trois"),
    ]


def test_third_line_beyond_three_line_heights_is_not_reconstructed():
    """Documented ceiling, not a defect in the ordering: a line whose gap to its parent exceeds three
    parent line-heights is outside the cell, so production keeps it out of the description instead of
    guessing.  The same text at tight leading is merged by the test above."""
    description, attached = secondary([FIRST_ROW, (65, [(0, "ligne deux")]), (100, [(0, "ligne trois")])])[0]
    assert (description, attached) == ("Service A", "ligne deux")


def test_long_wrapped_description_keeps_every_printed_token():
    rows = [
        (30, [(0, "Portrait de decoration peint a la main sur toile"), (240, "1"), (340, "83.33"), (480, "83.33")]),
        (65, [(0, "format 20x10cm avec cadre")]),
    ]
    description, attached = secondary(rows)[0]
    assert description == "Portrait de decoration peint a la main sur toile"
    assert attached == "format 20x10cm avec cadre"


def test_adjacent_similar_products_each_keep_their_own_continuation():
    rows = [
        (30, [(0, "Portrait 20x10"), (240, "1"), (340, "25"), (480, "25")]),
        (65, [(0, "cadre bois")]),
        (100, [(0, "Portrait 20x10"), (240, "2"), (340, "30"), (480, "60")]),
        (135, [(0, "cadre metal")]),
    ]
    assert secondary(rows) == [("Portrait 20x10", "cadre bois"), ("Portrait 20x10", "cadre metal")]


def test_continuation_attaches_only_to_the_row_directly_above_it():
    """A continuation under row 2 must not be appended to row 1, whose own cell is complete."""
    assert secondary([FIRST_ROW, SECOND_ROW, (135, [(0, "continuation B")])]) == [
        ("Service A", ""), ("Service B", "continuation B"),
    ]


def test_printed_reference_code_is_carried_verbatim_and_creates_no_row():
    description, attached = secondary([FIRST_ROW, (65, [(0, "Ref. A123")])])[0]
    assert (description, attached) == ("Service A", "Ref. A123")


def test_legitimate_punctuation_and_accents_survive_the_join():
    description, attached = secondary([FIRST_ROW, (65, [(0, "décoration, en bois; 20x10")])])[0]
    assert attached == "décoration, en bois; 20x10"


def test_row_with_empty_quantity_keeps_its_continuation_and_null_quantity():
    rows = [(30, [(0, "Service A"), (340, "400"), (480, "400")]), (65, [(0, "continuation")])]
    evidence = positioned(rows)
    assert [row.secondary_description for row in evidence.rows] == ["continuation"]
    assert not evidence.rows[0].cells["quantity"]


def test_row_without_unit_price_keeps_its_continuation():
    rows = [(30, [(0, "Service A"), (240, "1"), (480, "400")]), (65, [(0, "continuation")])]
    evidence = positioned(rows)
    assert [row.secondary_description for row in evidence.rows] == ["continuation"]
    assert not evidence.rows[0].cells["unit_price"]


# -------------------------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("line", [
    "IBAN FR76 1234 5678",
    "Thank you for your business",
    "Delivery date 2026-01-05",
    "Total 400",
])
def test_totals_payment_and_footer_lines_are_never_continuations(line):
    """The printed line below the last row belongs to the document, not to the row: a totals line carries
    values, and CONTINUATION_EXCLUDED rejects account, thanks and delivery wording."""
    assert secondary([FIRST_ROW, (65, [(0, line)])]) == [("Service A", "")]


def test_decorative_separator_line_is_not_a_continuation():
    assert secondary([FIRST_ROW, (65, [(0, "----------")])]) == [("Service A", "")]


def test_a_table_header_repeated_below_a_row_is_not_a_continuation():
    assert secondary([FIRST_ROW, (65, [(0, "Description"), (240, "Quantity"),
                                       (340, "Unit price"), (480, "Amount")])]) == [("Service A", "")]


def test_continuation_never_absorbs_a_line_that_prints_numbers():
    assert secondary([FIRST_ROW, (65, [(0, "lot de 2"), (240, "1")])]) == [("Service A", "")]


def test_low_confidence_continuation_is_refused():
    """The guard PR107 config B2 would have removed: merging a conf-40 read would export OCR noise as
    printed text, which is the one thing the description contract forbids."""
    assert secondary([FIRST_ROW, (65, [(0, "continuation")])], confidence=40) == [("Service A", "")]


def test_replacement_character_continuation_is_refused():
    assert secondary([FIRST_ROW, (65, [(0, "conta\ufffdination")])]) == [("Service A", "")]


def test_continuation_indented_into_a_neighbouring_column_is_refused():
    """The guard PR107 config B1 would have widened from 8 to 24 px.  At 20 px the line no longer starts
    under the description column, so production keeps it out of the cell."""
    rows = [FIRST_ROW, (65, [(0, "Service B"), (240, "2"), (340, "30"), (480, "60")]),
            (100, [(20, "continuation de B")])]
    assert positioned(rows).rows[1].secondary_description == ""


def test_large_vertical_gap_is_refused():
    assert secondary([FIRST_ROW, (200, [(0, "continuation")])]) == [("Service A", "")]


# --------------------------------------------------------------------------------- end-to-end export

def test_exported_rows_carry_the_reconstruction_with_numbers_untouched():
    evidence = positioned([FIRST_ROW, (65, [(0, "Continuation A")]),
                           SECOND_ROW, (135, [(0, "Continuation B")])])
    consensus = build_consensus({"gray": [copy.deepcopy(evidence)], "contrast": [copy.deepcopy(evidence)]})
    rows = reconcile(consensus, []).payload["line_items"]
    assert rows == [
        {"description": "Service A Continuation A", "quantity": 1.0, "unit_price": 400.0, "total_ht": 400.0},
        {"description": "Service B Continuation B", "quantity": 2.0, "unit_price": 30.0, "total_ht": 60.0},
    ]
