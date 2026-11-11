"""Descriptions are only ever exported when two families corroborate them - the accepted baseline's invariants.

Four of PR113's nine grid-edge tests assert nothing the owner declined: they pin when a line description is
*withheld*, which is exactly what PR112 Baseline B (the accepted production behaviour) does. PR114 Phase 2 keeps
them verbatim in the active regression suite and moves the five that assert the rejected candidate's recovery
behaviour to `results/product-review113/experimental/tests/test_pr113_grid_edge_description.py`, where they stay
runnable against the preserved `m4` source. Decision of record: `KEEP_PR112_BASELINE - R4 REJECTED` (2026-11-10).

Éditions Mirada prints its items inside a ruled table. The vertical rule beside the description column reaches
Tesseract as a character glued to the word it touches, so the same printed title arrives as

    deskew  "la cuisine de tous les jours"   (line confidence above the floor)
    scale3  "|la cuisine de tous les jours"  (the border token itself reads at confidence 0-20)

Baseline B silences the second family and requires a *folded* spelling the two families share, so one line loses
both its description and the row identity its printed total already carried. The owner declined changing that.
What this module does keep is the safety side of the rule: a description never reaches the payload on one
family's word alone, and never reaches it at all if every spelling carries a grid mark.

Synthetic and deterministic: no OCR, no model, no invoice data, no document-specific literal.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import from_text
from invoice_grounding import read_model, reconcile
from ocr_consensus import build_consensus

TABLE = (
    "Supplier: Acme Ltd\nInvoice date: 2026-01-01\nCurrency: GBP\n"
    "Description | Quantity | Net unit price | Net amount\n"
    "Clavier sans fil | 2 | 10 | 20\n"
    "Subtotal: 20\nVAT (0%): 0.00\nInvoice total: 20"
)
PRINTED = "Clavier sans fil"
GLUED = "|Clavier sans fil|"


def family(*, description=PRINTED, border_noise=False, item="Clavier sans fil"):
    """One preprocessing family's read of a ruled table line.

    `border_noise` is the printed rule: `from_text` puts the column separators in as their own tokens, so
    demoting them reproduces the low-confidence grid marks a real scan reports on the same OCR line.
    """
    evidence = from_text(TABLE.replace(PRINTED, item), locale="en-GB")
    row = evidence.rows[0]
    row.description = description
    if border_noise:
        row.line.tokens = [
            replace(token, confidence=12.0) if token.text == "|" else token for token in row.line.tokens
        ]
    return evidence


def vote(*families):
    readings = {f"family{index}": [evidence] for index, evidence in enumerate(families)}
    return build_consensus(readings, confidence=40).rows[0]


def consensus(*families):
    readings = {f"family{index}": [evidence] for index, evidence in enumerate(families)}
    return build_consensus(readings, confidence=40)


def test_a_description_only_ever_read_with_a_bar_attached_is_not_printed():
    row = vote(family(description=GLUED), family(description="||Clavier sans fil"))
    assert row.description == ""
    assert row.description_confident is False


def test_the_grid_tolerance_does_not_merge_two_different_words():
    assert vote(family(), family(description="|Clavier sans fixe|")).description == ""


def test_one_family_alone_still_establishes_nothing():
    assert vote(family()).description == ""


def test_two_families_that_agree_exactly_are_unchanged_and_reach_the_payload():
    item = reconcile(consensus(family(), family()), [read_model('{"line_items":[]}')]).payload["line_items"][0]
    assert (item["description"], item["quantity"], item["unit_price"], item["total_ht"]) == (
        PRINTED, 2, 10, 20,
    )
