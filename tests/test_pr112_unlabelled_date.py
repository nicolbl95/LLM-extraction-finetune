"""PR112 CAND-1: a standalone printed date with no date label anywhere is the document's issue date.

Future Graphics' frozen reads print

    clahe/psm6    "tax invoice invoice" | "09.01.2025" | "09.44.2025"
    scale3/psm6   "invoice tt"          | "09.01.2025"
    crop/psm6     "invoic je" | "due"   | "09.14.2025"

The label words are OCR garbage (`Invoice Date` reads as `sue` at confidence 3), so no invoice-kind reading
is ever built, `from_tsv` leaves `invoice_date` None, and consensus never sees the value - although the date
token itself is printed at confidence 94/95 in two distinct preprocessing families. Two general rules are
needed and neither alone is sufficient; both stay inside the existing corroboration budget:

* an unlabelled, self-standing numeric line becomes an invoice-kind reading, but only when the read carries
  no invoice label *and* no due/delivery/payment label, so a deadline can never be misread as the issue date;
* in the metadata agreement for the document date, a segmentation that read nothing is no evidence rather
  than a contradiction, so a blank read cannot silence a family that did read the value. The supplier name
  keeps the old veto - `test_abstention_corroborates_a_date_but_never_a_supplier_name`.

Synthetic and deterministic: no OCR, no model, no invoice data.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

TABLE = ["Logo design", "5", "200.00"]


def token(text, x, y, confidence=95.0):
    return Token(text, 0, (x, y, x + 8 + 6 * len(text), y + 16), confidence)


def read(lines, *, confidence=95.0):
    """One page: a dollar marker (so the printed order is month-first) then one metadata line each."""
    tokens = [token("$", 60, 20, confidence)]
    for index, text in enumerate(lines):
        tokens.append(token(text, 300, 60 + index * 30, confidence))
    return from_tokens(tokens, literal_columns=False)


def agreement(families):
    return build_consensus(families)


def payload(evidence):
    return reconcile(evidence, [{}])


def test_a_standalone_date_with_no_label_anywhere_is_read_as_the_issue_date():
    assert read(["Invoice", "09.01.2025", "09.44.2025"]).invoice_date == "2025-09-01"


def test_the_inferred_date_is_recovered_evidence_and_never_a_verified_one():
    result = payload(read(["Invoice", "09.01.2025", "09.44.2025"]))
    assert result.payload["invoice_date"] == "2025-09-01"
    assert result.statuses["invoice_date"] != "valide"


def test_a_date_inside_a_sentence_is_never_inferred():
    assert read(["Invoice", "Veuillez regler avant le 12.05.2024"]).invoice_date is None


def test_an_existing_label_wins_and_no_unlabelled_date_competes_with_it():
    assert read(["Invoice Date 01.02.2025", "09.01.2025"]).invoice_date == "2025-01-02"


def test_a_due_labelled_document_never_infers_its_issue_date_from_a_standalone_date():
    assert read(["Invoice", "Due", "09.14.2025"]).invoice_date is None


def test_two_distinct_standalone_dates_stay_unresolved():
    evidence = read(["Invoice", "09.01.2025", "09.05.2025"])
    assert evidence.invoice_date is None
    assert any("plusieurs dates" in issue for issue in evidence.issues)


def test_a_blank_segmentation_does_not_veto_a_date_corroborated_by_two_families():
    families = {
        family: [read(["Invoice", "09.01.2025", "09.44.2025"]), read(TABLE)]
        for family in ("deskew", "scale3")
    }
    combined = agreement(families)
    assert combined.invoice_date == "2025-09-01"
    assert payload(combined).payload["invoice_date"] == "2025-09-01"


def test_a_contradicting_segmentation_still_kills_its_family_vote():
    families = {
        "deskew": [read(["Invoice", "09.01.2025"]), read(["Invoice", "09.02.2025"])],
        "scale3": [read(["Invoice", "09.01.2025"]), read(["Invoice", "09.01.2025"])],
    }
    assert agreement(families).invoice_date is None


def test_one_family_alone_still_corroborates_nothing():
    families = {
        "deskew": [read(["Invoice", "09.01.2025"]), read(TABLE)],
        "scale3": [read(["Invoice"]), read(TABLE)],
    }
    assert agreement(families).invoice_date is None


def test_abstention_corroborates_a_date_but_never_a_supplier_name():
    """The two metadata fields are not alike, and the rule treats them differently.

    A date vote token *is* the exported ISO value, so two families that voted agree on the exported string.
    A name is folded for voting, so an accent- or border-contaminated spelling and a clean one share one
    token while their surface forms differ; letting a blank sibling abstain there exports the first spelling
    seen. `france/noise` measures exactly that: eight readings, three of which read no name at all, four
    carrying an accented or border-prefixed spelling and one spelling it correctly, and the strict gate is
    what keeps the misprint out.
    """
    named = read(["Supplier: Services Horizon SAS", "Invoice", "09.01.2025"])
    combined = agreement({"deskew": [read(TABLE), named], "scale3": [named, named]})
    assert combined.invoice_date == "2025-09-01"
    assert combined.supplier is None
