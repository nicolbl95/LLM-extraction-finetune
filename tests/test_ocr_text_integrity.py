import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import description_is_usable, from_text, has_replacement_character
from invoice_grounding import (
    NULL, RECOVERED, UNVERIFIED, VALID, read_model, reconcile,
)
from ocr_consensus import build_consensus
from ocr_fast_path import attempt_fast_path

CORRUPT = "Cr\ufffdation de logo"
CLEAN = "Création de logo"
UNACCENTED = "Creation de logo"
BASE = (
    "Invoice: FR-001\nSupplier: Acme Ltd\nInvoice date: 2026-01-01\n"
    "Due date: 2026-02-01\nCurrency: EUR\nDescription | Quantity | Unit price | Total HT\n"
    "{description} | 1 | 100 | 100\nTotal HT: 100\nVAT (0%): 0\nTotal TTC: 100"
)


def evidence(description):
    return from_text(BASE.format(description=description))


def corrupt_evidence():
    item = evidence(CLEAN)
    item.rows[0].description = CORRUPT
    item.rows[0].description_confident = True
    return item


def empty_result(item):
    return reconcile(item, [read_model('{"line_items":[]}')])


def prepared():
    return {
        "deskew": Image.new("RGB", (100, 100), "white"),
        "clahe": Image.new("RGB", (100, 100), "white"),
    }


def empty_read(image, psm, label):
    return from_text("")


def test_clean_accented_description_is_exported():
    item = evidence(CLEAN)
    assert description_is_usable(CLEAN)
    assert not has_replacement_character(CLEAN)
    result = empty_result(item)
    assert result.payload["line_items"][0]["description"] == CLEAN
    assert result.statuses["line_items.0.description"] in {VALID, RECOVERED}


def test_corrupted_description_is_never_exported_or_verified():
    assert not description_is_usable(CORRUPT)
    assert has_replacement_character(CORRUPT)
    item = corrupt_evidence()
    result = empty_result(item)
    assert result.payload["line_items"][0]["description"] is None
    assert result.statuses["line_items.0.description"] in {NULL, UNVERIFIED}
    assert CORRUPT not in json.dumps(result.payload, ensure_ascii=False)
    assert "\ufffd" not in json.dumps(result.payload, ensure_ascii=False)


def test_from_text_drops_a_corrupted_description_row():
    item = from_text(BASE.format(description=CORRUPT))
    assert not item.rows
    assert any("U+FFFD" in issue for issue in item.issues)


def test_conflicting_accented_and_unaccented_readings_keep_the_accent():
    consensus = build_consensus(
        {"deskew": [evidence(CLEAN)], "clahe": [evidence(UNACCENTED)]},
        confidence=40,
    )
    assert consensus.rows[0].description == CLEAN
    assert consensus.rows[0].description_confident


def test_corruption_in_one_family_is_not_selected():
    consensus = build_consensus(
        {
            "deskew": [evidence(CLEAN)],
            "clahe": [evidence(CLEAN)],
            "denoise": [corrupt_evidence()],
        },
        confidence=40,
    )
    assert consensus.rows[0].description == CLEAN
    assert consensus.rows[0].description_confident


def test_corruption_in_every_family_is_unverified():
    consensus = build_consensus(
        {"deskew": [corrupt_evidence()], "clahe": [corrupt_evidence()]},
        confidence=40,
    )
    assert consensus.rows[0].description == ""
    assert not consensus.rows[0].description_confident
    result = empty_result(consensus)
    assert result.payload["line_items"][0]["description"] is None
    assert result.statuses["line_items.0.description"] in {NULL, UNVERIFIED}


def test_fast_gate_rejects_a_corrupted_description():
    outcome = attempt_fast_path(
        [corrupt_evidence(), corrupt_evidence()], prepared(), empty_read,
    )
    assert not outcome.accepted
    assert "description" in outcome.reason or "replacement" in outcome.reason
    assert outcome.result is not None
    assert outcome.result.payload["line_items"][0]["description"] is None
    assert "\ufffd" not in json.dumps(outcome.result.payload, ensure_ascii=False)


def test_fast_gate_still_accepts_a_clean_accented_description():
    outcome = attempt_fast_path(
        [evidence(CLEAN), evidence(CLEAN)], prepared(), empty_read,
    )
    assert outcome.accepted
    assert outcome.result.payload["line_items"][0]["description"] == CLEAN


def test_reference_data_preserves_the_accent_when_read_as_utf8():
    rows = [
        json.loads(line)
        for line in (ROOT / "data/test.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    description = next(
        row for row in rows if row["id"] == "invoice-0406"
    )["output"]["line_items"][3]["description"]
    assert description == CLEAN
    assert not has_replacement_character(description)
