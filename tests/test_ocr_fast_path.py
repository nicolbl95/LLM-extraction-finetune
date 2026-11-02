import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import from_text
from ocr_fast_path import FLAG_NAME, FastPathOutcome, attempt_fast_path, fast_path_enabled

COMPLETE = (
    "Invoice: FR-001\nSupplier: Acme Ltd\nInvoice date: 2026-01-01\n"
    "Due date: 2026-02-01\nCurrency: EUR\n"
    "Description | Quantity | Unit price | Total HT\n"
    "Service | 2 | 10 | 20\n"
    "Total HT: 20.00\nVAT (0%): 0.00\nTotal TTC: 20.00"
)


def _prepared():
    return {
        "deskew": Image.new("RGB", (100, 100), "white"),
        "clahe": Image.new("RGB", (100, 100), "white"),
    }


def _read(image, psm, label):
    return from_text("")


def test_fast_path_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv(FLAG_NAME, raising=False)
    assert not fast_path_enabled()


def test_fast_path_accepts_explicit_truthy_values(monkeypatch):
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(FLAG_NAME, value)
        assert fast_path_enabled()


def test_fast_path_treats_other_values_as_disabled(monkeypatch):
    for value in ("", "0", "false", "off", "random"):
        monkeypatch.setenv(FLAG_NAME, value)
        assert not fast_path_enabled()


def test_fast_path_accepts_only_a_fully_verified_document():
    outcome = attempt_fast_path(
        [from_text(COMPLETE), from_text(COMPLETE)], _prepared(), _read,
    )

    assert outcome.accepted
    assert outcome.reason == "eligible"
    assert outcome.invoice == "FR-001"
    assert outcome.evidence is not None
    assert outcome.record(1.25) == {
        "enabled": True,
        "accepted": True,
        "reason": "eligible",
        "calls": 2,
        "elapsed_seconds": 1.25,
    }


def test_fast_path_rejects_an_unverified_document_despite_agreeing_readings():
    incomplete = COMPLETE.replace("Due date: 2026-02-01\n", "")
    outcome = attempt_fast_path(
        [from_text(incomplete), from_text(incomplete)], _prepared(), _read,
    )

    assert not outcome.accepted
    assert "not verified" in outcome.reason
    assert outcome.evidence is not None


def test_outcome_record_counts_header_reads():
    outcome = FastPathOutcome(accepted=False, reason="header failed", header_reads=2)
    assert outcome.record(0.5)["calls"] == 4
