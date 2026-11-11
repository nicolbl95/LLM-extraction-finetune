"""PR97 duration parser: the 19 PR96 cases plus ambiguous-context negatives."""

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from duration_parser import parse_duration


def hours(raw):
    return parse_duration(raw, hours_column=True, in_table=True)


def test_printed_durations():
    assert hours("03:00") == Decimal("3.000000")
    assert hours("05:00") == Decimal("5.000000")
    assert hours("06:30") == Decimal("6.500000")
    assert hours("04:20") == Decimal("4.333333")
    assert hours("02:00") == Decimal("2.000000")


def test_ocr_decimal_variants():
    assert hours("03.00") == Decimal("3.000000")
    assert hours("02.00") == Decimal("2.000000")
    assert hours("03,00") == Decimal("3.000000")


def test_compact_hhmm_requires_two_digit_hours():
    assert hours("0500") == Decimal("5.000000")
    assert hours("0630") == Decimal("6.500000")
    assert hours("2400") == Decimal("24.000000")
    assert hours("0000") is None


def test_rejections_keep_durations_separate_from_other_fields():
    for raw in ("2025-01-09", "1.234", "REF-001", "630", "25:00", "03:75", "$40", "40", "", None):
        assert hours(raw) is None, raw


def test_context_is_required():
    assert parse_duration("03:00", hours_column=False, in_table=True) is None
    assert parse_duration("03:00", hours_column=True, in_table=False) is None
    assert parse_duration("03:00", hours_column=False, in_table=False) is None


def test_ambiguous_forms_rejected():
    assert hours("3:5") is None
    assert hours("03:0") is None
    assert hours("3:00:00") is None
    assert hours("+3:00") is None
    assert hours("-03:00") is None
    assert hours("03 h") is None
    assert hours("3.5h") is None
    assert hours("4:20:00") is None


def test_boundaries():
    assert hours("24:00") == Decimal("24.000000")
    assert hours("24:01") == Decimal("24.016667")  # documented rule: hours <= 24, minutes <> 0
    assert hours("25:00") is None
    assert hours("00:30") == Decimal("0.500000")
    assert hours("0:30") == Decimal("0.500000")
    assert hours("00:59") == Decimal("0.983333")


def test_rounding_does_not_break_cents_arithmetic():
    duration = hours("04:20")
    assert duration == Decimal("4.333333")
    product = (duration * Decimal(25)).quantize(Decimal("0.01"))
    assert product == Decimal("108.33")
    assert product != Decimal("112.50")  # the printed row-4 discrepancy is preserved


def test_module_has_no_pipeline_side_effects():
    import invoice_evidence

    before_headers = dict(invoice_evidence.HEADERS)
    import importlib

    importlib.reload(sys.modules["duration_parser"])
    assert invoice_evidence.HEADERS == before_headers
