"""Duration parsing for invoice tables whose header declares an hours column.

Deliberately isolated: nothing imports this module from the extraction pipeline yet. Callers must
pass explicit column context, so a clock time, a date, a reference code or a plain quantity can never
be reinterpreted by accident.

A parsed duration is hours as a Decimal rounded to six decimals. Monetary validation must quantize
its own products to cents; the parser never adjusts a duration to make an amount reconcile.
"""

from __future__ import annotations

import re
from decimal import Decimal

_PRECISION = Decimal("0.000001")
_MAX_HOURS = 24

_COLON = re.compile(r"^(\d{1,2}):([0-5]\d)$")
_DECIMAL_SEPARATOR = re.compile(r"^(\d{1,2})[.,](\d{2})$")
_COMPACT = re.compile(r"^(\d{2})([0-5]\d)$")


def parse_duration(raw: str | None, *, hours_column: bool, in_table: bool) -> Decimal | None:
    """Return the duration in hours, or None when the evidence does not justify one.

    Accepted only when `hours_column` is true and the value sits inside a table row:
      HH:MM, H:MM, HH.MM / HH,MM (OCR decimal variants) and HHMM (H hours + MM minutes).
    Rejected: dates, values above 24 h, minutes above 59, monetary strings, reference codes,
    plain decimals without exactly two fractional digits, and ordinary quantities.
    """
    if not hours_column or not in_table:
        return None
    text = (raw or "").strip()
    if not text:
        return None
    for pattern in (_COLON, _DECIMAL_SEPARATOR):
        match = pattern.fullmatch(text)
        if match:
            hours, minutes = int(match.group(1)), int(match.group(2))
            if hours > _MAX_HOURS:
                return None
            return _hours(hours, minutes)
    match = _COMPACT.fullmatch(text)
    if match:
        hours, minutes = int(match.group(1)), int(match.group(2))
        if not 0 < hours <= _MAX_HOURS:
            return None
        return _hours(hours, minutes)
    return None


def _hours(hours: int, minutes: int) -> Decimal:
    return (Decimal(hours) + Decimal(minutes) / Decimal(60)).quantize(_PRECISION)
