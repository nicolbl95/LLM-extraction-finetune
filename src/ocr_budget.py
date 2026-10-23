"""A shared per-document deadline, including all scanned PDF pages."""

import time
import math
from collections.abc import Callable

from invoice_evidence import UnreadableOCR
from ocr_config import OCR_CONFIG


class OCRBudgetExceeded(UnreadableOCR):
    pass


class ProcessingBudget:
    def __init__(self, seconds: float = OCR_CONFIG.total_seconds,
                 *, clock: Callable[[], float] = time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("OCR budget must be positive")
        self.seconds = seconds
        self.clock = clock
        self.started = clock()

    def remaining(self) -> float:
        remaining = self.seconds - (self.clock() - self.started)
        if remaining <= 0:
            raise OCRBudgetExceeded(
                f"Budget documentaire OCR de {self.seconds:g} s dépassé : "
                "preuves OCR insuffisantes ; aucun prétraitement supplémentaire."
            )
        return remaining

    @property
    def expired(self) -> bool:
        return self.clock() - self.started >= self.seconds
