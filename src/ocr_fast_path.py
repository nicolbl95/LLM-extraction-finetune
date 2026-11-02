"""Feature-flagged four-call OCR early exit; disabled unless explicitly enabled."""

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass, field

from invoice_evidence import DocumentEvidence, has_replacement_character, invoice_number_from_text
from invoice_grounding import RECOVERED, VALID, read_model, reconcile
from ocr_consensus import build_consensus
from ocr_config import OCR_CONFIG
from ocr_quality import assess_ocr, signature

FLAG_NAME = "INVOICE_OCR_FAST_PATH"
SIGNATURE_ISSUE = (
    "Les lectures OCR indépendantes (segmentations 4 et 6) ne concordent pas."
)
_TRUTHY = {"1", "true", "yes", "on"}


def fast_path_enabled() -> bool:
    return os.environ.get(FLAG_NAME, "").strip().casefold() in _TRUTHY


@dataclass
class FastPathOutcome:
    accepted: bool
    reason: str
    evidence: DocumentEvidence | None = None
    result: object | None = None
    invoice: str | None = None
    header_reads: int = 0
    header_candidates: list[str | None] = field(default_factory=list)

    def record(self, elapsed_seconds: float) -> dict:
        return {
            "enabled": True,
            "accepted": self.accepted,
            "reason": self.reason,
            "calls": 2 + self.header_reads,
            "elapsed_seconds": elapsed_seconds,
        }


def _normalized_signature(evidence: DocumentEvidence, invoice: str | None) -> tuple:
    snapshot = copy.deepcopy(evidence)
    snapshot.invoice_number = invoice
    return signature(snapshot)


def attempt_fast_path(
    first_readings: list[DocumentEvidence],
    prepared_images: dict,
    read: Callable[[object, int, str], DocumentEvidence],
) -> FastPathOutcome:
    checked = assess_ocr(copy.deepcopy(first_readings[0]), copy.deepcopy(first_readings[1]))
    issues = list(checked.quality_issues)
    numbers = [evidence.invoice_number for evidence in first_readings]
    header_needed = len(set(numbers)) != 1 or not all(numbers)
    invoice = numbers[0] if numbers[0] and numbers[0] == numbers[1] else None
    # Header reads cannot rescue any other quality issue; skip them to keep the
    # rejected path inside the same OCR time budget as the full pipeline.
    if issues and issues != [SIGNATURE_ISSUE]:
        return FastPathOutcome(
            accepted=False,
            reason="fast quality issues: " + "; ".join(issues),
            invoice=invoice,
        )
    header_confirmed = None
    header_reads = 0
    candidates = []
    if header_needed:
        for family in ("deskew", "clahe"):
            crop = prepared_images[family].crop(
                (0, 0, prepared_images[family].width,
                 max(1, int(prepared_images[family].height * .22)))
            )
            header = read(crop, 11, f"header/{family}")
            header_reads += 1
            candidates.append(header.invoice_number or invoice_number_from_text(header.text))
        if len(candidates) == 2 and candidates[0] and candidates[0] == candidates[1]:
            header_confirmed = candidates[0]

    invoice = header_confirmed or (
        numbers[0] if numbers[0] and numbers[0] == numbers[1] else None
    )
    if issues and (
        issues != [SIGNATURE_ISSUE]
        or not header_confirmed
        or len(set(numbers)) == 1
    ):
        return FastPathOutcome(
            accepted=False,
            reason="fast quality issues: " + "; ".join(issues),
            invoice=invoice,
            header_reads=header_reads,
            header_candidates=candidates,
        )

    if _normalized_signature(first_readings[0], invoice) != _normalized_signature(
        first_readings[1], invoice
    ):
        return FastPathOutcome(
            accepted=False,
            reason="fast readings differ beyond invoice number",
            invoice=invoice,
            header_reads=header_reads,
            header_candidates=candidates,
        )

    fast = build_consensus(
        {"deskew": [first_readings[0]], "clahe": [first_readings[1]]},
        confidence=OCR_CONFIG.confidence,
    )
    if invoice:
        fast.invoice_number = invoice
    result = reconcile(fast, [read_model('{"line_items":[]}')])
    failures = []
    if not fast.rows:
        failures.append("no table rows")
    if not invoice:
        failures.append("invoice number not corroborated")
    if any(not row.description_confident for row in fast.rows):
        failures.append("description not confirmed")
    if any(has_replacement_character(row.description) for row in fast.rows):
        failures.append("replacement character in description")
    if any(
        len({number.value for number in row.cells.get(key, [])}) != 1
        for row in fast.rows
        for key in ("quantity", "unit_price", "total_ht")
    ):
        failures.append("numeric cells not uniquely corroborated")
    if result.failed_rows:
        failures.append(f"failed rows: {result.failed_rows}")
    bad_statuses = {
        path: status for path, status in result.statuses.items()
        if status not in {VALID, RECOVERED}
    }
    if bad_statuses:
        failures.append(f"unverified fields: {bad_statuses}")
    anomalies = [
        flag.code for flag in result.validation.flags if flag.severity != "information"
    ]
    if anomalies:
        failures.append(f"validation flags: {anomalies}")
    if result.validation.not_verified:
        failures.append(f"not verified: {result.validation.not_verified}")
    if any(
        value is None
        for item in result.payload["line_items"]
        for value in item.values()
    ):
        failures.append("line item contains null")
    if any(
        result.payload[key] is None
        for key in ("supplier_name", "invoice_date", "total_ht", "total_ttc")
    ):
        failures.append("document field contains null")
    if not result.payload["line_items"]:
        failures.append("no exported line items")
    exported_strings = [
        result.payload[key] for key in ("supplier_name", "invoice_date")
    ] + [
        value for item in result.payload["line_items"] for value in item.values()
    ]
    if any(
        isinstance(value, str) and has_replacement_character(value)
        for value in exported_strings
    ):
        failures.append("replacement character in exported field")

    return FastPathOutcome(
        accepted=not failures,
        reason="eligible" if not failures else "; ".join(failures),
        evidence=fast,
        result=result,
        invoice=invoice,
        header_reads=header_reads,
        header_candidates=candidates,
    )
