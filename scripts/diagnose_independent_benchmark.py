"""Read-only stage tracer for the frozen independent benchmark.

Observes the production pipeline (preprocessing -> Tesseract -> from_tsv ->
consensus -> grounding) and reports the earliest stage at which each missing
reference value was lost. Never imported by the application, never changes a
production decision, and never re-scores anything.
"""

import argparse
import json
import re
import sys
import time
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from unittest.mock import patch

import pytesseract

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from app import main
from evaluate_grounding import flatten
from invoice_evidence import from_tsv
from invoice_grounding import reconcile
from ocr_config import OCR_CONFIG

NUMBER_PATTERN = re.compile(
    r"\d{1,3}(?:[ \u00a0.,]\d{3})+(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,4})?"
)
MONTHS = [
    ("january", "janvier"), ("february", "fevrier"), ("march", "mars"), ("april", "avril"),
    ("may", "mai"), ("june", "juin"), ("july", "juillet"), ("august", "aout"),
    ("september", "septembre"), ("october", "octobre"), ("november", "novembre"),
    ("december", "decembre"),
]


def fold(text):
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[\s\u00a0'\-–—`´.:,]+", "", text).casefold()


def decimals(text):
    """Every numeric reading in *text*, with thousands separators resolved."""
    found = []
    for token in NUMBER_PATTERN.findall(text or ""):
        token = re.sub(r"[ \u00a0.,](?=\d{3}\b)", "", token.strip(" ,."))
        try:
            found.append(Decimal(token.replace(" ", "").replace(",", ".")))
        except InvalidOperation:
            continue
    return found


def number_seen(value, text):
    target = Decimal(str(value))
    return any(abs(found - target) <= Decimal("0.006") for found in decimals(text))


def date_seen(value, text):
    year, month, day = (int(value[:4]), int(value[5:7]), int(value[8:10]))
    folded = fold(text)
    forms = {f"{day:02d}{month:02d}{year}", f"{day}{month}{year}",
             f"{month:02d}{day:02d}{year}", f"{month}{day}{year}"}
    if any(form in folded for form in forms):
        return True
    names = MONTHS[month - 1]
    return any(f"{day}{name}{year}" in folded for name in names)


def gold_seen(field, value, text):
    if isinstance(value, (int, float, Decimal)):
        return number_seen(value, text)
    if field == "invoice_date":
        return date_seen(value, text)
    return fold(str(value)) in fold(text)


def trace(path):
    """Run the production pipeline once while recording every intermediate reading."""
    readings = []
    prepared_family = {}
    last_label = ["?"]

    def record_transform(*args, **kwargs):
        image, meta = original_transform(*args, **kwargs)
        prepared_family[id(image)] = meta["method"]
        return image, meta

    def record_from_tsv(data):
        evidence = from_tsv(data)
        evidence.family = last_label[0]
        readings.append(evidence)
        return evidence

    def reader(prepared, psm):
        last_label[0] = f"{prepared_family.get(id(prepared), 'crop')}|psm{psm}"
        return pytesseract.image_to_data(
            prepared, lang="fra+eng", config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
        )

    original_transform = main.recover_image.__globals__["transform"]
    started = time.perf_counter()
    with patch("ocr_recovery.transform", record_transform), patch("ocr_recovery.from_tsv", record_from_tsv):
        evidence = main.recover_image(main.open_image(str(path)), reader)
    elapsed = time.perf_counter() - started
    return evidence, readings, reconcile(evidence, [{}]).payload, elapsed


def structured_text(evidence):
    """Only what the pipeline turned into a usable candidate, never raw page text."""
    parts = [
        str(evidence.supplier or ""), str(evidence.supplier_raw or ""),
        str(evidence.invoice_date or ""), str(evidence.invoice_number or ""),
        *[str(number.value) for number in evidence.numbers],
        *[str(number.value) for values in evidence.totals.values() for number in values],
        *[str(tax.amount.value) for tax in evidence.taxes if tax.amount is not None],
    ]
    for row in evidence.rows:
        parts.append(str(row.description))
        parts.extend(str(number.value) for values in row.cells.values() for number in values)
    return "\n".join(parts)


def classify(gold, evidence, readings, payload):
    family_text = "\n".join(item.text for item in readings)
    family_candidates = "\n".join(structured_text(item) for item in readings)
    consensus_text = evidence.text
    consensus_lines = "\n".join(line.text for line in evidence.lines)
    consensus_candidates = structured_text(evidence)
    actual = flatten(payload)
    missing = []
    for field, value in gold.items():
        if value is None or actual.get(field) is not None:
            continue
        seen = lambda text: gold_seen(field, value, text)  # noqa: E731
        if not seen(family_text):
            stage = "1_never_recognized_by_any_family"
        elif not seen(consensus_lines):
            stage = "2_recognized_then_dropped_by_consensus_text"
        elif not seen(family_candidates):
            stage = "3_text_only_never_parsed_into_a_candidate"
        elif not seen(consensus_candidates):
            stage = "4_correct_family_candidate_lost_by_consensus"
        else:
            stage = "5_candidate_reached_grounding_and_was_nulled"
        missing.append({"field": field, "gold": str(value), "stage_lost": stage})
    return missing


def main_():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "real-scans")
    parser.add_argument("--truth", type=Path, default=ROOT / "real-scans-truth.json")
    parser.add_argument("--output", type=Path, default=ROOT / ".venv/diagnose-independent-benchmark.json")
    args = parser.parse_args()
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    report = {
        "purpose": "stage attribution of the 63 false nulls; production decisions untouched",
        "ocr_policy": OCR_CONFIG.version, "fingerprint": OCR_CONFIG.fingerprint(),
        "stage_labels": {
            "1_never_recognized_by_any_family": "no preprocessing family produced the value at all",
            "2_recognized_then_dropped_by_consensus_text": "a family saw it, the consensus page text does not contain it",
            "3_text_only_never_parsed_into_a_candidate": "in the consensus text but never became a labelled total, date, supplier or table cell",
            "4_correct_family_candidate_lost_by_consensus": "a family built the correct candidate; consensus discarded it",
            "5_candidate_reached_grounding_and_was_nulled": "the correct candidate survived to grounding and was removed",
        },
        "documents": [],
    }
    totals = {}
    for name in sorted(truth["documents"]):
        evidence, readings, payload, elapsed = trace(ROOT / "real-scans" / name)
        gold = flatten(truth["documents"][name])
        missing = classify(gold, evidence, readings, payload)
        for item in missing:
            totals[item["stage_lost"]] = totals.get(item["stage_lost"], 0) + 1
        report["documents"].append({
            "file": name,
            "seconds": round(elapsed, 1),
            "tesseract_calls": len(readings),
            "per_family": [
                {
                    "reading": item.family, "tokens": len(item.tokens), "lines": len(item.lines),
                    "rows": len(item.rows), "numbers": len(item.numbers),
                    "totals": {k: [str(n.value) for n in v] for k, v in item.totals.items()},
                    "supplier": item.supplier, "invoice_date": item.invoice_date,
                }
                for item in readings
            ],
            "consensus": {
                "lines": len(evidence.lines), "rows": len(evidence.rows),
                "numbers": len(evidence.numbers),
                "totals": {k: [str(n.value) for n in v] for k, v in evidence.totals.items()},
                "supplier": evidence.supplier, "supplier_raw": evidence.supplier_raw,
                "supplier_is_explicit": evidence.supplier_is_explicit,
                "invoice_date": evidence.invoice_date, "quality_issues": evidence.quality_issues,
            },
            "candidate_funnel": {
                "numbers_per_family": [len(item.numbers) for item in readings],
                "numbers_in_consensus": len(evidence.numbers),
                "rows_per_family": [len(item.rows) for item in readings],
                "rows_in_consensus": len(evidence.rows),
                "supplier_candidates_in_families": sorted({
                    str(item.supplier_raw or item.supplier) for item in readings
                    if (item.supplier_raw or item.supplier)
                }),
                "dates_in_families": sorted({
                    str(item.invoice_date) for item in readings if item.invoice_date
                }),
            },
            "payload": payload,
            "missing_fields": missing,
        })
        kept = sum(1 for k, v in gold.items() if v is not None and flatten(payload).get(k) is not None)
        gold_count = sum(1 for v in gold.values() if v is not None)
        print(f"{name}: kept {kept}/{gold_count} | rows={len(evidence.rows)} "
              f"| calls={len(readings)} | {elapsed:.1f}s")
        for item in missing:
            print(f"    {item['stage_lost'][0]} {item['field']} = {item['gold']}")
    report["stage_totals"] = totals
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("stage totals:", json.dumps(totals))
    print("written", args.output)


if __name__ == "__main__":
    main_()
