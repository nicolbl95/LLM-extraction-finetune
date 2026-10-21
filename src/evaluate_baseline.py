"""Evaluate a rule-based baseline extractor on data/test.jsonl."""

import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))

from schema import Invoice

TEST_PATH = ROOT / "data" / "test.jsonl"
REPORT_PATH = ROOT / "results" / "baseline_metrics.json"
FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc", "line_items")
TOLERANCE = 0.01
AMOUNT = r"\d+(?:[ \u202f]\d{3})*(?:[.,]\d+)?"
MONTHS_FR = {
    name: index
    for index, name in enumerate(
        [
            "janvier", "février", "mars", "avril", "mai", "juin", "juillet",
            "août", "septembre", "octobre", "novembre", "décembre",
        ],
        start=1,
    )
}
SUPPLIER_PATTERNS = (
    r"^Fournisseur[ \t]*:[ \t]*(.+)$",
    r"^Objet[ \t]*:[ \t]*Facture \S+ - (.+)$",
    r"Facture \S+ de (.+?) du ",
)
DATE_PATTERNS = (
    r"^Date[ \t]*:[ \t]*(.+)$",
    r" du (.+?)\. Articles",
    r"facture n° \S+ du ([^.\n]+)\.",
)
TOTAL_HT_PATTERNS = (rf"(?:Total|Montant) HT[ \t]*:?[ \t]*({AMOUNT})",)
TOTAL_TTC_PATTERNS = (rf"(?:Total|Montant) TTC[ \t]*:?[ \t]*({AMOUNT})",)
RAW_SEGMENT_PATTERN = r"Articles : (.*?)\. Total HT"
RAW_ITEM_PATTERN = rf"([^;]+?) x({AMOUNT}) PU ({AMOUNT}) (?:€|EUR) total ({AMOUNT}) (?:€|EUR)"
EMAIL_ITEM_PATTERN = rf"^- (.+?) : ({AMOUNT}) x ({AMOUNT}) (?:€|EUR) = ({AMOUNT}) (?:€|EUR)$"


def search(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.MULTILINE)


def first_group(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        match = search(pattern, text)
        if match:
            return match.group(1).strip()
    return None


def parse_amount(text: str | None) -> float | None:
    if not text:
        return None
    match = search(AMOUNT, text)
    if not match:
        return None
    return float(re.sub(r"[ \u202f]", "", match.group(0)).replace(",", "."))


def parse_date(text: str | None) -> str | None:
    if not text:
        return None
    if match := search(r"(\d{4})-(\d{2})-(\d{2})", text):
        return match.group(0)
    if match := search(r"(\d{2})/(\d{2})/(\d{4})", text):
        return f"{match[3]}-{match[2]}-{match[1]}"
    if match := search(r"(\d{1,2}) ([^\W\d_]+) (\d{4})", text):
        month = MONTHS_FR.get(match[2].lower())
        if month:
            return f"{match[3]}-{month:02d}-{int(match[1]):02d}"
    return None


def make_item(description: str, quantity: str, unit_price: str, total_ht: str) -> dict:
    return {
        "description": description.strip(),
        "quantity": parse_amount(quantity),
        "unit_price": parse_amount(unit_price),
        "total_ht": parse_amount(total_ht),
    }


def parse_line_items(text: str) -> list[dict]:
    items = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == 4 and parts[0] != "Désignation":
            items.append(make_item(*parts))
    segment = search(RAW_SEGMENT_PATTERN, text)
    if segment:
        for item in re.findall(RAW_ITEM_PATTERN, segment[1]):
            items.append(make_item(*item))
    for item in re.findall(EMAIL_ITEM_PATTERN, text, re.MULTILINE):
        items.append(make_item(*item))
    return items


def extract(text: str) -> dict:
    return {
        "supplier_name": first_group(SUPPLIER_PATTERNS, text),
        "invoice_date": parse_date(first_group(DATE_PATTERNS, text)),
        "total_ht": parse_amount(first_group(TOTAL_HT_PATTERNS, text)),
        "total_ttc": parse_amount(first_group(TOTAL_TTC_PATTERNS, text)),
        "line_items": parse_line_items(text),
    }


def close(pred: float | None, gold: float) -> bool:
    return pred is not None and abs(pred - gold) <= TOLERANCE


def item_correct(pred: dict, gold: dict) -> bool:
    return pred.get("description") == gold["description"] and all(
        close(pred.get(key), gold[key]) for key in ("quantity", "unit_price", "total_ht")
    )


def field_correct(field: str, pred, gold) -> bool:
    if pred is None:
        return False
    if field == "line_items":
        return len(pred) == len(gold) and all(item_correct(p, g) for p, g in zip(pred, gold))
    if isinstance(gold, float):
        return close(pred, gold)
    return pred == gold


def evaluate_record(record: dict) -> dict:
    raw = json.dumps(extract(record["input"]), ensure_ascii=False)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    try:
        Invoice.model_validate(payload)
        valid = True
    except ValidationError:
        valid = False
    fields = {
        field: field_correct(field, payload.get(field), record["output"][field])
        for field in FIELDS
    }
    return {"format": record["format"], "valid": valid, "fields": fields, "exact": all(fields.values())}


def rate(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def main() -> None:
    lines = TEST_PATH.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    results = [evaluate_record(record) for record in records]

    per_format = {}
    for fmt in sorted({r["format"] for r in results}):
        subset = [r for r in results if r["format"] == fmt]
        per_format[fmt] = {
            "num_examples": len(subset),
            "json_validity_rate": rate([r["valid"] for r in subset]),
            "exact_match_rate": rate([r["exact"] for r in subset]),
        }

    field_accuracy = {field: rate([r["fields"][field] for r in results]) for field in FIELDS}
    report = {
        "model": "rule_based_baseline",
        "split": "test",
        "num_examples": len(results),
        "json_validity_rate": rate([r["valid"] for r in results]),
        "field_accuracy": field_accuracy,
        "mean_field_accuracy": round(sum(field_accuracy.values()) / len(FIELDS), 4),
        "exact_match_rate": rate([r["exact"] for r in results]),
        "per_format": per_format,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
