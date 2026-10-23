"""Measure final grounded exports, independently from raw-model benchmark scores."""

import argparse
import copy
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from invoice_evidence import from_text, from_tsv
from invoice_grounding import read_model, reconcile


def flatten(payload: dict) -> dict:
    fields = {key: payload.get(key) for key in ("supplier_name", "invoice_date", "total_ht", "total_ttc")}
    for index, item in enumerate(payload.get("line_items") or []):
        for key in ("description", "quantity", "unit_price", "total_ht"):
            fields[f"line_items.{index}.{key}"] = (item or {}).get(key)
    return fields


def equal(left: object, right: object) -> bool:
    if isinstance(left, (int, float, Decimal)) and not isinstance(left, bool) and (
        isinstance(right, (int, float, Decimal)) and not isinstance(right, bool)
    ):
        return Decimal(str(left)) == Decimal(str(right))
    return left == right


def evaluate(records: list[dict]) -> dict:
    if not records:
        raise ValueError("No evaluation records")
    counts = defaultdict(lambda: {"count": 0, "correct": 0, "false_nulls": 0, "wrong_values_retained": 0})
    examples = []
    for record in records:
        evidence = (
            from_tsv(record["ocr_tsv"], locale=record.get("locale")) if "ocr_tsv" in record else
            from_text(record["ocr_text"], locale=record.get("locale"))
        )
        grounded = reconcile(evidence, [read_model(raw) for raw in record["model_responses"]])
        actual, expected = flatten(grounded.payload), flatten(record["expected"])
        for path in sorted(set(actual) | set(expected)):
            field = ".".join(part for part in path.split(".") if not part.isdigit())
            stats = counts[field]
            value, gold = actual.get(path), expected.get(path)
            stats["count"] += 1
            stats["correct"] += int(equal(value, gold))
            stats["false_nulls"] += int(value is None and gold is not None)
            stats["wrong_values_retained"] += int(value is not None and not equal(value, gold))
        examples.append({"id": record["id"], "payload": grounded.payload, "statuses": grounded.statuses})
    metrics = {
        field: {
            **stats,
            "exact_accuracy": stats["correct"] / stats["count"],
            "false_null_rate": stats["false_nulls"] / stats["count"],
            "wrong_value_retained_rate": stats["wrong_values_retained"] / stats["count"],
        } for field, stats in sorted(counts.items())
    }
    return {"num_examples": len(records), "model_called": False, "fields": metrics, "examples": examples}


def replay_records(path: Path, dataset: Path) -> list[dict]:
    rows = {row["id"]: row for row in map(json.loads, dataset.read_text(encoding="utf-8").splitlines())}
    records = []
    for case in json.loads(path.read_text(encoding="utf-8")):
        row = rows[case["id"]]
        model = copy.deepcopy(row["output"])
        for path, value in case["mutations"].items():
            parts = path.split(".")
            target = model
            for part in parts[:-1]:
                target = target[int(part)] if isinstance(target, list) else target[part]
            target[parts[-1]] = value
        records.append({
            "id": case["invoice"], "ocr_text": row["input"], "locale": row["locale"],
            "expected": row["output"], "model_responses": [json.dumps(model, ensure_ascii=False)],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--records", type=Path, help="JSONL: id, ocr_text, locale, model_responses, expected")
    inputs.add_argument("--replays", type=Path, help="Explicit synthetic replay mutations, for software regression only")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parents[1] / "data/test.jsonl")
    args = parser.parse_args()
    records = (
        replay_records(args.replays, args.dataset) if args.replays else
        [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    )
    report = evaluate(records)
    print("Evidence: dataset text / synthetic model errors" if args.replays else "Evidence: supplied OCR text / supplied model responses")
    print("No real model called. Rates use all reference fields, including absent predictions.")
    print(f"{'Field':32} {'Accuracy':>10} {'False null':>12} {'Wrong kept':>12} {'Count':>8}")
    for field, metrics in report["fields"].items():
        print(
            f"{field:32} {metrics['exact_accuracy']:10.2%} {metrics['false_null_rate']:12.2%} "
            f"{metrics['wrong_value_retained_rate']:12.2%} {metrics['count']:8d}"
        )


if __name__ == "__main__":
    main()
