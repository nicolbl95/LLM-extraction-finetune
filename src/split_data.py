"""Split invoices 80/10/10, balancing locale, format and template when available."""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def split_records(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row.get("locale", "")].append(row)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        items = groups[key]
        rng.shuffle(items)
        templates: dict[int, list[dict]] = defaultdict(list)
        for row in items:
            templates[row.get("template_id", 0)].append(row)

        def allocation(ratio: float) -> dict[int, int]:
            counts = {template: int(len(bucket) * ratio) for template, bucket in templates.items()}
            remaining = round(len(items) * ratio) - sum(counts.values())
            order = sorted(
                templates, key=lambda template: (-(len(templates[template]) * ratio - counts[template]), template),
            )
            for template in order[:remaining]:
                counts[template] += 1
            return counts

        train_counts, val_counts = allocation(TRAIN_RATIO), allocation(VAL_RATIO)
        for template in sorted(templates):
            formats: dict[str, list[dict]] = defaultdict(list)
            for row in templates[template]:
                formats[row["format"]].append(row)
            format_order = sorted(formats)
            rng.shuffle(format_order)
            ordered = []
            while any(formats.values()):
                for fmt in format_order:
                    if formats[fmt]:
                        ordered.append(formats[fmt].pop())
            n_train, n_val = train_counts[template], val_counts[template]
            splits["train"] += ordered[:n_train]
            splits["val"] += ordered[n_train:n_train + n_val]
            splits["test"] += ordered[n_train + n_val:]

    assert sum(len(items) for items in splits.values()) == len(rows)
    return splits


def main() -> None:
    lines = (DATA_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    splits = split_records(rows)
    for name, items in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} examples -> {path}")


if __name__ == "__main__":
    main()
