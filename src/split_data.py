"""Split data/dataset.jsonl into train/val/test (80/10/10), stratified by format."""

import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    lines = (DATA_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["format"]].append(row)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for fmt in sorted(groups):
        items = groups[fmt]
        rng.shuffle(items)
        n_train = int(len(items) * TRAIN_RATIO)
        n_val = int(len(items) * VAL_RATIO)
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    assert sum(len(items) for items in splits.values()) == len(rows)

    for name, items in splits.items():
        path = DATA_DIR / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in items:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)} examples -> {path}")


if __name__ == "__main__":
    main()
