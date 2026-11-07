"""Restore only missing deterministic dataset PNGs without rewriting JSON targets."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "src")]

from generate_dataset import generate_records
from render_invoice import render_png


def main():
    generated = generate_records()
    saved = [json.loads(line) for line in
             (ROOT / "data/dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    if [record for record, _ in generated] != saved:
        raise RuntimeError("Generated records differ from persisted targets; refusing asset restoration")
    restored = 0
    for record, document in generated:
        path = ROOT / record["image_path"]
        if path.exists():
            if not path.is_file():
                raise RuntimeError(f"Expected PNG path is not a file: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        render_png(document, path)
        restored += 1
    print(f"Restored {restored} missing PNGs; existing files and JSON targets unchanged.")


if __name__ == "__main__":
    main()
