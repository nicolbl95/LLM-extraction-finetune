#!/usr/bin/env python3
"""Frozen split and cached real OCR experiments. References are scoring-only."""

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests/stress"), str(ROOT / "scripts")]
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

from cases import image_cases, layout_cases
from invoice_evidence import from_tsv
from ocr_preprocessing import prepare
from ocr_quality import assess_ocr
from ocr_recovery import table_crop
from ocr_config import OCR_CONFIG
from run_stress_check import run_case
from invoice_grounding import read_model, reconcile
from ocr_consensus import build_consensus
from evaluate_grounding import equal, flatten

METHODS = ("deskew", "otsu", "adaptive", "denoise", "clahe", "scale3", "crop")
CACHE = ROOT / "results/ocr-experiment-cache"


def split_cases(names):
    groups = {}
    for name in names:
        groups.setdefault(name.split("/")[0], []).append(name)
    split = {}
    rng = random.Random(62419)
    for group, values in sorted(groups.items()):
        values = sorted(values)
        rng.shuffle(values)
        count = 6 if group == "layout" else 5
        for index, name in enumerate(values):
            split[name] = "tuning" if index < count else "validation"
    return split


def read_variant(image, method, psm):
    import pytesseract
    prepared, metadata = prepare(image, method)
    image_hash = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
    key = hashlib.sha256(f"{image_hash}/{method}/{psm}/v1".encode()).hexdigest()
    path = CACHE / f"{key}.json"
    CACHE.mkdir(exist_ok=True)
    if path.exists():
        return json.loads(path.read_text())
    data = pytesseract.image_to_data(
        prepared, lang="fra+eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT, timeout=60,
    )
    if method == "crop":
        data = table_crop(data, prepared, lambda crop, mode: pytesseract.image_to_data(
            crop, lang="fra+eng", config=f"--psm {mode}", output_type=pytesseract.Output.DICT, timeout=60,
        ))
    result = {"data": data, "metadata": metadata, "input_sha256": image_hash, "psm": psm}
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def score(case, evidence):
    result = reconcile(evidence, [read_model(json.dumps(case.model))])
    actual, expected = flatten(result.payload), flatten(case.expected)
    return {
        "case": case.name, "fields": len(expected),
        "false_nulls": sum(value is not None and actual.get(path) is None for path, value in expected.items()),
        "wrong_values": sum(value is not None and not equal(value, expected.get(path)) for path, value in actual.items()),
        "phantom_anomalies": sum(flag.severity != "information" for flag in result.validation.flags),
        "final": result.payload, "statuses": result.statuses,
    }


def summarize(records):
    return {key: sum(record[key] for record in records) for key in
            ("fields", "false_nulls", "wrong_values", "phantom_anomalies")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tuning", "validation"), required=True)
    parser.add_argument("--consensus", action="store_true")
    parser.add_argument("--families", default=",".join(OCR_CONFIG.families))
    parser.add_argument("--confidence", type=int, default=OCR_CONFIG.confidence)
    args = parser.parse_args()
    baseline = json.loads((ROOT / "results/stress-before-recovery.json").read_text())
    before = {case["case"]: case for case in baseline["cases"]}
    with tempfile.TemporaryDirectory(dir=ROOT / "tests/stress") as directory:
        cases = [*image_cases(ROOT), *layout_cases(Path(directory))]
        split = split_cases([case.name for case in cases])
        frozen = json.loads((ROOT / "tests/stress/split.json").read_text())
        if frozen["cases"] != split:
            raise ValueError("Case list no longer matches the frozen stress split")
        split_path = ROOT / "results/ocr-split.json"
        if split_path.exists() and json.loads(split_path.read_text())["cases"] != split:
            raise ValueError("Refusing to change the frozen tuning/validation split")
        split_path.write_text(json.dumps({
            "seed": 62419, "cases": split,
            "warning": "Baseline cases were previously observed; this is a frozen engineering holdout, not an unseen external dataset.",
        }, indent=2) + "\n")
        cases = [case for case in cases if split[case.name] == args.phase]
        trials = {}
        for method in (("consensus",) if args.consensus else METHODS):
            records = []
            for case in cases:
                if case.image is None:
                    record = before[case.name]
                    records.append({
                        "case": case.name, "fields": len(flatten(case.expected)),
                        "false_nulls": len(record["false_nulls"]), "wrong_values": len(record["wrong_values"]),
                        "phantom_anomalies": len(record["phantom_anomalies"]), "final": record["final"],
                        "statuses": record["statuses"],
                    })
                    continue
                if method == "consensus":
                    readings = {
                        family: [from_tsv(read_variant(case.image, family, psm)["data"]) for psm in (4, 6)]
                        for family in args.families.split(",")
                    }
                    evidence = build_consensus(readings, confidence=args.confidence)
                else:
                    first = read_variant(case.image, method, 4)
                    second = read_variant(case.image, method, 6)
                    evidence = assess_ocr(from_tsv(first["data"]), from_tsv(second["data"]))
                records.append(score(case, evidence))
                print(args.phase, method, case.name, records[-1]["false_nulls"], records[-1]["wrong_values"], flush=True)
            trials[method] = {"summary": summarize(records), "cases": records}
            output = ROOT / "results" / f"ocr-{args.phase}-{'consensus' if args.consensus else 'experiments'}.json"
            output.write_text(json.dumps(trials, ensure_ascii=False, indent=2) + "\n")
    baseline_records = [before[name] for name, partition in split.items() if partition == args.phase]
    baseline_nulls = sum(len(record["false_nulls"]) for record in baseline_records)
    decision = {}
    for method, trial in trials.items():
        metrics = trial["summary"]
        decision[method] = (
            "eligible" if metrics["wrong_values"] == metrics["phantom_anomalies"] == 0
            and metrics["false_nulls"] < baseline_nulls else "rejected"
        )
    print(json.dumps({"phase": args.phase, "baseline_nulls": baseline_nulls,
                      "trials": {method: {**trial["summary"], "decision": decision[method]} for method, trial in trials.items()}}))


if __name__ == "__main__":
    main()
