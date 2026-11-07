"""Replay captured model responses through the local pipeline and score them with the benchmark.

The model runs on the ZeroGPU Space; the evidence, grounding and validation must be scored here,
where the benchmark's verdict taxonomy lives. This adapter only feeds `bench_real_model.run_case`
the raw text captured from the Space - it never changes the extraction path.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]

import bench_real_model as bench  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    missing = [case["id"] for case in cases if case["id"] not in capture]
    if missing:
        raise SystemExit("capture is missing: %s" % ", ".join(missing))

    results = []
    for case in cases:
        captured = capture[case["id"]]
        raw = captured.get("first_pass_raw") or ""
        result = bench.run_case(case, predict=bench.SimulatedModel([{"raw": raw}]))
        result["space"] = {
            "wall_seconds": captured.get("wall_seconds"),
            "model_lines": captured.get("model_lines"),
            "final_payload": captured.get("payload"),
            "validation_flags": (captured.get("validation") or {}).get("flags"),
            "report_head": (captured.get("report") or "")[:300],
        }
        result["replay_matches_space_final"] = captured.get("payload") == result["final_payload"]
        results.append(result)
        print("%-24s model_used=%s raw_parse=%s final_matches_space=%s" % (
            case["id"], bool(captured.get("first_pass_raw")),
            result["model"]["raw_parse"], result["replay_matches_space_final"]))

    aggregate = bench.aggregate(results)
    report = {
        "harness": {"version": bench.VERSION, "mode": "replay-of-space-capture",
                    "cases": len(results), "manifest_sha256": bench.manifest_fingerprint(cases)},
        "space": {"requests": len(results),
                  "wall_seconds": round(sum(c.get("wall_seconds") or 0 for c in capture.values()), 2),
                  "generation_seconds": [c.get("model_lines") for c in capture.values()]},
        "aggregate": aggregate,
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False, indent=1)[:2500])
    print("report:", args.output)


if __name__ == "__main__":
    main()
