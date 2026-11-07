"""Smoke one invoice through a deployed Gradio Space and capture the model evidence.

Talks to the Space over HTTP only: it reads the app config to find the extraction endpoint, sends
one file, and prints the raw model output, the report excerpt, the model state and the timings.
`--list-api` performs no inference at all.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from gradio_client import Client, handle_file


def discover(client):
    info = client.view_api(return_format="dict", print_info=False)
    named = info.get("named_endpoints") or {}
    extraction = [name for name in named
                  if "extraction" in name.lower() or "extract" in name.lower()]
    if not extraction:
        raise SystemExit("No extraction endpoint found; endpoints: %s" % sorted(named))
    return extraction[0], named


def model_state(report, details):
    generation = re.search(r"génération\s*:\s*([0-9.]+)\s*s", details or "")
    passes = re.search(r"passes demandées\s*:\s*(\d+)\s*;\s*réussies\s*:\s*(\d+)", details or "")
    used = "non utilisé" not in (details or "") and "utilisé" in (details or "")
    return {
        "model_used": used,
        "generation_seconds": float(generation.group(1)) if generation else None,
        "passes_requested": int(passes.group(1)) if passes else None,
        "passes_succeeded": int(passes.group(2)) if passes else None,
        "report_head": (report or "")[:400],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True)
    parser.add_argument("--image", type=Path, help="invoice to upload")
    parser.add_argument("--api-name")
    parser.add_argument("--list-api", action="store_true", help="print endpoints and exit; no inference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--token", help="read from HF_TOKEN instead when omitted")
    args = parser.parse_args()

    client = Client(args.space, token=args.token or None)
    api_name, endpoints = discover(client)
    if args.list_api:
        print(json.dumps({"endpoints": sorted(endpoints), "chosen": api_name,
                          "signature": endpoints[api_name].get("parameters")}, ensure_ascii=False, indent=1))
        return 0
    if args.image is None:
        parser.error("--image is required unless --list-api is used")

    started = time.perf_counter()
    result = client.predict(handle_file(args.image), api_name=args.api_name or api_name)
    elapsed = round(time.perf_counter() - started, 2)
    payload, report, details, advanced, raw, validation, download = (list(result) + [None] * 7)[:7]
    summary = {
        "space": args.space, "api_name": args.api_name or api_name, "image": str(args.image),
        "http_elapsed_seconds": elapsed,
        "payload": payload,
        "first_pass_raw": raw,
        "validation": validation,
        "download": download,
        "advanced": advanced,
        **model_state(report, details),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    if args.output:
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
