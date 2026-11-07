"""Capture one authorized response from the existing invoice Gradio Space.

Run --self-test for an offline serialization check. A remote request requires
the explicit --invoke-space flag.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPACE = "https://nicolbl95-llm-extraction-finetune.hf.space"
DEFAULT_INPUT = ROOT / "tests" / "fixtures" / "pr75-synthetic-invoice.png"
DEFAULT_CAPTURE = ROOT / "diagnostics" / "pr76-space-canary.json"
API_NAME = "/run_extraction_ui"


def configure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Input and capture paths must remain inside the workspace.") from exc
    return resolved


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return legacy_dict()
    return {
        "__python_type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "__repr__": repr(value),
    }


def write_capture(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def exception_record(error: Exception) -> dict[str, str]:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def self_test() -> None:
    sample_outputs = (
        {"supplier_name": "Atelier Étoile", "total_ht": 37.0},
        "Résumé français : TVA vérifiée — sans erreur",
        "Performances : OCR 1,23 s ; génération 2,10 s",
        {"download": Path("résultat-démo.json"), "status": "prêt"},
    )
    with tempfile.TemporaryDirectory(prefix=".pr76-capture-test-", dir=ROOT) as temp_dir:
        capture_path = Path(temp_dir) / "four-outputs.json"
        write_capture(
            capture_path,
            {"status": "completed", "output_count": 4, "outputs": sample_outputs},
        )
        raw = capture_path.read_text(encoding="utf-8")
        restored = json.loads(raw)
        assert "Atelier Étoile" in raw
        assert "Résumé français" in raw
        assert "\\u00c9" not in raw
        assert restored["output_count"] == 4
        assert restored["outputs"][0]["supplier_name"] == "Atelier Étoile"
        assert restored["outputs"][3]["download"] == "résultat-démo.json"

        try:
            raise RuntimeError("Échec synthétique : sortie non disponible")
        except RuntimeError as error:
            failure_path = Path(temp_dir) / "exception.json"
            write_capture(
                failure_path,
                {"status": "error", "exception": exception_record(error)},
            )
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        assert "Échec synthétique" in failure["exception"]["message"]
        assert "Traceback" in failure["exception"]["traceback"]
    print("SELF-TEST PASS: UTF-8 accents, four outputs, full exception traceback.")


def invoke_space(args: argparse.Namespace) -> int:
    input_path = workspace_path(args.input)
    capture_path = workspace_path(args.capture)
    if not input_path.is_file():
        raise FileNotFoundError(f"Canary input not found: {input_path}")
    if capture_path.exists():
        raise FileExistsError(f"Refusing to overwrite capture: {capture_path}")

    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    phase = "client_initialization"
    client_stdout = io.StringIO()
    client_stderr = io.StringIO()
    try:
        with redirect_stdout(client_stdout), redirect_stderr(client_stderr):
            from gradio_client import Client, handle_file

            client = Client(
                args.space,
                token=False,
                max_workers=1,
                verbose=False,
                download_files=False,
                analytics_enabled=False,
                httpx_kwargs={"timeout": args.timeout},
            )
            phase = "run_extraction_ui"
            outputs = client.predict(handle_file(str(input_path)), api_name=API_NAME)
        elapsed = time.perf_counter() - started
        output_count = len(outputs) if isinstance(outputs, (list, tuple)) else None
        payload = {
            "capture_version": 1,
            "status": "completed",
            "started_at_utc": started_at,
            "client_elapsed_seconds": elapsed,
            "space": args.space,
            "api_name": API_NAME,
            "input": {
                "workspace_path": str(input_path.relative_to(ROOT)),
                "sha256": input_hash,
            },
            "output_count": output_count,
            "outputs": outputs,
            "client_stdout": client_stdout.getvalue(),
            "client_stderr": client_stderr.getvalue(),
        }
    except Exception as error:
        elapsed = time.perf_counter() - started
        payload = {
            "capture_version": 1,
            "status": "error",
            "started_at_utc": started_at,
            "client_elapsed_seconds": elapsed,
            "space": args.space,
            "api_name": API_NAME,
            "input": {
                "workspace_path": str(input_path.relative_to(ROOT)),
                "sha256": input_hash,
            },
            "phase": phase,
            "exception": exception_record(error),
            "client_stdout": client_stdout.getvalue(),
            "client_stderr": client_stderr.getvalue(),
        }
        write_capture(capture_path, payload)
        print(f"CAPTURE status=error phase={phase} elapsed_s={elapsed:.3f}")
        return 1

    write_capture(capture_path, payload)
    print(
        f"CAPTURE status=completed outputs={output_count} "
        f"elapsed_s={elapsed:.3f}"
    )
    return 0


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-test", action="store_true", help="Offline serialization test only.")
    mode.add_argument(
        "--invoke-space",
        action="store_true",
        help="Send one request to the existing Space; requires prior explicit authorization.",
    )
    parser.add_argument("--space", default=DEFAULT_SPACE)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--capture", default=str(DEFAULT_CAPTURE))
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    return invoke_space(args)


if __name__ == "__main__":
    raise SystemExit(main())
