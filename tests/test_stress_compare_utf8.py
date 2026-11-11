"""PR114: the stress harness must read its `--compare` baseline as utf-8, not as the Windows default codec.

`scripts/run_stress_check.py` writes its report with `json.dumps(..., ensure_ascii=False)` and an explicit
utf-8 encoding, so a report contains raw French diacritics. Reading that file back with a bare
`Path.read_text()` decodes it with `locale.getpreferredencoding(False)` - cp1252 on this machine - and cp1252
leaves 0x81, 0x8D, 0x8F, 0x90 and 0x9D undefined, so a "Ï" (utf-8 C3 8F) aborts the read. The abort happens
after the whole 38-case suite has already run, which makes the cheapest way to lose fifteen minutes of real
Tesseract time an encoding argument.

PR113 §13 made this one-expression fix while its R4 candidate was still under review; the owner rejected R4
and kept the tooling fix, so it now belongs to the accepted state and needs its own guard. The assertions here
are the ones that survive either decision: they pin the read to utf-8 and keep the acceptance arithmetic
intact. No OCR is run - the candidate report is handed to `main()` through a patched `run()`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DRIVER = r"""
import json, locale, sys
from pathlib import Path
from unittest import mock

sys.path[:0] = [{scripts!r}, {root!r}, {src!r}, {stress!r}]
sys.argv = ["run_stress_check.py", "--output", {output!r}, "--compare", {baseline!r}]

import run_stress_check as S
from app import main as app_main


def no_ocr(*args, **kwargs):
    raise AssertionError("the --compare path must not reach Tesseract")


app_main.pytesseract.image_to_data = no_ocr
S.run = lambda: json.loads(Path({candidate!r}).read_text(encoding="utf-8"))
code = 0
try:
    S.main()
except SystemExit as error:
    code = error.code or 0
report = json.loads(Path({output!r}).read_text(encoding="utf-8"))
print(json.dumps({{"preferred_encoding": locale.getpreferredencoding(False),
                  "utf8_mode": sys.flags.utf8_mode,
                  "exit_code": code,
                  "latency_acceptance": report.get("latency_acceptance")}}))
"""

# "MaÏsse" is deliberate: "Ï" is the two utf-8 bytes C3 8F, and 0x8F is undefined in cp1252.
CASE_NAME = "france/MaÏsse • éditeur"


def case(name):
    """A case record with every key main() prints or counts, so the driver reaches the compare read."""
    return {"case": name, "wrong_values": [], "false_nulls": [], "phantom_anomalies": [], "crash": None}


def fixtures(tmp_path: Path):
    baseline = {
        "scope": "fixture", "seed": 713,
        "summary": {"cases": 1, "wrong_values": 0, "false_nulls": 2, "phantom_anomalies": 0,
                    "crashes": 0, "expected_fields": 648},
        "cases": [dict(case(CASE_NAME), false_nulls=[{"field": "supplier"}])],
        "additional_model_errors": {"cases": 0, "wrong_values": 0, "phantom_anomalies": 0},
    }
    candidate = {
        "scope": "fixture", "seed": 713,
        "summary": {"cases": 1, "wrong_values": 0, "false_nulls": 1, "phantom_anomalies": 0,
                    "crashes": 0, "expected_fields": 648},
        "cases": [case(CASE_NAME)],
        "additional_model_errors": {"cases": 0, "wrong_values": 0, "phantom_anomalies": 0},
    }
    paths = {}
    for name, payload in (("baseline", baseline), ("candidate", candidate)):
        path = tmp_path / f"{name}.json"
        # Written exactly the way run_stress_check.py writes a report, so the read under test sees real bytes.
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
    output = tmp_path / "report.json"
    return baseline, candidate, paths["baseline"], paths["candidate"], output


def run_driver(tmp_path: Path) -> dict:
    _, _, baseline, candidate, output = fixtures(tmp_path)
    script = tmp_path / "driver.py"
    script.write_text(DRIVER.format(scripts=str(SCRIPTS), root=str(ROOT), src=str(ROOT / "src"),
                                    stress=str(ROOT / "tests" / "stress"), output=str(output),
                                    baseline=str(baseline), candidate=str(candidate)), encoding="utf-8")
    process = subprocess.run([sys.executable, str(script)], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=locale_env())
    assert process.returncode == 0, f"driver failed: {process.stdout[-1500:]} {process.stderr[-1500:]}"
    return json.loads(process.stdout.strip().splitlines()[-1])


def locale_env() -> dict:
    """The inherited environment with utf-8 mode switched off, so the platform codec is the one under test."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_stress_module_loads_under_the_windows_default_codec():
    check = ("import sys; sys.path[:0]=[r'%s', r'%s']; import run_stress_check as S, locale, json;"
             "print(json.dumps({'ok': callable(S.main) and callable(S.compare_latency_candidate),"
             "'encoding': locale.getpreferredencoding(False)}))" % (str(SCRIPTS), str(ROOT / "src")))
    process = subprocess.run([sys.executable, "-c", check], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=locale_env())
    assert process.returncode == 0, process.stderr[-1500:]
    assert json.loads(process.stdout.strip().splitlines()[-1])["ok"] is True


def test_compare_reads_a_diacritic_baseline_report_and_keeps_the_acceptance(tmp_path):
    result = run_driver(tmp_path)
    assert result["utf8_mode"] == 0, "the driver must run with the platform codec, not forced utf-8 mode"
    assert result["exit_code"] == 0, "`main()` must complete on a report the locale codec cannot decode"
    acceptance = result["latency_acceptance"]
    # One false null recovered out of the frozen 648-field denominator, inside the 2-point allowance.
    assert acceptance == {"baseline": str(tmp_path / "baseline.json"),
                          "false_null_delta_percentage_points": 100 * (1 - 2) / 648,
                          "allowed_increase_percentage_points": 2.0,
                          "passed": True}


def test_the_compare_read_names_utf8_explicitly():
    source = (SCRIPTS / "run_stress_check.py").read_text(encoding="utf-8")
    assert 'args.compare.read_text(encoding="utf-8")' in source
    assert "args.compare.read_text()" not in source, (
        "the bare read_text() is the defect: it decodes with locale.getpreferredencoding(False)")
    assert 'args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")' in source
