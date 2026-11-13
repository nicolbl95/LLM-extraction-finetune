import hashlib
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "PRODUCT_REVIEW_64_REPRODUCTION_RESULTS.json"
MATRIX_PATH = ROOT / "PRODUCT_REVIEW_62_REGRESSION_MATRIX.json"
WORKTREE_SRC = ROOT / "src"
TESSDATA_DIR = ROOT / ".venv" / "product-review64" / "tessdata"
CASES = (
    "microspective",
    "mon_entreprise",
    "mirada",
    "future_graphics",
    "fast_photography",
)


@pytest.fixture(scope="module")
def results():
    if not RESULTS_PATH.is_file():
        pytest.skip(
            "Run scripts/capture_product_review64_ocr_baseline.py first."
        )
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cases(results):
    return {case["case_id"]: case for case in results["cases"]}


@lru_cache(maxsize=1)
def run_worktree_header_probe(header: str) -> dict:
    probe = (
        "import json,sys;"
        "sys.path.insert(0,sys.argv[1]);"
        "from invoice_evidence import _header_matches,is_line_total_header,total_label_field;"
        "s=sys.argv[2];"
        "print(json.dumps({'header_matches':{k:_header_matches(k,s) for k in "
        "('description','quantity','unit_price','total_ht')},"
        "'subtotal_is_line_total_header':is_line_total_header('SUBTOTAL'),"
        "'standalone_subtotal_global_field':total_label_field('SUBTOTAL 772.50')}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(WORKTREE_SRC), header],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def run_text_parser_probe(text: str) -> dict:
    probe = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from invoice_evidence import from_text
from invoice_grounding import reconcile

evidence = from_text(sys.stdin.read())
result = reconcile(evidence, [])

def values(mapping):
    return {
        key: [str(number.value) for number in numbers]
        for key, numbers in mapping.items()
    }

print(json.dumps({
    "rows": [
        {
            "description": row.description,
            "cells": values(row.cells),
        }
        for row in evidence.rows
    ],
    "totals": values(evidence.totals),
    "payload": result.payload,
    "statuses": result.statuses,
    "warnings": result.warnings,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(WORKTREE_SRC)],
        cwd=ROOT,
        input=text,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def test_hash_verified_full_pipeline_baseline_captured_all_five_without_models(
    results, cases
):
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    expected_hashes = {
        case["case_id"]: case["candidate_asset"]["sha256"]
        for case in matrix["cases"]
        if case["case_id"] in CASES
    }

    assert set(cases) == set(CASES)
    assert results["capture_summary"]["successful_pipeline_captures"] == 5
    assert results["capture_summary"]["model_calls"] == 0
    assert results["model_inference"]["attempted"] is False
    for case_id in CASES:
        case = cases[case_id]
        assert case["capture_status"] == "captured"
        assert case["candidate_image"]["sha256"] == expected_hashes[case_id]
        assert case["reader_call_count"] > 0
        assert case["model_calls"] == 0
        assert case["model_output"] is None


def test_microspective_rate_price_header_should_identify_unit_price_column(cases):
    header = cases["microspective"]["diagnostic_probes"][
        "microspective_header_parser"
    ]["source_text"]
    probe = run_worktree_header_probe(header)

    assert probe["header_matches"]["unit_price"] is True
    assert probe["header_matches"]["quantity"] is True
    assert probe["header_matches"]["description"] is True


def test_microspective_captured_tokens_do_not_assemble_incomplete_rows(results):
    # `results` is requested, not read: this probe shells out to the capture file itself, so the
    # module's own skip guard is the only thing that keeps a checkout without it from hard-failing.
    probe = (
        "import json,sys;"
        "from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);"
        "from invoice_evidence import Token,from_tokens;"
        "doc=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'));"
        "c=next(x for x in doc['cases'] if x['case_id']=='microspective');"
        "r=next(x for x in c['reads'] if x['read_index']==0);"
        "tokens=[Token(t['text'],0,tuple(t['box']),t['confidence']) "
        "for t in r['tokens'] if t['text'].strip() and t['confidence']>=0 "
        "and t['text'].strip() not in {'|','||','[',']'}];"
        "e=from_tokens(tokens,literal_columns=False);"
        "print(json.dumps({'row_count':len(e.rows),'rows':[x.description for x in e.rows]}))"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            probe,
            str(WORKTREE_SRC),
            str(RESULTS_PATH),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    actual = json.loads(completed.stdout)

    assert actual["row_count"] == 0


def test_ocr_contextual_rows_require_confident_complete_numeric_cells():
    probe = """
import json
import sys

sys.path.insert(0, sys.argv[1])
from invoice_evidence import Token, from_tokens

def parse(include_subtotal):
    words = [
        ("Description", 0, 0), ("Quantity", 120, 0),
        ("RATE", 220, 0), ("/", 252, 0), ("PRICE", 265, 0),
        ("Subtotal", 380, 0), ("Service", 0, 20), ("A", 55, 20),
        ("2", 120, 20), ("15", 220, 20),
    ]
    if include_subtotal:
        words.append(("30", 380, 20))
    tokens = [
        Token(text, 0, (left, top, left + max(8, len(text) * 7), top + 10), 95)
        for text, left, top in words
    ]
    evidence = from_tokens(tokens, locale="en-US", literal_columns=False)
    return {
        "rows": [
            {
                "description": row.description,
                "cells": {
                    key: [str(number.value) for number in values]
                    for key, values in row.cells.items()
                    if values
                },
            }
            for row in evidence.rows
        ],
        "issues": evidence.issues,
    }

print(json.dumps({
    "complete": parse(True),
    "incomplete": parse(False),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(WORKTREE_SRC)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    actual = json.loads(completed.stdout)

    assert actual["complete"]["rows"] == [
        {
            "description": "Service A",
            "cells": {
                "quantity": ["2"],
                "unit_price": ["15"],
                "total_ht": ["30"],
            },
        }
    ]
    assert actual["incomplete"]["rows"] == []
    assert any(
        "Ligne de tableau contextuel ignorée" in issue
        for issue in actual["incomplete"]["issues"]
    )


@pytest.mark.parametrize("price_header", ["RATE", "PRICE", "RATE / PRICE"])
def test_contextual_price_headers_use_the_correct_column_geometry(price_header):
    result = run_text_parser_probe(
        "Description | Quantity | "
        f"{price_header} | Subtotal\nService A | 2 | 15 | 30"
    )

    assert len(result["rows"]) == 1
    row = result["rows"][0]
    assert row["description"] == "Service A"
    assert row["cells"]["quantity"] == ["2"]
    assert row["cells"]["unit_price"] == ["15"]
    assert row["cells"]["total_ht"] == ["30"]
    assert result["payload"]["line_items"][0]["quantity"] == 2.0
    assert result["payload"]["line_items"][0]["unit_price"] == 15.0
    assert result["payload"]["line_items"][0]["total_ht"] == 30.0


def test_rate_price_words_in_unrelated_text_do_not_create_table_columns():
    result = run_text_parser_probe(
        "The rate / price was discussed alongside the subtotal estimate."
    )
    probe = run_worktree_header_probe(
        "The rate / price was discussed alongside the subtotal estimate."
    )

    assert result["rows"] == []
    assert result["payload"]["line_items"] == []
    assert probe["header_matches"]["unit_price"] is False


def test_rate_price_before_quantity_is_not_accepted_as_swapped_columns():
    result = run_text_parser_probe(
        "Description | Rate / Price | Quantity | Subtotal\n"
        "Service A | 15 | 2 | 30"
    )

    assert result["rows"] == []
    assert result["payload"]["line_items"] == []


def test_subtotal_and_previous_balance_summary_rows_are_not_products():
    result = run_text_parser_probe(
        "Description | Quantity | Rate / Price | Subtotal\n"
        "Service A | 2 | 15 | 30\n"
        "Subtotal: 30\n"
        "Previous balance: 5\n"
        "Invoice total: 35"
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["description"] == "Service A"
    assert result["totals"]["total_ht"] == ["30"]
    assert result["totals"]["total_ttc"] == ["35"]
    assert result["payload"]["line_items"][0]["total_ht"] == 30.0


def test_missing_subtotal_amount_remains_null():
    result = run_text_parser_probe(
        "Description | Quantity | RATE / PRICE | SUBTOTAL\n"
        "Service A | 2 | 15 |"
    )

    assert len(result["rows"]) == 1
    assert result["rows"][0]["cells"]["quantity"] == ["2"]
    assert result["rows"][0]["cells"]["unit_price"] == ["15"]
    assert result["rows"][0]["cells"].get("total_ht", []) == []
    assert result["payload"]["line_items"][0]["total_ht"] is None


def test_microspective_header_gap_does_not_weaken_null_or_subtotal_controls(
    cases
):
    case = cases["microspective"]
    probe = run_worktree_header_probe(
        case["diagnostic_probes"]["microspective_header_parser"]["source_text"]
    )
    evidence = case["consensus_documentary_evidence"]
    output = case["ocr_only_reconciliation"]["payload"]

    assert probe["subtotal_is_line_total_header"] is False
    assert probe["standalone_subtotal_global_field"] == "total_ht"
    assert evidence["rows"] == []
    assert all(
        output[field] is None
        for field in ("supplier_name", "invoice_date", "total_ht", "total_ttc")
    )
    assert output["line_items"] == []
    assert case["model_calls"] == 0


def test_mon_conflicting_sixty_and_unprinted_line_ht_remain_unverified(cases):
    case = cases["mon_entreprise"]
    evidence = case["consensus_documentary_evidence"]
    payload = case["ocr_only_reconciliation"]["payload"]
    seen_prices = {
        number["value"]
        for read in case["reads"]
        for row in read["parsed_candidates"]["rows"]
        for number in row["cells"].get("unit_price", [])
    }

    assert {"60.00", "0.00"}.issubset(seen_prices)
    assert evidence["rows"][0]["cells"]["unit_price"] == []
    assert evidence["rows"][0]["family_votes"]["unit_price"] == {}
    assert payload["line_items"][0]["unit_price"] is None
    assert payload["line_items"][0]["total_ht"] is None
    assert payload["line_items"][1]["total_ht"] is None
    assert payload["total_ht"] == 1350.0
    assert payload["total_ttc"] == 1620.0


def test_mirada_historical_wrong_amount_is_not_replayed_as_new_qwen(
    results, cases
):
    capture = cases["mirada"]["prior_pr63_evidence"]["historical_qwen_capture"]
    first_pass = json.loads(capture["mirada_capture"]["attempt_responses"][0]["raw"])
    strict = results["historical_mirada_capture"]["strict_payload"]
    live = cases["mirada"]["consensus_documentary_evidence"]
    live_output = cases["mirada"]["ocr_only_reconciliation"]["payload"]

    assert capture["revision"] is None
    assert capture["input_sha256"] == cases["mirada"]["candidate_image"]["sha256"]
    assert first_pass["line_items"][2]["unit_price"] == 114.5
    assert first_pass["line_items"][2]["total_ht"] == 1145.0
    assert strict["line_items"][2]["unit_price"] is None
    assert strict["line_items"][2]["total_ht"] == 114.5
    assert live["rows"][0]["cells"]["unit_price"][0]["value"] == "15.955"
    assert live_output["line_items"][0]["unit_price"] is None
    assert live["rows"][2]["cells"]["quantity"][0]["value"] == "10"
    assert live_output["line_items"][2]["quantity"] is None
    assert live_output["line_items"][2]["total_ht"] is None
    assert cases["mirada"]["model_calls"] == 0


def test_future_due_date_evidence_does_not_invent_an_invoice_date(cases):
    case = cases["future_graphics"]
    evidence = case["consensus_documentary_evidence"]
    payload = case["ocr_only_reconciliation"]["payload"]
    read_text = "\n".join(read["raw_text"] for read in case["reads"])

    assert "09.01.2025" in read_text
    assert "09.14.2025" in read_text
    assert evidence["date_locale"] == "en-US"
    assert evidence["invoice_date"] is None
    assert payload["invoice_date"] is None


def test_fast_supplier_spelling_conflicts_do_not_create_verified_identity(cases):
    case = cases["fast_photography"]
    evidence = case["consensus_documentary_evidence"]
    payload = case["ocr_only_reconciliation"]["payload"]
    supplier_candidates = {
        read["parsed_candidates"]["supplier"]
        for read in case["reads"]
        if read["parsed_candidates"]["supplier"] is not None
    }

    assert "Fast Photohgraphy" in supplier_candidates
    assert "Fost Photohgraphy" in supplier_candidates
    assert evidence["supplier"] is None
    assert payload["supplier_name"] is None
    assert evidence["invoice_date"] == "2025-09-01"
    assert case["model_calls"] == 0


def test_isolated_tessdata_files_are_hash_verified_and_load_both_languages(
    monkeypatch,
):
    import pytesseract

    expected = {
        "fra": (
            1_130_365,
            "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca",
        ),
        "eng": (
            4_113_088,
            "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
        ),
    }
    if not TESSDATA_DIR.is_dir():
        pytest.skip("Isolated PR64 Tesseract traineddata is not installed.")
    monkeypatch.setenv("TESSDATA_PREFIX", str(TESSDATA_DIR))
    for language, (size, digest) in expected.items():
        path = TESSDATA_DIR / f"{language}.traineddata"
        assert path.stat().st_size == size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert {"eng", "fra"}.issubset(set(pytesseract.get_languages(config="")))
