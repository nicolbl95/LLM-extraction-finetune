import copy
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from invoice_evidence import (  # noqa: E402
    DocumentEvidence, EvidenceText, Number, TableRow, Tax, TextLine, Token,
)
from experiments.review_assessment import assess  # noqa: E402
from scripts.bench_real_model import run_case  # noqa: E402
from scripts.evidence_capture import SCHEMA_VERSION, serialize_evidence  # noqa: E402


def number(raw, value, page=0, box=(0, 0, 10, 10)):
    return Number(raw, Decimal(str(value)), page, box)


def synthetic_evidence():
    supplier_token = Token("Acme", 0, (10, 10, 50, 20), 96.5)
    qty_token = Token("1", 0, (80, 40, 90, 50), 94.0)
    line = TextLine("Service 1 10.00 10.00", [qty_token], 0, (60, 38, 220, 52))
    quantity = number("1", "1", 0, (80, 40, 90, 50))
    unit = number("10.00", "10.00", 0, (120, 40, 160, 50))
    total = number("10.00", "10.00", 0, (190, 40, 220, 50))
    row = TableRow(
        description="Service",
        cells={"quantity": [quantity], "unit_price": [unit], "total_ht": [total]},
        line=line,
        row_evidence={"method": "test", "positions": {"quantity": [80, 40, 90, 50]},
                      "printed": {"quantity": ["1"]}},
        row_index=0,
        family_votes={"quantity": {Decimal("1"): ("family-a", "family-b")}},
        corroborated_amount_mismatch=(Decimal("1"), Decimal("2")),
        secondary_description="Service alt",
    )
    total_ht = number("20.00", "20.00", 0, (0, 70, 40, 80))
    tax_amount = number("1.00", "1.00", 0, (0, 90, 30, 100))
    return DocumentEvidence(
        text="Acme\nDate: January 1, 2025\nService 1 10.00 10.00\nTotal HT 20.00",
        locale="en-US",
        tokens=[supplier_token, qty_token],
        lines=[
            TextLine("Acme", [supplier_token], 0, (10, 10, 50, 20)),
            line,
        ],
        rows=[row],
        totals={"total_ht": [total_ht]},
        taxes=[Tax(rate=number("5", "5", 0, (0, 85, 20, 95)), amount=tax_amount)],
        discounts=[number("0.00", "0.00")],
        numbers=[quantity, unit, total, total_ht],
        supplier="Acme",
        supplier_raw="ACME",
        supplier_is_explicit=True,
        invoice_number="INV-1",
        corroborated_identifiers=[("invoice", "INV-1")],
        invoice_date="2025-01-01",
        issues=["test issue"],
        quality_issues=["test quality issue"],
        ocr_calls=10,
        ocr_pages=1,
        ocr_elapsed_seconds=1.25,
        preprocessing_used=["deskew", "clahe"],
        fast_path={"used": False},
        table_shadow={"rows": 1},
        rows_provenance="recovered",
    )


def synthetic_diagnostics():
    return {
        "field_statuses": {"supplier_name": "vérifié"},
        "provenance": {"supplier_name": [{"raw": "Acme", "value": "Acme", "page": 0,
                                          "box": [10, 10, 50, 20]}]},
        "warnings": ["warning"],
        "validation": {"flags": [], "passed": [], "not_verified": []},
        "model_info": {"used": True, "reason": None},
        "unverified_model_readings": {"supplier_name": "Acme"},
        "arithmetic_warnings": [],
        "ocr_configuration": {"version": "test"},
    }


def test_serializer_preserves_lines_tokens_rows_candidates_and_provenance():
    evidence = synthetic_evidence()
    diagnostics = synthetic_diagnostics()
    evidence_snapshot = copy.deepcopy(evidence)
    diagnostics_snapshot = copy.deepcopy(diagnostics)
    serialized = serialize_evidence(evidence, diagnostics=diagnostics)

    assert serialized["schema_version"] == SCHEMA_VERSION
    assert serialized["text_sha256"]
    assert serialized["supplier"] == "Acme"
    assert serialized["supplier_raw"] == "ACME"
    assert serialized["supplier_is_explicit"] is True
    assert serialized["invoice_date"] == "2025-01-01"
    assert serialized["locale"] == "en-US"
    assert serialized["counts"] == {"tokens": 2, "lines": 2, "rows": 1}

    assert serialized["lines"][0]["text"] == "Acme"
    assert serialized["lines"][0]["confidence"] == 96.5
    assert serialized["lines"][0]["page"] == 0
    assert serialized["lines"][0]["box"] == [10, 10, 50, 20]

    assert serialized["totals"]["total_ht"] == ["20.00"]
    assert serialized["totals"]["total_ttc"] == []
    assert serialized["totals"]["total_candidate"] == []
    assert serialized["rows"][0]["description"] == "Service"
    assert serialized["rows"][0]["cells"]["quantity"] == ["1"]

    details = serialized["details"]
    assert details["tokens"][0]["text"] == "Acme"
    assert details["tokens"][0]["box"] == [10, 10, 50, 20]
    assert details["tokens"][0]["confidence"] == 96.5
    assert details["lines"][1]["tokens"][0]["text"] == "1"
    assert details["rows"][0]["cells"]["quantity"][0]["raw"] == "1"
    assert details["rows"][0]["cells"]["quantity"][0]["value"] == "1"
    assert details["rows"][0]["row_evidence"]["method"] == "test"
    assert details["rows"][0]["family_votes"]["quantity"]["1"] == ["family-a", "family-b"]
    assert details["rows"][0]["corroborated_amount_mismatch"] == ["1", "2"]
    assert details["field_candidates"]["totals"]["total_ht"][0]["value"] == "20.00"
    assert details["taxes"][0]["amount"]["value"] == "1.00"
    assert details["discounts"][0]["value"] == "0.00"
    assert details["ocr"]["calls"] == 10
    assert details["ocr"]["preprocessing_used"] == ["deskew", "clahe"]
    assert details["rows_provenance"] == "recovered"
    assert details["grounding"]["provenance"]["supplier_name"][0]["raw"] == "Acme"
    assert details["grounding"]["field_statuses"]["supplier_name"] == "vérifié"

    assert evidence == evidence_snapshot
    assert diagnostics == diagnostics_snapshot
    json.dumps(serialized, ensure_ascii=False)


def test_serialized_evidence_is_accepted_by_frozen_p1_assessor():
    evidence = synthetic_evidence()
    serialized = serialize_evidence(evidence)
    raw = {
        "supplier_name": "Acme",
        "invoice_date": "2025-01-01",
        "total_ht": 20.0,
        "total_ttc": None,
        "line_items": [{"description": "Service", "quantity": 1,
                        "unit_price": 10.0, "total_ht": 10.0}],
    }
    strict = {
        "payload": {"supplier_name": None, "invoice_date": None, "total_ht": None,
                    "total_ttc": None,
                    "line_items": [{"description": None, "quantity": None,
                                    "unit_price": None, "total_ht": None}]},
        "statuses": {"supplier_name": "non lu", "invoice_date": "non lu",
                     "total_ht": "non lu", "total_ttc": "non lu",
                     "line_items[0].description": "non lu",
                     "line_items[0].quantity": "non lu",
                     "line_items[0].unit_price": "non lu",
                     "line_items[0].total_ht": "non lu"},
    }
    result = assess(serialized, raw, strict)
    assert result["supplier_name"].status == "proposed"
    assert result["invoice_date"].status == "proposed"
    assert result["line_items[0].description"].status == "proposed"
    assert result["line_items[0].quantity"].status == "proposed"
    assert result["line_items[0].unit_price"].status == "proposed"
    assert result["line_items[0].total_ht"].status == "proposed"


def test_run_case_parity_with_evidence_capture_enabled_and_disabled():
    evidence = synthetic_evidence()
    payload = {
        "supplier_name": "Acme", "invoice_date": "2025-01-01",
        "total_ht": 20.0, "total_ttc": None,
        "line_items": [{"description": "Service", "quantity": 1,
                        "unit_price": 10.0, "total_ht": 10.0}],
    }
    truth = copy.deepcopy(payload)
    raw = json.dumps(payload, ensure_ascii=False)
    case = {"id": "synthetic", "path": "synthetic.png", "truth": {"inline": truth}}

    def fake_extract(text, *, warnings=None, diagnostics=None):
        if warnings is not None:
            warnings.extend([])
        if diagnostics is not None:
            diagnostics.update({
                "first_pass_raw": raw,
                "payload": payload,
                "field_statuses": {key: "vérifié" for key in (
                    "supplier_name", "invoice_date", "total_ht", "total_ttc")},
                "validation": {"flags": [], "passed": [], "not_verified": []},
                "provenance": {"supplier_name": [{"raw": "Acme", "value": "Acme",
                                                  "page": 0, "box": [10, 10, 50, 20]}]},
                "unverified_model_readings": {},
                "warnings": [],
                "arithmetic_warnings": [],
                "attempts": [],
                "model_info": {"used": True, "reason": None, "passes_requested": 1,
                               "passes_succeeded": 1},
                "request_elapsed_seconds": 0.0,
                "ocr_configuration": {"version": "test"},
            })
        return payload

    with patch("app.main.document_text", return_value=EvidenceText(evidence)), \
            patch("app.main.extract_invoice", side_effect=fake_extract):
        off = run_case(case, capture_evidence=False)
        on = run_case(case, capture_evidence=True)

    assert "evidence_capture" not in off
    assert on["evidence_capture"]["schema_version"] == SCHEMA_VERSION
    assert on["evidence_capture"]["details"]["grounding"]["provenance"]

    comparable_off = {key: value for key, value in off.items() if key not in {"runtime", "evidence_capture"}}
    comparable_on = {key: value for key, value in on.items() if key not in {"runtime", "evidence_capture"}}
    assert comparable_off == comparable_on
