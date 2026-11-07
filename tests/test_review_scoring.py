import copy
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from invoice_evidence import DocumentEvidence, Number, TextLine, Token  # noqa: E402
from scripts.evidence_capture import serialize_evidence  # noqa: E402
from scripts.score_review_validation import (  # noqa: E402
    ScoringRefusal, flatten_truth, manifest_fingerprint, score_validation,
)

POLICY = ROOT / "experiments/review_assessment_policy.json"
LOCK = ROOT / "experiments/review_assessment_policy.sha256"
ASSESSOR = ROOT / "experiments/review_assessment.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_evidence(lines, *, supplier=None, invoice_date=None, totals=None):
    tokens, text_lines = [], []
    for index, text in enumerate(lines):
        token = Token(text, 0, (10, 10 + index * 15, 300, 22 + index * 15), 95.0)
        tokens.append(token)
        text_lines.append(TextLine(text, [token], 0, (10, 10 + index * 15, 300, 22 + index * 15)))
    return DocumentEvidence(
        text="\n".join(lines),
        locale="en-US",
        tokens=tokens,
        lines=text_lines,
        totals=totals or {"total_ht": [], "total_ttc": [], "total_candidate": []},
        supplier=supplier,
        supplier_is_explicit=bool(supplier),
        invoice_date=invoice_date,
        ocr_calls=10,
        ocr_pages=1,
        preprocessing_used=["deskew"],
    )


def number(value):
    return Number(str(value), Decimal(str(value)), 0, (0, 0, 10, 10))


def truth_doc(*, supplier="Acme", date="2025-01-01", total_ht=20.0, total_ttc=None,
              rows=None):
    return {
        "supplier_name": supplier,
        "invoice_date": date,
        "total_ht": total_ht,
        "total_ttc": total_ttc,
        "line_items": rows or [],
    }


def empty_strict(rows, statuses):
    return {
        "supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
        "line_items": [{"description": None, "quantity": None, "unit_price": None,
                        "total_ht": None} for _ in range(rows)],
    }, statuses


def all_unread(rows):
    statuses = {key: "non lu" for key in ("supplier_name", "invoice_date", "total_ht", "total_ttc")}
    for index in range(rows):
        for field in ("description", "quantity", "unit_price", "total_ht"):
            statuses[f"line_items[{index}].{field}"] = "non lu"
    return statuses


def make_case(tmp_path, case_id, *, truth, raw, strict, statuses, evidence,
              country="FR", layout="synthetic-layout", document_class="genuine_issued",
              eligible=True, omit_provenance=False, omit_approval=False,
              provenance=None, approval=None):
    base = tmp_path / "validation"
    for directory in ("images", "truth", "provenance", "approval"):
        (base / directory).mkdir(parents=True, exist_ok=True)
    image = base / "images" / f"{case_id}.png"
    image.write_bytes(f"image:{case_id}".encode("utf-8"))
    truth_file = base / "truth" / f"{case_id}.json"
    truth_file.write_text(json.dumps({"documents": {case_id: truth}}, ensure_ascii=False), encoding="utf-8")
    provenance = provenance or {
        "genuinely_issued": True,
        "previously_exposed_development": False,
        "used_in_p1_development": False,
        "truth_independent_of_model": True,
    }
    provenance_path = base / "provenance" / f"{case_id}.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    approval = approval or {"approved": True, "scope": "independent validation"}
    approval_path = base / "approval" / f"{case_id}.json"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    diagnostics = {"field_statuses": statuses,
                   "provenance": {"supplier_name": [{"raw": "Acme", "value": "Acme",
                                                     "page": 0, "box": [10, 10, 100, 22]}]},
                   "warnings": [], "validation": {"flags": [], "passed": [], "not_verified": []}}
    capture_case = {
        "id": case_id,
        "path": str(image),
        "model": {"raw": json.dumps(raw, ensure_ascii=False) if isinstance(raw, dict) else str(raw)},
        "final_payload": strict,
        "evidence_capture": serialize_evidence(evidence, diagnostics=diagnostics),
    }
    manifest_case = {
        "id": case_id,
        "path": str(image),
        "document_class": document_class,
        "eligible_independent": eligible,
        "country": country,
        "language": "fr" if country == "FR" else "en",
        "layout": layout,
        "truth": {"file": str(truth_file), "pointer": f"documents/{case_id}",
                  "sha256": sha(truth_file)},
        "provenance": {"file": str(provenance_path), "sha256": sha(provenance_path)},
        "privacy_approval": {"file": str(approval_path), "sha256": sha(approval_path)},
        "original_sha256": hashlib.sha256(f"original:{case_id}".encode()).hexdigest(),
        "anonymized_sha256": sha(image),
    }
    if omit_provenance:
        manifest_case["provenance"] = {}
    if omit_approval:
        manifest_case["privacy_approval"] = {}
    return manifest_case, capture_case, base


def build_fixture(tmp_path, cases, *, target=None, omit_provenance=False, omit_approval=False):
    manifest_cases, capture_cases = [], []
    for case in cases:
        manifest_case, capture_case, base = make_case(
            tmp_path, case["id"], truth=case["truth"], raw=case["raw"], strict=case["strict"],
            statuses=case["statuses"], evidence=case["evidence"], country=case["country"],
            layout=case.get("layout", "synthetic-layout"),
            document_class=case.get("document_class", "genuine_issued"),
            eligible=case.get("eligible", True),
            omit_provenance=omit_provenance, omit_approval=omit_approval,
            provenance=case.get("provenance"), approval=case.get("approval"),
        )
        manifest_cases.append(manifest_case)
        capture_cases.append(capture_case)
    manifest = {
        "schema_version": "validation-manifest-1",
        "frozen": True,
        "policy_sha256": sha(POLICY),
        "assessor_sha256": sha(ASSESSOR),
        "validation_target": target or {
            "documents": len(manifest_cases),
            "countries": {case["country"]: 1 for case in manifest_cases},
            "fields": 1, "rows": 0, "layouts": 1,
        },
        "cases": manifest_cases,
    }
    manifest_path = tmp_path / "validation" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    capture = {
        "manifest": {"sha256": manifest_fingerprint(manifest_cases)},
        "cases": capture_cases,
    }
    capture_path = tmp_path / "validation" / "bench.json"
    capture_path.write_text(json.dumps(capture, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest_path, capture_path


def test_incorrect_verified_total_is_counted_with_exact_denominators(tmp_path):
    truth = truth_doc(total_ht=100.0, total_ttc=120.0,
                      rows=[{"description": "Service", "quantity": 1,
                             "unit_price": 100.0, "total_ht": 100.0}])
    strict = {"supplier_name": "Acme", "invoice_date": "2025-01-01",
              "total_ht": 90.0, "total_ttc": 120.0,
              "line_items": [{"description": None, "quantity": None,
                              "unit_price": None, "total_ht": None}]}
    statuses = {"supplier_name": "vérifié", "invoice_date": "vérifié",
                "total_ht": "vérifié", "total_ttc": "vérifié",
                "line_items[0].description": "non lu", "line_items[0].quantity": "non lu",
                "line_items[0].unit_price": "non lu", "line_items[0].total_ht": "non lu"}
    evidence = make_evidence(["Acme", "Date: January 1, 2025", "Service 1 100.00 100.00"],
                             supplier="Acme", totals={"total_ht": [number("100.00")],
                                                      "total_ttc": [number("120.00")],
                                                      "total_candidate": []})
    raw = {"supplier_name": "Acme", "invoice_date": "2025-01-01",
           "total_ht": 100.0, "total_ttc": 120.0, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-001", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
    }])
    result = score_validation(manifest, capture, allow_partial=False)
    metrics = result["summary"]["overall"]
    assert metrics["verified_total"] == 4
    assert metrics["verified_correct"] == 3
    assert metrics["verified_incorrect"] == 1
    assert metrics["verified_precision"] == 0.75
    assert metrics["printed_fields"] == 8
    assert metrics["verified_printed_coverage"] == 0.5
    assert any(field["path"] == "total_ht" and field["outcome"] == "incorrect"
               for field in result["details"]["documents"][0]["fields"])


def test_quantity_price_swap_is_rejected_and_not_counted_as_visible(tmp_path):
    truth = truth_doc(total_ht=70.0, rows=[{"description": "Service", "quantity": 2,
                                            "unit_price": 35.0, "total_ht": 70.0}])
    strict, statuses = empty_strict(1, all_unread(1))
    evidence = make_evidence(["Acme", "Service 35 35.00 1225.00"], supplier="Acme",
                             totals={"total_ht": [number("70.00")], "total_ttc": [],
                                     "total_candidate": []})
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
           "line_items": [{"description": "Service", "quantity": 35,
                           "unit_price": 35.0, "total_ht": 1225.0}]}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-swap", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
    }])
    result = score_validation(manifest, capture, allow_partial=False)
    metrics = result["summary"]["overall"]
    assert metrics["rejected_total"] == 1
    assert metrics["correctly_rejected"] == 1
    assert metrics["proposed_total"] == 3
    assert metrics["proposed_correct"] == 2
    assert metrics["proposed_incorrect"] == 1
    assert result["summary"]["rows"]["exact_rows_same_index"] == 0


def test_missing_and_duplicated_rows_are_distinct(tmp_path):
    missing_truth = truth_doc(total_ht=None, total_ttc=None, rows=[
        {"description": "Service A", "quantity": 1, "unit_price": 10.0, "total_ht": 10.0},
        {"description": "Service B", "quantity": 1, "unit_price": 20.0, "total_ht": 20.0},
    ])
    strict, statuses = empty_strict(2, all_unread(2))
    missing_evidence = make_evidence(["Service A 1 10.00 10.00"], totals={
        "total_ht": [], "total_ttc": [], "total_candidate": []})
    missing_raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
                   "total_ttc": None,
                   "line_items": [{"description": "Service A", "quantity": 1,
                                   "unit_price": 10.0, "total_ht": 10.0}]}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "missing-rows", "country": "FR", "truth": missing_truth, "raw": missing_raw,
        "strict": strict, "statuses": statuses, "evidence": missing_evidence,
    }])
    result = score_validation(manifest, capture, allow_partial=False)
    assert result["summary"]["rows"]["missing_rows"] == 1
    assert result["summary"]["overall"]["missing_printed"] >= 4

    duplicate_truth = truth_doc(total_ht=None, total_ttc=None, rows=[
        {"description": "Service", "quantity": 1, "unit_price": 10.0, "total_ht": 10.0},
        {"description": "Service", "quantity": 1, "unit_price": 10.0, "total_ht": 10.0},
    ])
    duplicate_evidence = make_evidence(["Service 1 10.00 10.00", "Service 1 10.00 10.00"],
                                       totals={"total_ht": [], "total_ttc": [], "total_candidate": []})
    duplicate_raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
                     "total_ttc": None,
                     "line_items": [{"description": "Service", "quantity": 1,
                                     "unit_price": 10.0, "total_ht": 10.0},
                                    {"description": "Service", "quantity": 1,
                                     "unit_price": 10.0, "total_ht": 10.0}]}
    manifest, capture = build_fixture(tmp_path / "duplicate", [{
        "id": "duplicate-rows", "country": "FR", "truth": duplicate_truth,
        "raw": duplicate_raw, "strict": strict, "statuses": statuses,
        "evidence": duplicate_evidence,
    }])
    result = score_validation(manifest, capture, allow_partial=False)
    assert result["summary"]["rows"]["exact_rows_same_index"] == 2
    assert result["summary"]["rows"]["exact_rows_any_index"] == 2
    assert result["summary"]["rows"]["extra_rows"] == 0


def test_correct_abstention_and_unsupported_supplier_are_scored_separately(tmp_path):
    truth = truth_doc(supplier="Consult Me", date=None, total_ht=None, total_ttc=None, rows=[])
    strict, statuses = empty_strict(0, all_unread(0))
    evidence = make_evidence(["FACTURE"], supplier=None)
    raw = {"supplier_name": "SERVICES", "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-unsupported", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
    }])
    result = score_validation(manifest, capture, allow_partial=False)
    metrics = result["summary"]["overall"]
    assert metrics["correctly_rejected"] == 1
    assert metrics["correct_abstention"] == 3
    assert metrics["visible_total"] == 0


def test_refuses_missing_provenance_and_privacy_approval(tmp_path):
    truth = truth_doc()
    strict, statuses = empty_strict(0, all_unread(0))
    evidence = make_evidence([], supplier=None)
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-refuse", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
    }], omit_provenance=True, omit_approval=True)
    with pytest.raises(ScoringRefusal) as exc:
        score_validation(manifest, capture, allow_partial=True)
    problems = " ".join(exc.value.problems)
    assert "provenance" in problems
    assert "privacy" in problems


def test_refuses_incomplete_target_unless_partial_is_explicit(tmp_path):
    truth = truth_doc()
    strict, statuses = empty_strict(0, all_unread(0))
    evidence = make_evidence([], supplier=None)
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    target = {"documents": 15, "countries": {"FR": 5, "UK": 5, "US": 5},
              "fields": 120, "rows": 30, "layouts": 8}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-partial", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
    }], target=target)
    with pytest.raises(ScoringRefusal) as exc:
        score_validation(manifest, capture, allow_partial=False)
    assert "target" in " ".join(exc.value.problems)
    result = score_validation(manifest, capture, allow_partial=True)
    assert result["summary"]["status"] == "scored"


def test_refuses_templates_and_development_documents(tmp_path):
    truth = truth_doc()
    strict, statuses = empty_strict(0, all_unread(0))
    evidence = make_evidence([], supplier=None)
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "template", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
        "document_class": "template", "eligible": False,
    }])
    with pytest.raises(ScoringRefusal) as exc:
        score_validation(manifest, capture, allow_partial=True)
    problems = " ".join(exc.value.problems)
    assert "genuine_issued" in problems
    assert "eligible_independent" in problems


def test_refuses_provenance_content_contradicting_the_declaration(tmp_path):
    """A document cannot be scored as independent on declaration alone."""
    clean = {"genuinely_issued": True, "previously_exposed_development": False,
             "used_in_p1_development": False, "truth_independent_of_model": True}
    variants = {
        "genuinely_issued": ({**clean, "genuinely_issued": False}, "does not confirm genuine issuance"),
        "previously_exposed_development": ({**clean, "previously_exposed_development": True},
                                           "marks previous development exposure"),
        "used_in_p1_development": ({**clean, "used_in_p1_development": True},
                                   "marks P1 development exposure"),
        "truth_independent_of_model": ({**clean, "truth_independent_of_model": False},
                                       "does not confirm independent truth"),
    }
    for index, (field, (flags, expected)) in enumerate(variants.items()):
        case_dir = tmp_path / f"variant-{index}"
        truth = truth_doc()
        strict, statuses = empty_strict(0, all_unread(0))
        raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
               "total_ttc": None, "line_items": []}
        manifest, capture = build_fixture(case_dir, [{
            "id": f"fr-{field}", "country": "FR", "truth": truth, "raw": raw,
            "strict": strict, "statuses": statuses, "evidence": make_evidence([], supplier=None),
            "provenance": flags,
        }])
        with pytest.raises(ScoringRefusal) as exc:
            score_validation(manifest, capture, allow_partial=True)
        assert expected in " ".join(exc.value.problems), (field, exc.value.problems)


def test_refuses_unapproved_privacy_content(tmp_path):
    truth = truth_doc()
    strict, statuses = empty_strict(0, all_unread(0))
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-no-approval", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": make_evidence([], supplier=None),
        "approval": {"approved": False, "reason": "no owner privacy approval record"},
    }])
    with pytest.raises(ScoringRefusal) as exc:
        score_validation(manifest, capture, allow_partial=True)
    assert "privacy approval is not approved" in " ".join(exc.value.problems)


def test_refuses_falsified_truth_hash(tmp_path):
    truth = truth_doc()
    strict, statuses = empty_strict(0, all_unread(0))
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-hash", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": make_evidence([], supplier=None),
    }])
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cases"][0]["truth"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScoringRefusal) as exc:
        score_validation(manifest, capture, allow_partial=True)
    problems = " ".join(exc.value.problems)
    assert "truth" in problems and "mismatch" in problems


def test_scoring_is_deterministic_private_and_read_only(tmp_path):
    truth = truth_doc()
    strict, statuses = empty_strict(0, all_unread(0))
    evidence = make_evidence([], supplier=None)
    raw = {"supplier_name": None, "invoice_date": None, "total_ht": None,
           "total_ttc": None, "line_items": []}
    manifest, capture = build_fixture(tmp_path, [{
        "id": "fr-deterministic", "country": "FR", "truth": truth, "raw": raw,
        "strict": strict, "statuses": statuses, "evidence": evidence,
    }])
    before = {str(path): sha(path) for path in (manifest, capture)}
    first = score_validation(manifest, capture, allow_partial=False)
    second = score_validation(manifest, capture, allow_partial=False)
    assert json.dumps(first["summary"], sort_keys=True) == json.dumps(second["summary"], sort_keys=True)
    assert json.dumps(first["details"], sort_keys=True) == json.dumps(second["details"], sort_keys=True)
    after = {str(path): sha(path) for path in (manifest, capture)}
    assert before == after
    public = json.dumps(first["summary"], ensure_ascii=False)
    assert "Acme" not in public
    assert '"value"' not in public
    assert '"truth"' not in public
    assert "2025-01-01" not in public
    assert "fields" in first["details"]["documents"][0]
    assert "input_hashes" in first["summary"]


def test_private_output_root_is_git_ignored_and_unignored_paths_are_refused(tmp_path):
    import subprocess
    from scripts.score_review_validation import _default_output_root, _validate_output_path

    root = _default_output_root()
    ignored = subprocess.run(["git", "check-ignore", "-q", str(root / "example-run")],
                             cwd=ROOT, check=False)
    assert ignored.returncode == 0
    with pytest.raises(ScoringRefusal):
        _validate_output_path(tmp_path / "unignored-output", allow_unignored=False)
    _validate_output_path(tmp_path / "explicit-test-output", allow_unignored=True)
