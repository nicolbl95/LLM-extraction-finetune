"""PR110 Phase 6: focused tests for the benchmark-preparation tooling.

Every test here makes 0 OCR calls and 0 model calls: the intake is a temporary directory of synthetic images
that carry no printed invoice text, and the pipeline is never reached - the runner refuses before it imports
anything. What is under test is the part PR110 adds (eligibility classification, the annotation walkthrough,
the seal, the projector, the three readiness gates, freeze verification) plus the one behaviour PR110 must not
disturb: PR103's published validator on a `version: 1` reference.

`test_guide_records_the_owners_answers_and_not_the_questions` exists because the first version of the
walkthrough passed the builtin `str` where the answer reader belonged, which recorded the prompt text as the
owner's arithmetic basis and desynchronised the rest of the session.

The three source-version tests exist because a freeze that demanded zero drift against PR103's 29 sources
could only ever be satisfied by reverting every production change accepted since PR103. Readiness now
compares the tree with the source version the caller declares, PR103 when nothing is declared, and the
run-time refusal in `verify_freeze` stays the check that protects an evaluation from a source that moved
after its cohort was frozen.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr103_validate_reference as V  # noqa: E402
import pr106_harness as P  # noqa: E402
import pr109_intake as I  # noqa: E402
import pr110_annotate as A  # noqa: E402
import pr110_authority as AU  # noqa: E402
import pr110_eligibility as E  # noqa: E402
import pr110_evaluate as G  # noqa: E402
import pr110_harness as K  # noqa: E402
import pr110_ready as R  # noqa: E402


@pytest.fixture(autouse=True)
def declared_empty_document_authority(monkeypatch):
    """PR164: the screen now decides identity from a document authority.

    These tests measure the policy, not the corpora restored on the machine they run on, so they declare an
    authority that knows no document. That is different from the authority being absent: an absent registry
    holds every document for review (tested in test_pr164_eligibility_hardening.py).
    """
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: AU.empty())

DOCUMENT_DEFAULTS = {
    "language": "French", "region": "FR", "input_kind": "scanned_image",
    "genuine_business_invoice": True, "quality_declaration": "clear",
    "source_origin": "provided by the project owner", "annotators": ["owner"],
    "annotation_date": "2026-11-09",
    "owner_permission": {"authorized_for_local_evaluation": True,
                         "evidence": "the owner's own company's document", "date": "2026-11-09"},
    "second_review": {"performed": False, "reviewer": None, "disagreements_resolved": None},
}


# --------------------------------------------------------------------------- synthetic fixtures


def synthetic_image(path: Path, seed: int) -> Path:
    """A low-ink patterned page: distinct bytes per seed, no printed text, and not a blank page.

    A blank page would be a bad fixture - an average hash of whitespace sits within the duplicate threshold of
    unrelated blank documents, which is a documented limitation of the screen rather than something to pin.
    """
    image = Image.new("L", (640, 820), 255)
    draw = ImageDraw.Draw(image)
    for index in range(18):
        left = 30 + ((seed * 53 + index * 29) % 480)
        top = 40 + ((seed * 97 + index * 61) % 640)
        draw.rectangle([left, top, left + 70, top + 14], fill=60)
    draw.line([0, 800 - seed, 640, 780 - seed], fill=20, width=3)
    image.save(path)
    return path


def filled_annotation(document_id: str, image: Path, **document) -> dict:
    """A complete version-2 annotation - the object the guided CLI holds just before it seals."""
    annotation = K.blank(document_id, image.name, [1])
    annotation["document"].update({"page_count_observed": 1, **DOCUMENT_DEFAULTS, **document})
    K.set_field(annotation, "supplier_name", "Atelier Test", "printed")
    K.set_field(annotation, "invoice_date", "2026-01-15", "printed")
    K.set_field(annotation, "currency", "EUR", "printed")
    K.set_field(annotation, "total_ht", 1000.0, "printed")
    K.set_field(annotation, "total_ttc", 1200.0, "printed")
    K.set_field(annotation, "tax.rate", 20.0, "printed")
    K.set_field(annotation, "tax.amount", 200.0, "printed")
    K.set_field(annotation, "shipping", None, "absent")
    K.add_row(annotation)
    for field, value in (("description", "Prestation de service"), ("quantity", 5.0),
                         ("unit_price", 200.0), ("total_ht", 1000.0)):
        K.set_field(annotation, f"line_items[0].{field}", value, "printed")
    return annotation


def sealed_annotation(document_id: str, image: Path, **document) -> dict:
    annotation = filled_annotation(document_id, image, **document)
    K.seal(annotation, image)
    return annotation


def write_intake(tmp_path: Path) -> tuple[Path, Path]:
    invoices, references = tmp_path / "invoices", tmp_path / "references"
    invoices.mkdir(exist_ok=True)
    references.mkdir(exist_ok=True)
    return invoices, references


def add_document(invoices: Path, references: Path, document_id: str, seed: int, **document) -> dict:
    image = synthetic_image(invoices / f"{document_id}.png", seed)
    annotation = sealed_annotation(document_id, image, **document)
    (references / f"{document_id}.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return {"image": image, "annotation": annotation}


def status_of(report: dict, name: str) -> dict:
    """Look a screened document up by file name rather than by hash: an exact duplicate shares its hash with
    the file it duplicates, so the hash is not a unique key inside a report that deliberately contains copies."""
    matches = [document for document in report["documents"] if document["document"] == name]
    assert len(matches) == 1, matches
    return matches[0]


def write_freeze(directory: Path, freeze: dict) -> Path:
    path = directory / "cohort-freeze.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    return path


def scale_protocol(monkeypatch, documents: int) -> None:
    """Shrink the readiness protocol to `documents` for a fixture that cannot hold twelve.

    `TARGET_DOCUMENTS` and `COHORT_QUOTA` are one protocol, not two knobs: the quota sums to the target, and
    `select_cohort` reports a protocol that no longer adds up as a finding. A test that lowers one without the
    other stops measuring the gate it names and measures PR167's coverage refusal instead - so both move
    together, onto a single country whose slot the fixture's FR documents fill.
    """
    monkeypatch.setattr(K, "TARGET_DOCUMENTS", documents)
    monkeypatch.setattr(K, "COHORT_QUOTA", {"FR": documents})


def write_source_manifest(directory: Path, **changes) -> Path:
    """Declare the source version the cohort is frozen against, as the current tree measures it.

    `changes` rewrite individual entries, so a test can declare a version the tree does not match and see
    whether the gate still says so.
    """
    hashes = {name: dict(value) for name, value in P.source_hashes().items()}
    for name, value in changes.items():
        if value is None:
            hashes.pop(name)
        else:
            hashes[name] = {"sha256": value, "bytes": 1}
    path = directory / "source-manifest.json"
    path.write_text(json.dumps({"source_hashes": hashes}), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- validator: v1 stays published


def v1_reference() -> dict:
    """A reference in PR103's vocabulary that carries two v2-only states on purpose."""
    return {
        "version": 1,
        "document": {
            "id": "fr-001", "source_file": "fr-001.png", "sha256": "deadbeef", "language": "fr",
            "region": "CA", "input_kind": "mixed", "annotation_date": "2026-11-09",
            "annotators": ["owner"], "ocr_output_seen_before_annotation": False,
            "owner_permission": {"authorized_for_local_evaluation": True, "evidence": "own document"},
            "disagreements": [], "pages": [1], "quality_declaration": "nonsense",
            "second_review": {"performed": None},
        },
        "reference": {
            "supplier_name": {"value": "Atelier Dupont", "provenance": "printed"},
            "invoice_date": {"value": "2025-03-04", "provenance": "printed"},
            "currency": {"value": "EUR", "provenance": "printed"},
            "total_ht": {"value": 100, "provenance": "derived"},
            "total_ttc": {"value": 120, "provenance": "printed"},
            "tax": {"rate": {"value": 20, "provenance": "derived"},
                    "amount": {"value": 20, "provenance": "printed"}},
            "shipping": {"value": None, "provenance": "ambiguous"},
            "line_items": [{"description": {"value": "Prestation", "provenance": "printed"},
                            "quantity": {"value": 2, "provenance": "printed"},
                            "unit_price": {"value": 50, "provenance": "printed"},
                            "total_ht": {"value": 100, "provenance": "printed"},
                            "is_product_or_service": True}],
            "document_inconsistencies": [],
        },
        "freeze": {"reference_sha256": None, "frozen_before_pipeline_run": None},
    }


def test_version_one_reference_keeps_the_published_pr103_problems():
    """PR110's additive edit must not add, drop or reword a rule for a version-1 reference."""
    assert V.validate(v1_reference()) == [
        "document.region must be FR, UK, US or other",
        "document.input_kind must be scanned_image, native_pdf or photograph",
        "reference.total_ht.provenance must be one of ['absent', 'illegible', 'printed']",
        "reference.total_ht is marked derived but records no derived_from arithmetic basis",
        "reference.shipping.provenance must be one of ['absent', 'illegible', 'printed']",
        "reference.shipping is marked ambiguous but records no note explaining what is undecidable",
        "freeze.reference_sha256 is required",
        "freeze.frozen_before_pipeline_run must be true",
    ]


def test_version_one_asks_for_none_of_the_version_two_metadata():
    """pages, second review and genuine-invoice are v2 requirements; v1 accepts the same body unchanged."""
    reference = v1_reference()
    reference["document"].update({"region": "FR", "input_kind": "photograph", "pages": None,
                                  "quality_declaration": None, "second_review": {"performed": None}})
    reference["reference"].update({"total_ht": {"value": 100, "provenance": "printed"},
                                   "shipping": {"value": None, "provenance": "absent"},
                                   "tax": {"rate": {"value": 20, "provenance": "printed"},
                                           "amount": {"value": 20, "provenance": "printed"}}})
    reference["freeze"] = {"reference_sha256": "abc", "frozen_before_pipeline_run": True}
    assert V.validate(reference) == []
    reference["version"] = 2
    assert V.validate(reference) == [
        "document.pages must be a non-empty list of 1-based page numbers covering this invoice and no other",
        "document.second_review.performed must be recorded true or false; PR109 found 0 of 11 inherited "
        "references with a second reviewer",
        "document.genuine_business_invoice must be recorded true or false: PR109 could only guess whether a "
        "cohort document was a template",
    ]


def test_version_two_closes_the_tax_object_gap_that_v1_left_open():
    tax = {"rate": {"value": 20, "provenance": "guessed"}, "amount": {"value": 20, "provenance": "printed"}}
    reference = v1_reference()
    reference["document"].update({"region": "FR", "input_kind": "photograph", "quality_declaration": None})
    reference["reference"].update({"tax": tax, "total_ht": {"value": 100, "provenance": "printed"},
                                   "shipping": {"value": None, "provenance": "absent"}})
    reference["freeze"] = {"reference_sha256": "abc", "frozen_before_pipeline_run": True}
    assert V.validate(reference) == []
    reference["version"] = 2
    reference["document"].update({"pages": [1], "second_review": {"performed": False},
                                  "genuine_business_invoice": True})
    assert V.validate(reference) == [
        "reference.tax.rate.provenance must be one of ['absent', 'ambiguous', 'derived', 'illegible', "
        "'printed']"]


# --------------------------------------------------------------------------- projector


def test_project_scores_printed_and_derived_and_nulls_the_rest():
    annotation = filled_annotation("doc-x", Path(__file__))
    K.set_field(annotation, "total_ht", None, "absent")
    K.set_field(annotation, "invoice_date", None, "ambiguous", note="two digits overlap")
    K.set_field(annotation, "line_items[0].unit_price", None, "illegible")
    K.set_field(annotation, "line_items[0].total_ht", 1000.0, "derived",
                derived_from="quantity x unit_price")

    projected = K.project(annotation)
    assert projected["reference"]["total_ht"] is None
    assert projected["reference"]["invoice_date"] is None
    assert projected["reference"]["line_items"][0]["unit_price"] is None
    assert projected["reference"]["line_items"][0]["total_ht"] == 1000.0
    counts = projected["counts"]
    assert counts["by_provenance"] == {"printed": 4, "derived": 1, "absent": 1, "illegible": 1,
                                       "ambiguous": 1}
    assert counts["not_annotated"] == 0
    assert counts["scored_values"] == 5
    assert projected["unverified_fields"] == ["invoice_date", "total_ht", "line_items[0].unit_price"]


def test_an_unannotated_field_is_reported_and_never_guessed():
    annotation = filled_annotation("doc-x", Path(__file__))
    annotation["reference"]["invoice_date"] = {"value": None, "provenance": None}
    projected = K.project(annotation)
    assert projected["counts"]["not_annotated"] == 1
    assert "invoice_date" in projected["unverified_fields"]
    assert sorted(A.missing_provenance(annotation)) == ["invoice_date"]


# --------------------------------------------------------------------------- seal


def test_seal_then_verify_seal_detects_an_edit_after_freezing(tmp_path):
    invoices, _ = write_intake(tmp_path)
    image = synthetic_image(invoices / "doc-1.png", 3)
    annotation = sealed_annotation("doc-1", image)
    assert K.verify_seal(annotation, image)["intact"] is True

    edited = copy.deepcopy(annotation)
    edited["reference"]["total_ttc"]["value"] = 1300.0
    state = K.verify_seal(edited, image)
    assert state["intact"] is False
    assert state["reference_sha256_matches"] is False
    assert state["body_sha256_matches"] is False

    other = synthetic_image(invoices / "other.png", 9)
    assert K.verify_seal(annotation, other)["sha256_matches_image_on_disk"] is False


def test_seal_refuses_while_the_validator_reports_a_problem(tmp_path):
    invoices, _ = write_intake(tmp_path)
    image = synthetic_image(invoices / "doc-2.png", 4)
    annotation = filled_annotation("doc-2", image)
    annotation["reference"]["supplier_name"]["provenance"] = "guessed"
    with pytest.raises(SystemExit) as refused:
        K.seal(annotation, image)
    assert "not sealable" in str(refused.value)
    assert annotation.get("freeze", {}).get("reference_sha256") is None


# --------------------------------------------------------------------------- the guided walkthrough

# DOCUMENT_STEPS minus `pages`, which blank() already fills, and minus the two second_review follow-ups,
# which the guide skips after a `n` answer to `performed`.
DOCUMENT_ANSWERS = [
    "owner's own accounting records",  # source_origin
    "y",                               # genuine_business_invoice
    "y",                               # owner_permission.authorized_for_local_evaluation
    "it is the owner's own company's document",  # owner_permission.evidence
    "French",                          # language
    "FR",                              # region
    "scanned_image",                   # input_kind
    "1",                               # page_count_observed
    "clear",                           # quality_declaration
    "owner, reviewer-b",               # annotators
    "2026-11-09",                      # annotation_date
    "n",                               # second_review.performed
]


def reader(answers: list[str]):
    def next_answer(prompt: str) -> str:
        assert answers, f"the guide asked an unexpected question: {prompt!r}"
        return answers.pop(0)

    return next_answer


HEADER_ANSWERS = [
    "p", "Atelier Test",                 # supplier_name
    "p", "2026-01-15",                   # invoice_date
    "p", "EUR",                          # currency
    "d", "100,00", "total_ttc - tax_amount",  # total_ht, derived with its basis
    "p", "1 200,00",                     # total_ttc
    "p", "20",                           # tax.rate
    "a",                                 # tax.amount absent
    "m", "the figure sits under two column headers",  # shipping ambiguous
]
ROW_ANSWERS = [
    "y",                                 # add line item 0
    "p", "Prestation de service",        # description
    "p", "6",                            # quantity
    "p", "200,00",                       # unit_price
    "p", "1 200,00",                     # total_ht
    "CMD-2026-04",                       # printed reference number
    "n",                                 # no further line items
]


def test_guide_records_the_owners_answers_and_not_the_questions():
    annotation = K.blank("doc-guide", "doc-guide.png", [1])
    answers = DOCUMENT_ANSWERS + HEADER_ANSWERS + ROW_ANSWERS
    A.guide(annotation, reader(answers))
    assert not answers, "the walkthrough asked fewer questions than the test answered"

    document = annotation["document"]
    assert document["genuine_business_invoice"] is True
    assert document["owner_permission"]["authorized_for_local_evaluation"] is True
    assert document["owner_permission"]["evidence"] == "it is the owner's own company's document"
    assert document["annotators"] == ["owner", "reviewer-b"]
    assert document["second_review"]["performed"] is False

    body = annotation["reference"]
    assert (body["total_ht"]["value"], body["total_ht"]["provenance"],
            body["total_ht"]["derived_from"]) == (100.0, "derived", "total_ttc - tax_amount")
    assert body["total_ttc"]["value"] == 1200.0
    assert (body["tax"]["amount"]["value"], body["tax"]["amount"]["provenance"]) == (None, "absent")
    assert body["shipping"]["provenance"] == "ambiguous"
    assert body["shipping"]["note"] == "the figure sits under two column headers"
    row = body["line_items"][0]
    assert (row["quantity"]["value"], row["unit_price"]["value"], row["total_ht"]["value"]) == (
        6.0, 200.0, 1200.0)
    assert row["reference_number"] == "CMD-2026-04"

    serialised = json.dumps(annotation)
    for question in ("which printed values and what arithmetic?", "what is undecidable here?",
                     "an order/reference number printed on this line", "<class 'str'>"):
        assert question not in serialised, f"the guide recorded the question instead of the answer"


def test_the_guide_asks_nothing_it_has_already_answered():
    annotation = K.blank("doc-guide", "doc-guide.png", [1])
    A.guide(annotation, reader(DOCUMENT_ANSWERS + HEADER_ANSWERS), rows=False)

    prompts: list[str] = []
    A.guide(annotation, lambda prompt: prompts.append(prompt) or "", rows=False)
    assert prompts == [], prompts

    A.guide(annotation, lambda prompt: prompts.append(prompt) or "", rows=False, redo=True)
    assert "Where did this invoice come from" in prompts[0]


def test_the_annotation_tool_cannot_see_an_extraction_result():
    """Blind by construction: the CLI that writes references never imports the pipeline."""
    source = (ROOT / "scripts" / "pr110_annotate.py").read_text(encoding="utf-8")
    for forbidden in ("app.main", "document_text", "extract_invoice", "pytesseract", "evaluation_metrics",
                      "torch", "pr106_harness"):
        assert forbidden not in source, forbidden


def test_parse_value_keeps_the_printed_form_of_a_french_number():
    assert A.parse_value("1 200,00") == 1200.0
    assert A.parse_value("1 234,56 €") == "1 234,56 €"
    assert A.parse_value("Atelier 2") == "Atelier 2"
    assert A.parse_value("20") == 20.0


# --------------------------------------------------------------------------- eligibility


def test_eligibility_separates_eligible_excluded_and_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "signed_population", lambda: [])
    monkeypatch.setattr(K, "TARGET_DOCUMENTS", 2)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "elig-1", 1)
    add_document(invoices, references, "elig-2", 2)

    # the copy is named so it sorts after the file it duplicates: the screen flags whichever copy of a byte
    # sequence it meets second, and that file adds nothing to the cohort
    shutil.copyfile(invoices / "elig-1.png", invoices / "zz-copy-of-elig-1.png")

    template = synthetic_image(invoices / "template.png", 7)
    (references / "template.json").write_text(json.dumps(
        sealed_annotation("template", template, genuine_business_invoice=False,
                          template_description="vendor demo sheet"), ensure_ascii=False), encoding="utf-8")

    # an annotation edited after sealing to withdraw permission: the validator stops it, the screen excludes it
    refused = add_document(invoices, references, "refused", 5)
    denied = copy.deepcopy(refused["annotation"])
    denied["document"]["owner_permission"]["authorized_for_local_evaluation"] = False
    (references / "refused.json").write_text(json.dumps(denied, ensure_ascii=False), encoding="utf-8")

    unannotated = synthetic_image(invoices / "unannotated.png", 11)
    unsealed = synthetic_image(invoices / "unsealed.png", 13)
    (references / "unsealed.json").write_text(
        json.dumps(filled_annotation("unsealed", unsealed), ensure_ascii=False), encoding="utf-8")

    used = ROOT / "tests" / "fixtures" / "france-blur-light.png"
    shutil.copyfile(used, invoices / "previously-used.png")

    report = E.eligibility(invoices, references, geometry=False)
    assert status_of(report, "elig-1")["status"] == E.ELIGIBLE

    copy_of_known = status_of(report, "zz-copy-of-elig-1")
    assert copy_of_known["status"] == E.EXCLUDED
    assert copy_of_known["hard_reasons"] == ["duplicate_inside_the_intake"]
    assert "elig-1.png" in next(check["detail"] for check in copy_of_known["checks"]
                                if check["check"] == "duplicate_inside_the_intake")

    already_used = status_of(report, "previously-used")
    assert already_used["status"] == E.EXCLUDED
    assert "exact_duplicate" in already_used["hard_reasons"]
    assert "previous_use" in already_used["hard_reasons"]

    assert status_of(report, "template")["hard_reasons"] == ["genuine_invoice"]
    assert status_of(report, "refused")["hard_reasons"] == ["permission"]
    assert status_of(report, "unannotated")["open_questions"] == ["reference_annotation"]
    assert status_of(report, "unsealed")["open_questions"] == ["reference_annotation", "seal"]

    assert report["counts"] == {E.ELIGIBLE: 2, E.EXCLUDED: 4, E.PENDING_REVIEW: 2}
    assert len(report["eligible_ids"]) == 2 and report["shortage"] == 0


def test_eligibility_catches_a_recompressed_copy_and_names_what_it_matched(tmp_path, monkeypatch):
    """PR164: a perceptual resemblance no longer excludes on its own, so this copy is held for review.

    PR163 measured two different documents inside the same 10-bit radius; the finding still blocks automatic
    admission, and the registered-variant check is what turns a resemblance into an exclusion
    (test_pr164_eligibility_hardening.py measures both).
    """
    invoices, references = write_intake(tmp_path)
    original = add_document(invoices, references, "orig", 21)
    signature = I.perceptual(original["image"])
    monkeypatch.setattr(I, "signed_population", lambda: [
        {"path": "somewhere/already-measured.png", "sha256": "0" * 64,
         "ahash": signature["ahash"], "dhash": signature["dhash"]}])

    recompressed = invoices / "recompressed.jpg"
    Image.open(original["image"]).convert("RGB").save(recompressed, "JPEG", quality=75)

    report = E.eligibility(invoices, references, geometry=False)
    document = status_of(report, "recompressed")
    assert document["status"] == E.PENDING_REVIEW
    assert document["review_label"] == "MANUAL_REVIEW_REQUIRED"
    assert document["hard_reasons"] == []
    assert "visually_similar_duplicate" in document["open_questions"]
    finding = next(check for check in document["checks"] if check["check"] == "visually_similar_duplicate")
    assert "somewhere/already-measured.png" in finding["detail"], "the finding must name what it matched"
    assert "resemblance only" in finding["detail"]


def test_eligibility_pairs_a_reference_to_a_renamed_document_by_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "signed_population", lambda: [])
    invoices, references = write_intake(tmp_path)
    document = add_document(invoices, references, "renamed", 31)
    document["image"].rename(invoices / "totally-different-name.png")

    report = E.eligibility(invoices, references, geometry=False)
    record = report["documents"][0]
    assert record["status"] == E.ELIGIBLE and record["id"] == "renamed"


def test_a_file_holding_more_than_one_invoice_is_held_for_review(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "signed_population", lambda: [])
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "partial", 41, pages=[1], page_count_observed=2)

    report = E.eligibility(invoices, references, geometry=False)
    record = report["documents"][0]
    assert record["status"] == E.PENDING_REVIEW
    assert record["open_questions"] == ["one_invoice_per_file"]


# --------------------------------------------------------------------------- readiness gates


def test_readiness_refuses_an_empty_cohort_and_writes_no_freeze(tmp_path, monkeypatch):
    invoices, references = write_intake(tmp_path)
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 2)
    monkeypatch.setattr(R, "ARTIFACT_DIR", tmp_path)

    report = R.readiness(invoices, references, geometry=False)
    assert report["verdict"] == R.NOT_READY
    assert report["intake_gate"]["passed"] is False and report["intake_gate"]["shortage"] == 2
    assert report["annotation_gate"]["passed"] is False
    assert report["annotation_gate"]["references"] == 0
    assert any("no reference file exists yet" in finding for finding in report["blocking_findings"])
    assert report["freeze"] is None and report["launch_command"] is None

    assert R.main(["--invoices", str(invoices), "--references", str(references),
                   "--output-name", "readiness.json", "--no-geometry"]) == 1
    assert (tmp_path / "readiness.json").is_file()
    assert not (tmp_path / "cohort-freeze.json").exists()


def test_readiness_freezes_a_cohort_that_passes_all_three_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 2)
    invoices, references = write_intake(tmp_path)
    first = add_document(invoices, references, "f-01", 51)
    add_document(invoices, references, "f-02", 52)

    report = R.readiness(invoices, references, geometry=False,
                         source_manifest=write_source_manifest(tmp_path))
    assert report["verdict"] == R.READY, report["blocking_findings"]
    assert report["coverage_gate"]["passed"] is True
    freeze = report["freeze"]
    assert [entry["id"] for entry in freeze["documents"]] == ["f-01", "f-02"]
    assert [entry["region"] for entry in freeze["documents"]] == ["FR", "FR"]
    assert freeze["selection"]["selected_ids"] == ["f-01", "f-02"]
    assert {key: value for key, value in freeze["gates"].items() if key != "note"} == {
        "intake_gate": True, "annotation_gate": True, "coverage_gate": True}
    assert freeze["document_authority"]["scope"] == "fixture"
    assert freeze["frozen_before_any_pipeline_run"] is True
    assert freeze["experimental_switches"]["all_off"] is True
    assert freeze["production_sources"]["pr103_drift"]["drifted"] == []
    assert freeze["cohort_sha256"] == K.digest([entry["reference_sha256"] for entry in freeze["documents"]])
    expected_freeze = K.display(R.ARTIFACT_DIR / "pr110-phase5-cohort-freeze.json")
    assert report["launch_command"] == (f'"{sys.executable}" scripts/pr110_evaluate.py '
                                        f"--freeze {expected_freeze}"), "the command must name its interpreter"

    freeze_path = write_freeze(tmp_path, freeze)
    verification = G.verify_freeze(freeze, invoices, references)
    assert verification["ok"] is True, verification["problems"]
    assert {document["id"] for document in verification["documents"]} == {"f-01", "f-02"}

    edited = copy.deepcopy(first["annotation"])
    edited["reference"]["total_ttc"]["value"] = 1201.0
    (references / "f-01.json").write_text(json.dumps(edited), encoding="utf-8")
    tampered = G.verify_freeze(json.loads(freeze_path.read_text(encoding="utf-8")), invoices, references)
    assert tampered["ok"] is False
    assert any("seal does not recompute" in problem for problem in tampered["problems"])

    (references / "f-01.json").write_text(json.dumps(first["annotation"]), encoding="utf-8")
    scale_protocol(monkeypatch, 3)
    short = R.readiness(invoices, references, geometry=False)
    assert short["verdict"] == R.NOT_READY and short["freeze"] is None
    assert short["intake_gate"]["eligible"] == 2 and short["intake_gate"]["shortage"] == 1
    assert short["coverage_gate"]["report"] == "FR: 2/3" and short["coverage_gate"]["missing"] == {"FR": 1}


# --------------------------------------------------------------------------- the declared source version


def test_readiness_names_pr103_when_no_source_version_is_declared(tmp_path, monkeypatch):
    """PR103 stays the historical baseline by default; nothing here re-freezes it."""
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 71)

    default = R.readiness(invoices, references, geometry=False)["freeze"]["production_sources"]["pr103_drift"]
    published = json.loads((ROOT / "results/product-review103/pr103-implementation-manifest.json")
                           .read_text(encoding="utf-8"))["source_hashes"]
    current = P.source_hashes()
    assert default["manifest"] == "results/product-review103/pr103-implementation-manifest.json"
    assert default["manifest_defaulted_to_pr103"] is True
    assert default["pr103_sources_hashed"] == 29 == len(published)
    # the drift is reported, not assumed: whichever of PR103's sources the tree now differs from are named
    assert default["drifted"] == sorted(name for name, value in published.items()
                                        if name in current and current[name]["sha256"] != value["sha256"])
    assert default["manifest_sha256"] == hashlib.sha256(
        (ROOT / "results/product-review103/pr103-implementation-manifest.json").read_bytes()).hexdigest()

    declared = R.readiness(invoices, references, geometry=False,
                           source_manifest=write_source_manifest(tmp_path))["freeze"]["production_sources"]
    assert declared["pr103_drift"]["manifest_defaulted_to_pr103"] is False
    assert declared["pr103_drift"]["drifted"] == [] and declared["pr103_drift"]["added_since_pr103"] == []
    assert declared["sha256"] == K.digest(P.source_hashes())


def test_readiness_still_reports_a_source_that_does_not_match_its_declared_version(tmp_path, monkeypatch):
    """A declared manifest is a claim the tree has to answer, not a way to make drift disappear."""
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 72)
    renamed = sorted(P.source_hashes())[0]

    drifted = R.readiness(invoices, references, geometry=False, source_manifest=write_source_manifest(
        tmp_path, **{renamed: "0" * 64}))["freeze"]["production_sources"]["pr103_drift"]
    assert drifted["drifted"] == [renamed]
    assert drifted["manifest_defaulted_to_pr103"] is False

    added = R.readiness(invoices, references, geometry=False, source_manifest=write_source_manifest(
        tmp_path, **{renamed: None}))["freeze"]["production_sources"]["pr103_drift"]
    assert added["drifted"] == [] and added["added_since_pr103"] == [renamed]
    assert added["manifest_sha256"] != drifted["manifest_sha256"]


def test_evaluation_refuses_sources_that_changed_after_the_freeze(tmp_path, monkeypatch):
    """The run-time anti-tamper is independent of which manifest readiness compared against."""
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 73)
    freeze = R.readiness(invoices, references, geometry=False,
                         source_manifest=write_source_manifest(tmp_path))["freeze"]
    assert G.verify_freeze(freeze, invoices, references)["ok"] is True

    changed = {name: {"sha256": "1" * 64, "bytes": 1} for name in P.source_hashes()}
    monkeypatch.setattr(P, "source_hashes", lambda: changed)  # an edit landed after the cohort was frozen
    verification = G.verify_freeze(freeze, invoices, references)
    assert verification["ok"] is False
    assert any("production sources changed after the freeze" in problem
               for problem in verification["problems"])
    refused = G.run(write_freeze(tmp_path, freeze), invoices, references)
    assert refused["verdict"] == "REFUSED" and refused["documents"] == [] and refused["aggregate"] is None


# --------------------------------------------------------------------------- the evaluation runner


def test_run_refuses_a_freeze_that_does_not_recompute(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 61)
    freeze = R.readiness(invoices, references, geometry=False)["freeze"]

    synthetic_image(invoices / "solo.png", 62)  # the bytes on disk are no longer the frozen document

    report = G.run(write_freeze(tmp_path, freeze), invoices, references)
    assert report["verdict"] == "REFUSED"
    assert report["documents"] == [] and report["aggregate"] is None
    assert any("seal does not recompute" in problem for problem in report["verification"]["problems"])
    assert any("not present in the intake directory" in problem
               for problem in report["verification"]["problems"])


def test_run_refuses_a_french_cohort_when_the_french_data_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "signed_population", lambda: [])
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "fr-01", 71, language="French", region="FR")
    freeze = R.readiness(invoices, references, geometry=False)["freeze"]

    import app.main as application

    monkeypatch.setattr(application, "ocr_language_plan", lambda **kwargs: {
        "requested": ["fra", "eng"], "effective": ["eng"], "missing": ["fra"], "fallback": True,
        "installed": ["eng"], "tessdata_prefix": None})

    def never_called(*args, **kwargs):
        raise AssertionError("an English-only read of a French document must never be attempted")

    monkeypatch.setattr(application, "document_text", never_called)
    report = G.run(write_freeze(tmp_path, freeze), invoices, references)
    assert report["verdict"] == "REFUSED"
    assert "fra" in report["reason"]
    assert report["french_documents"] == ["fr-01"]
    assert report["language_plan"]["missing"] == ["fra"]
    assert report["documents"] == [] and report["aggregate"] is None


def test_evaluate_reports_a_missing_freeze_without_running_anything(tmp_path, capsys):
    assert G.main(["--freeze", str(tmp_path / "no-such-freeze.json")]) == 1
    assert "REFUSED" in capsys.readouterr().out


# --------------------------------------------------------------------------- safety of the ground state


def test_experimental_paths_are_off_and_the_model_guard_is_measured():
    switches = K.experimental_switches_off()
    assert switches["all_off"] is True, switches
    assert switches["policy_version"], "the OCR policy fingerprint must be readable"
    guard = P.environment().get("full_precision_qwen_on_cpu")
    assert guard and guard.get("verdict"), guard


def test_the_owner_guide_names_only_commands_that_exist():
    guide = (ROOT / "benchmark_prep" / "OWNER_GUIDE.md").read_text(encoding="utf-8")
    for script in ("pr110_status.py", "pr110_annotate.py", "pr110_eligibility.py", "pr110_ready.py",
                   "pr110_evaluate.py"):
        assert f"scripts/{script}" in guide, script
    for sub_command in ("new", "guide", "check", "seal", "summary", "show"):
        assert f"pr110_annotate.py {sub_command}" in guide, sub_command
    for token in ("benchmark_prep/invoices", "benchmark_prep/references", "TESSDATA_PREFIX", "printed",
                  "derived", "absent", "illegible", "ambiguous", "pending_review", "eligible", "excluded"):
        assert token in guide, token
    assert K.gitignore_covered()["all_covered"] is True
