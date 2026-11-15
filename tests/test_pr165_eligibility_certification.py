"""PR165: the two defects the certification of the PR110 eligibility control found.

Both were demonstrated before being fixed, and both let a document reach the frozen cohort that should never
have reached it:

* `pr110_ready.readiness()` counted the references that validate and seal, and the eligibility screen that
  decides independence was reported beside that number without constraining it. A reference can be perfectly
  annotated on a document the screen excludes as already used, or holds for manual review, and the cohort was
  frozen anyway (Phases 4 and 6).
* A document that entered development use after the document authority was written is absent from it, so
  nothing recognises it and the screen admits it as independent. The registry cannot see that from its own
  contents; the directories it names images from are the claim it makes, and an image in one of them that it
  does not name is now reported (Phase 5). PR167 moved that registry to
  `benchmark_prep/document-authority.private.json`, ignored by Git, and left the schema and this policy in the
  tracked demonstration registry.

Everything here runs on controlled fixtures under `tmp_path`. Nothing writes into the real intake, and the
PR110 cohort stays empty until the owner supplies twelve independent documents. 0 OCR calls, 0 model calls,
0 pipeline runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr109_intake as I  # noqa: E402
import pr110_authority as A  # noqa: E402
import pr110_eligibility as E  # noqa: E402
import pr110_harness as K  # noqa: E402
import pr110_ready as R  # noqa: E402

from test_pr110_benchmark_prep import add_document, scale_protocol, write_intake  # noqa: E402
from pr110_prior_use import install as install_prior_use  # noqa: E402


@pytest.fixture(autouse=True)
def no_campaign_population(monkeypatch):
    """The fixtures are synthetic; the repository's own signed images must not collide with them."""
    monkeypatch.setattr(I, "signed_population", lambda: [])


@pytest.fixture(autouse=True)
def declared_empty_prior_use(monkeypatch, tmp_path):
    """PR169: the screen reads an inventory of past use, and PR103's published one is not in a clone.

    Nothing in this file measures that inventory - it measures the gates built on top of the screen - so the
    fixture supplies an empty one in PR103's own format. A test that was passing before keeps passing for the
    same reason: the documents it builds are still new to the record it now reads.
    """
    install_prior_use(monkeypatch, tmp_path)


@pytest.fixture
def declared_authority(monkeypatch):
    """Independence decided by an authority that knows no document, so the admissibility layer is what acts."""
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: A.empty())


# --------------------------------------------------------------------------- the gate, not the report


def test_a_document_held_for_manual_review_is_neither_counted_nor_frozen(declared_authority, tmp_path,
                                                                        monkeypatch):
    """Phase 4: `MANUAL_REVIEW_REQUIRED` has to stop the cohort, not merely decorate the report."""
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "good", 201)
    add_document(invoices, references, "held", 202, pages=[1], page_count_observed=2)

    report = R.readiness(invoices, references, geometry=False)
    frozen = [entry["id"] for entry in report["freeze"]["documents"]]
    assert frozen == ["good"], f"the freeze admitted a document the screen holds: {frozen}"
    assert report["intake_gate"]["eligible"] == 1 == len(frozen)
    held = report["intake_gate"]["annotated_not_admitted"]
    assert [entry["file"] for entry in held] == [K.display(references / "held.json")]
    assert held[0]["screen_status"] == E.PENDING_REVIEW
    assert held[0]["open_questions"] == ["one_invoice_per_file"]


def test_a_document_the_screen_excludes_is_neither_counted_nor_frozen(declared_authority, tmp_path,
                                                                     monkeypatch):
    """Phase 6, question B: a document can pass the independence screen and still be inadmissible."""
    scale_protocol(monkeypatch, 1)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "good", 203)
    add_document(invoices, references, "pro-forma", 204, genuine_business_invoice=False,
                 region="other", language="English")

    report = R.readiness(invoices, references, geometry=False)
    frozen = [entry["id"] for entry in report["freeze"]["documents"]]
    assert frozen == ["good"], f"the freeze admitted a document the screen excludes: {frozen}"
    excluded = report["intake_gate"]["annotated_not_admitted"]
    assert excluded[0]["screen_status"] == E.EXCLUDED
    assert "genuine_invoice" in excluded[0]["hard_reasons"], "the finding must name why it was refused"


def test_the_intake_gate_counts_admitted_documents_rather_than_sealed_references(declared_authority, tmp_path,
                                                                               monkeypatch):
    """One number, one meaning: `eligible` is the cohort size, so a shortage cannot hide behind it."""
    scale_protocol(monkeypatch, 3)
    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "good", 205)
    add_document(invoices, references, "held", 206, pages=[1], page_count_observed=2)
    add_document(invoices, references, "pro-forma", 207, genuine_business_invoice=False,
                 region="other", language="English")

    report = R.readiness(invoices, references, geometry=False)
    gate = report["intake_gate"]
    assert gate["eligible"] == 1
    assert gate["shortage"] == 2 == K.TARGET_DOCUMENTS - gate["eligible"]
    assert gate["passed"] is False and report["verdict"] == R.NOT_READY
    assert report["freeze"] is None
    assert report["annotation_gate"]["validated_and_sealed"] == 3, (
        "all three references seal: it is the screen that refuses two of them")


def test_an_incomplete_authority_is_a_named_blocking_finding(tmp_path, monkeypatch):
    """The gate has to say which authority it could not trust, not only that a document is missing.

    PR167 moved the real authority off Git, so this reads the tracked demonstration registry: what the test
    measures is the drift branch of `load()`, and the demonstration registry carries the same schema and the
    same on-disk variants for it.
    """
    scale_protocol(monkeypatch, 1)
    registry = json.loads(A.DEMO_REGISTRY_PATH.read_text(encoding="utf-8"))
    declared = sum(1 for document in registry["documents"] for variant in document["variants"]
                   if variant["expect_present"])
    assert declared > 0
    for document in registry["documents"]:
        for variant in document["variants"]:
            if variant["expect_present"]:
                variant["sha256"] = "0" * 64
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    damaged = A.load(path)
    assert damaged["available"] is True and damaged["complete"] is False
    assert len(damaged["drift"]) == declared
    assert "no longer match their recorded hash" in damaged["reason"]
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: damaged)

    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 208)
    report = R.readiness(invoices, references, geometry=False)
    assert report["verdict"] == R.NOT_READY and report["freeze"] is None
    assert any("authority is present but incomplete" in finding for finding in report["blocking_findings"]), (
        report["blocking_findings"])


def test_an_absent_prior_use_inventory_admits_nothing_and_says_why(declared_authority, tmp_path, monkeypatch):
    """PR169: a machine without PR103's inventory has no record of past use, which is not a record of none.

    The screen read that artefact through an import-time default, so a clone that does not carry it crashed on
    the first document instead of refusing. An absent inventory now reaches the same place as an unavailable
    authority: `previous_use` stays unanswered, every candidate is held for review and the readiness report
    names the missing file. The point of the test is that the absence is *reported* - an empty record read as
    an absent one would admit the cohort, and that is the mistake PR165 exists to keep out.
    """
    scale_protocol(monkeypatch, 1)
    absent = tmp_path / "pr103-dataset-manifest.json"
    assert not absent.exists()
    monkeypatch.setattr(I, "PR103_INVENTORY", absent)

    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 209)
    screen = E.eligibility(invoices, references, geometry=False)
    assert screen["prior_use_inventory"]["present"] is False
    assert screen["counts"] == {E.ELIGIBLE: 0, E.EXCLUDED: 0, E.PENDING_REVIEW: 1}, screen["counts"]
    checks = {check["check"]: check for check in screen["documents"][0]["checks"]}
    assert checks["previous_use"]["status"] == "unanswered"
    assert checks["previous_use"]["blocks"] == "pending"

    report = R.readiness(invoices, references, geometry=False)
    assert report["verdict"] == R.NOT_READY and report["freeze"] is None
    assert any("prior-use inventory is not on this machine" in finding
               for finding in report["blocking_findings"]), report["blocking_findings"]


# --------------------------------------------------------------------------- the stale registry

def test_an_unregistered_development_image_makes_the_registry_incomplete(tmp_path):
    """The one hazard a registry cannot see from its own contents: a document that arrived after it."""
    directory = tmp_path / "tests" / "fixtures"
    directory.mkdir(parents=True)
    for name in ("already-registered.png", "arrived-afterwards.png"):
        (directory / name).write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    registry = {"built_from": [{"path": "benchmark_prep/example-source-manifest.json"}],
                "documents": [{"id": "one", "variants": [
                    {"path": "tests/fixtures/already-registered.png", "sha256": None,
                     "expect_present": False}]}]}

    assert A.undeclared(registry, root=tmp_path) == ["tests/fixtures/arrived-afterwards.png"]


def test_a_directory_the_registry_names_nothing_from_makes_no_claim(tmp_path):
    """Coverage is judged only where the registry says it looked, so a partial corpus is not a false alarm."""
    directory = tmp_path / "real-scans"
    directory.mkdir()
    (directory / "unseen.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    registry = {"built_from": [{"path": "benchmark_prep/example-source-manifest.json"}],
                "documents": []}
    assert A.undeclared(registry, root=tmp_path) == []

    # and a fixture authority that declares no source version at all claims nothing, so it is never stale
    assert A.undeclared({"built_from": [], "documents": []}) == []


@pytest.mark.skipif(not A.REGISTRY_PATH.is_file(),
                    reason="the private document authority is local by design since PR167: a clone has none "
                           "and `--check` is the command that reports this machine's")
def test_the_private_registry_omits_nothing_it_covers():
    """The new check must not turn the local authority into a blocking one on the day PR167 moved it.

    Only `undeclared` and the declared scope are asserted: whether the registry's variants are on disk depends
    on which private development corpora this machine holds, and that is what
    `scripts/pr110_authority.py --check` reports. The registry itself is gitignored, so a clone has none.
    """
    authority = A.load(A.REGISTRY_PATH)
    assert authority["available"] is True
    assert authority["scope"] == "repository"
    assert authority["undeclared"] == []


def test_a_variant_dropped_from_the_registry_is_detected_and_holds_the_intake(tmp_path):
    """End to end: an image that was used in development and left out of the authority cannot be admitted.

    PR167 moved the real registry off Git, so this runs on the tracked demonstration one. `undeclared()`
    judges only a registry that names the sources it was built from, and the demonstration declares none, so
    the fixture gives it a coverage claim over `app/examples` before dropping a variant from it.
    """
    registry = json.loads(A.DEMO_REGISTRY_PATH.read_text(encoding="utf-8"))
    registry["built_from"] = [{"path": "app/examples/us.png",
                               "sha256": A.sha256_file(A.ROOT / "app/examples/us.png")}]
    dropped = next(document for document in registry["documents"]
                   if any(variant["path"] == "app/examples/us.png" for variant in document["variants"]))
    dropped["variants"] = [variant for variant in dropped["variants"]
                           if variant["path"] != "app/examples/us.png"]
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(registry), encoding="utf-8")

    authority = A.load(path)
    assert "app/examples/us.png" in authority["undeclared"]
    assert authority["complete"] is False
    assert "rebuilt with scripts/pr110_authority.py" in authority["reason"]

    invoices, references = write_intake(tmp_path)
    add_document(invoices, references, "solo", 209)
    report = E.eligibility(invoices, references, geometry=False, authority=authority)
    assert report["counts"][E.ELIGIBLE] == 0
    assert "document_authority" in report["documents"][0]["open_questions"]
    assert "app/examples/us.png" in report["document_authority"]["undeclared"]
