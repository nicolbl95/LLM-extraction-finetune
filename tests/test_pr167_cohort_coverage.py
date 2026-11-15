"""PR167 Phase 3 and 5: what the cohort is made of is a gate, not a description.

PR164 made the screen decide *which documents are independent* and PR166 measured that twelve independent,
admissible documents were still not twelve invoices of the kind the protocol promises. `readiness()` gated on a
count (intake) and on the shape of the references (annotation); neither of them looked at which countries those
documents are, so an intake of twelve French invoices froze as READY and would have reported one billing
convention as three. The composition table below is the one the protocol states - four invoices under each of
the three conventions the campaign was built for - and every row is measured against `TARGET_DOCUMENTS` and
`COHORT_QUOTA` at their production values. Nothing here scales the protocol down to make a gate pass; where a
fixture cannot hold twelve documents, that is `test_pr110_benchmark_prep.py`'s business, not this file's.

Two fixture properties make a twelve-document cohort measurable at all, and both are kept true by a test rather
than assumed:

* the pages are mutually distinct well beyond the screen's 10-bit band, so the screen admits them and the
  composition is what decides;
* independence is decided by an authority that knows no document, so what acts is the gate, not the corpora
  restored on the machine the tests happen to run on.

0 OCR calls, 0 model calls, 0 pipeline runs. Nothing is written into the real intake, and the twelve documents
of the PR110 cohort stay uncollected.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr109_intake as I  # noqa: E402
import pr110_authority as AU  # noqa: E402
import pr110_eligibility as E  # noqa: E402
import pr110_evaluate as G  # noqa: E402
import pr110_harness as K  # noqa: E402
import pr110_ready as R  # noqa: E402

from test_pr110_benchmark_prep import filled_annotation, synthetic_image, write_freeze, write_intake  # noqa: E402
from pr110_prior_use import install as install_prior_use  # noqa: E402

# The sixteen pages this file can build, and the only seeds it uses. `distinct_page()` keeps them mutually far
# enough for the screen; see `test_the_fixture_pages_are_far_enough_apart_for_the_screen_to_admit_them`.
SEEDS = list(range(16))
# Each convention declares the language, currency and tax rate its invoices carry, so a cohort of 4/4/4 is three
# kinds of document rather than one document relabelled. The arithmetic stays one body: 1000.00 HT, tax on it.
CONVENTIONS = {"FR": ("French", "EUR", 20.0), "UK": ("English", "GBP", 20.0),
               "US": ("English", "USD", 8.0), "CA": ("English", "CAD", 5.0),
               "other": ("French", "EUR", 20.0)}
# The loader as the module ships it, captured before the fixtures replace it. A test that wants the real
# absent-registry branch cannot reach it by patching `AU.load` and calling `AU.load`: the patch makes the name
# point at itself.
REAL_LOAD = AU.load


@pytest.fixture(autouse=True)
def no_campaign_population(monkeypatch):
    """The fixtures are synthetic; the repository's own signed images must not collide with them."""
    monkeypatch.setattr(I, "signed_population", lambda: [])


@pytest.fixture(autouse=True)
def declared_empty_document_authority(monkeypatch):
    """Independence is settled elsewhere; here every fixture document is a new one, so admissibility is equal."""
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: AU.empty())


@pytest.fixture(autouse=True)
def declared_empty_prior_use(monkeypatch, tmp_path):
    """PR169: the same reasoning as the authority above, for the screen's other per-machine input.

    `eligibility()` reads an inventory of what the campaign used before, and PR103's published one is a local
    artefact that no clone carries. This file gates on coverage, selection and the freeze, not on that record,
    so the fixture declares an empty one in PR103's format and every cohort stays as admissible as it was.
    """
    install_prior_use(monkeypatch, tmp_path)


# --------------------------------------------------------------------------- fixtures


def distinct_page(path: Path, seed: int) -> Path:
    """A page whose ink is a seed-determined subset of an 8x10 grid, so twelve of them are twelve documents.

    PR110's `synthetic_image()` places its blocks on a two-term congruence, and consecutive seeds land inside
    the screen's 10-bit band - measured, the closest of fourteen of them is 6 bits from another. That is the
    right fixture for a test that pins the band and unusable for one that has to reach the quota gate: the
    screen holds each near neighbour for manual review, so the cohort never arrives at `coverage_gate()`. These
    pages are at least 21 bits apart over `SEEDS`, in both hashes.
    """
    rng = random.Random(seed)
    image = Image.new("L", (640, 820), 255)
    draw = ImageDraw.Draw(image)
    for row in range(10):
        for column in range(8):
            if rng.random() < 0.40:
                left, top = column * 80 + 10, row * 82 + 10
                draw.rectangle([left, top, left + rng.randint(28, 62), top + rng.randint(10, 44)],
                               fill=rng.choice((30, 70, 120)))
    image.save(path)
    return path


def add_reference(references: Path, document_id: str, image: Path, region: str) -> dict:
    """Seal and store the reference for a page that is already on disk."""
    language, currency, rate = CONVENTIONS[region]
    annotation = filled_annotation(document_id, image, region=region, language=language)
    K.set_field(annotation, "currency", currency, "printed")
    K.set_field(annotation, "tax.rate", rate, "printed")
    K.set_field(annotation, "tax.amount", round(1000.0 * rate / 100, 2), "printed")
    K.set_field(annotation, "total_ttc", 1000.0 + round(1000.0 * rate / 100, 2), "printed")
    K.seal(annotation, image)
    (references / f"{document_id}.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return {"id": document_id, "region": region, "image": image, "annotation": annotation}


def add_invoice(invoices: Path, references: Path, document_id: str, region: str, seed: int) -> dict:
    """One eligible, admissible, sealed reference for a document declared under `region`'s convention."""
    return add_reference(references, document_id, distinct_page(invoices / f"{document_id}.png", seed), region)


def plan(composition: dict[str, int]) -> list[tuple[str, str, int]]:
    """The `(id, region, seed)` of every document a composition calls for, in a fixed sequence."""
    entries = []
    for region, count in sorted(composition.items()):
        for number in range(1, count + 1):
            entries.append((f"{region.lower()}-{number:02d}", region, SEEDS[len(entries)]))
    assert len(entries) <= len(SEEDS), f"a {len(entries)}-document cohort needs more than {len(SEEDS)} seeds"
    return entries


def write_cohort(tmp_path: Path, composition: dict[str, int], *,
                 order: range | None = None) -> tuple[Path, Path, list[dict]]:
    """Build `composition` as an intake of twelve-plus documents, sealed and ready for the gates.

    `order` writes the same documents in another sequence, which is how a test shows the selection does not
    depend on the order the files were found in.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    invoices, references = write_intake(tmp_path)
    entries = plan(composition)
    documents = [add_invoice(invoices, references, *entries[position])
                 for position in (order or range(len(entries)))]
    return invoices, references, documents


def ids(cohort: list[dict]) -> list[str]:
    return [entry["id"] for entry in cohort]


def frozen_cohort(tmp_path: Path, composition: dict[str, int] = None) -> tuple[Path, Path, dict]:
    """A real freeze, written by the real gates over a balanced cohort: the starting point of Phase 4."""
    invoices, references, _ = write_cohort(tmp_path, composition or {"FR": 4, "UK": 4, "US": 4})
    result = R.readiness(invoices, references, geometry=False)
    assert result["verdict"] == R.READY, result["blocking_findings"]
    return invoices, references, result["freeze"]


# --------------------------------------------------------------------------- the fixtures are what the gate needs


def test_the_fixture_pages_are_far_enough_apart_for_the_screen_to_admit_them(tmp_path):
    """The 10-bit band is PR164's and unchanged; a fixture inside it would exercise the screen, not the quota.

    This is a test of the fixtures rather than of the code: if a later edit makes `distinct_page()` produce near
    neighbours, the twelve-document READY case below stops measuring the coverage gate and starts measuring
    `visually_similar_duplicate`. The screen's decision is the smaller of the two hash distances, so both are
    taken together here, exactly as `pr110_eligibility.screen_file()` takes them.
    """
    signatures = {seed: I.perceptual(distinct_page(tmp_path / f"page-{seed}.png", seed)) for seed in SEEDS}
    closest = min(min(I.hamming(signatures[left]["ahash"], signatures[right]["ahash"]),
                      I.hamming(signatures[left]["dhash"], signatures[right]["dhash"]))
                  for left, right in itertools.combinations(SEEDS, 2))
    assert closest > I.PERCEPTUAL_MAX_DISTANCE, f"fixture pages sit inside the {I.PERCEPTUAL_MAX_DISTANCE}-bit band"


def test_a_quarter_of_the_cohort_is_declared_under_each_of_the_three_conventions(tmp_path):
    """The fixture is honest about what it claims: twelve documents that really are 4 FR + 4 UK + 4 US.

    A region a validator would refuse is worse than a fake one, so this checks the references themselves are
    admissible before any gate runs - the quota can only be spent by documents that clear the validator.
    """
    _, _, documents = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    assert [entry["region"] for entry in documents].count("FR") == 4
    for entry in documents:
        problems = K.validate(entry["annotation"], entry["image"])
        assert problems == [], problems
        assert K.verify_seal(entry["annotation"], entry["image"])["intact"] is True
    language, currency = CONVENTIONS["US"][:2]
    us = next(entry for entry in documents if entry["region"] == "US")
    assert us["annotation"]["document"]["language"] == language
    assert us["annotation"]["reference"]["currency"]["value"] == currency


# --------------------------------------------------------------------------- the table the protocol states


COMPOSITIONS = [
    pytest.param({"FR": 4, "UK": 4, "US": 4}, "FR: 4/4, UK: 4/4, US: 4/4", {}, True, id="four-each"),
    pytest.param({"FR": 3, "UK": 4, "US": 4}, "FR: 3/4, UK: 4/4, US: 4/4", {"FR": 1}, False,
                 id="three-french"),
    pytest.param({"FR": 4, "UK": 3, "US": 4}, "FR: 4/4, UK: 3/4, US: 4/4", {"UK": 1}, False,
                 id="three-british"),
    pytest.param({"FR": 4, "UK": 4, "US": 3}, "FR: 4/4, UK: 4/4, US: 3/4", {"US": 1}, False,
                 id="three-american"),
    pytest.param({"FR": 12}, "FR: 4/4, UK: 0/4, US: 0/4", {"UK": 4, "US": 4}, False, id="twelve-french"),
    pytest.param({"FR": 4, "UK": 4, "US": 3, "CA": 1}, "FR: 4/4, UK: 4/4, US: 3/4", {"US": 1}, False,
                 id="canadian-patch"),
    pytest.param({"FR": 4, "UK": 4, "US": 3, "other": 1}, "FR: 4/4, UK: 4/4, US: 3/4", {"US": 1}, False,
                 id="regionless-patch"),
]


@pytest.mark.parametrize("composition, report, missing, ready", COMPOSITIONS)
def test_the_coverage_table_is_decided_on_the_selected_cohort(tmp_path, composition, report, missing, ready):
    """Every row of PR167 Phase 5, at the protocol's own numbers.

    `geometry=False` because these rows are about quota arithmetic and the cheap screen mode is the campaign's
    documented way to isolate it; the production default, with the geometric distinctness test and the ruled
    region categories live, is the next test.
    """
    invoices, references, _ = write_cohort(tmp_path, composition)
    result = R.readiness(invoices, references, geometry=False)
    gate = result["coverage_gate"]

    assert gate["report"] == report
    assert gate["missing"] == missing
    assert gate["passed"] is ready
    assert result["verdict"] == (R.READY if ready else R.NOT_READY)
    assert (result["freeze"] is not None) is ready
    if not ready:
        text = " | ".join(result["blocking_findings"])
        assert f"{report} — {R.NOT_READY}" in text
        for region, count in missing.items():
            assert f"missing {region}: {count}" in text


def test_the_cohort_is_ready_only_when_all_three_gates_pass(tmp_path):
    """The production default: 4 FR + 4 UK + 4 US, geometric distinctness computed, verdict READY.

    This is the run that would have been READY before PR167 for any twelve admissible documents, so it asserts
    the composition itself: the freeze carries a region per document, the three gate results, and the selection
    the evaluator has to reproduce.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    result = R.readiness(invoices, references)
    gate = result["coverage_gate"]

    assert result["verdict"] == R.READY
    assert gate["requirement"] == ("4 FR + 4 UK + 4 US = 12 admissible, sealed, mutually distinct invoices")
    assert gate["report"] == "FR: 4/4, UK: 4/4, US: 4/4"
    assert gate["counts"] == {"FR": 4, "UK": 4, "US": 4}
    assert gate["findings"] == [] and gate["passed"] is True
    assert gate["selected"] == 12 == result["intake_gate"]["eligible"]

    freeze = result["freeze"]
    assert [entry["region"] for entry in freeze["documents"]].count("US") == 4
    assert sorted(entry["region"] for entry in freeze["documents"]) == ["FR"] * 4 + ["UK"] * 4 + ["US"] * 4
    assert ids(freeze["documents"]) == gate["selected_ids"] == freeze["selection"]["selected_ids"]
    assert freeze["selection"]["selection_sha256"] == gate["selection_sha256"]
    assert {key: value for key, value in freeze["gates"].items() if key != "note"} == {
        "intake_gate": True, "annotation_gate": True, "coverage_gate": True}


def test_the_gate_comes_after_two_gates_that_cannot_see_it(tmp_path):
    """Problème A, stated as a result: twelve French invoices clear the intake and the annotation gate.

    Before PR167 these three booleans were the whole verdict, so this cohort froze. It is the one case where the
    other gates must be shown passing rather than assumed, because it is the case where passing them means
    nothing.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 12})
    result = R.readiness(invoices, references, geometry=False)

    assert result["intake_gate"]["passed"] is True and result["intake_gate"]["eligible"] == 12
    assert result["annotation_gate"]["passed"] is True and result["annotation_gate"]["invalid"] == []
    assert result["coverage_gate"]["passed"] is False
    assert result["verdict"] == R.NOT_READY


# --------------------------------------------------------------------------- a slot is spent by its own country


def test_a_canadian_invoice_does_not_become_a_french_one(tmp_path):
    """`CA` is a region the v2 validator accepts and the protocol gives no quota to.

    Measuring it as `other` would be a false statement about the corpus and measuring it as a French invoice
    would be a worse one, so it is kept out of the cohort and named, with the reason, in the reserve.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 3, "CA": 1})
    gate = R.readiness(invoices, references, geometry=False)["coverage_gate"]

    assert "ca-01" not in gate["selected_ids"]
    assert gate["off_quota_candidates"] == {"CA": ["ca-01"]}
    assert gate["counts"] == {"FR": 4, "UK": 4, "US": 3} and gate["missing"] == {"US": 1}
    reserve = next(entry for entry in gate["reserve"] if entry["id"] == "ca-01")
    assert "cannot replace a required invoice" in reserve["reason"]
    assert gate["passed"] is False


def test_the_reserve_is_named_and_never_evaluated(tmp_path):
    """Phase 3: more than twelve documents are allowed; more than twelve evaluated are not.

    The gate counts the cohort it selected, so `candidates` describes the intake while `counts` decides the
    verdict - which is the difference between an inventory and a benchmark.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 6, "UK": 4, "US": 4})
    result = R.readiness(invoices, references, geometry=False)
    gate = result["coverage_gate"]

    assert gate["candidates"]["FR"] == ["fr-01", "fr-02", "fr-03", "fr-04", "fr-05", "fr-06"]
    assert gate["counts"] == {"FR": 4, "UK": 4, "US": 4} and gate["passed"] is True
    assert result["verdict"] == R.READY and gate["selected"] == 12
    assert [entry["id"] for entry in gate["reserve"]] == ["fr-05", "fr-06"]
    assert "not evaluated" in gate["reserve"][0]["reason"]
    frozen = ids(result["freeze"]["documents"])
    assert len(frozen) == 12 and "fr-05" not in frozen and "fr-06" not in frozen
    assert sorted(frozen) == sorted(gate["selected_ids"])


def test_the_selection_is_by_document_id_and_survives_a_new_competitor(tmp_path):
    """Deterministic, traceable, reproducible: within a country the order is the document id, then its bytes.

    Adding a French invoice whose id sorts first moves the boundary, so a candidate that was evaluated drops to
    the reserve and the selection digest changes with it. That is what lets a later agent re-derive the twelve
    from the freeze instead of trusting the run that wrote it.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 5, "UK": 4, "US": 4})
    before = R.readiness(invoices, references, geometry=False)["coverage_gate"]
    assert before["selected_ids"].count("fr-01") == 1 and before["counts"]["FR"] == 4
    assert [entry["id"] for entry in before["reserve"]] == ["fr-05"]

    add_invoice(invoices, references, "fr-00", "FR", SEEDS[15])
    after = R.readiness(invoices, references, geometry=False)["coverage_gate"]

    assert after["candidates"]["FR"] == ["fr-00", "fr-01", "fr-02", "fr-03", "fr-04", "fr-05"]
    assert [entry["id"] for entry in after["reserve"]] == ["fr-04", "fr-05"]
    assert "fr-00" in after["selected_ids"] and "fr-04" not in after["selected_ids"]
    assert after["selection_sha256"] != before["selection_sha256"]


def test_the_same_cohort_written_in_a_different_order_is_the_same_selection(tmp_path):
    """The freeze's `selection_sha256` must not depend on the directory walk.

    Both intakes hold the same twelve documents with the same ids and the same pages; they are written in
    opposite sequences. The digest also has to survive a fresh process reading a fresh directory, so this
    compares two independent readiness runs rather than two reports from one run.
    """
    forward, forward_references, _ = write_cohort(tmp_path / "forward", {"FR": 4, "UK": 4, "US": 4})
    backward, backward_references, _ = write_cohort(tmp_path / "backward", {"FR": 4, "UK": 4, "US": 4},
                                                   order=range(11, -1, -1))
    assert [path.name for path in sorted(forward_references.glob("*.json"))] == \
        [path.name for path in sorted(backward_references.glob("*.json"))]

    first = R.readiness(forward, forward_references, geometry=False)["coverage_gate"]
    second = R.readiness(backward, backward_references, geometry=False)["coverage_gate"]

    assert first["selected_ids"] == second["selected_ids"]
    assert first["selection_sha256"] == second["selection_sha256"]


def test_a_protocol_that_does_not_add_up_is_refused_rather_than_measuring_less(tmp_path, monkeypatch):
    """`COHORT_QUOTA` and `TARGET_DOCUMENTS` are one statement; a quota that contradicts it is a finding.

    A 4+4+3 quota under a target of twelve would freeze eleven documents and still call the benchmark PR110's,
    so `select_cohort()` reports the inconsistency and the gate refuses on it.
    """
    monkeypatch.setattr(K, "COHORT_QUOTA", {"FR": 4, "UK": 4, "US": 3})
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    result = R.readiness(invoices, references, geometry=False)
    gate = result["coverage_gate"]

    assert gate["passed"] is False and result["verdict"] == R.NOT_READY
    assert result["freeze"] is None
    assert any("the protocol does not add up" in finding for finding in gate["findings"])


# --------------------------------------------------------------------------- twelve slots, eleven invoices


def test_two_references_to_one_document_id_are_one_invoice_in_two_slots(tmp_path):
    """A duplicated identity survives the screen and stops at the gate.

    The two references scan different pages, so nothing the intake looks at excludes either one - no shared
    bytes, no registered variant, no near neighbour in the signed population. What makes them one invoice is
    that the campaign recorded both as `uk-02`, which is a property of the cohort rather than of either file,
    and it is the coverage gate that has the whole cohort in front of it.
    """
    invoices, references, documents = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    original = next(entry for entry in documents if entry["id"] == "uk-02")
    second = add_invoice(invoices, references, "uk-02-second-scan", "UK", SEEDS[12])
    refiled = json.loads((references / "uk-02-second-scan.json").read_text(encoding="utf-8"))
    refiled["document"]["id"] = "uk-02"
    second["image"] = distinct_page(invoices / "uk-02-second-scan.png", SEEDS[12])
    refiled["document"]["source_file"] = "uk-02-second-scan.png"
    refiled["document"]["sha256"] = I.sha256_file(second["image"])
    K.seal(refiled, second["image"])
    (references / "uk-02-second-scan.json").write_text(json.dumps(refiled, ensure_ascii=False, indent=2),
                                                       encoding="utf-8")
    assert original["image"] != second["image"]

    result = R.readiness(invoices, references, geometry=False)
    gate = result["coverage_gate"]

    assert result["intake_gate"]["eligible"] == 13 and result["annotation_gate"]["passed"] is True
    assert gate["counts"] == {"FR": 4, "UK": 4, "US": 4}
    assert gate["missing"] == {}
    assert [entry["pair"] for entry in gate["distinct"]["proven"]] == [["uk-02", "uk-02"]]
    assert "the same document id is annotated twice" in gate["distinct"]["proven"][0]["basis"]
    assert gate["passed"] is False and result["verdict"] == R.NOT_READY


def test_two_members_the_screen_cannot_separate_are_left_unproven(tmp_path):
    """PR164's collision case, now inside the cohort: a 6-bit resemblance is neither a duplicate nor a pass.

    The screen compares each document with the signed population, not with its cohort mates, so two near-
    identical pages in one intake both reach the selection. `cohort_variant_pairs()` is where the pair is
    measured, and PR164's decision stands: two documents of one template can sit this close, so the campaign
    reports the resemblance and asks the owner instead of asserting either answer.

    The pair sits in the middle of the French slots on purpose. Placed last in its region it would fall to the
    reserve, the cohort would read 4/4/4 and the gate would never see the resemblance at all.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    first = add_reference(references, "fr-02", synthetic_image(invoices / "fr-02.png", 1), "FR")
    second = add_reference(references, "fr-03", synthetic_image(invoices / "fr-03.png", 2), "FR")
    assert min(I.hamming(I.perceptual(first["image"])["ahash"], I.perceptual(second["image"])["ahash"]),
               I.hamming(I.perceptual(first["image"])["dhash"],
                         I.perceptual(second["image"])["dhash"])) <= I.PERCEPTUAL_MAX_DISTANCE

    result = R.readiness(invoices, references)
    gate = result["coverage_gate"]

    assert result["intake_gate"]["eligible"] == 12 and result["annotation_gate"]["passed"] is True
    assert gate["counts"] == {"FR": 4, "UK": 4, "US": 4} and gate["missing"] == {}
    assert [entry["pair"] for entry in gate["distinct"]["unconfirmed"]] == [["fr-02", "fr-03"]]
    assert gate["distinct"]["proven"] == [] and gate["distinct"]["errors"] == []
    assert gate["passed"] is False and result["verdict"] == R.NOT_READY
    assert any("not proven distinct and need the owner" in finding for finding in gate["findings"])


def test_a_gate_that_trusted_its_own_input_would_not_be_a_gate(tmp_path):
    """The byte-duplicate basis is checked on the selected cohort even though the screen refuses that shape.

    `pr110_eligibility.screen_file()` excludes a second copy of the same bytes as `duplicate_inside_the_intake`
    before any selection exists, so the intake never reaches this branch. It stays tested through
    `coverage_gate()` directly, because the gate's own claim is that it decides on what it is handed: two
    records carrying one document digest are one invoice whichever file name they were saved under.

    The second reference is sealed rather than edited, so the seal is intact and the digest is the only thing
    that gives the pair away - which is exactly the input the gate is asked to distrust.
    """
    invoices, references = write_intake(tmp_path)
    first = add_invoice(invoices, references, "fr-01", "FR", SEEDS[0])
    twin = json.loads((references / "fr-01.json").read_text(encoding="utf-8"))
    twin["document"]["id"] = "fr-02"
    K.seal(twin, first["image"])
    (references / "fr-02.json").write_text(json.dumps(twin, ensure_ascii=False, indent=2), encoding="utf-8")

    records = [record for record in K.cohort(references, invoices) if record["eligible"]]
    assert [record["id"] for record in records] == ["fr-01", "fr-02"]
    gate = R.coverage_gate(K.select_cohort(records, {"FR": 2}, target=2), geometry=False)

    assert gate["missing"] == {} and gate["selected"] == 2 and gate["not_admissible"] == []
    assert [entry["basis"] for entry in gate["distinct"]["proven"]] == ["identical document sha256"]
    assert gate["passed"] is False and len(gate["findings"]) == 1


# --------------------------------------------------------------------------- the three refusals that are not quota


def test_a_reference_that_is_not_sealed_is_not_a_slot(tmp_path):
    """Row 9 of the table, end to end: an unsealed reference fails the annotation gate and freezes nothing.

    Phase 4 of the brief wants the seal carried into the evaluation, so this also records that the freeze is
    absent rather than present-but-short: eleven sealed documents are not a smaller version of twelve.
    """
    invoices, references, documents = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    victim = next(entry for entry in documents if entry["id"] == "uk-03")
    unsealed = filled_annotation("uk-03", victim["image"], region="UK", language="English")
    K.set_field(unsealed, "currency", "GBP", "printed")
    K.set_field(unsealed, "tax.rate", 20.0, "printed")
    K.set_field(unsealed, "tax.amount", 200.0, "printed")
    K.set_field(unsealed, "total_ttc", 1200.0, "printed")
    (references / "uk-03.json").write_text(json.dumps(unsealed, ensure_ascii=False, indent=2), encoding="utf-8")

    result = R.readiness(invoices, references, geometry=False)

    assert result["annotation_gate"]["passed"] is False and result["intake_gate"]["eligible"] == 11
    assert result["intake_gate"]["shortage"] == 1
    assert result["coverage_gate"]["counts"] == {"FR": 4, "UK": 3, "US": 4}
    assert result["verdict"] == R.NOT_READY and result["freeze"] is None
    assert any("broken seal" in finding for finding in result["blocking_findings"])


def test_a_document_pending_review_is_counted_neither_in_the_cohort_nor_in_the_freeze(tmp_path):
    """Row 10 of the table: a held document is a slot nobody filled.

    The declaration mismatch is the screen's own reason to hold one, and PR165 already proved a held reference
    is not frozen; what is new is that a held document leaves a *country* short, which is the coverage gate's
    business rather than the intake's.
    """
    invoices, references, documents = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    victim = next(entry for entry in documents if entry["id"] == "us-02")
    held = filled_annotation("us-02", victim["image"], region="US", language="English",
                             page_count_observed=2)
    K.set_field(held, "currency", "USD", "printed")
    K.set_field(held, "tax.rate", 8.0, "printed")
    K.set_field(held, "tax.amount", 80.0, "printed")
    K.set_field(held, "total_ttc", 1080.0, "printed")
    K.seal(held, victim["image"])
    (references / "us-02.json").write_text(json.dumps(held, ensure_ascii=False, indent=2), encoding="utf-8")

    result = R.readiness(invoices, references, geometry=False)
    gate = result["coverage_gate"]

    assert result["intake_gate"]["pending_review"] == 1 and result["intake_gate"]["eligible"] == 11
    assert "us-02" not in gate["selected_ids"]
    assert gate["report"] == "FR: 4/4, UK: 4/4, US: 3/4" and gate["missing"] == {"US": 1}
    assert result["verdict"] == R.NOT_READY and result["freeze"] is None


def test_the_freeze_names_the_authority_the_documents_were_certified_against(tmp_path):
    """Phase 4: twelve frozen references say nothing about what the campaign had already used.

    Without the authority block an evaluator could reproduce the same twelve document ids while standing on a
    different claim about their independence - including the claim that there was nothing to check against. The
    report the screen decided on is copied whole into the freeze, so a run that happens after the private
    registry moved can be told apart from one that happened before it.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    authority = R.readiness(invoices, references, geometry=False)["freeze"]["document_authority"]

    assert authority["available"] is True and authority["complete"] is True
    assert authority["scope"] == "fixture" and authority["documents"] == 0
    assert authority["version"] == AU.REGISTRY_VERSION


def test_a_missing_authority_freezes_no_document_at_all(tmp_path, monkeypatch):
    """PR167's fail-closed rule, measured on a complete twelve-document cohort.

    A clone that does not hold the private registry sees no antecedents, which is the condition that would
    otherwise admit twelve invoices on the strength of nobody having recorded them. The absence is reported,
    nothing is eligible, and the composition gate is never reached: the cohort is not frozen at all rather than
    frozen with a note.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: REAL_LOAD(Path("nowhere") / "registry.json"))
    result = R.readiness(invoices, references, geometry=False)

    assert result["intake_gate"]["eligible"] == 0 and result["intake_gate"]["shortage"] == 12
    assert result["verdict"] == R.NOT_READY and result["freeze"] is None
    assert result["coverage_gate"]["counts"] == {"FR": 0, "UK": 0, "US": 0}
    assert any("document authority" in finding for finding in result["blocking_findings"])


@pytest.mark.parametrize("sources", ["built_from_a_manifest", "no_sources_at_all"])
def test_an_authority_that_names_no_document_admits_nothing(tmp_path, monkeypatch, sources):
    """PR167's other fail-closed rule: an authority that records nothing certifies nothing.

    The residue of a clone that got the private file but not the corpora behind it - or of a rebuild started
    before the manifests were read - is a registry that exists, verifies itself against the disk and names no
    document. `previous_use` can never fire against an empty record, so without this branch the twelve fixtures
    would be admitted as independent on the strength of a registry that has never heard of anything, which is
    the same claim the missing file already refuses. The manifest arm hashes as recorded on purpose: the stale
    source is not what is wrong here, the emptiness is.
    """
    invoices, references, _ = write_cohort(tmp_path, {"FR": 4, "UK": 4, "US": 4})
    manifest = ROOT / "benchmark_prep" / "OWNER_GUIDE.md"
    registry = {
        "version": AU.REGISTRY_VERSION, "scope": "repository", "documents": [],
        "built_from": [{"path": manifest.relative_to(ROOT).as_posix(), "sha256": AU.sha256_file(manifest)}]
                      if sources == "built_from_a_manifest" else [],
    }
    authority = tmp_path / "authority.json"
    authority.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: REAL_LOAD(authority))
    result = R.readiness(invoices, references, geometry=False)

    reason = next(finding for finding in result["blocking_findings"] if "document authority" in finding)
    assert "names no document" in reason
    assert result["intake_gate"]["eligible"] == 0 and result["intake_gate"]["shortage"] == 12
    assert result["verdict"] == R.NOT_READY and result["freeze"] is None


def test_the_tracked_demonstration_registry_names_only_what_a_clone_receives():
    """The one registry that is in Git must not accuse itself on a machine that holds only what Git ships.

    `demo_registry()` walks the published roots, so a variant that is tracked here but ignored there lands as
    `missing` on a clean clone and the demonstration is then a record of a defect rather than of the format. The
    scope cause is the single one it may carry: `demo` is not this repository's record, and a screen that loads
    it admits nothing whatever the variants do.
    """
    authority = REAL_LOAD(AU.DEMO_REGISTRY_PATH)

    assert authority["available"] is True and authority["complete"] is False
    assert authority["scope"] == "demo" and authority["documents"]
    assert not authority["missing"] and not authority["drift"] and not authority["undeclared"]
    assert authority["reason"].startswith("this registry declares scope 'demo'")
    assert "names no document" not in authority["reason"]


# --------------------------------------------------------------------------- Phase 4: what the evaluator re-checks


def reseal(references: Path, invoices: Path, document_id: str, **document) -> dict:
    """Change one frozen reference the way an owner editing after the freeze would, and seal it again.

    Re-sealing is the point: a tampering that left the seal broken is caught by the check PR110 already had, and
    proves nothing about PR167. Here the reference validates and seals, and only the composition gives it away.
    """
    annotation = json.loads((references / f"{document_id}.json").read_text(encoding="utf-8"))
    annotation["document"].update(document)
    K.seal(annotation, invoices / f"{document_id}.png")
    (references / f"{document_id}.json").write_text(json.dumps(annotation, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    return annotation


def test_an_untouched_cohort_still_selects_the_frozen_twelve(tmp_path):
    """The baseline the refusals below are measured against: re-running the screen and the gate changes nothing.

    A composition check that fires on an intact intake would be a false refusal, and a false refusal on a real
    twelve-document cohort is how an evaluation gets postponed forever, so this asserts the clean case first.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    verification = G.verify_freeze(freeze, invoices, references)

    assert verification["ok"] is True, verification["problems"]
    assert verification["problems"] == []
    assert [document["id"] for document in verification["documents"]] == freeze["selection"]["selected_ids"]


def test_a_french_slot_relabelled_after_the_freeze_is_refused(tmp_path):
    """Phase 4's first refusal: one invoice stops being French, and the cohort stops being 4/4/4.

    The reference still validates and still seals, because the seal covers the reference, not the claim it makes
    about the country. Without the composition check the run would have measured three French invoices and
    reported four.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    reseal(references, invoices, "fr-04", region="other")
    verification = G.verify_freeze(freeze, invoices, references)
    text = " | ".join(verification["problems"])

    assert verification["ok"] is False
    assert "declares region 'other' where the freeze certified 'FR'" in text
    assert "these bytes no longer select the frozen cohort" in text and "withdrawn: fr-04" in text
    assert "leaves a quota short: missing FR: 1" in text


def test_an_invoice_added_after_the_freeze_that_takes_a_slot_is_refused(tmp_path):
    """Phase 4's second refusal, part one: a thirteenth document can displace one of the certified twelve.

    The order is not cosmetic - the selection is `(document id, document sha256)` within a country - so `fr-00`
    sorts ahead of `fr-01` and pushes `fr-04` out. The run refuses rather than measuring a cohort that was never
    frozen.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    add_invoice(invoices, references, "fr-00", "FR", SEEDS[12])
    verification = G.verify_freeze(freeze, invoices, references)
    text = " | ".join(verification["problems"])

    assert verification["ok"] is False
    assert "newly selected: fr-00" in text and "withdrawn: fr-04" in text
    assert "leaves a quota short" not in text, "the composition is complete; it is a different twelve"


def test_a_reserve_document_added_after_the_freeze_does_not_invalidate_the_run(tmp_path):
    """The boundary PR167 has to hold in both directions: a reserve is allowed, a different cohort is not.

    A thirteenth French invoice that sorts after the four selected ones changes the reserve and nothing else, so
    the evaluator runs the same frozen twelve. Refusing here would say the protocol forbids what it states.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    add_invoice(invoices, references, "fr-05", "FR", SEEDS[12])

    assert G.verify_freeze(freeze, invoices, references)["ok"] is True


def test_a_document_withdrawn_after_the_freeze_is_refused(tmp_path):
    """Phase 4's second refusal, part two: the certified document is gone from the intake's references.

    This is the shape PR109 and PR129 kept producing - a cohort that reads as complete because nobody looked -
    and the eleven that remain are not a smaller version of the twelve that were frozen.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    (references / "us-02.json").unlink()
    verification = G.verify_freeze(freeze, invoices, references)
    text = " | ".join(verification["problems"])

    assert verification["ok"] is False
    assert "no reference file carries this document id" in text
    assert "withdrawn: us-02" in text and "leaves a quota short: missing US: 1" in text
    assert any("cohort digest does not recompute" in problem for problem in verification["problems"])


def test_a_freeze_that_records_no_selection_is_refused(tmp_path):
    """PR167 refuses the pre-PR167 freeze shape rather than running it.

    Twelve sealed references and a cohort digest say nothing about the composition, and a freeze that cannot
    show which twelve countries it certified is exactly the ambiguity Phase 4 exists to close. Everything else
    in the verification stays clean, so the missing block is the only reason for the refusal.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    verification = G.verify_freeze({key: value for key, value in freeze.items() if key != "selection"},
                                    invoices, references)

    assert verification["ok"] is False and len(verification["problems"]) == 1
    assert "records no selection" in verification["problems"][0]


def test_an_evaluation_without_the_document_authority_is_refused(tmp_path, monkeypatch):
    """The fail-closed rule one layer further down: no registry, no independence claim, no run.

    The bytes on disk are untouched here, so every other check passes. What is missing is the authority the
    freeze names as the basis for calling these twelve independent, and a clone that cannot read it must not
    conclude that nothing was ever used.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: REAL_LOAD(Path("nowhere") / "registry.json"))
    verification = G.verify_freeze(freeze, invoices, references)

    assert verification["ok"] is False and len(verification["problems"]) == 1
    assert "document authority" in verification["problems"][0]
    assert "silence of a clone" in verification["problems"][0]


def test_the_demonstration_registry_used_by_error_still_refuses_admission(tmp_path, monkeypatch):
    """PR167 Phase 10's wrong-file scenario, through `run()`: a registry that exists and says nothing.

    A clone carries `document-authority.demo.json` and can point `$PR110_DOCUMENT_AUTHORITY` at it by mistake.
    That is the hazard PR164's `scope: "demo"` was written for, and the answer has to be the same at both layers:
    the screen admits nothing, so the evaluator finds an empty cohort where twelve documents were certified, and
    the run is refused before the application is imported - 0 OCR calls, not one extraction scored against a
    demonstration corpus. The registry is generated here rather than read from disk so the test says nothing
    about which authority this machine happens to hold.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    demo = AU.write(tmp_path / "authority.demo.json", AU.demo_registry())
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: REAL_LOAD(demo))
    report = G.run(write_freeze(tmp_path, freeze), invoices, references)

    assert report["verdict"] == "REFUSED" and report["documents"] == [] and report["aggregate"] is None
    problems = report["verification"]["problems"]
    assert len(problems) == 1 and "document authority" in problems[0]
    assert "'demo'" in problems[0] and not any("no longer select" in p for p in problems)


def test_the_refusal_happens_before_the_first_pipeline_call(tmp_path):
    """Phase 11's condition on the refusal path: `run()` returns REFUSED without importing the application.

    The brief asks for an end-to-end sequence with 0 real OCR. This is the one place a refusal could still be
    followed by an extraction, so the test pins the verdict and the empty document list rather than trusting the
    order of two statements in `run()`.
    """
    invoices, references, freeze = frozen_cohort(tmp_path)
    reseal(references, invoices, "uk-01", region="US")
    report = G.run(write_freeze(tmp_path, freeze), invoices, references)

    assert report["verdict"] == "REFUSED" and report["documents"] == [] and report["aggregate"] is None
    assert any("declares region 'US' where the freeze certified 'UK'" in problem
               for problem in report["verification"]["problems"])
