"""PR164 Phase 6: the eligibility screen recognises documents, not just bytes.

PR163 measured two opposite failures of the same check. Four supplied files were documents the campaign
already used, re-rendered small enough that every byte and every global perceptual hash differed, so the
screen called them eligible. Two genuinely different documents sat inside the 10-bit perceptual radius of
an unrelated one, so the screen excluded them as duplicates. A single threshold cannot fix both; the
screen now asks what document a page is, and reports a resemblance it cannot resolve instead of deciding.

Groups A-F of the PR164 brief map to the tests below. Everything here runs on tracked fixtures except the
two tests guarded by `PR164_ATTACHMENT_DIR`, which need the owner's private files and skip without them.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "app")]

import pr109_intake as I  # noqa: E402
import pr110_authority as A  # noqa: E402
import pr110_eligibility as E  # noqa: E402
import pr110_harness as K  # noqa: E402

FRANCE = ROOT / "app" / "examples" / "france.png"
US = ROOT / "app" / "examples" / "us.png"
BLUR_FR = ROOT / "tests" / "fixtures" / "france-blur-light.png"
BLUR_US = ROOT / "tests" / "fixtures" / "us-blur-light.png"


# --------------------------------------------------------------------------- fixtures

def write_registry(tmp_path: Path, documents: list[dict]) -> Path:
    """A registry in the committed schema, pointing at whatever paths the test chose."""
    payload = {"version": A.REGISTRY_VERSION, "purpose": "test fixture", "built_from": [],
               "policy": {"similarity_threshold_bits": I.PERCEPTUAL_MAX_DISTANCE,
                          "geometry_prefilter_bits": A.GEOMETRY_PREFILTER_BITS,
                          "geometry_min_inliers": A.MIN_INLIERS},
               "documents": documents}
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def document(document_id: str, *variants: dict, **metadata) -> dict:
    return {"id": document_id, "category": metadata.get("category", "test_fixture"),
            "source_type": metadata.get("source_type"), "document_type": metadata.get("document_type"),
            "genuine_business_invoice": metadata.get("genuine_business_invoice"),
            "layout_family": None, "region": None, "currency": None,
            "history": [{"kind": "test", "record": document_id}],
            "variants": list(variants)}


def screen(path: Path, authority: dict, population: list[dict] | None = None) -> tuple[dict, dict]:
    """The bytes-only screen; the population is a parameter so a test can declare it explicitly."""
    record = {"by_sha": {}, "used_hashes": set(), "manifest": "test", "manifest_file": "test",
              "files": 0, "verdict": "test"}
    block = E.screen_file(path, record, population or [], {}, authority)
    return block, E.decide(block["checks"])


def population_of(path: Path) -> list[dict]:
    """What `signed_population()` would contribute for one image, without scanning the whole repository."""
    signature = I.perceptual(path)
    return [{"path": path.relative_to(ROOT).as_posix(), "sha256": signature["sha256"],
             "ahash": signature["ahash"], "dhash": signature["dhash"]}]


def re_render(source: Path, target: Path, *, scale: float = 0.5, jpeg: bool = False,
              crop_ratio: float = 0.0) -> Path:
    with Image.open(source) as image:
        rendered = image.convert("RGB") if jpeg else image
        if crop_ratio:
            rendered = rendered.crop((0, int(rendered.height * crop_ratio), rendered.width,
                                      rendered.height))
        rendered = rendered.resize((max(1, int(rendered.width * scale)),
                                    max(1, int(rendered.height * scale))), Image.LANCZOS)
        if jpeg:
            rendered.save(target, "JPEG", quality=80)
        else:
            rendered.save(target)
    return target


def blank_annotation(document_id: str, image: Path) -> dict:
    """The minimum a v2 reference needs to pass, so 'new document' can be tested end to end."""
    annotation = K.blank(document_id, image.name, [1])
    annotation["document"].update({
        "page_count_observed": 1, "language": "French", "region": "FR", "input_kind": "scanned_image",
        "genuine_business_invoice": True, "quality_declaration": "clear",
        "source_origin": "provided by the project owner", "annotators": ["owner"],
        "annotation_date": "2026-11-10",
        "owner_permission": {"authorized_for_local_evaluation": True,
                             "evidence": "the owner's own document", "date": "2026-11-10"},
        "second_review": {"performed": False, "reviewer": None, "disagreements_resolved": None}})
    for field, value in (("supplier_name", "Atelier Test"), ("invoice_date", "2026-01-15"),
                         ("currency", "EUR"), ("total_ht", 1000.0), ("total_ttc", 1200.0),
                         ("tax.rate", 20.0), ("tax.amount", 200.0)):
        K.set_field(annotation, field, value, "printed")
    K.set_field(annotation, "shipping", None, "absent")
    K.add_row(annotation)
    for field, value in (("description", "Prestation"), ("quantity", 5.0), ("unit_price", 200.0),
                         ("total_ht", 1000.0)):
        K.set_field(annotation, f"line_items[0].{field}", value, "printed")
    return K.seal(annotation, image)


# --------------------------------------------------------------------------- A. exact identity

def test_a_byte_identical_copy_of_a_registered_variant_is_excluded(tmp_path):
    copy = shutil.copyfile(FRANCE, tmp_path / "copy.png")
    authority = A.load(write_registry(tmp_path, [document(
        "app-examples-france", A.variant_record(FRANCE, "demo_asset"))]))

    block, decision = screen(Path(copy), authority)
    assert decision["status"] == E.EXCLUDED
    assert "exact_duplicate" in decision["hard_reasons"]
    finding = next(check for check in block["checks"] if check["check"] == "exact_duplicate")
    assert "app-examples-france" in finding["detail"]


def test_an_original_download_is_recognised_by_its_recorded_hash_alone(tmp_path):
    """The original may live outside the repository; its recorded hash still excludes a byte copy."""
    source = shutil.copyfile(FRANCE, tmp_path / "original-download.webp")
    recorded = A.variant_record(FRANCE, "original_download", expect_present=False)
    recorded["path"] = "original-download.webp"      # the real file is not in the repository
    recorded["expect_present"] = False

    authority = A.load(write_registry(tmp_path, [document("external-source", recorded)]))
    assert authority["complete"] is True, "a variant that is absent by design must not read as missing"

    _, decision = screen(Path(source), authority)
    assert decision["status"] == E.EXCLUDED
    assert decision["hard_reasons"] == ["exact_duplicate"]


# --------------------------------------------------------------------------- B/C. re-renders

@pytest.mark.parametrize("label,scale,jpeg,crop_ratio", [
    ("resized_50_percent", 0.5, False, 0.0),
    ("recompressed_jpeg", 1.0, True, 0.0),
    ("cropped_four_percent", 1.0, False, 0.04),
    ("screenshot_small_and_compressed", 0.45, True, 0.0),
])
def test_a_re_render_of_a_registered_document_is_excluded(tmp_path, label, scale, jpeg, crop_ratio):
    variant = re_render(FRANCE, tmp_path / f"{label}{'.jpg' if jpeg else '.png'}",
                        scale=scale, jpeg=jpeg, crop_ratio=crop_ratio)
    authority = A.load(write_registry(tmp_path, [document(
        "app-examples-france", A.variant_record(FRANCE, "demo_asset"))]))

    block, decision = screen(variant, authority)
    assert decision["status"] == E.EXCLUDED, label
    assert decision["hard_reasons"] == ["registered_variant"], label
    match = block["variant_matches"][0]
    assert match["document"] == "app-examples-france"
    assert match["inliers"] >= A.MIN_INLIERS and match["fraction"] >= A.MIN_INLIER_FRACTION


def test_the_four_pr163_false_negatives_are_recognised_as_variants(tmp_path):
    """The four documents PR163 measured as wrongly eligible, re-rendered exactly as they were supplied."""
    cases = [
        ("photographer", "diagnostics/holdout/new-layout/photographer-anon.png"),
        ("image-proforma", "diagnostics/holdout/image-proforma-anon.png"),
        ("graphic-designers", "diagnostics/holdout/graphic-designers-anon.png"),
        ("service", "diagnostics/holdout/new-layout/service-anon.png"),
    ]
    missing = [path for _, path in cases if not (ROOT / path).is_file()]
    if missing:
        pytest.skip(f"campaign corpora not restored: {missing}")

    for identity, relative in cases:
        source = ROOT / relative
        # PR163 measured these as downscaled screenshots of the original download: new bytes, same page
        variant = re_render(source, tmp_path / f"{identity}.png", scale=0.42, jpeg=True)
        authority = A.load(write_registry(tmp_path, [document(
            identity, A.variant_record(source, "evaluated_copy"))]))
        block, decision = screen(variant, authority)
        assert decision["status"] == E.EXCLUDED, identity
        assert decision["hard_reasons"] == ["registered_variant"], identity
        assert block["variant_matches"][0]["document"] == identity


def test_an_unregistered_re_render_is_still_held_for_review(tmp_path, monkeypatch):
    """Without a registered variant, geometry cannot confirm identity; the screen must not admit the file."""
    def weak(left, right, long_side=A.MAX_SIDE):
        return {"good": 200, "inliers": 70, "scale": 1.0, "covers": 0.3}

    monkeypatch.setattr(A, "correspondence", weak)
    authority = A.load(write_registry(tmp_path, [document(
        "app-examples-france", A.variant_record(FRANCE, "demo_asset"))]))
    variant = re_render(FRANCE, tmp_path / "unknown-render.png", scale=0.5)

    _, decision = screen(variant, authority)
    assert decision["status"] == E.PENDING_REVIEW
    assert decision["review_label"] == "MANUAL_REVIEW_REQUIRED"
    assert decision["hard_reasons"] == []
    assert "registered_variant" in decision["open_questions"]


# --------------------------------------------------------------------------- D. look-alikes

def test_two_different_documents_with_the_same_layout_are_not_a_duplicate(tmp_path):
    """PR163's false positive: 3 bits apart, different invoices. Resemblance is a suspicion, not identity."""
    authority = A.load(write_registry(tmp_path, [document(
        "fixture-france-blur", A.variant_record(BLUR_FR, "regression_fixture"))]))

    block, decision = screen(BLUR_US, authority, population_of(BLUR_FR))
    assert decision["status"] == E.PENDING_REVIEW
    assert decision["review_label"] == "MANUAL_REVIEW_REQUIRED"
    assert decision["hard_reasons"] == [], "a 3-bit resemblance must not exclude a different document"
    assert decision["open_questions"] == ["visually_similar_duplicate"]
    finding = next(check for check in block["checks"] if check["check"] == "visually_similar_duplicate")
    assert "resemblance only" in finding["detail"]
    assert "fixture-france-blur" in finding["detail"] or "france-blur-light" in finding["detail"]


def test_a_same_template_sibling_is_held_rather_than_excluded(tmp_path, monkeypatch):
    """Two invoices from one template share their furniture; that is not proof they are one document."""
    def sibling(left, right, long_side=A.MAX_SIDE):
        return {"good": 242, "inliers": 116, "scale": 1.0, "covers": 0.29}

    monkeypatch.setattr(A, "correspondence", sibling)
    authority = A.load(write_registry(tmp_path, [document(
        "service", A.variant_record(FRANCE, "evaluated_copy"))]))

    _, decision = screen(US, authority)
    assert decision["status"] == E.PENDING_REVIEW
    assert decision["hard_reasons"] == []
    assert "registered_variant" in decision["open_questions"]


# --------------------------------------------------------------------------- E. genuinely new

def test_a_new_document_is_not_treated_as_a_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(E.A, "load", lambda *args, **kwargs: A.empty())
    monkeypatch.setattr(I, "signed_population", lambda: [])
    invoices, references = tmp_path / "invoices", tmp_path / "references"
    invoices.mkdir()
    references.mkdir()

    image = invoices / "new-invoice.png"
    page = Image.new("L", (640, 820), 255)
    draw = ImageDraw.Draw(page)
    for index in range(20):
        draw.rectangle([30 + (index * 37) % 460, 60 + (index * 71) % 620,
                        90 + (index * 37) % 460, 74 + (index * 71) % 620], fill=50)
    page.save(image)
    (references / "new-invoice.json").write_text(
        json.dumps(blank_annotation("new-invoice", image), ensure_ascii=False), encoding="utf-8")

    report = E.eligibility(invoices, references, geometry=False)
    record = report["documents"][0]
    assert record["status"] == E.ELIGIBLE, record["checks"]
    assert record["review_label"] is None
    assert record["registered_identity"] is None


# --------------------------------------------------------------------------- F. no private data

def test_a_missing_authority_holds_every_document_and_says_why(tmp_path):
    authority = A.load(tmp_path / "no-such-registry.json")
    assert authority["available"] is False and authority["complete"] is False
    assert "must not admit anything" in authority["reason"]

    block, decision = screen(FRANCE, authority)
    assert decision["status"] == E.PENDING_REVIEW
    assert decision["hard_reasons"] == []
    assert decision["open_questions"] == ["document_authority"]
    assert decision["review_label"] == "MANUAL_REVIEW_REQUIRED"


def test_a_missing_registered_variant_holds_the_documents(tmp_path):
    variant = A.variant_record(FRANCE, "evaluated_copy")
    variant["path"] = "nowhere/this-file-is-absent.png"
    authority = A.load(write_registry(tmp_path, [document("absent-document", variant)]))

    assert authority["available"] is True and authority["complete"] is False
    assert [entry["path"] for entry in authority["missing"]] == ["nowhere/this-file-is-absent.png"]
    _, decision = screen(US, authority)
    assert decision["status"] == E.PENDING_REVIEW
    assert "document_authority" in decision["open_questions"]


def test_a_drifted_registered_variant_holds_the_documents(tmp_path):
    variant = A.variant_record(FRANCE, "evaluated_copy")
    variant["sha256"] = "0" * 64
    authority = A.load(write_registry(tmp_path, [document("changed-document", variant)]))

    assert authority["complete"] is False
    assert [entry["path"] for entry in authority["drift"]] == [A.variant_record(FRANCE, "x")["path"]]
    _, decision = screen(US, authority)
    assert decision["status"] == E.PENDING_REVIEW
    assert "document_authority" in decision["open_questions"]


# --------------------------------------------------------------------------- the registries on this machine

# PR167 split the authority in two: the demonstration registry is tracked and must satisfy these guards in a
# clone, the private one is gitignored and can only be measured where the campaign's corpora live. A missing
# private registry skips; it never passes unseen.
REGISTRIES = [pytest.param(A.DEMO_REGISTRY_PATH, "demo", id="tracked-demo"),
              pytest.param(A.REGISTRY_PATH, "repository", id="local-private",
                           marks=pytest.mark.skipif(
                               not A.REGISTRY_PATH.is_file(),
                               reason="the private document authority is local by design since PR167"))]


@pytest.mark.parametrize("path, scope", REGISTRIES)
def test_a_registry_loads_and_states_its_policy(path, scope):
    authority = A.load(path)
    assert authority["available"] is True
    assert authority["version"] == A.REGISTRY_VERSION
    assert authority["scope"] == scope
    assert authority["documents"], "a registry that names no document certifies nothing"

    registry = json.loads(path.read_text(encoding="utf-8"))
    policy = registry["policy"]
    assert policy["similarity_threshold_bits"] == I.PERCEPTUAL_MAX_DISTANCE == 10
    assert policy["geometry_min_inliers"] == A.MIN_INLIERS
    for entry in registry["documents"]:
        assert entry["id"] and entry["variants"], entry["id"]
        for variant in entry["variants"]:
            assert variant["role"] and variant["path"], entry["id"]
            assert variant["sha256"] or not variant["expect_present"], entry["id"]


@pytest.mark.parametrize("path, scope", REGISTRIES)
def test_a_registry_carries_no_supplier_name_or_printed_text(path, scope):
    """A registry that reaches Git names no party and repeats nothing printed on a page.

    PR166 Phase 2 separated what this guard can confuse. A party's name and a value read off a document must
    never appear, and the two loops below keep them out. Where a variant is kept has to appear, because
    verifying it means opening that file, so a corpus whose pages are named after the supplier that issued
    them leaves that name in a location field. The guard therefore also pins what a location field may be,
    which is the boundary PR166 judged acceptable for publication.
    """
    text = path.read_text(encoding="utf-8")
    forbidden = ("Prestige Services", "Zylker", "Picapot", "Studio Lumière", "CONSULT ME",
                 "Lawn maintenance", "Mon Entreprise", "GRAPHIC DESIGNERS", "Garden Repairs",
                 "Adventure Club")
    for name in forbidden:
        assert name not in text, name
    for token in ("total_ttc", "total_ht", "invoice_number", "supplier_name", "tva", "TVA"):
        assert token not in text, token

    registry = json.loads(text)
    declared_roots = {root.split("/")[0] for root, _ in A.DEVELOPMENT_ROOTS} | {"diagnostics"}
    for entry in registry["documents"]:
        for variant in entry["variants"]:
            location = variant["path"]
            assert ":" not in location and "\\" not in location and not location.startswith("/")
            if "/" in location:
                assert location.split("/")[0] in declared_roots, location
            else:
                assert not variant["expect_present"], f"{location} names a file the campaign never stored"


# --------------------------------------------------------------------------- the owner's own files

@pytest.mark.skipif(not __import__("os").environ.get("PR164_ATTACHMENT_DIR"),
                    reason="set PR164_ATTACHMENT_DIR to the folder of the PR163 attachments to re-measure them")
def test_the_pr163_attachments_cannot_be_admitted():
    directory = Path(__import__("os").environ["PR164_ATTACHMENT_DIR"])
    files = sorted((path for path in directory.glob("*.png") if path.stem.isdigit()),
                   key=lambda path: int(path.stem))
    assert len(files) == 11, files

    record = E.pr103_record()
    authority = A.load()
    siblings = {I.sha256_file(path): path for path in files}
    decisions = {}
    for path in files:
        block = E.screen_file(path, record, I.signed_population(), siblings, authority)
        decisions[path.name] = E.decide(block["checks"])

    excluded = [name for name, decision in decisions.items() if decision["status"] == E.EXCLUDED]
    assert len(excluded) == 10, decisions
    assert decisions["1.png"]["status"] == E.ELIGIBLE, "the garden-repairs credit note is genuinely new"
    for name in ("4.png", "6.png", "8.png", "10.png"):
        assert "registered_variant" in decisions[name]["hard_reasons"], decisions[name]
    for name in ("5.png", "11.png"):
        assert "registered_variant" in decisions[name]["hard_reasons"], decisions[name]
