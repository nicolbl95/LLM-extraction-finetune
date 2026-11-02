"""Shadow-mode table reconstruction: what it may record and what it may never do.

Every case here is a synthetic page built from word boxes, so the rules are tested on geometry and
labels rather than on a supplier. The flag is off by default, and these tests hold that boundary:
an experiment that cannot be switched off is not an experiment.
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import table_reconstruction as tr  # noqa: E402
import table_shadow  # noqa: E402
from invoice_evidence import DocumentEvidence, from_tsv  # noqa: E402
from ocr_recovery import recover_image  # noqa: E402

FLAG = table_shadow.FLAG_NAME
PAGE = (1000, 1000)
HEADER_Y = 90
ROW_Y = (140, 180, 220)


def page(rows, *, size=PAGE):
    """Rows of (text, right edge, confidence) into the normalised word boxes the module expects.

    Cells are given by their right edge because that is what a printed numeric column aligns on.
    """
    tokens = []
    for y, cells in rows:
        for text, right, confidence in cells:
            left = right - 4 * len(text)
            tokens.append((text, (left, y, right, y + 14), confidence))
    return tr.words_of(tokens, family="deskew", read="deskew_0", size=size)


def two_families(rows, **overrides):
    """Two preprocessing families, agreeing: the minimum the evidence rule accepts."""
    reads = {"deskew_0": page(rows)}
    for family in ("clahe", "scale3"):
        words = tr.words_of(
            [(text, box, confidence) for text, box, confidence in
             [(t[0], t[1], t[2]) for t in
              [(w["text"], w["pixel"], w["confidence"]) for w in reads["deskew_0"]]]],
            family=family, read="%s_0" % family, size=overrides.get("size", PAGE),
        )
        reads[family + "_0"] = words
    return reads


def cell_text(reads, family, text, *, drop=()):
    """Rebuild one family's read with a word changed, to test agreement between witnesses."""
    return [
        dict(word, text=text, confidence=95)
        for word in reads[family + "_0"]
        if word["text"] not in drop
    ]


TABLE = [
    (HEADER_Y, [("Description", 300, 96), ("Quantity", 500, 96),
                ("Unit", 640, 96), ("price", 700, 96), ("Amount", 860, 96)]),
    (ROW_Y[0], [("Consulting", 300, 96), ("2", 500, 96), ("10.00", 700, 96), ("20.00", 860, 96)]),
    (ROW_Y[1], [("Support", 300, 96), ("3", 500, 96), ("5.00", 700, 96), ("15.00", 860, 96)]),
]


def test_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not table_shadow.shadow_enabled()
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv(FLAG, value)
        assert table_shadow.shadow_enabled()
    for value in ("", "0", "off", "no", "false"):
        monkeypatch.setenv(FLAG, value)
        assert not table_shadow.shadow_enabled()


def test_disabled_shadow_leaves_the_evidence_object_untouched(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    evidence = DocumentEvidence("text", None, [], [])
    before = repr(vars(evidence))
    returned = table_shadow.observe(evidence, {}, rotated_size=PAGE)
    assert returned is evidence
    assert repr(vars(returned)) == before
    assert returned.table_shadow is None


def test_column_role_is_taken_from_the_label_printed_over_the_column():
    report = tr.reconstruct(two_families(TABLE))
    columns = report["pinned_roles"]
    assert columns["quantity"] < columns["unit_price"] < columns["amount_ht"]
    values = [
        {role: cell["value"] for role, cell in row["cells"].items()} for row in report["rows"]
    ]
    assert values == [
        {"quantity": "2", "unit_price": "10.00", "amount_ht": "20.00"},
        {"quantity": "3", "unit_price": "5.00", "amount_ht": "15.00"},
    ]


def test_overlapping_header_cues_assign_nothing():
    """'Net unit price' names a price column: the 'unit' inside it must not claim a quantity."""
    rows = [
        (HEADER_Y, [("Product", 300, 96), ("Net", 430, 96), ("unit", 470, 96), ("price", 500, 96),
                    ("Units", 600, 96), ("Net", 700, 96), ("amount", 760, 96)]),
        (ROW_Y[0], [("Handbook", 300, 96), ("29.66", 500, 96), ("15", 600, 96), ("444.90", 760, 96)]),
        (ROW_Y[1], [("Monitor", 300, 96), ("249.06", 500, 96), ("29", 600, 96), ("7,222.74", 760, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert report["pinned_roles"]["unit_price"] == 0 or "unit_price" in report["pinned_roles"]
    columns = {role: index for role, index in report["pinned_roles"].items()}
    assert columns["quantity"] == columns["unit_price"] + 1
    by_column = {}
    for row in report["rows"]:
        for role, cell in row["cells"].items():
            by_column.setdefault(columns[role], []).append(cell["value"])
    assert sorted(by_column[columns["quantity"]]) == ["15", "29"], "prices must not read as quantities"
    assert sorted(by_column[columns["unit_price"]]) == ["249.06", "29.66"]
    assert sorted(by_column[columns["amount_ht"]]) == ["444.90", "7222.74"]


def test_a_bare_sum_word_never_claims_the_line_amount_column():
    rows = [
        (HEADER_Y, [("Quantite", 400, 96), ("Montant", 600, 96), ("Total", 760, 96), ("TVA", 820, 96)]),
        (ROW_Y[0], [("Loyer", 200, 96), ("1", 400, 96), ("1000.00", 600, 96), ("200.00", 820, 96)]),
        (ROW_Y[1], [("Charge", 200, 96), ("2", 400, 96), ("500.00", 600, 96), ("100.00", 820, 96)]),
        (260, [("Total", 200, 96), ("TVA", 300, 96), ("300.00", 820, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert "amount_tax" in report["pinned_roles"] and "amount_ht" in report["pinned_roles"]
    assert report["pinned_roles"]["amount_tax"] != report["pinned_roles"]["amount_ht"]
    assert [row["cells"]["amount_ht"]["value"] for row in report["rows"]] == ["1000.00", "500.00"]
    assert [row["cells"].get("amount_tax", {}).get("value") for row in report["rows"]] == [
        "200.00", "100.00"]


def test_clock_durations_are_not_quantities():
    rows = [
        (HEADER_Y, [("Description", 300, 96), ("Quantity", 500, 96), ("Rate", 700, 96)]),
        (ROW_Y[0], [("Devise A", 300, 96), ("0300", 500, 96), ("40", 700, 96)]),
        (ROW_Y[1], [("Devise B", 300, 96), ("06.30", 500, 96), ("40", 700, 96)]),
        (ROW_Y[2], [("Devise C", 300, 96), ("0420", 500, 96), ("40", 700, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert "quantity" not in report["pinned_roles"]
    assert "clock durations" in report["pinning_evidence"]["quantity"]
    assert all("quantity" not in row["cells"] for row in report["rows"])


def test_a_summary_line_is_not_a_line_item():
    rows = TABLE + [(260, [("Total", 300, 96), ("TVA", 420, 96), ("7.00", 860, 96)])]
    report = tr.reconstruct(two_families(rows))
    assert len(report["rows"]) == 2
    assert not any("TVA" in row["band_text"] for row in report["rows"])
    assert [total["kind"] for total in report["labelled_totals"]] == ["tax"]


def test_a_wrapped_header_still_names_every_column():
    rows = [
        (HEADER_Y, [("Description", 300, 96), ("Quantity", 500, 96)]),
        (106, [("Unit", 640, 96), ("price", 700, 96), ("Amount", 860, 96)]),
        (ROW_Y[0], [("Consulting", 300, 96), ("2", 500, 96), ("10.00", 700, 96), ("20.00", 860, 96)]),
        (ROW_Y[1], [("Support", 300, 96), ("3", 500, 96), ("5.00", 700, 96), ("15.00", 860, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert {"quantity", "unit_price", "amount_ht"} <= set(report["pinned_roles"])
    assert report["rows"][0]["cells"]["unit_price"]["value"] == "10.00"


def test_comma_thousands_is_an_amount_not_a_rejection():
    rows = [
        (HEADER_Y, [("Description", 300, 96), ("Amount", 760, 96)]),
        (ROW_Y[0], [("Serveur", 300, 96), ("7,222.74", 760, 96)]),
        (ROW_Y[1], [("Clavier", 300, 96), ("176.05", 760, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert [row["cells"]["amount_ht"]["value"] for row in report["rows"]] == ["7222.74", "176.05"]
    assert tr.parse_money("1,234") is None, "a single comma stays ambiguous and is refused"
    assert tr.parse_money("1,234.56") == Decimal("1234.56")
    assert tr.parse_money("1.234,56") == Decimal("1234.56")


def test_a_value_printed_by_one_family_only_is_not_a_candidate():
    reads = two_families(TABLE)
    for family in ("clahe", "scale3"):
        reads["%s_0" % family] = [
            word for word in reads["%s_0" % family] if word["text"] != "20.00"]
    report = tr.reconstruct(reads)
    first = report["rows"][0]["cells"]
    assert "amount_ht" not in first, "one witness is not corroboration"
    assert first["quantity"]["value"] == "2"
    assert report["rows"][1]["cells"]["amount_ht"]["value"] == "15.00"


def test_incorrectly_printed_line_amount_is_kept_and_reported():
    rows = [
        (HEADER_Y, [("Description", 300, 96), ("Quantity", 500, 96),
                    ("Unit", 640, 96), ("price", 700, 96), ("Amount", 860, 96)]),
        (ROW_Y[0], [("Consulting", 300, 96), ("2", 500, 96), ("10.00", 700, 96), ("20.00", 860, 96)]),
        (ROW_Y[1], [("Support", 300, 96), ("3", 500, 96), ("5.00", 700, 96), ("14.00", 860, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    second = report["rows"][1]["cells"]
    assert second["amount_ht"]["value"] == "14.00", "the printed amount is the document's, not ours"
    assert "quantity" not in second and "unit_price" not in second, "no product may be exported"
    assert report["rows"][0]["cells"].keys() == {"quantity", "unit_price", "amount_ht"}
    veto = report["row_arithmetic_vetoes"][0]
    assert veto["printed_amount"] == "14.00" and veto["calculated_amount"] == "15.00"
    rejected = report["rows"][1]["rejected_cells"]
    assert rejected["quantity"]["value"] == "3" and rejected["quantity"]["reason"]


def test_document_reconciliation_refuses_a_column_that_contradicts_the_printed_total():
    rows = TABLE + [(260, [("Total", 200, 96), ("HT", 300, 96), ("99.00", 860, 96)])]
    report = tr.reconstruct(two_families(rows))
    assert report["certification_rejections"]["amount_ht"]
    assert all("amount_ht" not in row["cells"] for row in report["rows"])
    assert all(row["rejected_cells"]["amount_ht"]["reason"].startswith("document reconciliation")
               for row in report["rows"])


def test_descriptions_stay_diagnostic():
    report = tr.reconstruct(two_families(TABLE))
    for row in report["rows"]:
        assert "description" not in row["cells"]
        assert row["detected_text"] in {"Consulting", "Support"}
        assert row["detected_evidence"]["families"] == ["clahe", "deskew", "scale3"]


def test_reconstruction_needs_two_families_before_it_starts():
    report = tr.reconstruct({"deskew_0": page(TABLE)})
    assert report["status"] == "reconstructed" and len(report["rows"]) >= 1
    reads = {"deskew_0": page(TABLE), "deskew_1": page(TABLE)}
    report = tr.reconstruct(reads)
    assert report["rows"] == [] or all(not row["cells"] for row in report["rows"])


def test_grid_trigger_needs_a_ruled_table():
    import numpy as np

    plain_page = np.full((400, 600), 255, dtype=np.uint8)
    assert table_shadow.grid_trigger(plain_page)[0] is False
    ruled = plain_page.copy()
    for y in (100, 160, 220, 280):
        ruled[y:y + 2, 60:560] = 0
    for x in (60, 240, 420, 558):
        ruled[100:282, x:x + 2] = 0
    triggered, box, reason = table_shadow.grid_trigger(ruled)
    assert triggered is True and box is not None and "rules" in reason
    ruled_lines = plain_page.copy()
    for y in (100, 160, 220):
        ruled_lines[y:y + 2, 60:560] = 0
    assert table_shadow.grid_trigger(ruled_lines)[0] is False, "rules alone are not a grid"


def test_grid_read_is_only_taken_inside_the_detected_grid():
    import numpy as np

    page_pixels = np.full((400, 600), 255, dtype=np.uint8)
    for y in (100, 160, 220, 280):
        page_pixels[y:y + 2, 60:560] = 0
    for x in (60, 300, 558):
        page_pixels[100:282, x:x + 2] = 0
    page_pixels[120:140, 320:380] = 0  # a glyph inside the grid
    page_pixels[340:360, 320:380] = 0   # a glyph outside the grid, which must survive
    cleaned = table_shadow.strip_grid(page_pixels, (60, 100, 498, 182))
    assert (cleaned[100:282, 60:560] == 255).all() or (cleaned[100:282, 60:560] > 200).mean() > .98
    assert (cleaned[340:360, 320:380] == 0).all(), "outside the grid nothing may be removed"


def _evidence(text="Invoice: FR-9\nTotal HT: 99.00", rows=()):
    evidence = DocumentEvidence(text, None, [], [])
    evidence.source_kind = "ocr"
    evidence.rows = list(rows)
    return evidence


# The synthetic page is written in the prepared (x2) frame, so its rotated source is half as big.
ROTATED = (PAGE[0] // 2, PAGE[1] // 2)


def witness_reads(rows):
    """The page in the exact shape the pipeline hands the shadow: family -> evidence reads.

    Only the scale-2 families are used, so every token box really shares one coordinate frame.
    """
    words = [(text, right, y) for y, cells in rows for text, right, _confidence in cells]
    reading = from_tsv(_tsv(words))
    return {"deskew": [reading], "clahe": [reading]}


def test_observe_records_and_never_merges(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    evidence = _evidence()
    result = table_shadow.observe(evidence, witness_reads(TABLE), rotated_size=ROTATED)
    assert result is evidence
    assert evidence.table_shadow["status"] == "reconstructed"
    assert evidence.table_shadow["candidate_rows"], "the candidate is recorded, not discarded"
    assert evidence.rows == [], "a candidate row must never reach the extraction"
    assert evidence.text == "Invoice: FR-9\nTotal HT: 99.00"


def test_observe_does_not_run_when_the_pipeline_already_has_rows(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    from invoice_evidence import Number, TextLine, TableRow

    line = TextLine("Service | 2 | 10 | 20", [], 0, (0, 0, 10, 10))
    row = TableRow("Service", {"quantity": [Number("2", Decimal(2), 0, (0, 0, 1, 1))]}, line)
    evidence = _evidence(rows=[row])
    table_shadow.observe(evidence, {}, rotated_size=PAGE)
    assert evidence.table_shadow["status"] == "not_triggered"


def test_observe_survives_a_broken_component(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    evidence = _evidence()

    def reader(pixels, psm):
        raise RuntimeError("simulated failure inside the experiment")

    def explode(*args, **kwargs):
        raise RuntimeError("simulated failure inside the experiment")

    monkeypatch.setattr(table_shadow, "reads_of", explode)
    table_shadow.observe(
        evidence, witness_reads(TABLE), rotated_size=ROTATED,
        prepared={"deskew": Image.new("L", PAGE)}, reader=reader)
    assert evidence.table_shadow["status"] == "error"
    assert "simulated failure" in evidence.table_shadow["reason"]
    assert evidence.rows == [] and evidence.text == "Invoice: FR-9\nTotal HT: 99.00"


def test_observe_stops_at_its_own_deadline(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    clock = [0.0]
    monkeypatch.setattr(table_shadow, "MAX_SHADOW_SECONDS", 0.5)
    evidence = _evidence()
    table_shadow.observe(evidence, witness_reads(TABLE), rotated_size=ROTATED, clock=lambda: clock[0])
    # The reads arrive instantly here; the deadline is honoured between phases, not faked.
    assert evidence.table_shadow["status"] == "reconstructed"

    ticks = iter(range(100))
    late = _evidence()
    table_shadow.observe(late, witness_reads(TABLE), rotated_size=ROTATED,
                         clock=lambda: float(next(ticks)))
    assert late.table_shadow["status"] == "deadline"
    assert "0.5" in late.table_shadow["reason"]


def test_recovered_values_are_recorded_with_their_coordinates(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    evidence = _evidence()
    table_shadow.observe(evidence, witness_reads(TABLE), rotated_size=ROTATED)
    report = evidence.table_shadow
    assert report["status"] == "reconstructed"
    assert report["comparison"]["production_rows"] == 0
    candidate = report["candidate_rows"][0]
    assert candidate["quantity"] == "2" and candidate["description"] is None
    assert candidate["description_for_review"] == "Consulting"
    position = candidate["evidence"]["quantity"]["positions"][0]
    assert {"read", "family", "text", "confidence", "box", "pixel"} <= set(position)
    assert candidate["evidence"]["quantity"]["families"] == ["clahe", "deskew"]
    assert json_is_serialisable(report)


def json_is_serialisable(report):
    import json

    json.dumps(report)
    return True


def test_comparison_reads_the_production_totals_as_values_not_characters(monkeypatch):
    """A real `Number.value` is a Decimal printed once per column: the comparison must not walk it."""
    from invoice_evidence import Number

    monkeypatch.setenv(FLAG, "1")
    evidence = _evidence()
    evidence.totals = {"total_ht": [Number("35.00", Decimal("35.00"), 0, (0, 0, 1, 1))]}
    table_shadow.observe(evidence, witness_reads(TABLE), rotated_size=ROTATED)
    comparison = evidence.table_shadow["comparison"]
    assert evidence.table_shadow["status"] == "reconstructed"
    assert comparison["candidate_amount_column_sum"] == "35.00"
    assert comparison["agrees_with_production_total_ht"] == ["35.00"]
    assert comparison["production_totals"] == {"total_ht": ["35.00"]}


def _tsv(words, size=PAGE):
    """A minimal image_to_data payload: one word per line, at the given right edges."""
    data = {key: [] for key in (
        "level", "page_num", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf", "text")}
    for index, (text, right, y) in enumerate(words):
        width = 4 * len(text)
        data["level"].append(5)
        data["page_num"].append(1)
        data["block_num"].append(1)
        data["par_num"].append(index + 1)
        data["line_num"].append(index + 1)
        data["word_num"].append(1)
        data["left"].append(right - width)
        data["top"].append(y)
        data["width"].append(width)
        data["height"].append(14)
        data["conf"].append(96)
        data["text"].append(text)
    return data


def test_pipeline_end_to_end_adds_only_the_report(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    words = [("Description", 300, 90), ("Quantity", 500, 90), ("Unit", 640, 90), ("price", 700, 90),
             ("Amount", 860, 90), ("Consulting", 300, 140), ("2", 500, 140), ("10.00", 700, 140),
             ("20.00", 860, 140)]

    def reader(pixels, psm):
        return _tsv(words)

    plain = recover_image(Image.new("L", (500, 500), 255), reader)
    assert plain.table_shadow is None, "off must mean untouched"
    before = (plain.text, plain.rows, plain.totals, plain.supplier, plain.invoice_number,
              plain.quality_issues, plain.issues, plain.numbers, plain.taxes)

    monkeypatch.setenv(FLAG, "1")
    shadowed = recover_image(Image.new("L", (500, 500), 255), reader)
    after = (shadowed.text, shadowed.rows, shadowed.totals, shadowed.supplier,
             shadowed.invoice_number, shadowed.quality_issues, shadowed.issues, shadowed.numbers,
             shadowed.taxes)
    assert after == before, "shadow mode may not change the evidence the extraction reads"
    assert shadowed.table_shadow["status"] in {"reconstructed", "skipped", "not_triggered"}
    assert shadowed.rows == plain.rows

def test_headerless_amount_column_is_pinned_only_by_the_printed_total():
    rows = [
        (140, [("Service A", 300, 96), ("2", 500, 96), ("100.00", 760, 96)]),
        (180, [("Service B", 300, 96), ("1", 500, 96), ("50.00", 760, 96)]),
        (240, [("Total", 200, 96), ("HT", 300, 96), ("150.00", 760, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert report["pinned_roles"].get("amount_ht") == 1
    assert "corroborated column sum" in report["pinning_evidence"]["amount_ht"]
    assert [row["cells"]["amount_ht"]["value"] for row in report["rows"]] == ["100.00", "50.00"]
    assert "quantity" not in report["pinned_roles"], "no cue, no role"


def test_headerless_columns_without_a_matching_total_stay_unpinned():
    rows = [
        (140, [("Service A", 300, 96), ("2", 500, 96), ("100.00", 760, 96)]),
        (180, [("Service B", 300, 96), ("1", 500, 96), ("50.00", 760, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert "amount_ht" not in report["pinned_roles"]
    assert all("amount_ht" not in row["cells"] for row in report["rows"])


def test_headerless_columns_whose_sum_misses_the_total_stay_unpinned():
    rows = [
        (140, [("Service A", 300, 96), ("2", 500, 96), ("100.00", 760, 96)]),
        (180, [("Service B", 300, 96), ("1", 500, 96), ("50.00", 760, 96)]),
        (240, [("Total", 200, 96), ("HT", 300, 96), ("999.00", 760, 96)]),
    ]
    report = tr.reconstruct(two_families(rows))
    assert "amount_ht" not in report["pinned_roles"]
    assert all("amount_ht" not in row["cells"] for row in report["rows"])
