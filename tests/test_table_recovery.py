"""Experimental table recovery: an engine can veto a cell, never supply one.

Every case here runs the recovery hook with a queued stand-in for the independent OCR engine, so the
rules are exercised without RapidOCR and without a document.
"""

import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import table_recovery  # noqa: E402
from invoice_evidence import DocumentEvidence, from_tsv  # noqa: E402

FLAG = table_recovery.FLAG_NAME
PAGE = (1000, 1000)
ROTATED = (500, 500)


def _tsv(words):
    data = {key: [] for key in (
        "level", "page_num", "block_num", "par_num", "line_num", "word_num",
        "left", "top", "width", "height", "conf", "text")}
    for index, (text, right, y) in enumerate(words):
        width = 4 * len(text)
        data["level"].append(5); data["page_num"].append(1); data["block_num"].append(1)
        data["par_num"].append(index + 1); data["line_num"].append(index + 1)
        data["word_num"].append(1)
        data["left"].append(right - width); data["top"].append(y)
        data["width"].append(width); data["height"].append(14)
        data["conf"].append(96); data["text"].append(text)
    return data


TABLE = [
    ("Description", 300, 90), ("Quantity", 500, 90), ("Unit", 640, 90), ("price", 700, 90),
    ("Amount", 860, 90),
    ("Consulting", 300, 140), ("2", 500, 140), ("10.00", 700, 140), ("20.00", 860, 140),
    ("Support", 300, 180), ("3", 500, 180), ("5.00", 700, 180), ("15.00", 860, 180),
]


class QueueEngine:
    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def __call__(self, image, *, use_det=False, use_cls=False, use_rec=True):
        self.calls += 1
        text = self.texts[(self.calls - 1) % len(self.texts)] if self.texts else ""
        return SimpleNamespace(txts=(text,), scores=(0.9,))


def reads():
    reading = from_tsv(_tsv(TABLE))
    return {"deskew": [reading], "clahe": [reading]}


def evidence(*, rows=()):
    document = DocumentEvidence("", None, [], [])
    document.source_kind = "ocr"
    document.rows = list(rows)
    return document


def prepared():
    return {"deskew": Image.new("L", PAGE, 255)}


def test_recovery_is_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    document = evidence()
    returned = table_recovery.recover(document, reads(), rotated_size=ROTATED,
                                      prepared=prepared(), engine=QueueEngine(["2"]))
    assert returned is document
    assert document.rows == []
    assert getattr(document, "table_recovery", None) is None


def test_recovery_requires_the_independent_engine(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(table_recovery, "get_engine", lambda: None)
    document = evidence()
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=None)
    assert document.rows == []
    assert "engine" in document.table_recovery["reason"]


def test_rows_are_built_only_from_confirmed_cells(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    engine = QueueEngine(["2", "10.00", "20.00", "3", "5.00", "15.00"])
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=engine)
    assert document.table_recovery["status"] == "recovered"
    assert len(document.rows) == 2
    first = document.rows[0].cells
    assert {role: str(cells[0].value) for role, cells in first.items()} == {
        "quantity": "2", "unit_price": "10.00", "amount_ht": "20.00"}
    assert first["unit_price"][0].raw == "10.00"
    assert isinstance(first["quantity"][0], type(first["unit_price"][0]))


def test_a_conflicting_engine_rejects_the_cell(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    engine = QueueEngine(["999"])
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=engine)
    assert document.rows == []
    refusals = document.table_recovery["refused"]
    assert refusals and all(item.get("reason") in {"engines disagree", "no confirmed cell"}
                            for item in refusals)


def test_the_engine_text_never_becomes_the_value(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    # Same number, different spelling: the documentary text must survive untouched.
    engine = QueueEngine(["2", "10", "20", "3", "5", "15"])
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=engine)
    assert document.rows[0].cells["unit_price"][0].raw == "10.00"


def test_documents_with_production_rows_are_left_alone(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    from invoice_evidence import Number, TableRow, TextLine

    line = TextLine("Service | 2 | 10 | 20", [], 0, (0, 0, 10, 10))
    row = TableRow("Service", {"quantity": [Number("2", Decimal(2), 0, (0, 0, 1, 1))]}, line)
    document = evidence(rows=[row])
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=QueueEngine(["2"]))
    assert document.rows == [row]
    assert document.table_recovery["status"] == "not_triggered"

def _consensus_with_table_issue(rows_provenance):
    from invoice_evidence import DocumentEvidence, Token
    document = DocumentEvidence("", None, [Token("x", 0, (0, 0, 1, 1), 95)], [])
    document.source_kind = "consensus"
    document.ocr_cross_checked = True
    document.quality_issues = ["Tableau non reconstruit malgré les prétraitements."]
    document.rows_provenance = rows_provenance
    from invoice_evidence import TableRow, TextLine
    line = TextLine("x", [], 0, (0, 0, 1, 1))
    document.rows = [TableRow("x", {}, line)]
    return document


def test_recovered_rows_keep_the_partial_layout_allowance():
    from invoice_grounding import partial_layout_is_usable
    from ocr_quality import document_quality

    recovered = _consensus_with_table_issue("recovered")
    issues = document_quality(recovered)
    assert partial_layout_is_usable(recovered, issues), "recovered rows must not close the gate"

    production = _consensus_with_table_issue(None)
    assert not partial_layout_is_usable(production, document_quality(production))


def test_recovered_rows_do_not_satisfy_the_model_quality_gate():
    from ocr_quality import document_quality

    document = _consensus_with_table_issue("recovered")
    document.source_kind = "ocr"
    model = {"line_items": [{"description": "a"}, {"description": "b"}]}
    issues = document_quality(document, model)
    assert not any("Nombre d'articles" in issue for issue in issues)

    document.rows_provenance = None
    issues = document_quality(document, model)
    assert any("Nombre d'articles" in issue for issue in issues), "production rows are still counted"


def test_recovery_marks_its_rows_as_recovered(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=QueueEngine(["2", "10.00", "20.00", "3", "5.00", "15.00"]))
    assert document.rows and document.rows_provenance == "recovered"

def test_recovered_rows_keep_json_safe_ordinals(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=QueueEngine(["2", "10.00", "20.00", "3", "5.00", "15.00"]))
    for row in document.rows:
        assert row.row_index is None or isinstance(row.row_index, int)
        assert set(row.row_evidence) >= {"page", "row_index", "row_index_confidence",
                                         "printed", "positions", "refused_cells"}
    assert all(0 <= value <= 1 for box in
               (cell[0].box for row in document.rows for cell in row.cells.values())
               for value in box)


def test_ordinals_come_from_the_pitch_grid_and_stay_unresolved_when_unprovable():
    def band(top, height=0.03):
        return {"band_top": top, "band_bottom": top + height}

    ordinals, confidences = table_recovery._row_ordinals(
        [band(0.12), band(0.16), band(0.20)], 0.10)
    assert ordinals == [0, 1, 2] and confidences == ["high"] * 3

    # A single step after the first row, with irregular spacing: the ordinal is not guessed.
    ordinals, confidences = table_recovery._row_ordinals([band(0.12), band(0.20)], 0.10)
    assert ordinals == [0, None] and confidences == ["high", None]

    # A row that does not sit under the table floor is unresolved even alone.
    ordinals, _ = table_recovery._row_ordinals([band(0.30)], 0.10)
    assert ordinals == [None]


def _recovered_document(ordinals):
    from invoice_evidence import DocumentEvidence, Number, TableRow, TextLine
    from decimal import Decimal as D

    document = DocumentEvidence("", None, [], [])
    document.source_kind = "ocr"
    document.ocr_cross_checked = True
    document.rows_provenance = "recovered"
    for ordinal in ordinals:
        line = TextLine("Service", [], 0, (0, 0.1 * (ordinal + 1), 1, 0.1 * (ordinal + 1) + 0.02))
        document.rows.append(TableRow(
            "Service %d" % ordinal,
            {"quantity": [Number("10", D(10), 0, (0.5, 0.1, 0.6, 0.12))]},
            line, row_index=ordinal,
        ))
    return document


def _model(items):
    from invoice_grounding import read_model
    import json as _json
    payload = {"supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
               "line_items": [{"description": "Service %d" % index, "quantity": 10,
                               "unit_price": 1, "total_ht": 10} for index in range(items)]}
    return read_model(_json.dumps(payload))


def test_a_recovered_third_row_is_never_exported_as_the_first():
    from invoice_grounding import reconcile
    result = reconcile(_recovered_document([2]), [_model(3)])
    assert [item["quantity"] for item in result.payload["line_items"]] == [None, None, None]
    assert all(status != "valide" for status in result.statuses.values())


def test_a_missing_middle_row_blocks_the_export():
    from invoice_grounding import reconcile
    result = reconcile(_recovered_document([0, 2]), [_model(3)])
    assert [item["quantity"] for item in result.payload["line_items"]] == [None, None, None]


def test_a_complete_recovered_set_is_exported():
    from invoice_grounding import reconcile
    result = reconcile(_recovered_document([0]), [_model(1)])
    assert result.payload["line_items"][0]["quantity"] == 10


def test_reordered_model_proposals_cannot_shift_rows():
    from invoice_grounding import reconcile
    document = _recovered_document([0, 1])
    document.rows[0].description = "Alpha"
    document.rows[1].description = "Beta"
    import json as _json
    from invoice_grounding import read_model
    model = read_model(_json.dumps({
        "supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
        "line_items": [{"description": "Beta", "quantity": 10, "unit_price": 1, "total_ht": 10},
                       {"description": "Alpha", "quantity": 10, "unit_price": 1, "total_ht": 10}],
    }))
    result = reconcile(document, [model])
    assert [item["description"] for item in result.payload["line_items"]] == ["Alpha", "Beta"]

def test_rule_lattice_lists_only_full_width_rules():
    import numpy as np
    from table_shadow import rule_lattice

    pixels = np.full((400, 600), 255, dtype=np.uint8)
    for y in (100, 160, 220, 280):
        pixels[y:y + 2, 60:560] = 0
    pixels[340:342, 60:200] = 0  # a decorative short line is not a row boundary
    lattice = rule_lattice(pixels, (60, 100, 500, 182))
    assert len(lattice["rules"]) == 4
    assert lattice["rules"] == sorted(lattice["rules"])
    assert all(0.2 < rule < 0.8 for rule in lattice["rules"])


def test_lattice_stays_a_diagnostic_until_the_origin_is_anchored(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    document = evidence()
    table_recovery.recover(document, reads(), rotated_size=ROTATED, prepared=prepared(),
                           engine=QueueEngine(["2", "10.00", "20.00", "3", "5.00", "15.00"]))
    assert "lattice" in document.table_recovery, "the ruling is reported for diagnostics"
    # A blank prepared page has no rules, so nothing may be attributed from a lattice.
    assert all(row.row_index is None or isolate(row.row_index) for row in document.rows)


def isolate(value):
    return isinstance(value, int) and value >= 0
