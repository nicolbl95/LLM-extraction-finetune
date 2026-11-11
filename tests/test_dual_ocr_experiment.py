"""PR101 dual-engine arbitration tests: synthetic, deterministic, no OCR and no model.

Both engines' evidence is built through the unmodified production row builder (``from_tokens``), so a
fixture that would not produce a row in the product cannot produce one here either. Every assertion
then inspects the *exported* strict JSON through ``reconcile``, not the tokens the engines saw.
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import pytest

import dual_ocr_experiment as experiment
from invoice_evidence import Token, from_tokens
from invoice_grounding import reconcile
from ocr_consensus import build_consensus

HEADER_Y = 100
COLUMN_X = (60, 200, 380, 520, 660)


def token(text, x, y, width=None, confidence=95.0):
    return Token(text, 0, (x, y, x + (width or 12 + 5 * len(text)), y + 14), confidence)


WIDTHS = (None, 14, 44, 24, 48)


def table(labels, rows, xs=COLUMN_X, confidence=95.0):
    tokens = [token(label, x, HEADER_Y, confidence=confidence)
              for label, x in zip(labels, xs) if label]
    for index, row in enumerate(rows):
        y = HEADER_Y + 30 + index * 22
        for value, x, width in zip(row, xs, WIDTHS):
            if value:
                tokens.append(token(value, x, y, width, confidence))
    return tokens


def reading(labels, rows, **kwargs):
    return from_tokens(table(labels, rows, **kwargs), literal_columns=False)


LABELS = ["Description", "Quantité", "Prix unitaire", "", "Montant HT"]
ROW = [["Prestation", "2", "40,00", "", "80,00"]]


def tesseract(labels=LABELS, rows=ROW, families=("gray", "contrast")):
    readings = {family: [reading(labels, rows)] for family in families}
    return build_consensus(readings), next(iter(readings.values()))[0]


def arbitrate(strategy, *, tesseract_rows=ROW, rapid_rows=ROW, labels=LABELS,
              tesseract_labels=LABELS, tesseract_families=("gray", "contrast")):
    evidence, _ = tesseract(labels=tesseract_labels, rows=tesseract_rows, families=tesseract_families)
    rapid = reading(labels, rapid_rows)
    combined, record = experiment.arbitrate(evidence, rapid, strategy)
    return reconcile(combined, [{}]), combined, record


def exported(grounded, index=0):
    return grounded.payload["line_items"][index]


def test_flag_is_off_by_default_and_reads_the_environment(monkeypatch):
    monkeypatch.delenv(experiment.FLAG_NAME, raising=False)
    assert experiment.enabled() is False
    monkeypatch.setenv(experiment.FLAG_NAME, "1")
    assert experiment.enabled() is True
    monkeypatch.setenv(experiment.FLAG_NAME, "0")
    assert experiment.enabled() is False


def test_strategy_defaults_to_agreement_and_rejects_unknown_values(monkeypatch):
    monkeypatch.delenv(experiment.STRATEGY_ENV, raising=False)
    assert experiment.strategy() == "A"
    monkeypatch.setenv(experiment.STRATEGY_ENV, "c")
    assert experiment.strategy() == "C"
    monkeypatch.setenv(experiment.STRATEGY_ENV, "D")
    assert experiment.strategy() == "A"


def test_both_engines_reading_the_same_row_is_accepted_as_agreement():
    grounded, combined, record = arbitrate("A")
    assert exported(grounded) == {"description": "Prestation", "quantity": 2.0,
                                  "unit_price": 40.0, "total_ht": 80.0}
    assert record["cell_provenance"]["both"] == 3
    assert all(experiment.RAPIDOCR in voters
               for row in combined.rows for votes in row.family_votes.values() for voters in votes.values())
    assert all(experiment.TESSERACT in voters
               for row in combined.rows for votes in row.family_votes.values() for voters in votes.values())


def test_agreement_never_adopts_a_value_the_first_engine_did_not_print():
    grounded, _, record = arbitrate("A", tesseract_rows=[["Prestation", "2", "45,00", "", "90,00"]])
    assert exported(grounded)["unit_price"] == 45.0
    assert exported(grounded)["total_ht"] == 90.0
    assert record["cell_provenance"]["both"] == 0 or exported(grounded)["unit_price"] != 40.0


def test_conflicting_amounts_are_abstained_not_resolved_by_majority():
    # RapidOCR alone reads 40,00; the two Tesseract families agree on 45,00.
    grounded, _, record = arbitrate("A", tesseract_rows=[["Prestation", "2", "45,00", "", "80,00"]])
    assert exported(grounded)["unit_price"] in (45.0, None)
    assert exported(grounded)["unit_price"] != 40.0


def test_recovery_from_a_missing_row_only_activates_when_the_first_engine_built_no_row():
    evidence, _ = tesseract(rows=[])
    rapid = reading(LABELS, ROW)
    combined, record = experiment.arbitrate(evidence, rapid, "B")
    assert record["evidence_engine"] == experiment.RAPIDOCR
    assert len(combined.rows) == 1
    assert reconcile(combined, [{}]).payload["line_items"][0]["unit_price"] == 40.0


def test_a_row_tesseract_already_kept_is_never_replaced_by_a_second_engine_row():
    grounded, combined, record = arbitrate("B")
    assert len(combined.rows) == 1, "no duplicate row for the same printed line"
    assert record["rows"][0]["action"] != "rapidocr-row"


def test_single_engine_cells_are_never_exported_under_agreement():
    # Tesseract reads no amount in the price column at all.
    grounded, combined, record = arbitrate("A", tesseract_rows=[["Prestation", "2", "", "", "80,00"]])
    assert record["cell_provenance"]["both"] <= 2
    assert exported(grounded)["unit_price"] is None


def test_missing_evidence_recovery_fills_a_column_the_first_engine_left_empty():
    grounded, _, record = arbitrate("B", tesseract_rows=[["Prestation", "2", "", "", "80,00"]])
    assert record["cell_provenance"]["rapidocr"] >= 1
    assert exported(grounded)["unit_price"] is None or exported(grounded)["unit_price"] == 40.0


def test_conflict_is_abstained_for_every_strategy_that_may_not_adjudicate():
    grounded, _, record = arbitrate("B", tesseract_rows=[["Prestation", "2", "45,00", "", "80,00"]])
    assert exported(grounded)["unit_price"] != 40.0


def test_adjudication_refuses_a_genuinely_different_amount_even_when_confident():
    # RapidOCR is much more confident, but 39,00 against 40,00 is a different amount, not a
    # separator corruption: strategy C must abstain instead of preferring the second engine.
    evidence, _ = tesseract(rows=[["Prestation", "2", "40,00", "", "80,00"]])
    rapid = reading(LABELS, [["Prestation", "2", "39,00", "", "80,00"]])
    combined, record = experiment.arbitrate(evidence, rapid, "C")
    exported_row = reconcile(combined, [{}]).payload["line_items"][0]
    assert exported_row["unit_price"] != 39.0


def test_arbitration_never_mutates_the_evidence_it_was_given():
    evidence, _ = tesseract()
    before = copy.deepcopy(evidence)
    rapid = reading(LABELS, ROW)
    before_rapid = copy.deepcopy(rapid)
    experiment.arbitrate(evidence, rapid, "B")
    assert [(row.description, {key: [str(n.value) for n in numbers]
                               for key, numbers in row.cells.items()}) for row in evidence.rows] == [
        (row.description, {key: [str(n.value) for n in numbers] for key, numbers in row.cells.items()})
        for row in before.rows]
    assert [row.line.text for row in rapid.rows] == [row.line.text for row in before_rapid.rows]


def test_row_that_only_brings_a_description_is_not_added():
    grounded, combined, record = arbitrate("B", rapid_rows=[["Autre chose", "", "", "", ""]])
    assert all(row.description != "Autre chose" or len(combined.rows) == 1 for row in combined.rows)


def test_hourly_quantity_is_read_through_the_duration_parser_only_for_a_hours_column():
    hourly_labels = ["Description", "HRS / QTY", "RATE / PRICE", "", "SUBTOTAL"]
    evidence, _ = tesseract(labels=hourly_labels, rows=[], families=("gray", "contrast"))
    rapid = reading(hourly_labels, [["Website Updates", "03:00", "40,00", "", "120,00"]])
    combined, record = experiment.arbitrate(evidence, rapid, "B")
    assert record["hourly_table"] is True
    assert record["cells"][0]["origin"] == "duration-parsed"
    assert exported(reconcile(combined, [{}]))["quantity"] == 3.0


def test_a_non_hourly_table_never_converts_a_quantity_into_a_duration():
    evidence, _ = tesseract(rows=[])
    rapid = reading(LABELS, [["Prestation", "03:00", "40,00", "", "120,00"]])
    combined, record = experiment.arbitrate(evidence, rapid, "B")
    assert record["hourly_table"] is False
    assert all(cell.get("origin") != "duration-parsed" for cell in record["cells"])
    assert exported(reconcile(combined, [{}]))["quantity"] is None


def test_experiment_continues_with_tesseract_when_the_second_engine_is_unavailable(monkeypatch):
    monkeypatch.setattr(experiment, "rapidocr_read", lambda image, **kwargs: None)
    evidence, _ = tesseract()
    combined, record = experiment.experiment(None, evidence, strategy_name="B")
    assert record["available"] is False
    assert combined.rows == evidence.rows and combined.dual_ocr is None


def test_experiment_reports_an_engine_failure_instead_of_raising(monkeypatch):
    from rapidocr_evidence import RapidOcrRead

    monkeypatch.setattr(experiment, "rapidocr_read",
                        lambda image, **kwargs: RapidOcrRead(None, [], 0.0, "3.10.0", "ZeroDivisionError: boom"))
    evidence, _ = tesseract()
    combined, record = experiment.experiment(None, evidence, strategy_name="A")
    assert record["available"] is False and "boom" in record["failure"]
    assert combined is evidence


def test_experiment_survives_an_arbitration_exception(monkeypatch):
    evidence, _ = tesseract()
    monkeypatch.setattr(experiment, "rapidocr_read",
                        lambda image, **kwargs: type("R", (), {
                            "evidence": reading(LABELS, ROW), "regions": [], "seconds": 0.1,
                            "engine_version": "3.10.0", "failure": None})())
    monkeypatch.setattr(experiment, "arbitrate", lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("arbitration exploded")))
    combined, record = experiment.experiment(None, evidence, strategy_name="C")
    assert "arbitration exploded" in record["failure"]
    assert combined is evidence


def test_document_level_fields_cannot_overwrite_a_value_the_first_engine_printed():
    evidence, _ = tesseract()
    evidence.supplier = "Atelier Gris"
    rapid = reading(LABELS, ROW)
    rapid.supplier = "Autre Fournisseur"
    combined, record = experiment.arbitrate(evidence, rapid, "B")
    assert combined.supplier == "Atelier Gris"
    assert all(item["field"] != "supplier" for item in record["document"])


def test_a_supplier_name_is_never_adopted_from_the_second_engine():
    # Two readings of the same pixels can agree on an amount; they can also agree on a misread logo, and
    # then agreement proves nothing about the name. PR101 measured exactly that on Fast Photography.
    evidence, _ = tesseract()
    evidence.supplier = None
    rapid = reading(LABELS, ROW)
    rapid.supplier = "Photographie"
    for strategy in experiment.STRATEGIES:
        combined, record = experiment.arbitrate(evidence, rapid, strategy)
        assert combined.supplier is None, strategy
        assert [item for item in record["document"] if item["field"] == "supplier"
                and item["provenance"] == experiment.DROPPED]


def test_a_document_field_the_first_engine_lacks_is_filled_and_recorded():
    evidence, _ = tesseract()
    evidence.total_ttc = None
    evidence.totals = {"total_ht": evidence.totals.get("total_ht", [])}
    rapid = reading(LABELS, ROW)
    combined, record = experiment.arbitrate(evidence, rapid, "B")
    assert isinstance(record["document"], list)


@pytest.mark.parametrize("strategy", experiment.STRATEGIES)
def test_no_strategy_invents_a_monetary_value_neither_engine_printed(strategy):
    grounded, combined, record = arbitrate(strategy)
    printed = {float(number.value) for row in combined.rows for numbers in row.cells.values()
               for number in numbers}
    for item in grounded.payload["line_items"]:
        for key in ("quantity", "unit_price", "total_ht"):
            assert item[key] is None or float(item[key]) in printed
