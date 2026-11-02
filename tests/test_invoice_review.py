import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from invoice_evidence import Token, from_text, from_tokens  # noqa: E402
import invoice_review as review  # noqa: E402

TEXT = (
    "Fournisseur : Batisolite\n"
    "Date de facture : 12/03/2026\n"
    "Désignation | Quantité | PU HT | Montant HT\n"
    "Pose isolation 7,2 12,50 90,00\n"
    "Total HT : 90,00\n"
    "Total TTC : 108,00\n"
)
STRICT = {
    "supplier_name": "Batisolite",
    "invoice_date": "2026-03-12",
    "total_ht": 90.0,
    "total_ttc": None,
    "line_items": [
        {"description": "Pose isolation", "quantity": 7.2, "unit_price": 12.5, "total_ht": 90.0}
    ],
}
RAW = {
    "supplier_name": "Batisolite SARL",
    "invoice_date": "2026-03-12",
    "total_ht": 90.0,
    "total_ttc": 108.0,
    "line_items": [
        {"description": "Pose isolation", "quantity": 7.2, "unit_price": 12.5, "total_ht": 90.0}
    ],
}
STATUSES = {
    "supplier_name": "récupéré (corroboré par OCR)",
    "invoice_date": "valide",
    "total_ht": "valide",
    "total_ttc": "non vérifié",
    "line_items.0.description": "valide",
    "line_items.0.quantity": "non vérifié",
    "line_items.0.unit_price": "valide",
    "line_items.0.total_ht": "valide",
}
PROVENANCE = {
    "supplier_name": [{"raw": "Fournisseur : Batisolite", "value": "Batisolite",
                       "page": 0, "box": [10, 20, 300, 40]}],
    "invoice_date": [{"raw": "12/03/2026", "value": "2026-03-12", "page": 0,
                      "box": [10, 50, 150, 66]}],
    "total_ht": [{"raw": "90,00", "value": "90.0", "page": 0, "box": [200, 120, 260, 134]}],
}


def assessed(payload=None, raw=None, statuses=None, *, diagnostics=None):
    return review.assess_for_ui(
        from_text(TEXT), raw or RAW, payload or STRICT, statuses or STATUSES,
        diagnostics=diagnostics or {"provenance": PROVENANCE},
    )


def test_flag_is_off_by_default_and_only_truthy_values_enable_it():
    assert review.review_ui_enabled({}) is False
    assert review.review_ui_enabled({"INVOICE_REVIEW_UI": ""}) is False
    assert review.review_ui_enabled({"INVOICE_REVIEW_UI": "0"}) is False
    assert review.review_ui_enabled({"INVOICE_REVIEW_UI": "off"}) is False
    for value in ("1", "true", "TRUE", "yes", "on"):
        assert review.review_ui_enabled({"INVOICE_REVIEW_UI": value}) is True


def test_assess_for_ui_uses_the_frozen_p1_contract_without_touching_the_strict_payload():
    payload = json.loads(json.dumps(STRICT))
    result = assessed(payload=payload)
    assert payload == STRICT
    assert set(result) - set(STRICT) == {
        "mode", "schema_version", "field_assessments", "review_summary",
    }
    assert result["mode"] == "review_assisted"
    for key in STRICT:
        assert result[key] == STRICT[key]


def test_proposals_never_reach_the_primary_fields_and_stay_in_the_assessment_map():
    result = assessed()
    assert result["total_ttc"] is None
    assert result["line_items"][0]["quantity"] == 7.2
    assert result["field_assessments"]["total_ttc"]["status"] == "proposed"
    assert result["field_assessments"]["total_ttc"]["value"] == 108.0
    assert result["field_assessments"]["line_items[0].quantity"]["status"] == "proposed"
    rows = {row[0]: row for row in review.review_rows(result, provenance=PROVENANCE)}
    ttc = rows["Total TTC"]
    assert ttc[1] == "—", "the retained strict value stays empty for a proposal"
    assert ttc[2] == "108.0", "the model reading is displayed in its own column"
    assert ttc[3] == "proposition du modèle (non vérifiée)"


def test_verified_and_recovered_keep_distinct_labels():
    result = assessed()
    labels = {path: review.status_label(item)
              for path, item in result["field_assessments"].items()}
    assert labels["invoice_date"] == "vérifié"
    assert labels["supplier_name"] == "récupéré (document) — divergence avec le modèle"
    assert result["review_summary"]["verification_conflicts"] == 1
    counts = review.status_counts(result)
    assert counts["récupéré (document)"] == 1
    assert counts["vérifié"] == 5


def test_p1_reason_codes_are_displayed_with_their_french_explanation():
    result = assessed()
    rows = {row[0]: row for row in review.review_rows(result, provenance=PROVENANCE)}
    assert rows["Total TTC"][5].startswith("labelled_ttc —")
    assert rows["Ligne 1 — Quantité"][5].startswith("ocr_row_line —")
    assert "attribution : row_local_ocr_line" in rows["Ligne 1 — Quantité"][5]


def test_unlocatable_evidence_is_reported_as_a_limitation():
    result = assessed()
    without_provenance = review.review_rows(result)
    assert review.NO_EVIDENCE in {row[4] for row in without_provenance}
    with_provenance = {row[0]: row for row in review.review_rows(result, provenance=PROVENANCE)}
    assert with_provenance["Total HT"][4].startswith("zone ancrée par le recalage · page 1")
    assert "page ou zone indisponible" in with_provenance["Total TTC"][4]
    limitations = review.review_limitations(result, provenance=PROVENANCE)
    assert limitations and "sans page ni zone localisable" in limitations[0]
    assert "Aucune preuve documentaire disponible" not in " ".join(limitations)


def test_missing_evidence_is_displayed_as_absent_not_invented():
    result = assessed()
    limitations = review.review_limitations(result, evidence_available=False)
    assert any("Aucune preuve documentaire disponible" in item for item in limitations)
    summary = review.review_summary_markdown(result, evidence_available=False)
    assert "Aucune preuve documentaire disponible" in summary
    degraded = review.unavailable_review("Contrôle P1 indisponible")
    assert review.review_rows(degraded) == []
    assert "Contrôle P1 indisponible" in review.review_summary_markdown(degraded)


def test_multi_page_evidence_reports_the_page_of_each_proposal():
    words = ["Pose", "isolation", "7,2", "12,50", "90,00"]
    second = ["Autre", "prestation", "3,0", "20,00", "60,00"]
    tokens = [Token(word, page, (10 + 60 * index, 30, 60 + 60 * index, 44))
              for page, line in enumerate((words, second))
              for index, word in enumerate(line)]
    evidence = from_tokens(tokens)
    raw = {
        "supplier_name": None, "invoice_date": None, "total_ht": 150.0, "total_ttc": None,
        "line_items": [
            {"description": "Pose isolation", "quantity": 7.2, "unit_price": 12.5, "total_ht": 90.0},
            {"description": "Autre prestation", "quantity": 3.0, "unit_price": 20.0, "total_ht": 60.0},
        ],
    }
    strict = {"supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
              "line_items": [{}, {}]}
    result = review.assess_for_ui(evidence, raw, strict, {}, diagnostics={})
    assert {line.page for line in evidence.lines} == {0, 1}
    rows = {row[0]: row for row in review.review_rows(result)}
    assert "page 1" in rows["Ligne 1 — Quantité"][4]
    assert "page 2" in rows["Ligne 2 — Quantité"][4]
    assert result["field_assessments"]["line_items[1].quantity"]["evidence"]["page"] == 1
    assert result["field_assessments"]["line_items[1].description"]["status"] == "proposed"


def test_repeated_rows_are_assessed_and_labelled_separately():
    payload = json.loads(json.dumps(STRICT))
    payload["line_items"].append(dict(payload["line_items"][0]))
    raw = json.loads(json.dumps(RAW))
    raw["line_items"].append(dict(raw["line_items"][0]))
    statuses = dict(STATUSES, **{
        "total_ht": "valide",
        "line_items.1.description": "valide",
        "line_items.1.quantity": "non vérifié",
        "line_items.1.unit_price": "valide",
        "line_items.1.total_ht": "valide",
    })
    result = assessed(payload=payload, raw=raw, statuses=statuses)
    rows = [row for row in review.review_rows(result) if row[0].startswith("Ligne")]
    assert [row[0] for row in rows][:4] == [
        "Ligne 1 — Description", "Ligne 1 — Quantité", "Ligne 1 — Prix unitaire",
        "Ligne 1 — Total HT",
    ]
    assert {row[0].split(" — ")[0] for row in rows} == {"Ligne 1", "Ligne 2"}
    assert all(row[3] == "proposition du modèle (non vérifiée)" for row in rows
               if row[0].endswith("Quantité"))
    assert result["field_assessments"]["line_items[1].quantity"]["row"]["index"] == 1


def test_summary_states_the_read_only_scope_and_the_strict_disclaimer():
    result = assessed()
    summary = review.review_summary_markdown(result, document_name="facture.png",
                                            provenance=PROVENANCE)
    assert summary.startswith("### Révision — lecture seule")
    assert "facture.png" in summary
    assert review.PROPOSAL_DISCLAIMER in summary
    assert "Corriger, accepter ou exporter n'est pas disponible" in summary
    assert "vérifié 5 · récupéré (document) 1" in summary


def test_review_json_exposes_the_display_block_and_stays_serialisable():
    result = assessed()
    displayed = review.review_json(result, provenance=PROVENANCE)
    assert displayed["display"]["columns"] == list(review.COLUMN_HEADERS)
    assert set(displayed["display"]["status_counts"]) >= {"vérifié", "récupéré (document)"}
    assert json.dumps(displayed, ensure_ascii=False)
    for key in STRICT:
        assert displayed[key] == STRICT[key]


def test_paths_are_read_in_both_bracket_and_dot_notations():
    assert review.field_label("line_items[1].unit_price") == "Ligne 2 — Prix unitaire"
    assert review.field_label("line_items.1.unit_price") == "Ligne 2 — Prix unitaire"
    assert review._canonical_path("line_items[1].unit_price") == "line_items.1.unit_price"
    assert review._payload_value(STRICT, "line_items[0].description") == "Pose isolation"
    assert review._payload_value(STRICT, "line_items.4.quantity") is None
