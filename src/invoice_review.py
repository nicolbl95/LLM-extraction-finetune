"""Read-only review display over the frozen P1 assessment contract.

Importing this module never loads the research-only P1 assessor; that happens inside
:func:`assess_for_ui` so that a deployment without ``experiments/`` keeps working while
``INVOICE_REVIEW_UI`` stays off.
"""

from __future__ import annotations

import json
import os
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

REVIEW_UI_ENV = "INVOICE_REVIEW_UI"
REVIEW_FLAG_TRUE_VALUES = {"1", "true", "yes", "on"}
REVIEW_PAGE_MAX_SIDE = 1600
COLUMN_HEADERS = ("Champ", "Valeur retenue", "Lecture du modèle", "Statut",
                  "Preuve documentaire", "Motif")
FIELD_LABELS = {
    "supplier_name": "Fournisseur",
    "invoice_date": "Date de facture",
    "total_ht": "Total HT",
    "total_ttc": "Total TTC",
    "description": "Description",
    "quantity": "Quantité",
    "unit_price": "Prix unitaire",
}
LINE_ITEM_LABEL = "Ligne"
STATUS_LABELS = {
    "verified": "vérifié",
    "proposed": "proposition du modèle (non vérifiée)",
    "rejected": "rejeté",
    "missing_printed": "imprimé mais non extrait",
    "absent": "absent du document",
}
STRICT_STATUS_LABELS = {
    "valide": "vérifié",
    "récupéré (corroboré par ocr)": "récupéré (document)",
    "neutralisé (non fiable)": "incertain",
    "non vérifié": "non vérifié",
}
EVIDENCE_KIND_LABELS = {
    "ocr_line": "ligne OCR",
    "ocr_supplier": "bloc fournisseur",
    "ocr_date": "date OCR",
    "ocr_total": "total OCR",
    "grounding": "zone ancrée par le recalage",
}
REASON_LABELS = {
    "final_status": "statut de la vérification stricte",
    "ocr_complete_description": "description reprise d'une ligne OCR complète",
    "ocr_row_line": "nombre présent sur la ligne OCR de la ligne de facture",
    "ocr_supplier_text": "nom présent dans le bloc fournisseur",
    "ocr_supplier_block": "nom présent dans le bloc fournisseur",
    "ocr_supplier_header": "nom présent dans l'en-tête",
    "ocr_date": "date présente dans le document",
    "labelled_ht": "montant étiqueté HT",
    "labelled_ttc": "montant étiqueté TTC",
    "unqualified_total": "total sans libellé HT/TTC",
    "missing_label": "montant présent sans libellé de total",
    "no_model_value": "aucune valeur lue par le modèle",
    "no_evidence": "aucune preuve documentaire",
    "not_on_row_line": "nombre absent de la ligne OCR de la ligne de facture",
    "global_total_as_line": "total de facture utilisé comme montant de ligne",
    "quantity_equals_unit_price": "quantité identique au prix unitaire",
    "row_attribution_conflict": "ligne OCR attribuée à une autre ligne",
    "row_line_not_found": "ligne OCR de la ligne de facture introuvable",
    "incomplete_description": "description incomplète par rapport à la ligne OCR",
    "contradicted_supplier": "fournisseur contredit par le document",
    "supplier_not_found": "fournisseur absent du document",
    "contradicted_date": "date contredite par le document",
    "date_not_found": "date absente du document",
    "ttc_as_ht": "montant TTC lu comme HT",
    "ht_as_ttc": "montant HT lu comme TTC",
    "not_in_document": "montant absent du document",
    "unsupported_field": "champ non couvert par le contrôle",
    "unsupported_total": "total non exploitable",
}
NO_EVIDENCE = "aucune preuve renvoyée par le contrôle P1"
NO_LOCATION = "page ou zone indisponible"
PROPOSAL_DISCLAIMER = (
    "Aucune proposition n'est retenue dans le JSON strict : "
    "seuls les statuts « vérifié » et « récupéré (document) » sont exportés."
)
_ROW_PATH_RE = re.compile(r"^line_items[\[.](\d+)[\].]+([a-z_\[\]]+)$")
TRUNCATED_AT = 90


def review_ui_enabled(environ: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environ is None else environ
    return str(values.get(REVIEW_UI_ENV, "")).strip().casefold() in REVIEW_FLAG_TRUE_VALUES


def _p1_modules():
    root = Path(__file__).resolve().parents[1]
    for path in (str(root), str(root / "scripts")):
        if path not in sys.path:
            sys.path.append(path)
    from experiments.review_assessment import assess, serialize_review
    from scripts.evidence_capture import serialize_evidence
    return assess, serialize_review, serialize_evidence


def assess_for_ui(evidence: Any, raw_payload: Mapping[str, Any] | None,
                  strict_payload: Mapping[str, Any] | None,
                  statuses: Mapping[str, Any] | None, *,
                  diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify every field with the frozen P1 assessor; no OCR call and no inference."""
    assess, serialize_review, serialize_evidence = _p1_modules()
    captured = serialize_evidence(evidence, diagnostics=diagnostics or {})
    assessments = assess(captured, raw_payload or {}, {
        "payload": strict_payload or {}, "statuses": statuses or {},
    })
    return serialize_review(strict_payload or {}, assessments)


def unavailable_review(reason: str) -> dict[str, Any]:
    """A review result that displays the limitation instead of inventing assessments."""
    return {
        "supplier_name": None, "invoice_date": None, "total_ht": None, "total_ttc": None,
        "line_items": [], "mode": "review_assisted", "field_assessments": {},
        "review_summary": {"verified": 0, "proposed": 0, "rejected": 0, "missing_printed": 0,
                           "absent": 0, "review_required": 0, "verification_conflicts": 0},
        "limitations": [reason],
    }


def _path_parts(path: str) -> tuple[int | None, str]:
    match = _ROW_PATH_RE.match(path)
    if not match:
        return None, path
    return int(match.group(1)), match.group(2).strip("[]")


def field_label(path: str) -> str:
    row_index, field = _path_parts(path)
    label = FIELD_LABELS.get(field, field)
    if row_index is None:
        return label
    return f"{LINE_ITEM_LABEL} {row_index + 1} — {label}"


def _payload_value(review: Mapping[str, Any], path: str) -> Any:
    row_index, field = _path_parts(path)
    if row_index is None:
        return review.get(field)
    items = review.get("line_items") or []
    if row_index >= len(items) or not isinstance(items[row_index], Mapping):
        return None
    return items[row_index].get(field)


def display_value(path: str, value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, Decimal):
        if path.endswith("quantity") and value == value.to_integral_value():
            return format(value, ".0f")
        return format(value, "f")
    if path.endswith("quantity") and isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value).is_integer():
            return str(int(value))
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _canonical_path(path: str) -> str:
    row_index, field = _path_parts(path)
    if row_index is None:
        return field
    return f"line_items.{row_index}.{field}"


def status_group(assessment: Mapping[str, Any]) -> str:
    status = str(assessment.get("status") or "")
    if status == "verified":
        strict_status = str((assessment.get("checks") or {}).get("strict_status") or "").casefold()
        return STRICT_STATUS_LABELS.get(strict_status, STATUS_LABELS["verified"])
    return STATUS_LABELS.get(status, status or "statut inconnu")


def status_label(assessment: Mapping[str, Any]) -> str:
    label = status_group(assessment)
    if assessment.get("verification_conflict"):
        return f"{label} — divergence avec le modèle"
    return label


def _quote(evidence: Mapping[str, Any]) -> str:
    text = evidence.get("text")
    if text is None and evidence.get("value") is not None:
        text = f"valeur {evidence.get('value')}"
    if text is None:
        return "contenu de preuve non renvoyé"
    text = " ".join(str(text).split())
    if len(text) > TRUNCATED_AT:
        return f"«{text[:TRUNCATED_AT]}…»"
    return f"«{text}»"


def _grounding_evidence(provenance: Mapping[str, Any] | None, path: str) -> dict[str, Any]:
    """Recovery anchor recorded by the strict pipeline; P1 sends no evidence for verified fields."""
    if not isinstance(provenance, Mapping):
        return {}
    entries = provenance.get(_canonical_path(path)) or provenance.get(path)
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], Mapping):
        return {}
    first = entries[0]
    evidence = {"kind": "grounding", "text": first.get("raw"), "value": first.get("value"),
                "page": first.get("page"), "box": first.get("box")}
    if len(entries) > 1:
        evidence["candidates"] = len(entries)
    return {key: value for key, value in evidence.items() if value is not None}


def evidence_of(assessment: Mapping[str, Any], path: str = "",
                provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    evidence = assessment.get("evidence")
    if isinstance(evidence, Mapping) and evidence:
        return dict(evidence)
    if str(assessment.get("status") or "") == "verified":
        return _grounding_evidence(provenance, path)
    return {}


def evidence_label(evidence: Mapping[str, Any]) -> str:
    if not evidence:
        return NO_EVIDENCE
    kind = evidence.get("kind")
    parts = [EVIDENCE_KIND_LABELS.get(str(kind), str(kind or "preuve"))]
    page, box = evidence.get("page"), evidence.get("box")
    if page is None or box is None:
        parts.append(NO_LOCATION)
    else:
        parts.append(f"page {int(page) + 1} · zone {json.dumps(list(box), ensure_ascii=False)}")
    confidence = evidence.get("confidence")
    if isinstance(confidence, (int, float)) and confidence > 0:
        parts.append(f"confiance {confidence:.0f} %")
    if evidence.get("label"):
        parts.append(f"libellé {evidence['label']}")
    if evidence.get("candidates"):
        parts.append(f"{evidence['candidates']} sources")
    parts.append(_quote(evidence))
    return " · ".join(str(part) for part in parts)


def has_local_evidence(assessment: Mapping[str, Any], path: str = "",
                       provenance: Mapping[str, Any] | None = None) -> bool:
    evidence = evidence_of(assessment, path, provenance)
    return evidence.get("page") is not None and evidence.get("box") is not None


def reason_label(assessment: Mapping[str, Any]) -> str:
    code = str(assessment.get("reason") or "")
    if not code:
        return "aucun motif renvoyé"
    description = REASON_LABELS.get(code)
    checks = assessment.get("checks") or {}
    details = []
    if checks.get("arithmetic_consistent") is False:
        details.append("quantité × prix ≠ montant")
    if checks.get("row_attribution"):
        details.append(f"attribution : {checks['row_attribution']}")
    if checks.get("label_conflict"):
        details.append("libellé HT/TTC ambigu")
    if checks.get("documentary_candidate") is False and assessment.get("status") == "absent":
        details.append("aucun candidat imprimé")
    text = f"{code} — {description}" if description else code
    if details:
        text = f"{text} ({'; '.join(details)})"
    return text[:200]


def review_rows(review: Mapping[str, Any],
                provenance: Mapping[str, Any] | None = None) -> list[list[str]]:
    rows = []
    for path, assessment in (review.get("field_assessments") or {}).items():
        if not isinstance(assessment, Mapping):
            continue
        status = str(assessment.get("status") or "")
        model_value = assessment.get("value") if status == "proposed" else assessment.get("model_value")
        rows.append([
            field_label(path),
            display_value(path, _payload_value(review, path)),
            display_value(path, model_value),
            status_label(assessment),
            evidence_label(evidence_of(assessment, path, provenance)),
            reason_label(assessment),
        ])
    return rows


def status_counts(review: Mapping[str, Any]) -> dict[str, int]:
    counts = {
        "vérifié": 0, "récupéré (document)": 0,
        "proposition du modèle (non vérifiée)": 0, "rejeté": 0,
        "imprimé mais non extrait": 0, "absent du document": 0,
    }
    for assessment in (review.get("field_assessments") or {}).values():
        group = status_group(assessment)
        counts[group] = counts.get(group, 0) + 1
    return counts


def review_limitations(review: Mapping[str, Any], *, evidence_available: bool = True,
                       provenance: Mapping[str, Any] | None = None,
                       extra: tuple[str, ...] = ()) -> list[str]:
    limitations = [str(item) for item in (review.get("limitations") or [])]
    limitations.extend(extra)
    assessments = review.get("field_assessments") or {}
    unlocatable = sum(1 for path, item in assessments.items()
                      if isinstance(item, Mapping)
                      and str(item.get("status") or "") in {"verified", "proposed"}
                      and not has_local_evidence(item, path, provenance))
    if unlocatable:
        limitations.append(
            f"{unlocatable} champ(s) vérifié(s) ou proposé(s) sans page ni zone localisable : "
            "la preuve ne peut pas être montrée à l'écran."
        )
    if not evidence_available:
        limitations.append(
            "Aucune preuve documentaire disponible : les statuts ci-dessus ne reposent "
            "que sur la lecture du modèle."
        )
    return limitations


def review_summary_markdown(review: Mapping[str, Any], *, evidence_available: bool = True,
                            document_name: str | None = None,
                            provenance: Mapping[str, Any] | None = None,
                            limitations: tuple[str, ...] = ()) -> str:
    counts = status_counts(review)
    assessed = len(review.get("field_assessments") or {})
    summary = review.get("review_summary") or {}
    lines = ["### Révision — lecture seule"]
    if document_name:
        lines.append(f"- Document : {document_name}")
    if assessed:
        lines.append("- " + " · ".join(f"{label} {count}" for label, count in counts.items() if count))
        visible = (counts["vérifié"] + counts["récupéré (document)"]
                   + counts["proposition du modèle (non vérifiée)"])
        lines.append(f"- Champs affichés : {visible} sur {assessed} évalués.")
    else:
        lines.append("- Aucun champ évalué par le contrôle P1.")
    conflicts = int(summary.get("verification_conflicts") or 0)
    if conflicts:
        lines.append(f"- Divergences entre le document et la lecture du modèle : {conflicts}.")
    lines.append(f"- {PROPOSAL_DISCLAIMER}")
    issues = review_limitations(review, evidence_available=evidence_available,
                                provenance=provenance, extra=limitations)
    if issues:
        lines.append("- Limitations de preuve :")
        lines.extend(f"  - {issue}" for issue in issues)
    else:
        lines.append("- Limitations de preuve : aucune ; chaque champ affiché a une zone localisable.")
    lines.append("- Phase 1 : affichage seul. Corriger, accepter ou exporter n'est pas disponible.")
    return "\n".join(lines)


def review_json(review: Mapping[str, Any], *, evidence_available: bool = True,
                provenance: Mapping[str, Any] | None = None,
                limitations: tuple[str, ...] = ()) -> dict[str, Any]:
    """The P1 review contract plus a display block that belongs to the UI only."""
    payload = json.loads(json.dumps(review, ensure_ascii=False, default=str))
    payload["display"] = {
        "columns": list(COLUMN_HEADERS),
        "status_counts": status_counts(payload),
        "limitations": review_limitations(
            payload, evidence_available=evidence_available, provenance=provenance,
            extra=limitations,
        ),
    }
    return payload
