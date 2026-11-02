"""Fail closed when OCR does not provide a consistent, recouped document."""

import re

from invoice_evidence import (
    DocumentEvidence, folded, has_replacement_character, reading_issue_category,
)
from invoice_validation import document_validation
from ocr_config import OCR_CONFIG

TABLE_HEADER_CUES = re.compile(
    r"\b(?:description|quantity|qty|unit\s+price|amount|total)\b"
)


def quality_hint(
    evidence: DocumentEvidence, source_size: tuple[int, int] | None = None,
) -> str | None:
    """Advisory hint only; never changes fields or statuses."""
    if evidence.rows and not evidence.quality_issues:
        return None
    small_source = bool(
        source_size and source_size[0] * source_size[1] < 400_000
    )
    table_expected = any(evidence.totals.values()) or bool(evidence.taxes) or (
        bool(evidence.numbers)
        and (
            any(TABLE_HEADER_CUES.search(folded(line.text)) for line in evidence.lines)
            or "Tableau non reconstruit malgré les prétraitements." in evidence.quality_issues
        )
    ) or (small_source and bool(evidence.quality_issues))
    if not table_expected:
        return None
    category = reading_issue_category(evidence)
    if category == "timeout":
        return None
    confidences = [token.confidence for token in evidence.tokens]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0
    if small_source or category == "blur" or mean_confidence < OCR_CONFIG.fast_mean_confidence:
        return "low_quality"
    if category == "layout":
        return "layout"
    return None


def signature(evidence: DocumentEvidence) -> tuple:
    return (
        evidence.supplier, evidence.invoice_number, evidence.invoice_date,
        tuple((folded(row.description), tuple(
            (key, tuple(sorted({number.value for number in row.cells.get(key, [])})))
            for key in ("quantity", "unit_price", "total_ht")
        )) for row in evidence.rows),
        tuple((key, tuple(sorted({number.value for number in evidence.totals.get(key, [])})))
              for key in ("total_ht", "total_ttc")),
        tuple(sorted((str(tax.rate.value) if tax.rate else "", str(tax.amount.value)) for tax in evidence.taxes)),
        tuple(number.value for number in evidence.discounts),
    )


def assess_ocr(primary: DocumentEvidence, secondary: DocumentEvidence) -> DocumentEvidence:
    primary.source_kind = "ocr"
    primary.ocr_cross_checked = True
    issues = primary.quality_issues
    confidences = [token.confidence for token in primary.tokens]
    mean = sum(confidences) / len(confidences) if confidences else 0
    if mean < OCR_CONFIG.fast_mean_confidence:
        issues.append(f"Confiance OCR moyenne insuffisante : {mean:.1f}/100 (minimum {OCR_CONFIG.fast_mean_confidence}).")
    secondary_mean = sum(token.confidence for token in secondary.tokens) / max(1, len(secondary.tokens))
    if secondary_mean < OCR_CONFIG.fast_mean_confidence:
        issues.append("La lecture de confirmation a une confiance moyenne insuffisante.")
    if any(
        has_replacement_character(token.text)
        for evidence in (primary, secondary)
        for token in evidence.tokens
    ):
        issues.append("Caractère de remplacement OCR (U+FFFD) détecté dans le texte.")
    if not primary.rows:
        issues.append("Tableau OCR non reconnu.")
    if len(primary.rows) != len(secondary.rows):
        issues.append("Nombre de lignes incohérent entre les deux lectures OCR.")
    if signature(primary) != signature(secondary):
        issues.append("Les lectures OCR indépendantes (segmentations 4 et 6) ne concordent pas.")
    if any(not row.cells.get(key) for row in primary.rows for key in ("quantity", "unit_price", "total_ht")):
        issues.append("Cellules numériques manquantes : structure de ligne non recoupée.")
    if primary.supplier and folded(primary.supplier).endswith((" lid", " l1d", " lti")):
        issues.append("Nom de fournisseur OCR ambigu.")
    if any(token.confidence < OCR_CONFIG.fast_token_confidence
           for evidence in (primary, secondary) for row in evidence.rows for token in row.line.tokens):
        issues.append("Une ligne du tableau contient des tokens à faible confiance.")
    if primary.rows and any(
        len({n.value for n in row.cells.get(key, [])}) != 1
        for row in primary.rows for key in ("quantity", "unit_price", "total_ht")
    ):
        issues.append("Plusieurs interprétations des cellules OCR.")
    # A missing decimal can look exactly like a printed arithmetic error. Without
    # native text/manual confirmation, neither OCR nor the LLM can disambiguate it.
    possible_errors = [
        flag for flag in document_validation(primary).flags if flag.severity != "information"
    ]
    if possible_errors:
        issues.append(
            "Lectures OCR contradictoires avec les contrôles documentaires ; "
            "erreur imprimée ou erreur OCR indiscernables : "
            + ", ".join(sorted({flag.code for flag in possible_errors}))
        )
    if issues:
        primary.issues.extend(issue for issue in issues if issue not in primary.issues)
    return primary


def document_quality(evidence: DocumentEvidence, model: dict | None = None) -> list[str]:
    issues = list(evidence.quality_issues)
    if evidence.source_kind in {"ocr", "mixed"} and not evidence.ocr_cross_checked:
        issues.append("Preuves OCR non recoupées par une seconde lecture indépendante.")
    if model and evidence.source_kind in {"ocr", "mixed"} and evidence.rows_provenance != "recovered":
        items = model.get("line_items")
        if isinstance(items, list) and items and len(items) != len(evidence.rows):
            issues.append("Nombre d'articles modèle/OCR incohérent ; aucune correspondance supposée.")
    return issues
