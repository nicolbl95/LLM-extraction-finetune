"""Regional agreement across distinct preprocessing families, not model votes."""

import re
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from itertools import product

from invoice_evidence import DocumentEvidence, folded
from ocr_config import OCR_CONFIG


def _matching_row(anchor, evidence):
    scored = []
    for row in evidence.rows:
        score = SequenceMatcher(None, folded(anchor.description), folded(row.description)).ratio()
        shared = sum(
            bool({n.value for n in anchor.cells.get(key, [])} & {n.value for n in row.cells.get(key, [])})
            for key in ("quantity", "unit_price", "total_ht")
        )
        if score >= .9 or shared >= 2:
            scored.append((score + shared, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
        return scored[0][1]
    return None


def _family_votes(groups, key, value):
    votes = defaultdict(set)
    originals = {}
    for family, entries in groups.items():
        observations = []
        for entry in entries:
            family_values = set()
            for candidate in key(entry):
                token = value(candidate)
                family_values.add(token)
                originals.setdefault(token, candidate)
            observations.append(family_values)
        # Disagreement between segmentations of one family is not an extra vote.
        if observations and all(values == observations[0] for values in observations):
            for token in observations[0]:
                votes[token].add(family)
    return votes, originals


def _agreed(groups, key, value, *, unanimous=False):
    votes, originals = _family_votes(groups, key, value)
    candidates = [token for token, voters in votes.items() if len(voters) >= 2]
    if unanimous and len(votes) > 1:
        return []
    return [originals[token] for token in candidates]


def number_confident(row, number, threshold):
    left, top, right, bottom = number.box
    tokens = [
        token for token in row.line.tokens
        if left <= token.box[0] < right and top <= token.box[1] < bottom
    ]
    return bool(tokens) and all(token.confidence >= threshold for token in tokens)


def _siret_values(evidence: DocumentEvidence) -> list[str]:
    values = []
    for match in re.finditer(r"\bSIRET\s*:?\s*([0-9][0-9 ]{12,18}[0-9])\b", evidence.text, re.I):
        value = match.group(1).replace(" ", "")
        if len(value) == 14:
            values.append(value)
    return values


def build_consensus(readings: dict[str, list[DocumentEvidence]], *, confidence=OCR_CONFIG.confidence) -> DocumentEvidence:
    all_reads = [evidence for group in readings.values() for evidence in group]
    anchor = max(all_reads, key=lambda evidence: (
        sum(sum(bool(row.cells.get(key)) for key in ("quantity", "unit_price", "total_ht")) for row in evidence.rows),
        sum(token.confidence for token in evidence.tokens) / max(1, len(evidence.tokens)),
    ))
    result = deepcopy(anchor)
    result.source_kind = "consensus"
    result.ocr_cross_checked = True
    result.quality_issues = []
    result.issues = []
    result.corroborated_identifiers = [
        ("siret", value)
        for value in _agreed(readings, _siret_values, lambda candidate: candidate, unanimous=True)
    ]
    for key in ("supplier", "invoice_number", "invoice_date"):
        def metadata(evidence):
            candidate = getattr(evidence, key)
            if candidate is None:
                return []
            if key == "invoice_number":
                return [candidate]
            if key == "supplier":
                expected_name = folded(evidence.supplier_raw or candidate)
                if evidence.supplier_is_explicit:
                    relevant = [
                        line for line in evidence.lines
                        if "supplier" in folded(line.text) or "fournisseur" in folded(line.text)
                    ]
                else:
                    relevant = [
                        line for line in evidence.lines
                        if folded(line.text) == expected_name
                    ]
            else:
                relevant = [
                    line for line in evidence.lines
                    if (
                        "invoice date" in folded(line.text)
                        or "date de facture" in folded(line.text)
                        or (
                            re.search(r"\bdate\b", folded(line.text))
                            and not re.search(
                                r"\b(?:due\s+date|date\s+d[' ]?echeance|date\s+de\s+(?:livraison|paiement)|echeance)\b",
                                folded(line.text),
                            )
                        )
                    )
                ]
            if not relevant or any(token.confidence < confidence for line in relevant for token in line.tokens):
                return []
            return [candidate]
        candidates = _agreed(readings, metadata, folded, unanimous=True)
        setattr(result, key, candidates[0] if len(candidates) == 1 else None)
    if result.supplier:
        explicit_votes = _agreed(
            readings,
            lambda evidence: [evidence.supplier_is_explicit]
            if evidence.supplier and folded(evidence.supplier) == folded(result.supplier)
            else [],
            lambda value: value,
            unanimous=True,
        )
        result.supplier_is_explicit = explicit_votes == [True]
    if result.supplier and folded(result.supplier).endswith((" lid", " l1d", " lti")):
        result.supplier = None
    result.totals = {}
    for key in ("total_ht", "total_ttc", "total_candidate"):
        result.totals[key] = _agreed(
            readings, lambda evidence: [
                n for n in evidence.totals.get(key, [])
                if any(line.box == n.box and all(t.confidence >= confidence for t in line.tokens) for line in evidence.lines)
            ], lambda n: n.value,
        )
    result.taxes = _agreed(
        readings, lambda evidence: [
            tax for tax in evidence.taxes
            if any(line.box == tax.amount.box and all(t.confidence >= confidence for t in line.tokens) for line in evidence.lines)
        ], lambda tax: (tax.rate.value if tax.rate else None, tax.amount.value),
    )
    # Tax and identity contradictions from image OCR are never asserted as
    # invoice errors without native/manual confirmation.
    result.discounts = _agreed(readings, lambda evidence: evidence.discounts, lambda n: n.value)
    amount_mismatches = []
    for row in result.rows:
        matches = {
            family: [candidate for evidence in entries if (candidate := _matching_row(row, evidence)) is not None]
            for family, entries in readings.items()
        }
        descriptions = _agreed(
            matches, lambda candidate: [candidate.description] if all(
                token.confidence >= confidence for token in candidate.line.tokens
            ) else [], folded, unanimous=True,
        )
        row.description = descriptions[0] if len(descriptions) == 1 else ""
        row.description_confident = bool(row.description)
        for key in ("quantity", "unit_price", "total_ht"):
            candidates_for_family = lambda candidate: [
                    number for number in candidate.cells.get(key, [])
                    if number_confident(candidate, number, confidence)
                ]
            votes, _ = _family_votes(matches, candidates_for_family, lambda number: number.value)
            row.family_votes[key] = {
                value: tuple(sorted(families))
                for value, families in votes.items()
                if len(families) >= 2
            }
            row.cells[key] = _agreed(matches, candidates_for_family, lambda number: number.value)
        candidates = [
            {number.value for number in row.cells[key]} for key in ("quantity", "unit_price", "total_ht")
        ]
        if all(candidates):
            solutions = [
                values for values in product(*candidates)
                if abs((values[0] * values[1]).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) - values[2]) <= Decimal(".02")
            ]
            if not solutions:
                corroborated_quantities = set(row.family_votes["quantity"])
                corroborated_prices = set(row.family_votes["unit_price"])
                corroborated_totals = set(row.family_votes["total_ht"])
                calculated_totals = {
                    (quantity * price).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)
                    for quantity, price in product(corroborated_quantities, corroborated_prices)
                }
                if (
                    len(corroborated_totals) == 1
                    and len(calculated_totals) == 1
                    and all(
                        abs(total - calculated) > Decimal(".02")
                        for total, calculated in product(corroborated_totals, calculated_totals)
                    )
                ):
                    amount_mismatches.append(
                        (
                            row,
                            next(iter(corroborated_totals)),
                            next(iter(calculated_totals)),
                        )
                    )
                row.cells["quantity"] = []
                row.cells["unit_price"] = []
                result.issues.append(
                    f"Ligne « {row.description} » : lectures contradictoires avec le produit ; "
                    "quantité/prix non recoupés, pas d'anomalie de facture affirmée."
                )
    if not result.rows:
        confidence_values = [token.confidence for token in result.tokens]
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        if mean_confidence < OCR_CONFIG.fast_mean_confidence:
            result.quality_issues.append(
                f"Confiance OCR moyenne insuffisante : {mean_confidence:.1f}/100 "
                f"(minimum {OCR_CONFIG.fast_mean_confidence})."
            )
        else:
            result.quality_issues.append("Tableau non reconstruit malgré les prétraitements.")
    result.numbers = [
        n for row in result.rows for values in row.cells.values() for n in values
    ] + [n for values in result.totals.values() for n in values]
    line_amounts = []
    for row in result.rows:
        totals = row.family_votes.get("total_ht", {})
        if len(totals) != 1:
            line_amounts = []
            break
        line_amounts.append(next(iter(totals)))
    invoice_totals = {number.value for number in result.totals.get("total_ht", [])}
    if line_amounts and len(line_amounts) == len(result.rows) and len(invoice_totals) == 1:
        printed_sum = sum(line_amounts, Decimal(0)) - sum(
            (abs(number.value) for number in result.discounts), Decimal(0),
        )
        subtotal_delta = printed_sum - next(iter(invoice_totals))
        for row, printed_amount, calculated_amount in amount_mismatches:
            if abs(subtotal_delta - (printed_amount - calculated_amount)) <= Decimal(".02"):
                row.corroborated_amount_mismatch = (printed_amount, calculated_amount)
    return result
