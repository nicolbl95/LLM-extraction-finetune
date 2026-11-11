"""Regional agreement across distinct preprocessing families, not model votes."""

import re
from collections import defaultdict
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from itertools import product

from invoice_evidence import (
    DocumentEvidence, OPTIONAL_HEADERS, description_is_usable, document_date_readings,
    folded, has_replacement_character,
)
from invoice_validation import parse_date
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


def _accent_count(text: str) -> int:
    return sum(char.isalpha() and ord(char) > 127 for char in text)


def _family_votes(groups, key, value, prefer=None, skip_empty=False):
    votes = defaultdict(set)
    originals = {}
    for family, entries in groups.items():
        observations = []
        for entry in entries:
            family_values = set()
            for candidate in key(entry):
                token = value(candidate)
                family_values.add(token)
                if token not in originals or (
                    prefer is not None and prefer(candidate) > prefer(originals[token])
                ):
                    originals[token] = candidate
            if skip_empty and not family_values:
                continue
            observations.append(family_values)
        # Disagreement between segmentations of one family is not an extra vote.
        if observations and all(values == observations[0] for values in observations):
            for token in observations[0]:
                votes[token].add(family)
    return votes, originals


def _agreed(groups, key, value, *, unanimous=False, silence_votes=False, prefer=None):
    # A segmentation that read nothing at all about a field is silence, not a contradiction, so it may not
    # veto the sibling that did read it - but only when the vote token *is* the exported value. Free text
    # is folded for voting, so an accent- or border-contaminated spelling and a clean one share one token
    # while their surface forms differ; there a blank sibling keeps vetoing, as before.
    votes, originals = _family_votes(groups, key, value, prefer, skip_empty=silence_votes)
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


def _candidate_numbers(matches, field, confidence):
    candidates = {}
    families_by_value = defaultdict(set)
    for family, entries in matches.items():
        for evidence in entries:
            for number in evidence.cells.get(field, []):
                if number_confident(evidence, number, confidence):
                    candidates.setdefault(number.value, number)
                    families_by_value[number.value].add(family)
    return candidates, families_by_value


def _recover_quantity(row, matches, confidence):
    # Recovery must remain independently corroborated: a single OCR family is
    # insufficient even when arithmetic happens to agree with price and total.
    prices = {number.value for number in row.cells.get("unit_price", [])}
    totals = {number.value for number in row.cells.get("total_ht", [])}
    if len(prices) != 1 or len(totals) != 1:
        return None
    if len(row.family_votes.get("unit_price", {})) != 1 or len(row.family_votes.get("total_ht", {})) != 1:
        return None
    price, total = next(iter(prices)), next(iter(totals))
    candidates, families_by_value = _candidate_numbers(matches, "quantity", confidence)
    solving = [
        number for value, number in candidates.items()
        if len(families_by_value[value]) >= 2
        and abs((value * price).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) - total) <= Decimal(".02")
    ]
    return solving[0] if len(solving) == 1 else None


def _siret_values(evidence: DocumentEvidence) -> list[str]:
    values = []
    for match in re.finditer(r"\bSIRET\s*:?\s*([0-9][0-9 ]{12,18}[0-9])\b", evidence.text, re.I):
        value = match.group(1).replace(" ", "")
        if len(value) == 14:
            values.append(value)
    return values


def _confident_invoice_number(evidence: DocumentEvidence, candidate: str, confidence: int) -> bool:
    identifier = re.compile(rf"(?<!\w){re.escape(candidate)}(?!\w)", re.I)
    return any(
        token.confidence >= confidence and identifier.search(token.text)
        for line in evidence.lines
        if identifier.search(line.text)
        for token in line.tokens
    )


def _agreed_invoice_number(
    readings: dict[str, list[DocumentEvidence]], confidence: int,
) -> str | None:
    candidates = set()
    families_by_candidate = defaultdict(set)
    for family, entries in readings.items():
        for evidence in entries:
            candidate = evidence.invoice_number
            if candidate is None:
                continue
            candidates.add(candidate)
            if _confident_invoice_number(evidence, candidate, confidence):
                families_by_candidate[candidate].add(family)
    if len(candidates) != 1:
        return None
    candidate = next(iter(candidates))
    return candidate if len(families_by_candidate[candidate]) >= 2 else None


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
    result.invoice_number = _agreed_invoice_number(readings, confidence)
    for key in ("supplier", "invoice_date"):
        def metadata(evidence):
            candidate = getattr(evidence, key)
            if candidate is None:
                return []
            if key == "supplier":
                expected_name = folded(evidence.supplier_raw or candidate)
                if evidence.supplier_source is not None:
                    relevant = [evidence.supplier_source]
                elif evidence.supplier_is_explicit:
                    relevant = [
                        line for line in evidence.lines
                        if "supplier" in folded(line.text) or "fournisseur" in folded(line.text)
                    ]
                    if not relevant:
                        relevant = [line for line in evidence.lines if expected_name in folded(line.text)]
                else:
                    relevant = [
                        line for line in evidence.lines
                        if expected_name in folded(line.text)
                    ]
            else:
                for kind, raw, line in document_date_readings(evidence.lines):
                    date_value, _ = parse_date(raw, evidence.date_locale)
                    if kind != "invoice" or date_value is None:
                        continue
                    for token in line.tokens:
                        parsed, _ = parse_date(token.text, evidence.date_locale)
                        if (
                            parsed is not None
                            and parsed.isoformat() == candidate
                            and token.confidence >= confidence
                        ):
                            return [candidate]
                    if (
                        date_value.isoformat() == candidate and line.tokens
                        and all(token.confidence >= confidence for token in line.tokens)
                    ):
                        return [candidate]
                return []
            if not relevant or any(token.confidence < confidence for line in relevant for token in line.tokens):
                return []
            return [candidate]
        candidates = _agreed(readings, metadata, folded, unanimous=True,
                             silence_votes=key == "invoice_date")
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
            matches, lambda candidate: [candidate.description] if (
                all(token.confidence >= confidence for token in candidate.line.tokens)
                and not has_replacement_character(candidate.description)
            ) else [], folded, unanimous=True, prefer=_accent_count,
        )
        row.description = descriptions[0] if len(descriptions) == 1 else ""
        row.description_confident = bool(row.description)
        continuation_matches = {}
        for family, entries in matches.items():
            available = [entry for entry in entries if entry.secondary_description]
            if not available:
                continue
            longest = max(available, key=lambda entry: len(folded(entry.secondary_description)))
            if all(
                folded(longest.secondary_description) == folded(entry.secondary_description)
                or folded(longest.secondary_description).startswith(
                    folded(entry.secondary_description) + " "
                ) for entry in available
            ):
                continuation_matches[family] = [longest]
        secondary = _agreed(
            continuation_matches,
            lambda candidate: [candidate.secondary_description] if (
                candidate.secondary_description
                and description_is_usable(candidate.secondary_description)
                and not has_replacement_character(candidate.secondary_description)
            ) else [],
            folded, prefer=_accent_count,
        )
        if secondary:
            observed = [
                candidate.secondary_description for entries in matches.values() for candidate in entries
                if candidate.secondary_description
            ]
            # Missing trailing OCR words do not contradict an independently corroborated
            # longer reading; differing words still reject the continuation.
            compatible = [
                candidate for candidate in secondary if all(
                    folded(candidate) == folded(value)
                    or folded(candidate).startswith(folded(value) + " ")
                    or folded(value).startswith(folded(candidate) + " ")
                    for value in observed
                )
            ]
            secondary = [max(compatible, key=len)] if compatible else []
        if row.description_confident and len(secondary) == 1:
            row.secondary_description = secondary[0]
            row.description = f"{row.description} {secondary[0]}"
        else:
            row.secondary_description = ""
        for key in ("quantity", "unit_price", "total_ht", *OPTIONAL_HEADERS):
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
        if (
            {number.value for number in row.cells.get("unit_price", [])} == {Decimal(0)}
            and any(value > 0 for value in row.family_votes.get("gross_amount", {}))
        ):
            row.cells["unit_price"] = []
            result.issues.append(
                f"Ligne « {row.description} » : prix nul incompatible avec le montant TTC "
                "recoupé ; lecture du prix non confirmée."
            )
        candidates = [
            {number.value for number in row.cells[key]} for key in ("quantity", "unit_price", "total_ht")
        ]
        if all(candidates):
            solutions = [
                values for values in product(*candidates)
                if abs((values[0] * values[1]).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) - values[2]) <= Decimal(".02")
            ]
            if not solutions:
                recovered = _recover_quantity(row, matches, confidence)
                if recovered is not None:
                    row.cells["quantity"] = [recovered]
                    result.issues.append(
                        f"Ligne « {row.description} » : quantité {recovered.raw} retenue "
                        "depuis une lecture OCR isolée confirmée par le prix et le montant ; "
                        "lectures majoritaires contradictoires non retenues."
                    )
                else:
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
                    row.amount_operand_veto = True
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
