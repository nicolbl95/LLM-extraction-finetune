"""PR102: one deterministic evaluator shared by the baseline and the experimental pipeline.

The two historical scorers disagreed about what a number meant: PR91 counted a document against its own
output, and PR101's indexed view called a correct amount incorrect when it landed in the slot of a row that
was never recovered. This module replaces both with explicit, separately reported metrics.

Rules that matter:

* A row is matched on **printed evidence**, never on position: an agreeing description, or a full monetary
  triple that is unique among the reference rows. A description that disagrees blocks the match even when
  the amounts agree — that is a correct amount on the wrong product, and it is counted as an incorrect
  association, not as a correct cell.
* Each exported row and each reference row is used at most once, so a duplicate cannot be matched twice and
  a set-based view cannot hide it.
* A missing value and an incorrect value are never merged. A reference field that is null is unverified and
  is not counted at all.
* The positional view is still computed and returned, labelled, so the historical numbers stay comparable.

Pure functions, no I/O, no randomness: the same inputs always give the same verdicts.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

ROW_FIELDS = ("description", "quantity", "unit_price", "total_ht")
MONETARY_FIELDS = ("quantity", "unit_price", "total_ht")
HEADER_FIELDS = ("supplier_name", "invoice_date", "total_ht", "total_ttc")

CORRECT = "correct"
INCORRECT = "incorrect"
MISSING = "missing"
UNVERIFIED = "unverified"

IDENTITY_DESCRIPTION = "description"
IDENTITY_AMOUNTS = "unique-monetary-triple"
IDENTITY_PARTIAL = "unique-partial-amounts"
IDENTITY_NONE = "none"

# Two independent printed values are the minimum that can identify a product row. A lone total cannot:
# totals coincide across products, and crediting one would let an amount masquerade as a row.
MINIMUM_AGREEMENTS = 2


def _text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _number(value):
    """Decimal for anything numeric-looking, so 40, 40.0 and '40.00' compare equal."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    text = _text(value).replace("\u00a0", " ").replace(" ", "")
    text = re.sub(r"[^\d.,\-]", "", text)
    if not text or not re.search(r"\d", text):
        return None
    if "," in text and "." in text:
        text = text.replace(",", "") if text.rfind(".") > text.rfind(",") else text.replace(".", "").replace(",", ".")
    elif "," in text:
        text = text.replace(",", ".") if re.search(r",\d{1,2}$", text) else text.replace(",", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _equal(field: str, left, right) -> bool:
    if field == "description":
        return _text(left) == _text(right)
    a, b = _number(left), _number(right)
    if a is not None and b is not None:
        return a == b
    return _text(left) == _text(right)


def _signature(row: dict) -> tuple:
    return (_text(row.get("description")), *(_number(row.get(key)) for key in MONETARY_FIELDS))


def _triple(row: dict) -> tuple:
    return tuple(_number(row.get(key)) for key in MONETARY_FIELDS)


def _has_triple(row: dict) -> bool:
    return all(value is not None for value in _triple(row))


def _rows(payload: dict) -> list[dict]:
    return [row or {} for row in (payload.get("line_items") or [])]


def match_rows(exported: list[dict], reference: list[dict]) -> dict:
    """Association-aware pairing on printed evidence, with the rejected candidates kept visible.

    Identity is graded: an agreeing description is strongest, a complete unique monetary triple next, and
    two or more agreeing monetary values that only one reference row shares last. One agreeing value is
    never enough — a lone total coincides across products and would let an amount pose as a row.
    """
    agreements = [
        [tuple((key, _number(export.get(key))) for key in MONETARY_FIELDS
               if _number(export.get(key)) is not None and _number(gold.get(key)) is not None
               and _number(export.get(key)) == _number(gold.get(key)))
         for gold in reference]
        for export in exported
    ]
    partial_options = [
        [gold_index for gold_index, keys in enumerate(per_export) if len(keys) >= MINIMUM_AGREEMENTS]
        for per_export in agreements
    ]
    triple_counts: dict[tuple, int] = {}
    for row in reference:
        if _has_triple(row):
            triple_counts[_triple(row)] = triple_counts.get(_triple(row), 0) + 1

    candidates = []
    for export_index, export in enumerate(exported):
        for gold_index, gold in enumerate(reference):
            export_description, gold_description = _text(export.get("description")), _text(gold.get("description"))
            descriptions_agree = bool(export_description) and bool(gold_description) and export_description == gold_description
            descriptions_disagree = bool(export_description) and bool(gold_description) and not descriptions_agree
            keys = agreements[export_index][gold_index]
            amounts_agree = len(keys) == len(MONETARY_FIELDS)
            # A description naming this product while the amounts belong to a different reference row is a
            # wrong-product row, not a match with a few bad cells.
            borrowed = (
                not amounts_agree and _has_triple(export)
                and any(other_index != gold_index and _has_triple(other) and _triple(export) == _triple(other)
                        for other_index, other in enumerate(reference))
            )
            if descriptions_disagree or borrowed:
                identity, score = IDENTITY_NONE, 0
            elif descriptions_agree:
                identity, score = IDENTITY_DESCRIPTION, 4 + len(keys)
            elif amounts_agree and triple_counts.get(_triple(gold)) == 1:
                identity, score = IDENTITY_AMOUNTS, 3
            elif len(keys) >= MINIMUM_AGREEMENTS and len(partial_options[export_index]) == 1:
                identity, score = IDENTITY_PARTIAL, len(keys)
            else:
                identity, score = IDENTITY_NONE, 0
            candidates.append({
                "export_index": export_index, "gold_index": gold_index,
                "identity": identity, "score": score,
                "agreeing_monetary_fields": [key for key, _value in keys],
                "descriptions_disagree": descriptions_disagree,
                "amounts_belong_to_another_row": borrowed,
                "amounts_agree": amounts_agree,
            })

    admissible = sorted(
        (candidate for candidate in candidates if candidate["identity"] != IDENTITY_NONE),
        key=lambda candidate: (-candidate["score"], candidate["export_index"], candidate["gold_index"]),
    )
    used_export, used_gold, pairs = set(), set(), []
    for candidate in admissible:
        if candidate["export_index"] in used_export or candidate["gold_index"] in used_gold:
            continue
        used_export.add(candidate["export_index"])
        used_gold.add(candidate["gold_index"])
        pairs.append(candidate)
    pairs.sort(key=lambda pair: pair["export_index"])
    return {
        "pairs": pairs,
        "matched_export_indices": sorted(used_export),
        "matched_gold_indices": sorted(used_gold),
        "rejected_candidates": [candidate for candidate in candidates
                                if candidate["amounts_agree"]
                                and (candidate["descriptions_disagree"]
                                     or candidate["amounts_belong_to_another_row"])],
    }


def _field_verdicts(export: dict, gold: dict) -> dict:
    verdicts = {}
    for field in ROW_FIELDS:
        reference_value, exported_value = gold.get(field), export.get(field)
        if reference_value is None:
            verdicts[field] = UNVERIFIED
        elif exported_value is None:
            verdicts[field] = MISSING
        elif _equal(field, exported_value, reference_value):
            verdicts[field] = CORRECT
        else:
            verdicts[field] = INCORRECT
    return verdicts


def _header_verdicts(payload: dict, reference: dict) -> dict:
    verdicts = {}
    for field in HEADER_FIELDS:
        reference_value, exported_value = reference.get(field), payload.get(field)
        if reference_value is None:
            verdicts[field] = UNVERIFIED
        elif exported_value is None:
            verdicts[field] = MISSING
        else:
            if field in ("supplier_name", "invoice_date"):
                same = _text(exported_value) == _text(reference_value)
            else:
                left, right = _number(exported_value), _number(reference_value)
                same = (left == right if left is not None and right is not None
                        else _text(exported_value) == _text(reference_value))
            verdicts[field] = CORRECT if same else INCORRECT
    return verdicts


def _positional(exported: list[dict], reference: list[dict]) -> dict:
    """The historical index-based view, kept only so the old numbers remain explainable."""
    counts = {verdict: 0 for verdict in (CORRECT, INCORRECT, MISSING, UNVERIFIED)}
    monetary_incorrect = 0
    for index, gold in enumerate(reference):
        export = exported[index] if index < len(exported) else {}
        for field, verdict in _field_verdicts(export, gold).items():
            counts[verdict] += 1
            if verdict == INCORRECT and field in MONETARY_FIELDS:
                monetary_incorrect += 1
    return {"cells": counts, "incorrect_monetary_cells": monetary_incorrect,
            "note": "index-based; a correct amount in an unrecovered row's slot is reported incorrect here"}


def evaluate(payload: dict, reference: dict) -> dict:
    """Score one exported payload against one frozen reference. Deterministic and side-effect free."""
    exported, gold = _rows(payload), _rows(reference)
    matching = match_rows(exported, gold)
    pairs = matching["pairs"]
    matched_gold = set(matching["matched_gold_indices"])
    matched_export = set(matching["matched_export_indices"])

    cells = {field: {verdict: 0 for verdict in (CORRECT, INCORRECT, MISSING, UNVERIFIED)}
             for field in ROW_FIELDS}
    rows_correct, rows_with_errors, identity_by_amounts = 0, 0, 0
    pair_details = []
    for pair in pairs:
        export, gold_row = exported[pair["export_index"]], gold[pair["gold_index"]]
        verdicts = _field_verdicts(export, gold_row)
        for field, verdict in verdicts.items():
            cells[field][verdict] += 1
        if pair["identity"] in (IDENTITY_AMOUNTS, IDENTITY_PARTIAL):
            identity_by_amounts += 1
        if all(verdict in (CORRECT, UNVERIFIED) for verdict in verdicts.values()):
            rows_correct += 1
        else:
            rows_with_errors += 1
        pair_details.append({**pair, "verdicts": verdicts})

    # Unmatched exported rows: duplicates first, then wrong-product amounts, then unsupported noise.
    seen_signatures = {_signature(exported[index]) for index in matched_export}
    duplicates, misassociated, unsupported = [], [], []
    incorrect_association_cells = 0
    for index, export in enumerate(exported):
        if index in matched_export:
            continue
        signature = _signature(export)
        if signature in seen_signatures:
            duplicates.append(index)
            continue
        seen_signatures.add(signature)
        borrowed = [
            gold_index for gold_index, gold_row in enumerate(gold)
            if _has_triple(export) and _has_triple(gold_row) and _triple(export) == _triple(gold_row)
        ]
        if borrowed:
            misassociated.append({"export_index": index, "amounts_belong_to_gold_rows": borrowed})
            incorrect_association_cells += sum(
                1 for field in MONETARY_FIELDS if export.get(field) is not None)
        else:
            unsupported.append(index)

    missing_rows = [index for index in range(len(gold)) if index not in matched_gold]
    missing_row_fields = sum(
        1 for index in missing_rows for field in ROW_FIELDS if gold[index].get(field) is not None)

    field_totals = {verdict: sum(cells[field][verdict] for field in ROW_FIELDS)
                    for verdict in (CORRECT, INCORRECT, MISSING, UNVERIFIED)}
    incorrect_monetary = sum(cells[field][INCORRECT] for field in MONETARY_FIELDS)
    header = _header_verdicts(payload, reference)

    return {
        "rows": {
            "expected": len(gold),
            "exported": len(exported),
            "correctly_matched": rows_correct,
            "matched_with_field_errors": rows_with_errors,
            "matched_total": len(pairs),
            "identity_by_amounts_only": identity_by_amounts,
            "missing": len(missing_rows),
            "missing_indices": missing_rows,
            "duplicate": len(duplicates),
            "duplicate_indices": duplicates,
            "misassociated": len(misassociated),
            "misassociated_detail": misassociated,
            "unsupported": len(unsupported),
            "unsupported_indices": unsupported,
        },
        "fields": {
            "by_field": cells,
            "correct": field_totals[CORRECT],
            "incorrect": field_totals[INCORRECT],
            "missing": field_totals[MISSING] + missing_row_fields,
            "missing_in_matched_rows": field_totals[MISSING],
            "missing_in_unrecovered_rows": missing_row_fields,
            "unverified": field_totals[UNVERIFIED],
            "incorrect_monetary": incorrect_monetary + incorrect_association_cells,
            "incorrect_monetary_in_matched_rows": incorrect_monetary,
            "incorrect_association": incorrect_association_cells,
        },
        "header": header,
        "header_summary": {
            "correct": sum(1 for verdict in header.values() if verdict == CORRECT),
            "incorrect": sum(1 for verdict in header.values() if verdict == INCORRECT),
            "missing": sum(1 for verdict in header.values() if verdict == MISSING),
            "unverified": sum(1 for verdict in header.values() if verdict == UNVERIFIED),
            "incorrect_supplier": header["supplier_name"] == INCORRECT,
            "incorrect_date": header["invoice_date"] == INCORRECT,
        },
        "pairs": pair_details,
        "rejected_amount_agreements": matching["rejected_candidates"],
        "positional": _positional(exported, gold),
    }


def aggregate(results: list[dict]) -> dict:
    """Sum per-document evaluations. Header verdicts stay per document; only counts are added."""
    rows = {key: 0 for key in ("expected", "exported", "correctly_matched", "matched_with_field_errors",
                               "matched_total", "identity_by_amounts_only", "missing", "duplicate",
                               "misassociated", "unsupported")}
    fields = {key: 0 for key in ("correct", "incorrect", "missing", "missing_in_matched_rows",
                                 "missing_in_unrecovered_rows", "unverified", "incorrect_monetary",
                                 "incorrect_monetary_in_matched_rows", "incorrect_association")}
    header = {key: 0 for key in ("correct", "incorrect", "missing", "unverified")}
    positional = {verdict: 0 for verdict in (CORRECT, INCORRECT, MISSING, UNVERIFIED)}
    positional_monetary = 0
    incorrect_suppliers = incorrect_dates = 0
    for result in results:
        for key in rows:
            rows[key] += result["rows"][key]
        for key in fields:
            fields[key] += result["fields"][key]
        for key in header:
            header[key] += result["header_summary"][key]
        for verdict in positional:
            positional[verdict] += result["positional"]["cells"][verdict]
        positional_monetary += result["positional"]["incorrect_monetary_cells"]
        incorrect_suppliers += int(result["header_summary"]["incorrect_supplier"])
        incorrect_dates += int(result["header_summary"]["incorrect_date"])
    return {"documents": len(results), "rows": rows, "fields": fields, "header": header,
            "incorrect_supplier_documents": incorrect_suppliers,
            "incorrect_date_documents": incorrect_dates,
            "positional": {"cells": positional, "incorrect_monetary_cells": positional_monetary}}
