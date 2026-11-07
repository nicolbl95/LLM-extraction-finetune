"""Frozen tuning-selected OCR pipeline; never reads stress references or metrics."""

import logging
import re
import time
from decimal import Decimal, ROUND_HALF_UP

import cv2
import numpy as np
from PIL import Image

from invoice_evidence import (
    Number, Token, UnreadableOCR, from_tokens, from_tsv, invoice_number_from_text, number_tokens,
)
from ocr_consensus import _candidate_numbers, _matching_row, build_consensus
from ocr_fast_path import attempt_fast_path, fast_path_enabled
from ocr_preprocessing import deskew, transform
from ocr_quality import assess_ocr
from ocr_quality import signature
from invoice_validation import document_validation
from ocr_config import OCR_CONFIG
from ocr_budget import ProcessingBudget
import table_geometry
import table_recovery
import table_shadow

LOGGER = logging.getLogger(__name__)
FAMILIES = OCR_CONFIG.families
CONFIDENCE = OCR_CONFIG.confidence


def _grid_mask(gray, *, regional=False):
    ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    bridge = 9 if regional else 5
    horizontal = cv2.morphologyEx(
        cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, bridge), np.uint8)),
        cv2.MORPH_OPEN, np.ones((1, max(60, gray.shape[1] // 6)), np.uint8),
    )
    vertical = cv2.morphologyEx(
        cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((bridge, 1), np.uint8)),
        cv2.MORPH_OPEN,
        np.ones((max(60, gray.shape[0] // (2 if regional else 12)), 1), np.uint8),
    )
    return cv2.bitwise_or(horizontal, vertical)


def _ruled_region(prepared):
    mask = _grid_mask(np.asarray(prepared))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = [
        cv2.boundingRect(contour) for contour in contours
        if cv2.contourArea(contour) > 0
    ]
    candidates = [
        (x, y, width, height) for x, y, width, height in rectangles
        if width > prepared.width / 2 and height > 40
    ]
    if not candidates:
        return None
    x, y, width, height = max(candidates, key=lambda rect: rect[2] * rect[3])
    return max(0, x - 8), max(0, y - 8), min(prepared.width, x + width + 8), min(prepared.height, y + height + 8)


def _regional_table(prepared, region, original, reader):
    left, top, right, bottom = region
    if (right - left) * (bottom - top) * 4 > OCR_CONFIG.max_pixels:
        return None
    image = prepared.crop(region).resize(
        ((right - left) * 2, (bottom - top) * 2), Image.Resampling.BICUBIC,
    )
    gray = np.asarray(image).copy()
    mask = cv2.dilate(_grid_mask(gray, regional=True), np.ones((3, 3), np.uint8))
    gray[mask > 0] = 255
    reading = from_tsv(reader(Image.fromarray(gray), 6))
    tokens = [
        token for token in original.tokens
        if not (left <= (token.box[0] + token.box[2]) / 2 <= right
                and top <= (token.box[1] + token.box[3]) / 2 <= bottom)
    ]
    tokens.extend(
        Token(token.text, token.page,
              (left + token.box[0] / 2, top + token.box[1] / 2,
               left + token.box[2] / 2, top + token.box[3] / 2),
              token.confidence)
        for token in reading.tokens
    )
    return from_tokens(tokens, locale=original.locale, literal_columns=False)


def _clarify_decimal_quantity(evidence, readings, prepared_images, reader, budget):
    for row in evidence.rows:
        if row.cells.get("quantity") or row.corroborated_amount_mismatch is not None:
            continue
        matches = {
            family: [match for reading in entries if (match := _matching_row(row, reading)) is not None]
            for family, entries in readings.items()
        }
        quantities, _ = _candidate_numbers(matches, "quantity", CONFIDENCE)
        prices, _ = _candidate_numbers(matches, "unit_price", CONFIDENCE)
        price_votes = row.family_votes.get("unit_price", {})
        amount_votes = row.family_votes.get("total_ht", {})
        if not quantities or len(price_votes) != 1 or len(amount_votes) != 1:
            continue
        price, amount = next(iter(price_votes)), next(iter(amount_votes))
        if not any(
            abs((value * price).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) - amount) > Decimal(".02")
            for value in quantities
        ):
            continue
        candidate = None
        confirmed = []
        for family in OCR_CONFIG.fast_families:
            budget.remaining()
            source_row = _matching_row(row, readings[family][0])
            if source_row is None or len(source_row.cells.get("quantity", [])) != 1:
                break
            source = source_row.cells["quantity"][0]
            left, top, right, bottom = source.box
            prepared = prepared_images[family]
            crop_left, crop_top = max(0, int(left) - 8), max(0, int(top) - 8)
            crop = prepared.crop((
                crop_left, crop_top, min(prepared.width, int(right) + 8),
                min(prepared.height, int(bottom) + 8),
            ))
            if crop.width * crop.height * 4 > OCR_CONFIG.max_pixels:
                break
            crop = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.BICUBIC)
            local = from_tsv(reader(crop, 7), locale=evidence.locale)
            numbers = [number for line in local.lines for number in number_tokens(line.text, line, evidence.locale)]
            if (
                len(numbers) != 1 or not local.tokens
                or any(token.confidence < CONFIDENCE for token in local.tokens)
            ):
                break
            number = numbers[0]
            if candidate is None:
                candidate = number.value
                if (
                    not any(candidate != old and (candidate * 10 == old or old * 10 == candidate)
                            for old in quantities)
                    or abs((candidate * price).quantize(Decimal(".01"), rounding=ROUND_HALF_UP) - amount) > Decimal(".02")
                ):
                    break
            elif number.value != candidate:
                break
            confirmed.append(Number(
                number.raw, number.value, source.page,
                (crop_left + number.box[0] / 2, crop_top + number.box[1] / 2,
                 crop_left + number.box[2] / 2, crop_top + number.box[3] / 2),
            ))
        if len(confirmed) == 2 and candidate is not None:
            row.cells["quantity"] = confirmed
            row.family_votes["quantity"] = {candidate: OCR_CONFIG.fast_families}
            row.cells["unit_price"] = [prices[price]]
            row.corroborated_amount_mismatch = None
            evidence.numbers.extend([*confirmed, prices[price]])
            evidence.issues.append(
                f"Ligne « {row.description} » : décimale relue dans sa cellule "
                "par deux prétraitements indépendants ; aucune valeur calculée ajoutée."
            )
        # At most one ambiguous cell (two bounded calls) per document.
        return len(confirmed) == 2
    return False


def table_crop(data, prepared, reader):
    evidence = from_tsv(data)
    if not evidence.rows:
        return data
    top = min(row.line.box[1] for row in evidence.rows)
    bottom = max(row.line.box[3] for row in evidence.rows)
    headers = [line for line in evidence.lines if line.box[3] <= top and top - line.box[3] < 120]
    crop_top = int(min(line.box[1] for line in headers)) if headers else max(0, int(top) - 90)
    cropped = prepared.crop((0, crop_top, prepared.width, min(prepared.height, int(bottom) + 20)))
    cropped_data = reader(cropped, 6)
    keep = [index for index, text in enumerate(data["text"]) if text.strip() and not (
        crop_top <= float(data["top"][index]) <= bottom
    )]
    merged = {key: [values[index] for index in keep] for key, values in data.items()}
    for index, text in enumerate(cropped_data["text"]):
        if not text.strip():
            continue
        for key in merged:
            value = cropped_data[key][index]
            merged[key].append(value + crop_top if key == "top" else value)
    return merged


def recover_image(image, reader, *, budget=None):
    budget = budget or ProcessingBudget()
    started = time.perf_counter()
    budget.remaining()
    rotated, angle = deskew(image)
    budget.remaining()
    prepared_images = {
        "deskew": transform(rotated, angle, "deskew")[0],
        "clahe": transform(rotated, angle, "clahe")[0],
    }
    budget.remaining()
    first_readings = []
    readings = {"deskew": [], "clahe": []}
    completed = {"deskew": set(), "clahe": set()}
    for family, psm in zip(OCR_CONFIG.fast_families, OCR_CONFIG.segmentations):
        budget.remaining()
        evidence = from_tsv(reader(prepared_images[family], psm))
        first_readings.append(evidence)
        readings[family].append(evidence)
        completed[family].add(psm)
        budget.remaining()

    def record_ocr_metadata(evidence):
        evidence.ocr_pages = 1
        evidence.preprocessing_used = list(readings)
        return evidence

    def observe(evidence):
        """Shadow reconstruction (diagnostic) then optional recovery; both OFF by default."""
        evidence = table_shadow.observe(
            evidence, readings, rotated_size=rotated.size, prepared=prepared_images,
            reader=reader, budget=budget,
        )
        evidence = table_recovery.recover(
            evidence, readings, rotated_size=rotated.size, prepared=prepared_images)
        return table_geometry.recover(
            evidence, readings, rotated_size=rotated.size, prepared=prepared_images)

    if not any(evidence.rows for evidence in first_readings):
        budget.remaining()
        region = _ruled_region(prepared_images["deskew"])
        if region is not None:
            for family, original in zip(OCR_CONFIG.fast_families, first_readings):
                budget.remaining()
                regional = _regional_table(prepared_images[family], region, original, reader)
                if regional is not None:
                    readings[family].append(regional)
            budget.remaining()
            regional_consensus = build_consensus(readings, confidence=CONFIDENCE)
            if (
                regional_consensus.supplier and regional_consensus.invoice_date
                and regional_consensus.totals.get("total_ht")
                and regional_consensus.totals.get("total_ttc")
                and regional_consensus.rows and all(
                row.description_confident and all(
                    len(row.cells.get(key, [])) == 1
                    for key in ("quantity", "unit_price", "total_ht")
                ) for row in regional_consensus.rows
                )
            ):
                return observe(record_ocr_metadata(regional_consensus))

    fast_outcome = None
    fast_metadata = None
    if fast_path_enabled():
        def fast_reader(prepared, psm, label):
            return from_tsv(reader(prepared, psm))

        fast_outcome = attempt_fast_path(first_readings, prepared_images, fast_reader)
        fast_metadata = fast_outcome.record(time.perf_counter() - started)
        LOGGER.info(
            "OCR fast path %s: %s; calls=%d; decision=%.3fs",
            "accepted" if fast_outcome.accepted else "rejected",
            fast_outcome.reason,
            fast_metadata["calls"],
            fast_metadata["elapsed_seconds"],
        )
        if fast_outcome.accepted:
            evidence = record_ocr_metadata(fast_outcome.evidence)
            evidence.fast_path = fast_metadata
            return observe(evidence)

    primary = assess_ocr(first_readings[0], first_readings[1])
    has_invoice_heading = any(
        re.search(r"^\s*(?:invoice|facture)(?!\s+(?:date|total))\b", line.text, re.I)
        for evidence in first_readings
        for line in evidence.lines
    )
    invoice_numbers = [evidence.invoice_number for evidence in first_readings]
    needs_header_confirmation = len(set(invoice_numbers)) != 1 or not all(invoice_numbers)
    confirm_header = needs_header_confirmation and (has_invoice_heading or bool(primary.quality_issues))
    if has_invoice_heading and needs_header_confirmation and not primary.quality_issues:
        primary.quality_issues.append("Numéro de facture non lu.")

    if not primary.quality_issues:
        evidence = record_ocr_metadata(primary)
        if fast_metadata is not None:
            evidence.fast_path = fast_metadata
        return observe(evidence)
    # assess_ocr adds refusal annotations; consensus needs the original evidence.
    for evidence in first_readings:
        evidence.quality_issues = []
        evidence.issues = []
    header_candidates = (
        list(fast_outcome.header_candidates)
        if fast_outcome is not None and confirm_header else []
    )
    refinement_attempted = False

    def read_header(family, psm):
        prepared = prepared_images[family]
        crop = prepared.crop((0, 0, prepared.width, max(1, int(prepared.height * .22))))
        header = from_tsv(reader(crop, psm))
        budget.remaining()
        return header.invoice_number or invoice_number_from_text(header.text)

    pair = [signature(evidence) for evidence in first_readings]
    comparable = pair[0][:1] + pair[0][2:] == pair[1][:1] + pair[1][2:]
    legible = all(
        evidence.tokens
        and sum(token.confidence for token in evidence.tokens) / len(evidence.tokens) >= OCR_CONFIG.fast_mean_confidence
        and all(token.confidence >= OCR_CONFIG.fast_token_confidence
                for row in evidence.rows for token in row.line.tokens)
        for evidence in first_readings
    )
    if comparable and legible:
        candidate = build_consensus(readings, confidence=CONFIDENCE)
        recovery_needed = any(
            row.corroborated_amount_mismatch is not None or (
                not row.cells.get("quantity")
                and len(row.family_votes.get("unit_price", {})) == 1
                and len(row.family_votes.get("total_ht", {})) == 1
            ) for row in candidate.rows
        )
        if (
            recovery_needed and candidate.supplier and candidate.invoice_date
            and len(candidate.totals.get("total_ht", [])) == 1
            and len(candidate.totals.get("total_ttc", [])) == 1
        ):
            if confirm_header:
                for family in OCR_CONFIG.fast_families[len(header_candidates):]:
                    budget.remaining()
                    header_candidates.append(read_header(family, 11))
            invoice = (
                header_candidates[0] if len(header_candidates) == 2
                and header_candidates[0] == header_candidates[1]
                else invoice_numbers[0] if len(set(invoice_numbers)) == 1 else None
            )
            if invoice:
                refinement_attempted = True
                _clarify_decimal_quantity(candidate, readings, prepared_images, reader, budget)
                validation = document_validation(candidate)
                if (
                    candidate.rows and all(
                        row.description_confident and (
                            row.corroborated_amount_mismatch is not None
                            or all(len({n.value for n in row.cells.get(key, [])}) == 1
                                   for key in ("quantity", "unit_price", "total_ht"))
                        ) for row in candidate.rows
                    )
                    and all(flag.code in {"corroborated_line_amount_mismatch", "subtotal", "identifier_siret"}
                            for flag in validation.flags)
                    and all(reason.startswith("Ligne ") for reason in validation.not_verified)
                ):
                    candidate.invoice_number = invoice
                    return observe(record_ocr_metadata(candidate))
    for family in FAMILIES:
        budget.remaining()
        if family == "scale3" and rotated.width * rotated.height * 9 > OCR_CONFIG.max_pixels:
            LOGGER.warning("OCR scale3 omitted: source exceeds the 25 MP preprocessing budget")
            continue
        prepared = prepared_images.get(family)
        if prepared is None:
            prepared = transform(rotated, angle, family)[0]
        budget.remaining()
        readings.setdefault(family, [])
        for psm in OCR_CONFIG.segmentations:
            if psm in completed.get(family, set()):
                continue
            budget.remaining()
            header_psm = (
                11 if family in {"deskew", "clahe"} and (
                    (family == "deskew" and psm == 6)
                    or (family == "clahe" and psm == 4)
                ) else None
            )
            if confirm_header and header_psm is not None:
                if len(header_candidates) == 2:
                    continue
                header_candidates.append(read_header(family, header_psm))
                continue
            data = reader(prepared, psm)
            budget.remaining()
            if family == "crop":
                data = table_crop(data, prepared, reader)
            readings[family].append(from_tsv(data))
            budget.remaining()
    confirmed_invoice_number = None
    if (
        confirm_header and len(header_candidates) == 2 and header_candidates[0]
        and header_candidates[0] == header_candidates[1]
    ):
        confirmed_invoice_number = header_candidates[0]
    if not any(evidence.tokens for entries in readings.values() for evidence in entries):
        raise UnreadableOCR("Aucun texte détecté après les prétraitements OCR.")
    consensus = build_consensus(readings, confidence=CONFIDENCE)
    if not refinement_attempted:
        _clarify_decimal_quantity(consensus, readings, prepared_images, reader, budget)
    if confirmed_invoice_number is not None:
        consensus.invoice_number = confirmed_invoice_number
    evidence = record_ocr_metadata(consensus)
    if fast_metadata is not None:
        evidence.fast_path = fast_metadata
    return observe(evidence)
