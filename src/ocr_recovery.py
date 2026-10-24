"""Frozen tuning-selected OCR pipeline; never reads stress references or metrics."""

import logging
import re

from invoice_evidence import UnreadableOCR, from_tsv, invoice_number_from_text
from ocr_consensus import build_consensus
from ocr_preprocessing import deskew, transform
from ocr_quality import assess_ocr
from ocr_config import OCR_CONFIG
from ocr_budget import ProcessingBudget

LOGGER = logging.getLogger(__name__)
FAMILIES = OCR_CONFIG.families
CONFIDENCE = OCR_CONFIG.confidence


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

    def record_ocr_metadata(evidence):
        evidence.ocr_pages = 1
        evidence.preprocessing_used = list(readings)
        return evidence

    if not primary.quality_issues:
        return record_ocr_metadata(primary)
    # assess_ocr adds refusal annotations; consensus needs the original evidence.
    for evidence in first_readings:
        evidence.quality_issues = []
        evidence.issues = []
    header_candidates = []
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
                crop = prepared.crop(
                    (0, 0, prepared.width, max(1, int(prepared.height * .22)))
                )
                header = from_tsv(reader(crop, header_psm))
                header_candidates.append(
                    header.invoice_number or invoice_number_from_text(header.text)
                )
                budget.remaining()
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
    if confirmed_invoice_number is not None:
        consensus.invoice_number = confirmed_invoice_number
    return record_ocr_metadata(consensus)
