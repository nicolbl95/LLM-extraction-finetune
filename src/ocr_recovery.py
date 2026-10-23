"""Frozen tuning-selected OCR pipeline; never reads stress references or metrics."""

import logging

from invoice_evidence import UnreadableOCR, from_tsv
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
    if not primary.quality_issues:
        return primary
    # assess_ocr adds refusal annotations; consensus needs the original evidence.
    for evidence in first_readings:
        evidence.quality_issues = []
        evidence.issues = []
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
            data = reader(prepared, psm)
            budget.remaining()
            if family == "crop":
                data = table_crop(data, prepared, reader)
            readings[family].append(from_tsv(data))
            budget.remaining()
    if not any(evidence.tokens for entries in readings.values() for evidence in entries):
        raise UnreadableOCR("Aucun texte détecté après les prétraitements OCR.")
    return build_consensus(readings, confidence=CONFIDENCE)
