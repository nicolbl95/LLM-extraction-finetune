"""Deterministic image transformations, with no invoice-specific values."""

import cv2
import numpy as np
from PIL import Image
from ocr_config import OCR_CONFIG

cv2.setNumThreads(1)

def estimate_skew(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"))
    width = gray.shape[1]
    if width > 1200:
        gray = cv2.resize(gray, (1200, round(gray.shape[0] * 1200 / width)))
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60,
                            minLineLength=max(50, gray.shape[1] // 7), maxLineGap=25)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if -5 <= angle <= 5:
            angles.append(angle)
    return float(np.median(angles)) if len(angles) >= 3 else 0.0


def deskew(image: Image.Image) -> tuple[Image.Image, float]:
    angle = estimate_skew(image)
    if abs(angle) < .15:
        return image.convert("L"), angle
    # Image coordinates grow downwards; PIL rotation uses the opposite convention.
    return image.convert("L").rotate(angle, Image.Resampling.BICUBIC, expand=True, fillcolor=255), angle


def prepare(image: Image.Image, method: str) -> tuple[Image.Image, dict]:
    rotated, angle = deskew(image)
    return transform(rotated, angle, method)


def transform(rotated: Image.Image, angle: float, method: str) -> tuple[Image.Image, dict]:
    scale = 3 if method == "scale3" else 2
    if rotated.width * rotated.height * scale * scale > OCR_CONFIG.max_pixels:
        raise ValueError("Prétraitement OCR supérieur à la limite de 25 mégapixels.")
    output = rotated.resize((rotated.width * scale, rotated.height * scale), Image.Resampling.BICUBIC)
    pixels = np.asarray(output)
    if method == "otsu":
        _, pixels = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        pixels = cv2.adaptiveThreshold(pixels, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 41, 15)
    elif method == "denoise":
        pixels = cv2.medianBlur(pixels, 3)
    elif method == "clahe":
        pixels = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(pixels)
    elif method not in {"deskew", "scale3", "crop"}:
        raise ValueError(f"Unknown OCR preprocessing method: {method}")
    return Image.fromarray(pixels), {"angle_degrees": angle, "scale": scale, "method": method}
