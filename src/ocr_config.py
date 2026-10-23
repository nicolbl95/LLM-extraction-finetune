"""Single versioned OCR policy; changing it requires a new validation run."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OCRConfiguration:
    version: str = "ocr-policy-4-bounded-fastpath"
    confidence: int = 40
    families: tuple[str, ...] = ("deskew", "clahe", "crop", "denoise", "scale3")
    segmentations: tuple[int, ...] = (4, 6)
    fast_families: tuple[str, ...] = ("deskew", "clahe")
    fast_mean_confidence: int = 85
    fast_token_confidence: int = 60
    total_seconds: float = 45.0
    call_seconds: float = 60.0
    max_pixels: int = 25_000_000

    def export(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.export(), sort_keys=True).encode()).hexdigest()


OCR_CONFIG = OCRConfiguration()
