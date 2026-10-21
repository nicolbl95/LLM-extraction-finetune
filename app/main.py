"""Gradio demo: invoice document -> text/OCR -> fine-tuned LoRA -> Invoice."""

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

import spaces
import gradio as gr
import pymupdf
import pytesseract
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_finetuned import Invoice, LoRAInference, parse_object

LOGGER = logging.getLogger(__name__)
MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 10
MAX_PIXELS = 25_000_000
EXAMPLES = [
    ("France · EUR", ROOT / "app/examples/france.png"),
    ("Royaume-Uni · GBP", ROOT / "app/examples/uk.png"),
    ("États-Unis · USD", ROOT / "app/examples/us.png"),
]
DEFAULT_ADAPTER = (
    ROOT / "results/qwen25-3b-international-v3-20261021T053959288046Z/lora_adapter"
)


def checked_path(filename: str | None) -> Path:
    if not filename:
        raise ValueError("Importez une facture ou choisissez un exemple.")
    path = Path(filename)
    if path.suffix.lower() not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise ValueError("Formats acceptés : PDF, PNG et JPEG.")
    if not path.is_file():
        raise ValueError("Le fichier sélectionné est introuvable.")
    if not 0 < path.stat().st_size <= MAX_BYTES:
        raise ValueError("Le fichier doit être non vide et ne pas dépasser 20 Mo.")
    return path


def check_pixels(width: float, height: float) -> None:
    if width * height > MAX_PIXELS:
        raise ValueError("Image ou page trop grande : limite de 25 mégapixels.")


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        check_pixels(*image.size)
        return ImageOps.exif_transpose(image).convert("RGB")


def render_page(page: pymupdf.Page) -> Image.Image:
    scale = 2
    check_pixels(page.rect.width * scale, page.rect.height * scale)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def check_pdf(document: pymupdf.Document) -> None:
    if document.needs_pass:
        raise ValueError("Les PDF protégés par mot de passe ne sont pas acceptés.")
    if not 1 <= document.page_count <= MAX_PAGES:
        raise ValueError("Le PDF doit contenir entre 1 et 10 pages.")


def preview_document(filename: str | None):
    if not filename:
        return None, None, "Importez une facture ou choisissez un exemple."
    try:
        path = checked_path(filename)
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(path) as document:
                check_pdf(document)
                image = render_page(document[0])
                message = f"Aperçu : page 1 sur {document.page_count}. Toutes les pages seront analysées."
        else:
            image = open_image(path)
            message = "Facture prête. Cliquez sur « Extraire les données »."
        return image, None, message
    except (OSError, ValueError, RuntimeError, Image.DecompressionBombError) as exc:
        LOGGER.exception("Invoice preview failed")
        return None, None, f"Impossible de lire la facture : {exc}"


def ocr_image(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image, lang="fra+eng", timeout=60)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "OCR indisponible : installez Tesseract et les langues fra/eng (voir packages.txt)."
        ) from exc
    if not text.strip():
        raise ValueError("Aucun texte détecté. Utilisez une facture nette et lisible.")
    return text.strip()


def document_text(filename: str) -> str:
    path = checked_path(filename)
    if path.suffix.lower() != ".pdf":
        return ocr_image(open_image(path))
    with pymupdf.open(path) as document:
        check_pdf(document)
        pages = []
        for page in document:
            text = page.get_text("text").strip()
            pages.append(text or ocr_image(render_page(page)))
    return "\n\n".join(pages)


@lru_cache(maxsize=1)
def get_extractor() -> LoRAInference:
    adapter = Path(os.environ.get("INVOICE_ADAPTER_PATH", str(DEFAULT_ADAPTER))).expanduser()
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    device = os.environ.get("INVOICE_DEVICE", "cuda")
    if device not in {"cpu", "cuda"}:
        raise ValueError("INVOICE_DEVICE doit valoir cpu ou cuda.")
    return LoRAInference(
        adapter_path=adapter,
        device=device,
        load_in_4bit=device == "cuda",
        n_ctx=4096,
        max_tokens=2048,
    )


@spaces.GPU
def extract_invoice(text: str) -> dict:
    prediction = get_extractor().predict({"id": "live-upload", "input": text})
    if prediction["finish_reason"] != "stop":
        raise ValueError("Réponse du modèle incomplète. Essayez une facture plus courte.")
    invoice = Invoice.model_validate(parse_object(prediction["raw"]))
    payload = invoice.model_dump(mode="json")
    # Pydantic's shared schema allows non-finite floats; JSON output must not.
    json.dumps(payload, allow_nan=False)
    return payload


def run_extraction(filename: str | None):
    yield None, "Extraction en cours… Le premier chargement du modèle peut prendre plusieurs minutes."
    try:
        path = checked_path(filename)
        payload = extract_invoice(document_text(str(path)))
    except (OSError, ValueError, RuntimeError, ImportError,
            pytesseract.TesseractError, Image.DecompressionBombError) as exc:
        LOGGER.exception("Invoice extraction failed")
        yield None, f"Échec de l'extraction : {exc}"
        return
    yield payload, "Extraction terminée — JSON validé par Pydantic."


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Extraction intelligente de factures") as demo:
        gr.Markdown(
            "# Extraction intelligente de factures\n"
            "Transformez vos factures françaises, britanniques et américaines "
            "en données JSON structurées avec un modèle spécialisé."
        )
        with gr.Row():
            with gr.Column():
                upload = gr.File(
                    label="Importer une facture — PDF ou image",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg"],
                    type="filepath",
                )
                gr.Markdown("PDF de 10 pages maximum · 20 Mo maximum · OCR français/anglais")
                gr.Examples(
                    examples=[[str(path)] for _, path in EXAMPLES],
                    inputs=[upload],
                    label="Essayer une facture : France / Royaume-Uni / États-Unis",
                    example_labels=[label for label, _ in EXAMPLES],
                    cache_examples=False,
                )
                extract = gr.Button("Extraire les données", variant="primary")
                preview = gr.Image(label="Aperçu de la facture", type="pil", interactive=False)
            with gr.Column():
                status = gr.Markdown("Importez une facture ou choisissez un exemple.")
                result = gr.JSON(label="Résultat de l'extraction — JSON validé")
                gr.Markdown(
                    "**À vérifier avant utilisation :** la validation Pydantic garantit la structure, "
                    "pas l'exactitude des données. Cette démo utilise l'OCR puis le modèle LoRA ; "
                    "elle ne renvoie pas les réponses du dataset.\n\n"
                    "**Confidentialité :** utilisez de préférence des factures fictives ou anonymisées. "
                    "Les documents sont traités sur le serveur hébergeant la démo."
                )
        upload.change(
            preview_document, inputs=[upload], outputs=[preview, result, status],
            concurrency_limit=1,
        )
        extract.click(
            run_extraction, inputs=[upload], outputs=[result, status], concurrency_limit=1,
        )
    return demo.queue(default_concurrency_limit=1, max_size=8)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_app().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        max_file_size="20mb",
    )
