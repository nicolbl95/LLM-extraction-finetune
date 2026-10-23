"""Independent stress inputs; no generated training/test records are loaded."""

import copy
import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class StressCase:
    name: str
    expected: dict
    model: dict
    image: Image.Image | None = None
    text: str | None = None
    pdf: Path | None = None


def reference(supplier, invoice_date, ht, ttc, items):
    return {
        "supplier_name": supplier, "invoice_date": invoice_date, "total_ht": ht, "total_ttc": ttc,
        "line_items": [
            dict(zip(("description", "quantity", "unit_price", "total_ht"), item)) for item in items
        ],
    }


REFERENCES = {
    "france": reference("Services Horizon SAS", "2025-04-30", 5716.77, 6860.12, [
        ("Clavier sans fil", 30, 31.55, 946.50), ("Maintenance informatique", 7.2, 95.07, 684.50),
        ("Conseil informatique", 5.4, 84.78, 457.81), ("Création de logo", 4.2, 863.80, 3627.96),
    ]),
    "uk": reference("Oak Studio Ltd", "2025-05-18", 6527.52, 7743.58, [
        ("Logo design", 7.8, 757.97, 5912.17), ("Printed technical handbook", 20, 22.36, 447.20),
        ("Network cable", 15, 11.21, 168.15),
    ]),
    "us": reference("Cedar Office LLC", "2023-12-07", 8972.80, 8972.80, [
        ("Printed technical handbook", 15, 29.66, 444.90), ("27 inch monitor", 29, 249.06, 7222.74),
        ("Web development", 5.1, 77.66, 396.07), ("Wireless keyboard", 7, 25.15, 176.05),
        ("Printer ink cartridge", 22, 33.32, 733.04),
    ]),
}


def wrong_model(expected):
    model = copy.deepcopy(expected)
    model["line_items"][0].update(quantity=54, unit_price=77.68, total_ht=4210.32)
    return model


def error_profile(expected, profile):
    """Reconstructed user-reported failures, not locally captured generations."""
    model = copy.deepcopy(expected)
    items = model["line_items"]
    if profile == "reported":
        if expected["supplier_name"] == "Services Horizon SAS":
            items[0]["description"] = "Clavier sans il"
            items[1]["quantity"] = 142
            items[3]["description"] = "Édition de logo"
        elif expected["supplier_name"] == "Oak Studio Ltd":
            items[0].update(quantity="5,91217", unit_price="75,797")
        elif expected["supplier_name"] == "Cedar Office LLC":
            items[0].update(unit_price=29.68, total_ht=445.20)
            items[2].update(quantity=54, unit_price=77.68, total_ht=4210.32)
        else:
            items[0].update(quantity=54, unit_price=77.68, total_ht=4210.32)
    elif profile == "coherent":
        # Deliberately exact multiplication but unsupported source values.
        items[0].update(quantity=15, unit_price=29.68, total_ht=445.20)
    elif profile == "permuted":
        items[0].update(quantity=12, unit_price=95.70, total_ht=1148.40)
        items[0]["description"] = "Lecture inventée"
    else:
        raise ValueError(profile)
    return model


def degrade(image, operation):
    if operation.startswith("rotate"):
        return image.rotate(float(operation.split(":")[1]), Image.Resampling.BICUBIC, expand=True, fillcolor="white")
    if operation == "blur":
        return image.filter(ImageFilter.GaussianBlur(.8))
    if operation == "jpeg40":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=40)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if operation == "low_contrast":
        return ImageEnhance.Contrast(image).enhance(.25)
    if operation == "half_resolution":
        return image.resize((image.width // 2, image.height // 2), Image.Resampling.BICUBIC)
    if operation in {"noise", "destroyed"}:
        rng = random.Random(713)
        if operation == "destroyed":
            data = bytes(rng.randrange(256) for _ in range(image.width * image.height))
        else:
            data = bytes(max(0, min(255, pixel + round(rng.gauss(0, 14)))) for pixel in image.convert("L").tobytes())
        return Image.frombytes("L", image.size, data).convert("RGB")
    raise ValueError(operation)


def image_cases(root):
    operations = ("rotate:1.5", "rotate:-2", "blur", "jpeg40", "low_contrast", "half_resolution", "noise", "destroyed")
    for name, expected in REFERENCES.items():
        with Image.open(root / "app/examples" / f"{name}.png") as source:
            image = source.convert("RGB")
        for operation in operations:
            yield StressCase(f"{name}/{operation}", expected, wrong_model(expected), image=degrade(image, operation))


LAYOUT_REFERENCE = reference("Stress Services Ltd", "2026-01-01", 1284.56, 1531.47, [
    ("Installation", 2, 617.28, 1234.56), ("Manuel", 2, 25, 50),
])
LAYOUT_HEADER = "Supplier: Stress Services Ltd\nInvoice date: 2026-01-01\nCurrency: EUR\n"
LAYOUT_TOTALS = "\nSous-total HT : 1 284,56 €\nTVA (20%) sur 1 234,56 € : 246,91 €\nTVA (0%) sur 50,00 € : 0,00 €\nTotal TTC : 1 531,47 €"


def layout_cases(directory):
    layouts = {
        "reordered": (
            "Montant HT | Description | Quantité | PU net HT\n"
            "1 234,56 € | Installation | 2 | 617,28 €\n50,00 € | Manuel | 2 | 25,00 €"
        ),
        "qte_pu": "Article | Qté | P.U. | Total HT\nInstallation | 2 | 617,28 | 1 234,56\nManuel | 2 | 25,00 | 50,00",
        "qty_unit": "Item | Qty | Unit price | Line total\nInstallation | 2 | 617.28 | 1234.56\nManuel | 2 | 25 | 50",
        "separate_discount": (
            "Description | Quantity | Net unit price | Net amount\n"
            "Installation | 2 | 617,28 | 1 234,56\nManuel | 2 | 25,00 | 50,00\n"
            "Remise 10 % déjà incluse dans les prix nets"
        ),
        "mixed_vat": "Description | Quantity | Net unit price | Net amount\nInstallation | 2 | 617,28 | 1 234,56\nManuel | 2 | 25 | 50",
        "inverse_european": "Description | Quantity | Net unit price | Net amount\nInstallation | 2 | 617,28 | 1.234,56\nManuel | 2 | 25,00 | 50,00",
        "inverse_english": "Description | Quantity | Net unit price | Net amount\nInstallation | 2 | 617.28 | 1,234.56\nManuel | 2 | 25.00 | 50.00",
        "currency_prefix": "Description | Quantity | Net unit price | Net amount\nInstallation | 2 | EUR 617,28 | €1 234,56\nManuel | 2 | EUR 25 | €50",
        "currency_suffix": "Description | Quantity | Net unit price | Net amount\nInstallation | 2 | 617,28 EUR | 1 234,56 EUR\nManuel | 2 | 25 EUR | 50 EUR",
    }
    for name, table in layouts.items():
        text = LAYOUT_HEADER + table + LAYOUT_TOTALS
        yield StressCase(f"layout/{name}", LAYOUT_REFERENCE, wrong_model(LAYOUT_REFERENCE), text=text)
    discounted = reference("Stress Services Ltd", "2026-01-01", 135, 135, [
        ("Installation", 2, 50, 100), ("Manuel", 2, 25, 50),
    ])
    yield StressCase(
        "layout/separate_discount_amount", discounted, wrong_model(discounted),
        text=LAYOUT_HEADER + "Description | Quantity | Net unit price | Net amount\n"
        "Installation | 2 | 50 | 100\nManuel | 2 | 25 | 50\nRemise commerciale : -15,00 €\n"
        "Sous-total HT : 135,00 €\nTVA (0%) : 0,00 €\nTotal TTC : 135,00 €",
    )
    import pymupdf
    path = directory / "two_pages.pdf"
    table_header = "Description | Quantity | Net unit price | Net amount"
    pages = [
        LAYOUT_HEADER + table_header + "\nInstallation | 2 | 617,28 | 1 234,56",
        table_header + "\nManuel | 2 | 25 | 50" + LAYOUT_TOTALS,
    ]
    with pymupdf.open() as pdf:
        for text in pages:
            pdf.new_page().insert_text((40, 40), text, fontsize=10)
        pdf.save(path)
    yield StressCase("layout/two_page_pdf", LAYOUT_REFERENCE, wrong_model(LAYOUT_REFERENCE), pdf=path)
