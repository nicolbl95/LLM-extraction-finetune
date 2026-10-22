"""Generate 600 reproducible FR/UK/US invoice texts, labels and matching PNGs."""

import json
import random
from collections import Counter
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from itertools import product
from pathlib import Path

from render_invoice import Document, STYLES, render_png
from schema import Invoice, LineItem

SEED = 42
NUM_EXAMPLES = 600
MIN_ITEMS = 1
MAX_ITEMS = 12
DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "dataset.jsonl"
FORMATS = ("text_block", "raw_text", "email")
LOCALES = ("fr-FR", "en-GB", "en-US")
CURRENCIES = {"fr-FR": "EUR", "en-GB": "GBP", "en-US": "USD"}
SYMBOLS = {"EUR": "€", "GBP": "£", "USD": "$"}
MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)
MONTHS_EN = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
# French/English description, minimum/maximum unit price, fractional quantity, books.
PRODUCTS = (
    ("Ramette papier A4", "Office paper pack", 4, 9, False, False),
    ("Cartouche d'encre", "Printer ink cartridge", 25, 60, False, False),
    ("Conseil informatique", "IT consulting", 80, 150, True, False),
    ("Licence logicielle", "Software licence", 120, 480, False, False),
    ("Abonnement cloud", "Cloud subscription", 15, 60, False, False),
    ("Livraison express", "Express delivery", 15, 35, False, False),
    ("Maintenance informatique", "IT maintenance", 60, 110, True, False),
    ("Câble réseau", "Network cable", 6, 14, False, False),
    ("Nettoyage de bureaux", "Office cleaning", 90, 200, True, False),
    ("Développement web", "Web development", 55, 95, True, False),
    ("Écran 27 pouces", "27 inch monitor", 150, 290, False, False),
    ("Clavier sans fil", "Wireless keyboard", 18, 45, False, False),
    ("Disque SSD 1 To", "1 TB SSD drive", 60, 110, False, False),
    ("Création de logo", "Logo design", 200, 900, True, False),
    ("Impression de brochures", "Brochure printing", 8, 25, False, False),
    ("Manuel technique imprimé", "Printed technical handbook", 15, 55, False, True),
)
SUPPLIERS = {
    "fr-FR": ("Atelier Lumière", "Bureau du Parc", "Services Horizon", "Studio Océan"),
    "en-GB": ("Thames Office", "Bristol Digital", "Northbridge Services", "Oak Studio"),
    "en-US": ("Cedar Office", "Pacific Digital", "Hudson Services", "Maple Studio"),
}
ADDRESSES = {
    "fr-FR": ("rue des Artisans, 69003 Lyon, France", "avenue du Port, 44000 Nantes, France"),
    "en-GB": ("King Street, Bristol BS1 4EF, United Kingdom", "High Street, London SW1A 1AA, United Kingdom"),
    "en-US": (
        "Market Street, San Francisco, CA 94103, United States",
        "Broadway, New York, NY 10013, United States",
        "Oak Street, Portland, OR 97205, United States",
    ),
}


def rounded(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def money(value: float, locale: str, code: bool = False) -> str:
    currency = CURRENCIES[locale]
    amount = f"{value:,.2f}"
    if locale == "fr-FR":
        amount = amount.replace(",", " ").replace(".", ",")
        return f"{amount} {currency if code else SYMBOLS[currency]}"
    return f"{currency} {amount}" if code else f"{SYMBOLS[currency]}{amount}"


def format_date(value: date, locale: str, style: str) -> str:
    if style == "iso":
        return value.isoformat()
    if style == "long":
        months = MONTHS_FR if locale == "fr-FR" else MONTHS_EN
        if locale == "en-US":
            return f"{months[value.month - 1]} {value.day}, {value.year}"
        return f"{value.day} {months[value.month - 1]} {value.year}"
    separator = "." if style == "dots" else "/"
    pattern = "%m/%d/%Y" if locale == "en-US" else "%d/%m/%Y"
    return value.strftime(pattern).replace("/", separator)


def make_iban(rng: random.Random) -> str:
    bank, branch, account = rng.randint(10000, 99999), rng.randint(10000, 99999), rng.randint(10**10, 10**11 - 1)
    rib = 97 - (89 * bank + 15 * branch + 3 * account) % 97
    bban = f"{bank:05d}{branch:05d}{account:011d}{rib:02d}"
    iban = f"FR{98 - int(bban + '152700') % 97:02d}{bban}"
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))


def build_invoice(rng: random.Random, locale: str, index: int) -> tuple[Invoice, dict]:
    french = locale == "fr-FR"
    address_index = rng.randrange(len(ADDRESSES[locale]))
    standard_rate = Decimal("20")
    if locale == "en-US":
        standard_rate = (Decimal("7.25"), Decimal("8.875"), Decimal("0"))[address_index]
    items, details = [], []
    discount = rng.choice((0, 0, 5, 10, 15))
    for fr, en, low, high, fractional, book in rng.sample(PRODUCTS, rng.randint(MIN_ITEMS, MAX_ITEMS)):
        qty = Decimal(rng.randint(1, 80)) / 10 if fractional else Decimal(rng.randint(1, 30))
        price = Decimal(rng.randint(low * 100, high * 100)) / 100
        net_price = rounded(price * (100 - discount) / 100)
        net = rounded(qty * net_price)
        rate = Decimal("5.5") if book and french else Decimal("0") if book and locale == "en-GB" else standard_rate
        tax = rounded(net * rate / 100)
        items.append(LineItem(
            description=fr if french else en, quantity=float(qty),
            unit_price=float(net_price), total_ht=float(net),
        ))
        details.append({
            "list_price": float(price), "discount_percent": discount,
            "tax_rate": float(rate), "tax": float(tax),
        })
    total_ht = sum((Decimal(str(item.total_ht)) for item in items), Decimal(0))
    tax_total = sum((Decimal(str(detail["tax"])) for detail in details), Decimal(0))
    issued = date(2023, 1, 1) + timedelta(days=rng.randint(0, 1200))
    term_days = rng.choice((0, 14, 30, 45, 60))
    date_style = rng.choice(("slash", "iso", "long", "dots"))
    supplier = rng.choice(SUPPLIERS[locale])
    suffix = rng.choice(("SARL", "SAS", "EURL")) if french else "Ltd" if locale == "en-GB" else "LLC"
    invoice = Invoice(
        supplier_name=f"{supplier} {suffix}", invoice_date=issued,
        total_ht=float(total_ht), total_ttc=float(total_ht + tax_total), line_items=items,
    )
    ctx = {
        "locale": locale, "currency": CURRENCIES[locale],
        "number": f"{locale[-2:]}-{issued.year}-{index + 1:05d}",
        "date_style": date_style, "date_text": format_date(issued, locale, date_style),
        "due_text": format_date(issued + timedelta(days=term_days), locale, date_style),
        "address": f"{rng.randint(1, 199)} {ADDRESSES[locale][address_index]}",
        "customer": rng.choice(("Maison Rivage SAS", "Groupe Aurore SARL", "Studio Prairie EURL")) if french
        else rng.choice(("River Trading", "Atlas Partners", "Summit Design")) + (" Ltd" if locale == "en-GB" else " LLC"),
        "customer_address": f"{rng.randint(200, 399)} {rng.choice(ADDRESSES[locale])}",
        "email": f"billing{index + 1}@example.invalid",
        "term_days": term_days, "details": details, "tax_total": float(tax_total),
        "code_money": rng.choice((True, False)),
        "reference": f"PO-{rng.randint(10000, 99999)}",
    }
    if french:
        siren = rng.randint(100_000_000, 999_999_999)
        ctx["registration"] = f"SIRET : {siren}{rng.randint(10000, 99999)} | TVA : FR{(12 + 3 * (siren % 97)) % 97:02d}{siren}"
        ctx["bank"] = f"Virement : IBAN {make_iban(rng)} | BIC BNPAFRPP"
        ctx["legal"] = "Pénalités de retard : taux légal majoré. Indemnité de recouvrement : 40 EUR."
    elif locale == "en-GB":
        ctx["registration"] = f"Company no. {rng.randint(10000000, 99999999)} | VAT ID GB{rng.randint(100000000, 999999999)}"
        ctx["bank"] = f"Bank transfer | Sort code {rng.randint(10, 99)}-{rng.randint(10, 99)}-{rng.randint(10, 99)} | Account {rng.randint(10000000, 99999999)}"
        ctx["legal"] = "Late payment interest may apply under the agreed terms. Printed books are zero-rated."
    else:
        ctx["registration"] = f"EIN {rng.randint(10, 99)}-{rng.randint(1000000, 9999999)}"
        ctx["bank"] = f"ACH / bank transfer | Account ending {rng.randint(1000, 9999)} | Reference {ctx['number']}"
        ctx["legal"] = "Sales tax calculated for the seller's state; no interstate or local surtax simulated."
    return invoice, ctx


def build_document(invoice: Invoice, ctx: dict, template: int, rng: random.Random) -> Document:
    locale = ctx["locale"]
    fr = locale == "fr-FR"

    def amount(value: float) -> str:
        return money(value, locale, ctx["code_money"])

    labels = {
        "description": rng.choice(("Désignation", "Description", "Prestation")) if fr else rng.choice(("Description", "Item / service", "Product")),
        "quantity": rng.choice(("Qté", "Quantité")) if fr else rng.choice(("Qty", "Quantity", "Units")),
        "unit_price": "PU net HT" if fr else "Net unit price",
        "total_ht": "Montant HT" if fr else "Net amount",
        "list_price": "PU catalogue" if fr else "List price",
        "discount": "Remise" if fr else "Discount",
        "tax_rate": "TVA %" if fr else "VAT %" if locale == "en-GB" else "Tax %",
        "tax": "TVA" if fr else "VAT" if locale == "en-GB" else "Sales tax",
        "gross": "Total TTC" if fr else "Gross amount",
    }
    # Column orders reflect the nine reference structures, not their artwork.
    columns = (
        ("description", "unit_price", "quantity", "total_ht"),
        ("quantity", "description", "unit_price", "total_ht"),
        ("description", "quantity", "unit_price", "tax_rate", "tax", "gross"),
        ("description", "unit_price", "quantity", "tax_rate", "total_ht"),
        ("quantity", "description", "unit_price", "tax_rate", "total_ht", "tax"),
        ("description", "unit_price", "quantity", "tax_rate", "total_ht"),
        ("description", "quantity", "unit_price", "total_ht"),
        ("description", "list_price", "discount", "unit_price", "quantity", "total_ht"),
        ("description", "quantity", "unit_price", "tax_rate", "tax", "gross"),
    )[template]
    # Always expose the discount and net unit price, including on tax-focused layouts.
    if ctx["details"][0]["discount_percent"] and "discount" not in columns:
        columns = (*columns, "discount")
    rows = []
    for item, detail in zip(invoice.line_items, ctx["details"]):
        values = {
            "description": item.description,
            "quantity": f"{item.quantity:g}".replace(".", ",") if fr else f"{item.quantity:g}",
            "unit_price": amount(item.unit_price), "total_ht": amount(item.total_ht),
            "list_price": amount(detail["list_price"]),
            "discount": f"{detail['discount_percent']}%",
            "tax_rate": f"{detail['tax_rate']:g}%",
            "tax": amount(detail["tax"]), "gross": amount(item.total_ht + detail["tax"]),
        }
        rows.append([values[column] for column in columns])
    title = ("FACTURE" if fr else "INVOICE")
    if template == 8:
        title = "FACTURE PRO FORMA" if fr else "PRO FORMA INVOICE"
    elif template in (3, 5, 6) and not fr:
        title = "TAX INVOICE"
    totals = [(rng.choice(("Total HT", "Sous-total HT", "Net à payer HT")) if fr
               else rng.choice(("Subtotal (net)", "Total excl. tax", "Net total")), amount(invoice.total_ht))]
    for rate in sorted({d["tax_rate"] for d in ctx["details"]}):
        tax = sum((Decimal(str(d["tax"])) for d in ctx["details"] if d["tax_rate"] == rate), Decimal(0))
        totals.append((f"{labels['tax']} ({rate:g}%)", amount(float(tax))))
    totals.append((rng.choice(("Total TTC", "Net à payer TTC")) if fr
                   else rng.choice(("Invoice total", "Total due", "Grand total")), amount(invoice.total_ttc)))
    payment = [
        f"{'Échéance' if fr else 'Due date'} : {ctx['due_text']}",
        (f"Paiement sous {ctx['term_days']} jours" if fr else f"Payment terms: NET {ctx['term_days']}")
        if ctx["term_days"] else ("Paiement à réception" if fr else "Payment due on receipt"),
        ctx["bank"],
    ]
    if ctx["details"][0]["discount_percent"]:
        payment.append(
            "Remise déjà appliquée au PU net HT ; montants arrondis par ligne."
            if fr else "Discount already applied to net unit prices; amounts rounded per line."
        )
    return Document(
        title=f"{title} {ctx['number']}",
        seller=[("Fournisseur : " if fr else "Supplier: ") + invoice.supplier_name, ctx["address"], ctx["email"]],
        customer=[("Facturé à : " if fr else "Bill to: ") + ctx["customer"], ctx["customer_address"]],
        metadata=[
            f"{'Date de facture' if fr else 'Invoice date'} : {ctx['date_text']}",
            f"{'Commande' if fr else 'Order reference'} : {ctx['reference']}",
            f"{'Devise' if fr else 'Currency'} : {ctx['currency']}",
        ],
        headers=[labels[column] for column in columns], rows=rows,
        description_column=columns.index("description"),
        totals=totals, payment=payment, legal=[ctx["registration"], ctx["legal"]],
        template=template, font=rng.choice(("Helvetica", "Times-Roman", "Courier")),
        font_size=rng.choice((8.0, 8.5, 9.0)), accent=rng.choice(STYLES[template]),
        margin=rng.choice((28, 34, 40)), gap=rng.choice((8, 12, 16)),
    )


def render_text(document: Document, fmt: str, locale: str) -> str:
    sections = document.text_sections()
    if fmt == "raw_text":
        return " ; ".join(section.replace("\n", " ; ") for section in sections)
    body = "\n\n".join(sections)
    if fmt == "email":
        fr = locale == "fr-FR"
        return (
            f"{'Objet' if fr else 'Subject'}: {document.title}\n\n"
            f"{'Bonjour, voici le détail de votre facture.' if fr else 'Hello, please find your invoice details below.'}\n\n"
            f"{body}\n\n{'Cordialement' if fr else 'Kind regards'},\n{document.seller[0]}"
        )
    return body


def validate_record(record: dict) -> None:
    invoice = Invoice.model_validate(record["output"])
    details = record["line_details"]
    if record["locale"] not in LOCALES or record["currency"] != CURRENCIES[record["locale"]]:
        raise ValueError("Locale/currency mismatch")
    if not MIN_ITEMS <= len(invoice.line_items) <= MAX_ITEMS or len(details) != len(invoice.line_items):
        raise ValueError("Invalid number of line items")
    for item, detail in zip(invoice.line_items, details):
        price = rounded(Decimal(str(detail["list_price"])) * (100 - Decimal(str(detail["discount_percent"]))) / 100)
        net = rounded(Decimal(str(item.quantity)) * price)
        tax = rounded(net * Decimal(str(detail["tax_rate"])) / 100)
        if price != Decimal(str(item.unit_price)) or net != Decimal(str(item.total_ht)) or tax != Decimal(str(detail["tax"])):
            raise ValueError("Inconsistent line price, discount or tax")
    net_total = sum((Decimal(str(item.total_ht)) for item in invoice.line_items), Decimal(0))
    tax_total = sum((Decimal(str(d["tax"])) for d in details), Decimal(0))
    if net_total != Decimal(str(invoice.total_ht)) or net_total + tax_total != Decimal(str(invoice.total_ttc)):
        raise ValueError("Inconsistent net/gross totals")


def generate_records(seed: int = SEED) -> list[tuple[dict, Document]]:
    rng = random.Random(seed)
    visual_options = {
        template: list(product(
            ("Helvetica", "Times-Roman", "Courier"), (8.0, 8.5, 9.0),
            STYLES[template], (28, 34, 40), (8, 12, 16),
        ))
        for template in range(9)
    }
    for options in visual_options.values():
        rng.shuffle(options)
    records = []
    for index in range(NUM_EXAMPLES):
        locale = LOCALES[index % 3]
        template = (index // 3) % 9
        fmt = FORMATS[(index // 27 + index % 3) % 3]
        invoice, ctx = build_invoice(rng, locale, index)
        document = build_document(invoice, ctx, template, rng)
        # Sample without replacement so all 600 visual profiles are distinct.
        document.font, document.font_size, document.accent, document.margin, document.gap = visual_options[template].pop()
        record = {
            "id": f"invoice-{index + 1:04d}", "dataset_version": "international-v3",
            "format": fmt, "locale": locale, "currency": ctx["currency"],
            "template_id": template + 1, "template_reference": f"data/templates/{template + 1}.png",
            "date_style": ctx["date_style"],
            "visual": {"font": document.font, "font_size": document.font_size,
                       "accent": document.accent, "margin": document.margin, "gap": document.gap},
            "line_details": ctx["details"],
            "image_path": f"data/invoices/invoice-{index + 1:04d}.png",
            "input": render_text(document, fmt, locale),
            "output": invoice.model_dump(mode="json"),
        }
        validate_record(record)
        records.append((record, document))
    return records


def main() -> None:
    records = generate_records()
    if len({record["input"] for record, _ in records}) != NUM_EXAMPLES:
        raise ValueError("Duplicate invoice inputs")
    for record, document in records:
        render_png(document, DATA_DIR.parent / record["image_path"])
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for record, _ in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} invoices and matching PNGs")
    print(dict(Counter(record["locale"] for record, _ in records)))
    print(dict(Counter(record["format"] for record, _ in records)))


if __name__ == "__main__":
    main()
