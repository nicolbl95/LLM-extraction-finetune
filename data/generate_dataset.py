"""Generate seeded synthetic French invoices with validated JSON targets."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

from schema import Invoice, LineItem

SEED = 42
NUM_EXAMPLES = 60
OUTPUT_PATH = Path(__file__).resolve().parent / "dataset.jsonl"
FORMATS = ("text_block", "raw_text", "email")
TVA_RATES = (0.055, 0.10, 0.20)
MONTHS_FR = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)
SUPPLIERS = (
    "Atelier Lumière SARL",
    "Bureau Plus SAS",
    "Techno Nord SA",
    "Boulangerie Dupont SARL",
    "Transports Rapides SAS",
    "Imprimerie Océan",
    "Cabinet Conseil Martin",
    "Garage Central SARL",
    "Fleurs & Co",
    "Informatique Services SAS",
    "Nettoyage Pro SARL",
    "Électro Sud SA",
)
PRODUCTS = (
    ("Ramette papier A4", 4.0, 9.0),
    ("Cartouche d'encre noire", 25.0, 60.0),
    ("Prestation de conseil", 80.0, 150.0),
    ("Licence logicielle annuelle", 120.0, 480.0),
    ("Livraison express", 15.0, 35.0),
    ("Maintenance préventive", 60.0, 110.0),
    ("Câble réseau 10 m", 6.0, 14.0),
    ("Nettoyage de locaux", 90.0, 200.0),
    ("Pain de campagne", 3.0, 5.5),
    ("Bouquet de fleurs", 25.0, 45.0),
    ("Heure de développement", 55.0, 95.0),
    ("Pneu hiver 205/55", 70.0, 130.0),
)


def money(value: float, french: bool) -> str:
    if french:
        return f"{value:,.2f}".replace(",", " ").replace(".", ",") + " €"
    return f"{value:.2f} EUR"


def format_date(d: date, rng: random.Random) -> str:
    style = rng.choice(("slash", "iso", "long"))
    if style == "slash":
        return d.strftime("%d/%m/%Y")
    if style == "iso":
        return d.isoformat()
    return f"{d.day} {MONTHS_FR[d.month - 1]} {d.year}"


def build_invoice(rng: random.Random) -> tuple[Invoice, dict]:
    items = []
    for description, low, high in rng.sample(PRODUCTS, rng.randint(1, 4)):
        quantity = rng.randint(1, 20)
        unit_price = round(rng.uniform(low, high), 2)
        items.append(
            LineItem(
                description=description,
                quantity=quantity,
                unit_price=unit_price,
                total_ht=round(quantity * unit_price, 2),
            )
        )
    total_ht = round(sum(item.total_ht for item in items), 2)
    rate = rng.choice(TVA_RATES)
    total_ttc = round(total_ht * (1 + rate), 2)
    invoice_date = date(2023, 1, 1) + timedelta(days=rng.randint(0, 900))
    invoice = Invoice(
        supplier_name=rng.choice(SUPPLIERS),
        invoice_date=invoice_date,
        total_ht=total_ht,
        total_ttc=total_ttc,
        line_items=items,
    )
    ctx = {
        "number": f"FA-{rng.randint(1000, 9999)}",
        "rate_pct": f"{rate * 100:g}",
        "tva": round(total_ttc - total_ht, 2),
        "french": rng.random() < 0.5,
        "date_text": format_date(invoice_date, rng),
    }
    return invoice, ctx


def render_text_block(invoice: Invoice, ctx: dict) -> str:
    f = ctx["french"]
    lines = [
        f"FACTURE N° {ctx['number']}",
        f"Fournisseur : {invoice.supplier_name}",
        f"Date : {ctx['date_text']}",
        "",
        "Désignation | Qté | PU HT | Total HT",
    ]
    lines += [
        f"{i.description} | {i.quantity:g} | {money(i.unit_price, f)} | {money(i.total_ht, f)}"
        for i in invoice.line_items
    ]
    lines += [
        "",
        f"Total HT : {money(invoice.total_ht, f)}",
        f"TVA ({ctx['rate_pct']} %) : {money(ctx['tva'], f)}",
        f"Total TTC : {money(invoice.total_ttc, f)}",
    ]
    return "\n".join(lines)


def render_raw_text(invoice: Invoice, ctx: dict) -> str:
    f = ctx["french"]
    items = " ; ".join(
        f"{i.description} x{i.quantity:g} PU {money(i.unit_price, f)} total {money(i.total_ht, f)}"
        for i in invoice.line_items
    )
    return (
        f"Facture {ctx['number']} de {invoice.supplier_name} du {ctx['date_text']}. "
        f"Articles : {items}. Total HT {money(invoice.total_ht, f)}, "
        f"TVA {ctx['rate_pct']} % {money(ctx['tva'], f)}, "
        f"Total TTC {money(invoice.total_ttc, f)}."
    )


def render_email(invoice: Invoice, ctx: dict) -> str:
    f = ctx["french"]
    details = "\n".join(
        f"- {i.description} : {i.quantity:g} x {money(i.unit_price, f)} = {money(i.total_ht, f)}"
        for i in invoice.line_items
    )
    return (
        f"Objet : Facture {ctx['number']} - {invoice.supplier_name}\n\n"
        "Bonjour,\n\n"
        f"Veuillez trouver ci-dessous le détail de notre facture n° {ctx['number']} "
        f"du {ctx['date_text']}.\n\n"
        f"{details}\n\n"
        f"Montant HT : {money(invoice.total_ht, f)}\n"
        f"TVA ({ctx['rate_pct']} %) : {money(ctx['tva'], f)}\n"
        f"Montant TTC : {money(invoice.total_ttc, f)}\n\n"
        f"Cordialement,\n{invoice.supplier_name}"
    )


RENDERERS = {
    "text_block": render_text_block,
    "raw_text": render_raw_text,
    "email": render_email,
}


def main() -> None:
    rng = random.Random(SEED)
    records: list[dict] = []
    seen_inputs: set[str] = set()
    while len(records) < NUM_EXAMPLES:
        fmt = FORMATS[len(records) % len(FORMATS)]
        invoice, ctx = build_invoice(rng)
        text = RENDERERS[fmt](invoice, ctx)
        if text in seen_inputs:
            continue
        seen_inputs.add(text)
        target = Invoice.model_validate(invoice.model_dump(mode="json")).model_dump(mode="json")
        records.append({"format": fmt, "input": text, "output": target})

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
