#!/usr/bin/env python3
"""Render the tracked blank invoice template offered as a user-facing example.

The template prints every expected label and an empty value area, and never a number:
the reader must therefore confirm nothing, which is the ``0 valeur fausse`` demo of the
interface. Re-run this script only when the layout of the blank example must change.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "data")]

from render_invoice import Document, render_png

BLANK_TEMPLATE = Document(
    title="FACTURE",
    seller=["Fournisseur :", "Adresse :", "Contact :"],
    customer=["Facturé à :", "Adresse :"],
    metadata=["Date de facture :", "Commande :", "Devise :"],
    headers=["Désignation", "Quantité", "Prix unit. HT", "Montant HT"],
    rows=[["", "", "", ""], ["", "", "", ""], ["", "", "", ""]],
    description_column=0,
    totals=[("Sous-total HT", ""), ("TVA (0%)", ""), ("Total TTC", "")],
    payment=["Échéance :", "Règlement :"],
    legal=["Modèle vierge à compléter : aucune valeur n'est imprimée dans ce document."],
    template=0, font="Helvetica", font_size=10.0, accent="#34495e", margin=34, gap=12,
)


def main() -> None:
    target = ROOT / "app/examples/modele-vierge.png"
    render_png(BLANK_TEMPLATE, target)
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
