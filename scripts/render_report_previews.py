#!/usr/bin/env python3
"""Render the exact French report Markdown; no model, GPU, browser or CDN."""

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from invoice_evidence import from_text
from invoice_grounding import read_model, reconcile
from invoice_validation import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/report-previews")
    args = parser.parse_args()
    from markdown_it import MarkdownIt
    from app.main import deployed_revision
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "INVOICE GB-2026-00001\nSupplier: Exemple Services Ltd\nCurrency: GBP\n"
        "Invoice date: 2026-01-01\nDescription | Quantity | Net unit price | Net amount\n"
        "Service | 2 | 10 | 20\nSubtotal: 20\nVAT (20%): 4\nInvoice total: 24\n"
        "Due date: 2026-02-15\nPayment terms: NET 45"
    )
    model = read_model(
        '{"supplier_name":"Exemple Services Ltd","invoice_date":"2026-01-01",'
        '"total_ht":20,"total_ttc":24,"line_items":[{"description":"Service","quantity":2,"unit_price":10,"total_ht":20}]}'
    )
    for name, source in (("sans-anomalie", text), ("erreur-imprimee", text.replace("10 | 20", "10 | 19")),
                         ("ocr-insuffisant", "Texte OCR brouillé")):
        evidence = from_text(source)
        if name == "ocr-insuffisant":
            evidence.source_kind = "ocr"
            evidence.quality_issues = ["Confiance OCR moyenne insuffisante et tableau non reconnu."]
        result = reconcile(evidence, [model])
        suggestions = {"line_items.0.quantity": 54, "line_items.0.unit_price": "77,68",
                       "line_items.0.total_ht": "4210,32"} if name == "ocr-insuffisant" else {}
        markdown = render_report(
            evidence, result.validation, result.statuses, [], insufficient=bool(evidence.quality_issues),
            suggestions=suggestions,
        )
        technical = "\n\n".join([*result.warnings, *result.report()])
        if suggestions:
            technical += "\n\nLectures modèle simulées non fiables (hors JSON) : " + json.dumps(suggestions, ensure_ascii=False)
        version = deployed_revision()
        rendered = MarkdownIt("commonmark", {"html": False}).render(markdown)
        page = (
            '<!doctype html><html lang="fr"><meta charset="utf-8">'
            f"<title>Rapport — {html.escape(name)}</title>"
            "<style>body{font:16px/1.6 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;"
            "color:#172b4d;background:#f5f7fa}main{background:white;padding:2rem;border-radius:12px}"
            "h2,h3{color:#17365d}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef2f7;padding:1rem}"
            "details{border:1px solid #cbd5e1;padding:1rem;margin-top:1rem}footer{font-size:.85rem}"
            "</style><main><p><em>Rendu HTML statique du rapport exact ; document de démonstration, aucun modèle appelé.</em></p>"
            + rendered + "<h3>JSON standardisé</h3><pre>"
            + html.escape(json.dumps(result.payload, ensure_ascii=False, indent=2))
            + '</pre><details><summary>Détails techniques</summary><pre>' + html.escape(technical)
            + "</pre><h4>validation.flags — hors schéma scoré</h4><pre>"
            + html.escape(json.dumps(result.validation.export(), ensure_ascii=False, indent=2))
            + "</pre></details><footer>Version : " + html.escape(version)
            + " · Preuves : texte de démonstration (pas d'OCR réel pour ces rendus)<br>"
            + "Limite connue : les images floues ou inclinées peuvent donner plus de champs non vérifiés."
            + "</footer></main></html>"
        )
        (args.output_dir / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")
        (args.output_dir / f"{name}.html").write_text(page, encoding="utf-8")
        (args.output_dir / f"{name}.json").write_text(json.dumps({
            "model_called": False, "evidence": "handwritten demo text", "payload": result.payload,
            "validation": result.validation.export(), "report": markdown, "statuses": result.statuses,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(args.output_dir / f"{name}.html")


if __name__ == "__main__":
    main()
