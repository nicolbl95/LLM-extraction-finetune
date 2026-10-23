from datetime import date
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from invoice_evidence import from_text
from invoice_grounding import read_model, reconcile
from invoice_validation import document_validation, render_report


BASE = (
    "Supplier: Équinoxe Ltd\nInvoice date: 2026-01-01\nCurrency: EUR\n"
    "Description | Quantity | Net unit price | Net amount\nService | 2 | 10 | 20\n"
    "Subtotal: 20\nVAT (20%): 4\nInvoice total: 24\nDue date: 2026-02-15\nPayment terms: NET 45"
)

POSITIVE = [
    ("line_arithmetic", BASE.replace("Service | 2 | 10 | 20", "Service | 2 | 10 | 19"), "Ligne 1", "19"),
    ("subtotal", BASE.replace("Subtotal: 20", "Subtotal: 21"), "Sous-total", "21"),
    ("tax_total", BASE.replace("Invoice total: 24", "Invoice total: 25"), "Total TTC", "25"),
    ("payment_terms", BASE.replace("2026-02-15", "2026-02-16"), "Échéance", "2026-02-16"),
    ("name_variant", BASE + "\nSupplier: Équinoxes Ltd", "Nom du fournisseur", "Équinoxes"),
    ("identifier_iban", BASE + "\nIBAN FR1420041010050500013M02607", "IBAN", "FR1420041010050500013M02607"),
    ("duplicate_line", BASE.replace("Subtotal:", "Service | 2 | 10 | 20\nSubtotal:"), "Ligne 2", "Service"),
]


@pytest.mark.parametrize("code,text,location,printed", POSITIVE, ids=[case[0] for case in POSITIVE])
def test_true_positive_report_and_flags(code, text, location, printed):
    evidence = from_text(text)
    validation = document_validation(evidence, today=date(2026, 10, 23))
    flag = next(flag for flag in validation.flags if flag.code == code)
    assert flag.severity in {"erreur", "avertissement"}
    assert location in flag.location
    report = render_report(evidence, validation, {}, [])
    assert location in report
    assert printed in report
    assert "anomalie" in report
    assert code in {entry["code"] for entry in validation.export()["flags"]}


NEGATIVE = [
    ("limited", BASE + "\nSupplier: Équinoxe Limited"),
    ("sas", BASE.replace("Équinoxe Ltd", "Équinoxe SAS") + "\nSupplier: Équinoxe S.A.S."),
    ("case", BASE + "\nSupplier: ÉQUINOXE LTD"),
    ("accents", BASE + "\nSupplier: Equinoxe Ltd"),
    ("rounding", BASE.replace("Service | 2 | 10 | 20", "Service | 0.5 | 0.05 | 0.03")
     .replace("Subtotal: 20", "Subtotal: 0.03").replace("VAT (20%): 4", "VAT (0%): 0")
     .replace("Invoice total: 24", "Invoice total: 0.03")),
    ("net_discount", BASE + "\nRemise 10 % déjà incluse dans les prix nets"),
    ("zero_tax", BASE.replace("VAT (20%): 4", "VAT (0%): 0").replace("Invoice total: 24", "Invoice total: 20")),
    ("mixed_tax", BASE.replace("Subtotal:", "Book | 2 | 5 | 10\nSubtotal:")
     .replace("Subtotal: 20", "Subtotal: 30").replace("VAT (20%): 4", "VAT (20%): 4\nVAT (0%): 0")
     .replace("Invoice total: 24", "Invoice total: 34")),
]


@pytest.mark.parametrize("name,text", NEGATIVE, ids=[case[0] for case in NEGATIVE])
def test_true_negative_report_does_not_claim_a_document_error(name, text):
    evidence = from_text(text)
    validation = document_validation(evidence, today=date(2026, 10, 23))
    assert not [flag for flag in validation.flags if flag.severity != "information"], name
    report = render_report(evidence, validation, {"total_ht": "valide"}, [])
    assert "Aucune anomalie" in report
    assert "Erreur —" not in report
    assert "Avertissement —" not in report
    assert all(flag["severity"] == "information" for flag in validation.export()["flags"])


def test_native_printed_error_is_not_hidden_by_ocr_quality_gate():
    evidence = from_text(POSITIVE[0][1])
    result = reconcile(evidence, [read_model('{"line_items":[]}')])
    assert result.payload["line_items"][0]["total_ht"] == 19
    assert any(flag.code == "line_arithmetic" for flag in result.validation.flags)
    assert evidence.quality_issues == []
