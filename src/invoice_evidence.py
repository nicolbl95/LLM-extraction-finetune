"""Document-only evidence: positions, locale-aware decimals, table cells and taxes."""

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal, InvalidOperation


class UnreadableOCR(ValueError):
    """OCR ran but produced no usable text."""


def folded(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).casefold().split())


def parse_number(raw: str | Decimal | int, locale: str | None = None) -> Decimal:
    if isinstance(raw, bool):
        raise ValueError("Boolean is not a document number")
    if isinstance(raw, (Decimal, int)):
        value = Decimal(raw)
    else:
        text = re.sub(r"\s+", " ", raw.strip())
        parenthesized = text.startswith("(") and text.endswith(")")
        negative = parenthesized and "%" not in text
        if parenthesized:
            text = text[1:-1].strip()
        text = re.sub(r"^(?:EUR|GBP|USD|[€£$])\s*|\s*(?:EUR|GBP|USD|[€£$%])$", "", text).strip()
        if " " in text:
            if not re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+(?:[,.]\d+)?", text):
                raise ValueError(f"Ambiguous grouping: {raw!r}")
            text = text.replace(" ", "")
        if "," in text and "." in text:
            if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d+", text):
                text = text.replace(",", "")
            elif re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d+", text):
                text = text.replace(".", "").replace(",", ".")
            else:
                raise ValueError(f"Ambiguous separators: {raw!r}")
        elif "," in text:
            if locale in {"en-GB", "en-US"}:
                if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", text):
                    raise ValueError(f"Invalid thousands grouping for {locale}: {raw!r}")
                text = text.replace(",", "")
            else:
                if not re.fullmatch(r"[+-]?\d+,\d+", text):
                    raise ValueError(f"Ambiguous decimal: {raw!r}")
                if locale is None and re.fullmatch(r"[+-]?[1-9]\d{0,2},\d{3}", text):
                    raise ValueError(f"Locale required for {raw!r}")
                text = text.replace(",", ".")
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
            raise ValueError(f"Invalid document number: {raw!r}")
        try:
            value = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"Invalid decimal: {raw!r}") from exc
        if negative:
            value = -value
    if not value.is_finite():
        raise ValueError(f"Non-finite number: {raw!r}")
    if abs(value.adjusted()) > 1000 or len(value.as_tuple().digits) > 1000:
        raise ValueError(f"Number outside supported invoice precision: {raw!r}")
    return value


@dataclass(frozen=True)
class Token:
    text: str
    page: int
    box: tuple[float, float, float, float]
    confidence: float = 100


@dataclass(frozen=True)
class Number:
    raw: str
    value: Decimal
    page: int
    box: tuple[float, float, float, float]


@dataclass
class TextLine:
    text: str
    tokens: list[Token]
    page: int
    box: tuple[float, float, float, float]


@dataclass
class TableRow:
    description: str
    cells: dict[str, list[Number]]
    line: TextLine
    description_confident: bool = True


@dataclass(frozen=True)
class Tax:
    rate: Number | None
    amount: Number
    base: Number | None = None


@dataclass
class DocumentEvidence:
    text: str
    locale: str | None
    tokens: list[Token]
    lines: list[TextLine]
    rows: list[TableRow] = field(default_factory=list)
    totals: dict[str, list[Number]] = field(default_factory=dict)
    taxes: list[Tax] = field(default_factory=list)
    discounts: list[Number] = field(default_factory=list)
    numbers: list[Number] = field(default_factory=list)
    supplier: str | None = None
    supplier_raw: str | None = None
    invoice_number: str | None = None
    corroborated_identifiers: list[tuple[str, str]] = field(default_factory=list)
    invoice_date: str | None = None
    issues: list[str] = field(default_factory=list)
    available: bool = True
    source_kind: str = "text"
    ocr_cross_checked: bool = False
    quality_issues: list[str] = field(default_factory=list)
    ocr_calls: int = 0
    ocr_pages: int = 0
    ocr_elapsed_seconds: float = 0
    preprocessing_used: list[str] = field(default_factory=list)


class EvidenceText(str):
    evidence: DocumentEvidence

    def __new__(cls, evidence: DocumentEvidence):
        instance = super().__new__(cls, evidence.text)
        instance.evidence = evidence
        return instance


NUMERIC = re.compile(
    r"(?<![\w./-])(?:\(\s*)?(?:[€£$]\s*)?[+-]?\d+(?:[ \u00a0\u202f]\d{3})*"
    r"(?:[,.]\d+)*(?:\s*[€£$%])?(?:\s*\))?(?![\w./-])"
)
HEADERS = {
    "description": r"\b(?:description|product|designation|article|prestation|item(?: / service)?)\b",
    "quantity": r"\b(?:quantite|qte|quantity|units|qty)\b",
    "unit_price": r"(?:\bp\.u\.(?=\s|$)|\b(?:pu net(?: ht)?|prix unitaire(?: net)?(?: ht)?|net unit(?: price)?|unit price)\b)",
    "total_ht": r"\b(?:montant ht|net amount|line total|total ht)\b",
}
OPTIONAL_HEADERS = {
    "list_price": r"\b(?:pu catalogue|list (?:unit )?price)\b",
    "discount": r"\b(?:discount|remise)\b",
    "tax_rate": r"\b(?:vat|tva|tax)\s*%",
    "tax_amount": r"^(?:vat|tva|sales tax)$",
}
MONTHS = {
    word: index for index, words in enumerate((
        ("january", "janvier"), ("february", "fevrier"), ("march", "mars"),
        ("april", "avril"), ("may", "mai"), ("june", "juin"),
        ("july", "juillet"), ("august", "aout"), ("september", "septembre"),
        ("october", "octobre"), ("november", "novembre"), ("december", "decembre"),
    ), 1) for word in words
}


def invoice_number_from_text(text: str) -> str | None:
    match = re.search(r"\b(FR|GB|US)-(\d{4})-(\d{5})\b", text, re.I)
    if not match:
        return None
    return f"{match.group(1).upper()}-{match.group(2)}-{match.group(3)}"


def union_box(tokens: list[Token]) -> tuple[float, float, float, float]:
    return (
        min(t.box[0] for t in tokens), min(t.box[1] for t in tokens),
        max(t.box[2] for t in tokens), max(t.box[3] for t in tokens),
    )


def number_tokens(text: str, line: TextLine, locale: str | None) -> list[Number]:
    found = []
    for match in NUMERIC.finditer(text):
        raw = match.group().strip()
        try:
            value = parse_number(raw, locale)
        except ValueError:
            continue
        found.append(Number(raw, value, line.page, line.box))
    return found


def positioned_lines(tokens: list[Token]) -> list[TextLine]:
    bands: list[list[Token]] = []
    for token in sorted(tokens, key=lambda t: (t.page, (t.box[1] + t.box[3]) / 2, t.box[0])):
        middle = (token.box[1] + token.box[3]) / 2
        if bands:
            reference = bands[-1][0]
            ref_middle = (reference.box[1] + reference.box[3]) / 2
            tolerance = max(2, min(token.box[3] - token.box[1], reference.box[3] - reference.box[1]) * .6)
        if bands and token.page == reference.page and abs(middle - ref_middle) <= tolerance:
            bands[-1].append(token)
        else:
            bands.append([token])
    lines = []
    for band in bands:
        band.sort(key=lambda t: t.box[0])
        lines.append(TextLine(" ".join(t.text for t in band), band, band[0].page, union_box(band)))
    return lines


def from_tsv(data: dict, *, page: int = 0, locale: str | None = None) -> DocumentEvidence:
    tokens = []
    for index, text in enumerate(data["text"]):
        if not text.strip() or float(data["conf"][index]) < 0:
            continue
        left, top = float(data["left"][index]), float(data["top"][index])
        right, bottom = left + float(data["width"][index]), top + float(data["height"][index])
        tokens.append(Token(text, page, (left, top, right, bottom), float(data["conf"][index])))
    evidence = from_tokens(tokens, locale=locale)
    evidence.source_kind = "ocr"
    return evidence


def from_text(text: str, *, locale: str | None = None, page: int = 0) -> DocumentEvidence:
    lines = text.split(" ; ") if "\n" not in text and " ; " in text and "|" in text else text.splitlines()
    tokens = [
        Token(match.group(), page, (match.start(), index * 10, match.end(), index * 10 + 8))
        for index, line in enumerate(lines)
        for match in re.finditer(r"\S+", line)
    ]
    return from_tokens(tokens, locale=locale, raw_lines={(page, index * 10): line for index, line in enumerate(lines)})


def combine_pages(pages: list[DocumentEvidence], *, contains_ocr: bool = False) -> DocumentEvidence:
    """Merge selected evidence, never reparse raw OCR tokens past its quality gate."""
    result = DocumentEvidence("", None, [], [], source_kind="consensus" if contains_ocr else "pdf")
    result.ocr_cross_checked = contains_ocr
    suppliers, invoice_numbers, dates, locales = set(), set(), set(), set()
    texts = []
    for index, source in enumerate(pages):
        evidence = deepcopy(source)
        texts.append(evidence.text)
        if evidence.supplier:
            suppliers.add(evidence.supplier)
        if evidence.invoice_number:
            invoice_numbers.add(evidence.invoice_number)
        result.corroborated_identifiers.extend(
            identifier for identifier in evidence.corroborated_identifiers
            if identifier not in result.corroborated_identifiers
        )
        if evidence.invoice_date:
            dates.add(evidence.invoice_date)
        if evidence.locale:
            locales.add(evidence.locale)
        result.tokens.extend(replace(token, page=index) for token in evidence.tokens)
        for line in evidence.lines:
            line.page = index
            line.tokens = [replace(token, page=index) for token in line.tokens]
        result.lines.extend(evidence.lines)
        for row in evidence.rows:
            row.line.page = index
            row.cells = {key: [replace(number, page=index) for number in values] for key, values in row.cells.items()}
        result.rows.extend(evidence.rows)
        for key, values in evidence.totals.items():
            result.totals.setdefault(key, []).extend(replace(number, page=index) for number in values)
        result.numbers.extend(replace(number, page=index) for number in evidence.numbers)
        result.discounts.extend(replace(number, page=index) for number in evidence.discounts)
        result.taxes.extend(
            Tax(replace(tax.rate, page=index) if tax.rate else None, replace(tax.amount, page=index),
                replace(tax.base, page=index) if tax.base else None) for tax in evidence.taxes
        )
        result.quality_issues.extend(evidence.quality_issues)
        result.issues.extend(evidence.issues)
        result.ocr_calls += evidence.ocr_calls
        result.ocr_pages += evidence.ocr_pages
        result.ocr_elapsed_seconds += evidence.ocr_elapsed_seconds
        result.preprocessing_used.extend(
            family for family in evidence.preprocessing_used
            if family not in result.preprocessing_used
        )
    result.text = "\n\n".join(texts)
    result.supplier = next(iter(suppliers)) if len(suppliers) == 1 else None
    result.supplier_raw = result.supplier
    result.invoice_number = next(iter(invoice_numbers)) if len(invoice_numbers) == 1 else None
    result.invoice_date = next(iter(dates)) if len(dates) == 1 else None
    result.locale = next(iter(locales)) if len(locales) == 1 else None
    return result


def from_tokens(tokens: list[Token], *, locale: str | None = None,
                raw_lines: dict[tuple[int, float], str] | None = None) -> DocumentEvidence:
    lines = positioned_lines(tokens)
    if raw_lines:
        for line in lines:
            line.text = raw_lines.get((line.page, line.box[1]), line.text)
    text = "\n".join(line.text for line in lines)
    if locale is None:
        locale = "fr-FR" if re.search(r"\bEUR\b|€|\bFACTURE\b", text, re.I) else (
            "en-GB" if re.search(r"\bGBP\b|£", text) else
            "en-US" if re.search(r"\bUSD\b|\$", text) else None
        )
    evidence = DocumentEvidence(text, locale, tokens, lines)
    evidence.invoice_number = invoice_number_from_text(text)
    columns: dict[str, int] = {}
    starts: dict[str, float] = {}
    header_page = -1
    suppliers: dict[str, str] = {}
    dates: set[str] = set()
    for line in lines:
        normalized = folded(line.text)
        evidence.numbers.extend(number_tokens(line.text, line, locale))
        supplier = re.search(r"(?:supplier|fournisseur)\s*:\s*(.+?)(?:\s+(?:bill to|factur[ée] [aà])\s*:|$)", line.text, re.I)
        if supplier:
            name = supplier.group(1).strip()
            if name:
                key = re.sub(r"\b(?:ltd|limited)\.?$", "ltd", folded(name))
                key = re.sub(r"\bs[. ]*a[. ]*s\.?$", "sas", key)
                suppliers[key] = name
        if re.search(r"\b(?:invoice date|date de facture|date\s*:)\b", normalized) or normalized.startswith("date :") or normalized.startswith("date:"):
            date_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", line.text)
            local_date = re.search(r"\b(\d{2})[./](\d{2})[./](\d{4})\b", line.text)
            if date_match:
                dates.add(date_match.group())
            elif local_date:
                day, month, year = local_date.groups()
                if locale == "en-US":
                    day, month = month, day
                dates.add(f"{year}-{month}-{day}")
            else:
                day_first = re.search(r"\b(\d{1,2}) ([a-z]+) (\d{4})\b", normalized)
                month_first = re.search(r"\b([a-z]+) (\d{1,2}),? (\d{4})\b", normalized)
                if day_first and day_first[2] in MONTHS:
                    dates.add(f"{day_first[3]}-{MONTHS[day_first[2]]:02d}-{int(day_first[1]):02d}")
                elif month_first and month_first[1] in MONTHS:
                    dates.add(f"{month_first[3]}-{MONTHS[month_first[1]]:02d}-{int(month_first[2]):02d}")
        amounts = number_tokens(line.text, line, locale)
        if re.match(r"^(?:remise|discount|rabais)\b.*:", normalized) and not re.search(
            r"inclu|appliqu|already applied", normalized,
        ):
            money = [number for number in amounts if "%" not in number.raw]
            if len(money) == 1:
                evidence.discounts.append(money[0])
                continue
        tax_label = re.match(r"^(?:tva|vat|sales tax)\s*(?:\(|:|\d)", normalized)
        total_field = None
        if re.match(r"^(?:sous.total(?: ht)?|subtotal(?: \(net\))?|total excl\. tax|total ht|net total|net a payer ht)\s*:", normalized):
            total_field = "total_ht"
        elif re.match(r"^(?:net a payer ttc|invoice total|total ttc|grand total|total due)\s*:", normalized):
            total_field = "total_ttc"
        if tax_label or total_field:
            columns, starts = {}, {}
            if tax_label and amounts:
                rate = next((n for n in amounts if "%" in n.raw), None)
                money = [n for n in amounts if "%" not in n.raw]
                if money:
                    tax = Tax(rate, money[-1], money[0] if len(money) > 1 else None)
                    if tax not in evidence.taxes:
                        evidence.taxes.append(tax)
            if total_field and amounts:
                evidence.totals.setdefault(total_field, []).append(amounts[-1])
            continue
        if "|" in line.text:
            inline = line.text.split("|")
            inline_values = {}
            for cell in inline[1:]:
                label, separator, content = cell.partition(":")
                if not separator:
                    continue
                for key in ("quantity", "unit_price", "total_ht"):
                    if re.fullmatch(HEADERS[key], folded(label)):
                        inline_values[key] = number_tokens(content, line, locale)
            if len(inline_values) >= 2:
                evidence.rows.append(TableRow(inline[0].strip(), inline_values, line))
                continue
        matches = {key: re.search(pattern, normalized) for key, pattern in HEADERS.items()}
        if all(matches.values()):
            header_page = line.page
            if "|" in line.text:
                cells = line.text.split("|")
                columns = {
                    key: index for key, pattern in {**HEADERS, **OPTIONAL_HEADERS}.items()
                    for index, cell in enumerate(cells) if re.search(pattern, folded(cell))
                }
                starts = {}
            else:
                columns = {}
                starts = {}
                for key, pattern in HEADERS.items():
                    for index in range(len(line.tokens)):
                        suffix = " ".join(token.text for token in line.tokens[index:])
                        if re.match(pattern, folded(suffix)):
                            starts[key] = line.tokens[index].box[0]
                            break
                for token in line.tokens:
                    if folded(token.text) in {"discount", "remise"}:
                        starts["discount"] = token.box[0]
                        break
                for key, pattern in OPTIONAL_HEADERS.items():
                    for index in range(len(line.tokens)):
                        suffix = " ".join(token.text for token in line.tokens[index:])
                        if re.match(pattern, folded(suffix)):
                            starts[key] = line.tokens[index].box[0]
                            break
            continue
        if line.page != header_page or not (columns or starts):
            continue
        cell_texts: dict[str, str] = {}
        cell_lines: dict[str, TextLine] = {}
        if columns and "|" in line.text:
            cells = line.text.split("|")
            if len(cells) > max(columns.values()):
                cell_texts = {key: cells[index].strip() for key, index in columns.items()}
        elif starts:
            ordered = sorted(starts, key=starts.__getitem__)
            for index, key in enumerate(ordered):
                left = starts[key] - 4
                right = starts[ordered[index + 1]] - 4 if index + 1 < len(ordered) else float("inf")
                cell_tokens = [token for token in line.tokens if left <= token.box[0] < right]
                if cell_tokens:
                    cell_texts[key] = " ".join(t.text for t in cell_tokens)
                    cell_lines[key] = TextLine(cell_texts[key], cell_tokens, line.page, union_box(cell_tokens))
        if not cell_texts:
            continue
        values = {
            key: [
                number for number in number_tokens(cell_texts.get(key, ""), cell_lines.get(key, line), locale)
                if key in OPTIONAL_HEADERS or "%" not in number.raw
            ]
            for key in ("quantity", "unit_price", "total_ht", *OPTIONAL_HEADERS)
        }
        if sum(bool(values[key]) for key in ("quantity", "unit_price", "total_ht")) >= 2:
            low_confidence = any(t.confidence < 60 for t in line.tokens)
            if low_confidence:
                evidence.issues.append(f"Ligne OCR à faible confiance : {line.text}")
            description_line = cell_lines.get("description", line)
            evidence.rows.append(TableRow(
                cell_texts.get("description", ""), values, line,
                all(token.confidence >= 60 for token in description_line.tokens),
            ))
        elif evidence.rows and starts and cell_texts.get("description") and not any(values.values()):
            last = evidence.rows[-1]
            if line.page == last.line.page and line.box[1] - last.line.box[3] < 20:
                last.description += " " + cell_texts["description"]
    if len(suppliers) == 1:
        supplier = next(iter(suppliers.values()))
        evidence.supplier_raw = supplier
        if supplier.upper() in {"ERROR", "INVALID"} or re.search(r"\b(?:LId|L1d|LtI|Lt1)\.?$", supplier):
            evidence.issues.append("supplier_name : non vérifié — nom ou suffixe OCR ambigu, aucune substitution.")
        else:
            evidence.supplier = re.sub(r"\b(?:Ltd|Limited)\.?$", "Ltd", " ".join(supplier.split()), flags=re.IGNORECASE)
    elif suppliers:
        evidence.issues.append("supplier_name : plusieurs fournisseurs incompatibles dans le document.")
    if len(dates) == 1:
        candidate = next(iter(dates))
        try:
            date.fromisoformat(candidate)
            evidence.invoice_date = candidate
        except ValueError:
            evidence.issues.append(f"invoice_date : date documentaire invalide {candidate!r}.")
    elif dates:
        evidence.issues.append("invoice_date : plusieurs dates de facture incompatibles dans le document.")
    return evidence
