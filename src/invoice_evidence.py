"""Document-only evidence: positions, locale-aware decimals, table cells and taxes."""

import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal, InvalidOperation

from ocr_config import OCR_CONFIG


class UnreadableOCR(ValueError):
    """OCR ran but produced no usable text."""


def folded(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return " ".join(
        "".join(c for c in text if not unicodedata.combining(c))
        .casefold().replace("œ", "oe").replace("æ", "ae").split()
    )


LABEL_PREFIX = " \t:;,.‘’'\"`´£€$–—-"


def normalized_label(text: str) -> str:
    return folded(text).lstrip(LABEL_PREFIX)


TOTAL_HT_LABEL = re.compile(
    r"^(?:sous.total(?: ht)?|sub\s*total(?: \(net\))?|total excl\. tax|total ht|"
    r"total hors taxes?|total avant taxes?|total\s*\(\s*ht\s*\)|net total|net a payer ht)\s*:?\s*"
)
TOTAL_TTC_LABEL = re.compile(
    r"^(?:net a{1,2} payer ttc|invoice total|total ttc|grand total|total general|total due|"
    r"total tva comprises?|total toutes taxes comprises?|amount payable|"
    r"total\s*\(\s*ttc\s*\))\s*:?\s*"
)
INVOICE_TOTAL_LABELS = {"total_ht": TOTAL_HT_LABEL, "total_ttc": TOTAL_TTC_LABEL}
UNQUALIFIED_TOTAL_LABEL = re.compile(r"^total(?:\s*:\s*|\s+)")
TOTAL_QUALIFIER = re.compile(r"\b(?:ht|ttc|vat|tva)\b")


def total_label_field(text: str) -> str | None:
    """Which invoice total a line declares in its own label, or None for any other line.

    Only explicit wording decides: bare "total", "montant HT/TTC" and table column labels stay
    unqualified and must be resolved by structure or arithmetic, never by the label alone.
    """
    label = normalized_label(text)
    for key, pattern in INVOICE_TOTAL_LABELS.items():
        if pattern.match(label):
            return key
    if UNQUALIFIED_TOTAL_LABEL.match(label) and not TOTAL_QUALIFIER.search(label):
        return "total_candidate"
    return None


LINE_TOTAL_HEADER = re.compile(r"\b(?:amount|total|line total|amount due)\b")
LINE_TOTAL_EXCLUDED = re.compile(
    r"\b(?:gross|brut|ttc|subtotal|tax|vat|tva|discount|remise)\b"
)

# A tax line declares itself in its own label: TVA/VAT in French and English documents, and the Canadian
# acronyms printed in exactly the same position (TPS 13,42 $). The delimiter after the label is what separates
# a tax line from prose that merely mentions the acronym.
TAX_LINE_LABEL = re.compile(
    r"^(?:tva|vat|tax|sales tax|tps|tvq|qst|hst|gst)\s*(?:\(|:|\d|[$€£])"
)
# Quebec prints its TPS and TVQ registration numbers in the same `LABEL : value` syntax as a tax amount, so a
# bare integer of five or more digits is read as an identifier and never becomes a tax.
TAX_IDENTITY_NUMBER = re.compile(r"\d{5,}")


def is_identity_number(raw: str) -> bool:
    return bool(TAX_IDENTITY_NUMBER.fullmatch(raw.strip().replace(" ", "")))


def is_line_total_header(text: str, *, at_start: bool = False) -> bool:
    normalized = normalized_label(text)
    if LINE_TOTAL_EXCLUDED.search(normalized):
        return False
    matcher = LINE_TOTAL_HEADER.match if at_start else LINE_TOTAL_HEADER.search
    return bool(matcher(normalized))


DATE_EXCLUDED_LABEL = re.compile(
    r"\b(?:due\s+date|delivery\s+date|payment\s+date|paid\s+date|"
    r"date\s+d['’]?\s*echeance|date\s+de\s+(?:livraison|paiement)|echeance)\b"
)
DATE_INVOICE_LABEL = re.compile(
    r"\b(?:invoice\s+date|issue\s+date|date\s+of\s+invoice|date\s+of\s+issue|"
    r"date\s+de\s+facture|facture\s+date|issued|date)\b"
)
DATE_LABEL = re.compile(
    r"\b(?:(?P<due>due(?:\s+date)?|delivery\s+date|payment\s+date|paid\s+date|"
    r"date\s+d['’]?\s*echeance|date\s+de\s+(?:livraison|paiement)|echeance)"
    r"|(?P<invoice>invoice\s+date|issue\s+date|date\s+of\s+invoice|date\s+of\s+issue|"
    r"date\s+de\s+facture|facture\s+date|issued|date))\b"
)
UNLABELLED_DATE = re.compile(r"\d{1,2}([./])\d{1,2}\1\d{4}|\d{4}-\d{2}-\d{2}")


def is_invoice_date_line(text: str) -> bool:
    return any(match.lastgroup == "invoice" for match in DATE_LABEL.finditer(folded(text)))


REPLACEMENT_CHARACTER = "\ufffd"


def has_replacement_character(text: str | None) -> bool:
    return REPLACEMENT_CHARACTER in (text or "")


LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def description_is_usable(text: str | None) -> bool:
    """A description must carry a letter and must not be a whole monetary value.

    A number or currency-coded amount where the description column is expected proves the
    header/data alignment is broken, so the caller must keep the line unverified instead of
    exporting a false value.
    """
    stripped = (text or "").strip()
    if has_replacement_character(stripped) or not LETTER.search(stripped):
        return False
    try:
        parse_number(stripped)
    except ValueError:
        return True
    return False


def supplier_is_usable(name: str | None) -> bool:
    """A supplier name is a name, never a printed label.

    When the value area of ``Fournisseur :`` is empty, OCR glues the next printed label to
    it (``Facturé à :``), which must never be exported as a supplier.
    """
    stripped = (name or "").strip()
    return bool(stripped) and description_is_usable(stripped) and not stripped.endswith(":")


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
    # Document ordinal of the row (page-relative vertical order). None for production rows, whose
    # order is already the OCR line order.
    row_index: int | None = None
    row_evidence: dict | None = None
    family_votes: dict[str, dict[Decimal, tuple[str, ...]]] = field(default_factory=dict)
    corroborated_amount_mismatch: tuple[Decimal, Decimal] | None = None
    secondary_description: str = ""
    # Set only where the consensus product guard rejects corroborated quantity AND price, so the
    # export can abstain on the amount without inferring a veto from empty cells.
    amount_operand_veto: bool = False


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
    supplier_is_explicit: bool = False
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
    fast_path: dict | None = None
    table_shadow: dict | None = None
    # Populated only by the experimental dual-engine path (EXPERIMENTAL_DUAL_OCR); it records which
    # engine each row and cell came from. Production evidence never carries it.
    dual_ocr: dict | None = None
    # "recovered" marks rows injected by the experimental table recovery: they are evidence for
    # their own cells, never a completed production reconstruction of the table.
    rows_provenance: str | None = None
    date_locale: str | None = None
    supplier_source: TextLine | None = None


def reading_issue_category(evidence: DocumentEvidence) -> str | None:
    if not evidence.available:
        return "unavailable"
    quality_text = folded(" ".join(evidence.quality_issues))
    if any(term in quality_text for term in ("timeout", "delai", "budget documentaire")):
        return "timeout"
    confidences = [token.confidence for token in evidence.tokens]
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0
    if not confidences or mean_confidence < OCR_CONFIG.fast_mean_confidence:
        return "blur"
    if (
        not evidence.rows
        and (
            any(evidence.totals.values())
            or evidence.taxes
            or evidence.corroborated_identifiers
        )
    ):
        return "partial"
    if (
        evidence.source_kind != "text"
        and (
            "Tableau non reconstruit malgré les prétraitements." in evidence.quality_issues
            or not evidence.rows
        )
    ):
        # "layout" claims a reading failure: only evidence produced by an actual reading
        # step (OCR, consensus, native PDF) can back that claim. Directly supplied text
        # has no layout to recognise, so it is reported as a partial reading instead.
        return "layout"
    if evidence.quality_issues:
        return "partial"
    return None


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
    "description": (
        r"\b(?:description|product|produit|designation|article|prestation"
        r"|item(?:\s+description)?(?: / service)?|libelle|titre\s+du\s+livre)\b"
    ),
    "quantity": r"\b(?:quantite|qte|quantity|units|qty)\b|\bheures?\b|\bhrs?\b",
    "unit_price": (
        r"(?:\bp\s*\.?\s*u\s*\.?(?=\s|$)"
        r"|\bpu\b(?:\s+(?:net|ht))*(?=\s|$)"
        r"|\bp\.u\.(?:\s+(?:net|ht))*(?=\s|$)"
        r"|\bprix\s+unit(?:aire)?\.?(?:\s+(?:net|ht))*(?=\s|$)"
        r"|\bnet\s+unit(?:\s+price)?\b|\bunit\s+price\b|\bhourly\s+rate\b)"
        r"|(?:(?:[€$£]\s*)?/\s*(?:heure|heuree|hour|hr)\b|\bper\s+(?:heure|hour)\b)"
    ),
    "total_ht": r"\b(?:montant(?:\s+ht)?|net amount|line total|total\s+ht)\b",
}
OPTIONAL_HEADERS = {
    "list_price": r"\b(?:pu catalogue|list (?:unit )?price)\b",
    "discount": r"\b(?:discount|remise)\b",
    "tax_rate": r"(?:\b(?:vat|tva|tax)\s*(?:%|rate)|%\s*(?:tva|vat|tax)\b)",
    "tax_amount": r"^(?:total\s+(?:vat|tva|tax)\b|(?:vat|tva|tax|sales tax)$)",
    "gross_amount": r"\b(?:total\s+ttc|montant\s+ttc|gross\s+(?:amount|total))\b",
}
MONTHS = {
    word: index for index, words in enumerate((
        ("january", "janvier"), ("february", "fevrier"), ("march", "mars"),
        ("april", "avril"), ("may", "mai"), ("june", "juin"),
        ("july", "juillet"), ("august", "aout"), ("september", "septembre"),
        ("october", "octobre"), ("november", "novembre"), ("december", "decembre"),
    ), 1) for word in words
}


def parse_document_date(text: str, locale: str | None) -> tuple[date | None, str | None]:
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    local = re.search(r"\b(\d{1,2})([./])(\d{1,2})\2(\d{4})\b", text)
    normalized = folded(text)
    day_first = re.search(r"\b(\d{1,2}) ([a-z]+) (\d{4})\b", normalized)
    month_first = re.search(r"\b([a-z]+) (\d{1,2}),? (\d{4})\b", normalized)
    compact = re.fullmatch(r"\s*[:;]?\s*(\d{7,8})\s*", text)
    parts = None
    style = None
    if iso:
        parts, style = tuple(map(int, iso.groups())), "AAAA-MM-JJ"
    elif local:
        first, separator, second, year = local.groups()
        first, second = int(first), int(second)
        style = f"numérique ({separator})"
        if first > 12:
            day, month = first, second
        elif second > 12:
            day, month = second, first
        elif locale is None and first != second:
            return None, style
        else:
            day, month = (second, first) if locale == "en-US" else (first, second)
        parts = int(year), month, day
    elif day_first and day_first[2] in MONTHS:
        parts, style = (int(day_first[3]), MONTHS[day_first[2]], int(day_first[1])), "mois en lettres"
    elif month_first and month_first[1] in MONTHS:
        parts, style = (int(month_first[3]), MONTHS[month_first[1]], int(month_first[2])), "mois en lettres"
    elif compact:
        raw = compact[1]
        candidates = set()
        for split in (1, 2):
            first, second = raw[:split], raw[split:-4]
            if not 1 <= len(second) <= 2:
                continue
            for day, month in ((first, second), (second, first)):
                try:
                    candidates.add(date(int(raw[-4:]), int(month), int(day)))
                except ValueError:
                    continue
        return (next(iter(candidates)) if len(candidates) == 1 else None), "chiffres sans séparateurs"
    if parts is None:
        return None, None
    try:
        return date(*parts), style
    except ValueError:
        return None, style


def document_date_readings(lines: list[TextLine]) -> list[tuple[str, str, TextLine]]:
    readings = []
    labels = []
    deadline = False
    continued = set()
    for index, line in enumerate(lines):
        matches = list(DATE_LABEL.finditer(folded(line.text)))
        labels.append(matches)
        deadline = deadline or any(match.lastgroup == "due" for match in matches)
        for offset, label in enumerate(matches):
            end = matches[offset + 1].start() if offset + 1 < len(matches) else len(folded(line.text))
            text = folded(line.text)[label.end():end]
            source = line
            if not re.search(r"\d", text) and index + 1 < len(lines):
                following = lines[index + 1]
                height = line.box[3] - line.box[1]
                if (
                    following.page == line.page
                    and 0 <= following.box[1] - line.box[3] <= 3 * height
                    and abs(following.box[0] - line.box[0]) <= 2 * height
                    and not DATE_LABEL.search(folded(following.text))
                ):
                    text, source = following.text, following
                    continued.add(index + 1)
            readings.append((label.lastgroup or "invoice", text, source))
    # An unreadable date label is a recognition failure, not an absence of evidence: a numeric date
    # printed alone on its own line still dates the document. It is read as the issue date only while
    # nothing in the document labels a deadline, so an échéance can never be relabelled, and a date
    # inside a sentence stays prose.
    if deadline or any(kind == "invoice" for kind, _, _ in readings):
        return readings
    for index, line in enumerate(lines):
        if labels[index] or index in continued:
            continue
        standalone = folded(line.text)
        if UNLABELLED_DATE.fullmatch(standalone):
            readings.append(("invoice", standalone, line))
    return readings


def document_date_locale(readings: list[tuple[str, str, TextLine]], locale: str | None) -> str | None:
    orders = set()
    for _, text, _ in readings:
        for match in re.finditer(r"\b(\d{1,2})([./])(\d{1,2})\2(\d{4})\b", text):
            first, _, second, year = match.groups()
            first, second = int(first), int(second)
            if first > 12 and second <= 12 and parse_document_date(match.group(), "fr-FR")[0]:
                orders.add("fr-FR")
            elif second > 12 and first <= 12 and parse_document_date(match.group(), "en-US")[0]:
                orders.add("en-US")
    return next(iter(orders)) if len(orders) == 1 else None if orders else locale


def invoice_number_from_text(text: str) -> str | None:
    lines = text.splitlines()
    matches = []
    for line_index, line in enumerate(lines):
        for match in re.finditer(
            r"\b(FR|GB|US)-(?:((\d{4})-(\d{3,12}))|(\d{3,12}))\b",
            line,
            re.I,
        ):
            year = match.group(3)
            sequence = match.group(4) or match.group(5)
            candidate = (
                f"{match.group(1).upper()}-{year}-{sequence}"
                if year else f"{match.group(1).upper()}-{sequence}"
            )
            context = " ".join(lines[max(0, line_index - 2):line_index + 1])
            invoice_context = bool(re.search(
                r"\b(?:facture|invoice|invoice number|invoice no|numero de facture|n[°o]\s*(?:de\s*)?facture)\b",
                folded(context),
            ))
            year_form = year is not None
            if year_form or invoice_context:
                matches.append((int(invoice_context), int(year_form), candidate))
    if not matches:
        return None
    best_score = max((invoice_context, year_form) for invoice_context, year_form, _ in matches)
    candidates = {
        candidate for invoice_context, year_form, candidate in matches
        if (invoice_context, year_form) == best_score
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _header_matches(key: str, text: str, *, at_start: bool = False) -> bool:
    normalized = _header_text(text)
    if key == "total_ht":
        if re.search(r"\b(?:gross|brut)\s+(?:amount|montant)\b", normalized):
            return False
        cells = [folded(cell).strip() for cell in normalized.split("|")]
        if any(cell in {"montant", "montant ht", "amount", "amount ht"} for cell in cells):
            return True
        if at_start and re.match(r"(?:amount|montant)(?:\s+ht)?\b", normalized):
            return True
        if re.search(r"\bttc\b", normalized):
            return False
    pattern = HEADERS.get(key)
    matcher = re.match if at_start else re.search
    if pattern and matcher(pattern, normalized):
        return True
    return key == "unit_price" and not at_start and _contextual_rate_price_header(text)


def _header_text(text: str) -> str:
    vocabulary = {"description", "designation", "product", "quantity", "quantite", "hourly",
                  "unitaire", "amount", "montant"}

    def correct(match):
        word = match.group()
        candidates = [
            target for target in vocabulary if len(word) == len(target)
            and sum(left != right for left, right in zip(word, target)) == 1
        ]
        return candidates[0] if word not in vocabulary and len(candidates) == 1 else word

    normalized = re.sub(r"\b[a-z]{5,}\b", correct, normalized_label(text))
    # Header-only OCR repair: a known two-word label, never a body value or supplier name.
    return re.sub(r"\bhourly\s+[a-z]{4}\b", lambda match: (
        "hourly rate" if sum(a != b for a, b in zip(match.group(), "hourly rate")) <= 1
        else match.group()
    ), normalized)


def _contextual_rate_price_header(text: str) -> bool:
    normalized = _header_text(text)
    if not (
        re.search(HEADERS["description"], normalized)
        and re.search(HEADERS["quantity"], normalized)
        and (
            re.search(HEADERS["total_ht"], normalized)
            or is_line_total_header(text)
            or re.search(r"\b(?:subtotal|sub[\s-]+total)\b", normalized)
        )
    ):
        return False
    aliases = list(re.finditer(r"\b(rate|price)\b", normalized))
    if len(aliases) == 1:
        return True
    return (
        len(aliases) == 2
        and aliases[0].group(1) == "rate"
        and aliases[1].group(1) == "price"
        and bool(re.fullmatch(
            r"\s*/\s*", normalized[aliases[0].end():aliases[1].start()],
        ))
    )


def _header_token_index(line: TextLine, key: str) -> int | None:
    for index in range(len(line.tokens)):
        suffix = " ".join(token.text for token in line.tokens[index:])
        if _header_matches(key, suffix, at_start=True) or (
            key == "total_ht" and is_line_total_header(suffix, at_start=True)
        ):
            return index
    return None


def _rate_price_token_index(line: TextLine) -> int | None:
    labels = [normalized_label(token.text).strip() for token in line.tokens]
    compact = [
        index for index, label in enumerate(labels)
        if re.fullmatch(r"rate\s*/\s*price", label)
    ]
    if len(compact) == 1:
        return compact[0]
    if compact:
        return None
    rates = [index for index, label in enumerate(labels) if label == "rate"]
    prices = [index for index, label in enumerate(labels) if label == "price"]
    if len(rates) == 1 and len(prices) == 1 and rates[0] < prices[0]:
        if any(label == "/" for label in labels[rates[0] + 1:prices[0]]):
            return rates[0]
        return None
    if len(rates) + len(prices) == 1:
        return (rates or prices)[0]
    return None


def _contextual_table_header_starts(line: TextLine) -> dict[str, float]:
    description = _header_token_index(line, "description")
    quantity = _header_token_index(line, "quantity")
    unit_price = _header_token_index(line, "unit_price")
    if unit_price is None and _contextual_rate_price_header(line.text):
        unit_price = _rate_price_token_index(line)
    total_ht = _header_token_index(line, "total_ht")
    subtotal = [
        index for index, token in enumerate(line.tokens)
        if normalized_label(token.text).strip() in {"subtotal", "sub total", "sub-total"}
    ]
    subtotal_column = total_ht is None and len(subtotal) == 1
    if subtotal_column:
        total_ht = subtotal[0]
    if (
        description is None or quantity is None or unit_price is None or total_ht is None
        or not (
            line.tokens[description].box[0] < line.tokens[quantity].box[0]
            < line.tokens[unit_price].box[0] < line.tokens[total_ht].box[0]
        )
    ):
        return {}
    starts = {}
    if _header_token_index(line, "unit_price") is None:
        starts["unit_price"] = line.tokens[unit_price].box[0]
    if subtotal_column:
        starts["total_ht"] = line.tokens[total_ht].box[0]
    return starts


def _contextual_literal_header_columns(line: TextLine) -> dict[str, int]:
    if "|" not in line.text:
        return {}
    cells = [folded(cell).strip() for cell in line.text.split("|")]
    description = [
        index for index, cell in enumerate(cells)
        if _header_matches("description", cell)
    ]
    quantity = [
        index for index, cell in enumerate(cells)
        if _header_matches("quantity", cell)
    ]
    unit_price = [
        index for index, cell in enumerate(cells)
        if _header_matches("unit_price", cell)
        or re.fullmatch(r"(?:rate|price|rate\s*/\s*price)", cell)
    ]
    total_ht = [
        index for index, cell in enumerate(cells)
        if _header_matches("total_ht", cell)
        or is_line_total_header(cell)
        or cell in {"subtotal", "sub total", "sub-total"}
    ]
    if (
        len(description) == len(quantity) == len(unit_price) == len(total_ht) == 1
        and description[0] < quantity[0] < unit_price[0] < total_ht[0]
    ):
        return {"unit_price": unit_price[0], "total_ht": total_ht[0]}
    return {}


SUPPLIER_LEGAL_SUFFIX = re.compile(
    r"\b(?:ltd|limited|llc|inc|corp(?:oration)?|plc|llp|gmbh|sas|sarl|sa|bv|ag|co)\b\.?",
    re.I,
)
SUPPLIER_CLIENT_MARKER = re.compile(
    r"\b(?:bill(?:ed)?\s+to|ship\s+to|sold\s+to|deliver\s+to|invoice\s+to|"
    r"facture\s+a|facture\s+au)\b|^\s*(?:customer|client)\s*:"
)
SUPPLIER_CONTACT_MARKER = re.compile(
    r"\b(?:contact|email|e-mail|phone|telephone|tel|www)\b|@"
)
SUPPLIER_EXCLUDED_LINE = re.compile(
    r"\b(?:facture|invoice|date|currency|devise|order|commande|siret|iban|vat|tva|"
    r"subtotal|total|bill|payment|echeance|due|delivery|sincerely|thank|"
    r"registration|bank|account|email|phone|tel|website)\b"
)
CONTINUATION_EXCLUDED = re.compile(
    r"\b(?:payment|invoice|facture|total|subtotal|vat|tva|tax|bank|iban|"
    r"sort\s+code|account|due|delivery|date|reference|terms|sincerely|"
    r"thank|email|phone|tel|website|registration|siret)\b"
)
SUPPLIER_STOPWORDS = {
    "the", "and", "for", "from", "with", "you", "your", "our", "this", "that",
    "thank", "trusting", "sending", "dear", "please", "payment", "invoice",
    "is", "are", "we", "to", "of", "by", "at", "as", "a", "an",
    "issued", "signature", "sincerely",
}

# Form labels, section headings and table vocabulary: never a displayed supplier name.
SUPPLIER_CONTEXT_STOPWORDS = {
    "adresse", "adresses", "livraison", "facturation", "conditions", "description",
    "specification", "specifications", "incoterm", "pays", "origine", "coordonnees",
    "details", "siege", "merci", "signature", "devis", "reference", "nombre",
    "colis", "poids", "taille", "quantite", "unite", "unites", "informations",
    "renseignements", "modalites", "remarque",
}

ADDRESS_STREET = re.compile(
    r"\b\d{1,4}\s*(?:bis|ter|quater)?\s*,?\s*"
    r"(?:rue|avenue|av\.?|boulevard|bd\.?|chemin|route|place|impasse|all[ée]e|quai|cours|"
    r"street|st\.?|road|rd\.?|lane|drive|way|high\s+street|park|square|gardens?)\b",
    re.IGNORECASE,
)
ADDRESS_POSTAL = re.compile(r"\b\d{4,5}\s+[A-ZÀ-ÖØ-Ý]")
ADDRESS_UK_POSTCODE = re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b")
ADDRESS_HOUSE_NUMBER = re.compile(r"^\s*\d{1,4}\b")
ADDRESS_PLACE_NAME = re.compile(r"\b[A-ZÀ-ÖØ-Ý][\w'’.-]*")


@dataclass(frozen=True)
class SupplierCandidate:
    name: str
    line: TextLine
    tier: int


def _looks_table_header(line: TextLine) -> bool:
    core = all(
        _header_matches(key, line.text)
        for key in ("description", "quantity", "unit_price")
    )
    return core and (
        _header_matches("total_ht", line.text) or is_line_total_header(line.text)
        or bool(re.search(OPTIONAL_HEADERS["gross_amount"], folded(line.text)))
    )


def _issuer_names(lines: list[TextLine]) -> list[str]:
    names = []
    for index, line in enumerate(lines[:-1]):
        label = re.search(r"\b(?:billed\s+from|please\s+make\s+payment\s+to)\b", folded(line.text))
        if label is None:
            continue
        source = lines[index + 1]
        height = line.box[3] - line.box[1]
        if source.page != line.page or not 0 <= source.box[1] - line.box[3] <= 3 * height:
            continue
        # A neighbouring Billed-to column bounds the issuer region.
        markers = [
            token.box[0] for token in line.tokens
            if re.match(r"^bill(?:ed)?\b", folded(token.text))
        ]
        right = sorted(markers)[1] if len(markers) > 1 else float("inf")
        tokens = [token for token in source.tokens if line.box[0] - height <= token.box[0] < right]
        if tokens and all(token.confidence >= OCR_CONFIG.confidence for token in tokens):
            name = _clean_supplier_name(" ".join(token.text for token in tokens))
            if name and _supplier_context_ok(name) and not SUPPLIER_EXCLUDED_LINE.search(folded(name)):
                names.append(name)
    return names


def _registered_office_supplier(lines: list[TextLine]) -> SupplierCandidate | None:
    client = next((
        index for index, line in enumerate(lines)
        if re.search(r"^\s*client\b|(?:bill(?:ed)?|invoice|ship)\s+to\b", folded(line.text))
    ), None)
    if client is None:
        return None
    candidates = []
    for index, label in enumerate(lines[:-2]):
        words = [normalized_label(token.text) for token in label.tokens]
        for start in range(len(words) - 1):
            if words[start:start + 2] != ["siege", "social"]:
                continue
            label_tokens = label.tokens[start:start + 2]
            if any(token.confidence < OCR_CONFIG.confidence for token in label_tokens):
                continue
            left = label_tokens[0].box[0]
            height = max(token.box[3] - token.box[1] for token in label_tokens)
            right = next((
                token.box[0] for token in label.tokens[start + 2:]
                if token.box[0] - label_tokens[-1].box[2] > 2 * height
            ), float("inf"))

            def column(line):
                return [
                    token for token in line.tokens
                    if left - height <= token.box[0] < right
                ]

            name_line, address_line = lines[index + 1:index + 3]
            if (
                name_line.page != label.page or address_line.page != label.page
                or not 0 < name_line.box[1] - label.box[1] <= 3 * height
                or not 0 < address_line.box[1] - name_line.box[1] <= 3 * height
            ):
                continue
            name_tokens, address_tokens = column(name_line), column(address_line)
            name = _clean_supplier_name(" ".join(token.text for token in name_tokens))
            if (
                not name or not _supplier_context_ok(name)
                or SUPPLIER_EXCLUDED_LINE.search(folded(name))
                or not _address_shaped(" ".join(token.text for token in address_tokens))
                or any(token.confidence < OCR_CONFIG.fast_token_confidence for token in name_tokens)
            ):
                continue
            header = [column(line) for line in lines[:client] if line.page == label.page]
            header_name = any(
                folded(" ".join(token.text for token in tokens[offset:offset + len(name_tokens)])) == folded(name)
                and all(token.confidence >= OCR_CONFIG.fast_token_confidence
                        for token in tokens[offset:offset + len(name_tokens)])
                for tokens in header for offset in range(len(tokens) - len(name_tokens) + 1)
            )
            registration = any(
                re.search(r"\b(?:siren|siret)\b", folded(" ".join(token.text for token in tokens)))
                and any(re.search(r"\d", token.text) for token in tokens)
                and all(token.confidence >= OCR_CONFIG.confidence for token in tokens)
                for tokens in header
            )
            if header_name and registration:
                source = TextLine(name, name_tokens, name_line.page, union_box(name_tokens))
                candidates.append(SupplierCandidate(name, source, 3))
    names = {folded(candidate.name) for candidate in candidates}
    return candidates[0] if len(names) == 1 else None


def _client_line_indices(lines: list[TextLine]) -> set[int]:
    excluded = set()
    for start, line in enumerate(lines):
        if not SUPPLIER_CLIENT_MARKER.search(folded(line.text)):
            continue
        end = next(
            (index for index in range(start + 1, len(lines)) if _looks_table_header(lines[index])),
            len(lines),
        )
        excluded.update(range(start, end))
    return excluded


def _contact_line_indices(lines: list[TextLine]) -> set[int]:
    markers = [
        index for index, line in enumerate(lines)
        if SUPPLIER_CONTACT_MARKER.search(folded(line.text))
    ]
    excluded = set()
    for index in markers:
        excluded.update(range(max(0, index - 2), min(len(lines), index + 3)))
    return excluded


def _clean_supplier_name(name: str) -> str | None:
    name = " ".join(name.split()).strip(" ,;:.-–—")
    name = re.sub(r"\b(?:Ltd|Limited)\.?$", "Ltd", name, flags=re.I)
    words = name.split()
    if not 2 <= len(words) <= 4:
        return None
    if any(re.search(r"[\d@|/]", word) for word in words):
        return None
    normalized_words = {folded(word).strip(".,") for word in words}
    if normalized_words & {"logo", "invoice", "facture", "bill", "customer", "client"}:
        return None
    if normalized_words & SUPPLIER_STOPWORDS:
        return None
    return name or None


def _legal_supplier_name(line: TextLine) -> str | None:
    for match in SUPPLIER_LEGAL_SUFFIX.finditer(line.text):
        prefix = line.text[:match.start()]
        words = re.findall(r"[^\W\d_][\w'’.-]*", prefix, re.UNICODE)
        selected = []
        for word in reversed(words):
            normalized = folded(word).strip(".-")
            if not normalized or normalized in SUPPLIER_STOPWORDS or any(char.isdigit() for char in word):
                break
            selected.insert(0, word)
            if len(selected) == 4:
                break
        if selected:
            suffix = match.group().strip(".,;:")
            name = _clean_supplier_name(" ".join([*selected, suffix]))
            if name:
                return name
    return None


def _address_shaped(text: str) -> bool:
    # Streets, postcodes and house numbers only count when a capitalised place name follows:
    # "36 Terrick gt 2N" is an address, "3 tailles disponibles" and "75002 exemplaires" are prose.
    return bool(
        ADDRESS_STREET.search(text)
        or ADDRESS_UK_POSTCODE.search(text)
        or (ADDRESS_POSTAL.search(text) and ADDRESS_PLACE_NAME.search(text))
        or (ADDRESS_HOUSE_NUMBER.match(text) and ADDRESS_PLACE_NAME.search(text))
    )


def _supplier_context_ok(text: str) -> bool:
    """Reject labels and prose before they can become a displayed supplier name."""
    stripped = text.strip()
    if not stripped or stripped[-1] in ".:;!?" or ":" in stripped or "," in stripped:
        return False
    if re.search(r"\betc\b", stripped, re.IGNORECASE):
        return False
    words = stripped.split()
    if not any(word[:1].isupper() for word in words):
        return False
    return not {folded(word).strip(".,;:") for word in words} & SUPPLIER_CONTEXT_STOPWORDS


def _address_supplier_name(line: TextLine) -> str | None:
    text = " ".join(line.text.split()).strip()
    parts = re.split(r"\s*[,–—]\s*|\s+-\s+", text, maxsplit=1)
    if len(parts) < 2:
        return None
    prefix, remainder = parts
    if not _address_shaped(remainder) or not _supplier_context_ok(prefix):
        return None
    return _clean_supplier_name(prefix)


def _supplier_candidates(lines: list[TextLine]) -> list[SupplierCandidate]:
    client_lines = _client_line_indices(lines)
    contact_lines = _contact_line_indices(lines)
    header_index = next(
        (index for index, line in enumerate(lines) if _looks_table_header(line)),
        None,
    )
    candidates = []
    for index, line in enumerate(lines):
        if index in client_lines:
            continue
        text = " ".join(line.text.split()).strip()
        if not text or SUPPLIER_EXCLUDED_LINE.search(folded(text)):
            continue
        if not line.tokens or any(token.confidence < OCR_CONFIG.confidence for token in line.tokens):
            continue
        legal = _legal_supplier_name(line)
        if legal:
            candidates.append(SupplierCandidate(legal, line, 3))
            continue
        if index in contact_lines or (header_index is not None and index > header_index):
            continue
        address = _address_supplier_name(line)
        if address:
            candidates.append(SupplierCandidate(address, line, 2))
            continue
        words = text.split()
        if 2 <= len(words) <= 4 and _supplier_context_ok(text):
            name = _clean_supplier_name(text)
            if name:
                candidates.append(SupplierCandidate(name, line, 1))
    return candidates


def _implicit_supplier_name(lines: list[TextLine]) -> str | None:
    candidates = _supplier_candidates(lines)
    for tier in (3, 2, 1):
        names = {candidate.name for candidate in candidates if candidate.tier == tier}
        if len(names) == 1:
            return next(iter(names))
        if names:
            return None
    return None


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
        if not text.strip() or text.strip() in {"|", "||", "[", "]"} or float(data["conf"][index]) < 0:
            continue
        left, top = float(data["left"][index]), float(data["top"][index])
        right, bottom = left + float(data["width"][index]), top + float(data["height"][index])
        tokens.append(Token(text, page, (left, top, right, bottom), float(data["conf"][index])))
    evidence = from_tokens(tokens, locale=locale, literal_columns=False)
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
    suppliers, supplier_explicitness, invoice_numbers, dates, locales = set(), set(), set(), set(), set()
    texts = []
    for index, source in enumerate(pages):
        evidence = deepcopy(source)
        texts.append(evidence.text)
        if evidence.supplier:
            suppliers.add(evidence.supplier)
            supplier_explicitness.add(evidence.supplier_is_explicit)
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
        if evidence.rows:
            result.rows_provenance = (
                "recovered" if all(source.rows_provenance == "recovered"
                                   for source in pages if source.rows) else None)
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
        if evidence.table_shadow:
            if result.table_shadow is None:
                result.table_shadow = {
                    "flag": evidence.table_shadow.get("flag"), "enabled": True, "pages": [],
                }
            result.table_shadow["pages"].append(dict(evidence.table_shadow, page=index))
    result.text = "\n\n".join(texts)
    result.supplier = next(iter(suppliers)) if len(suppliers) == 1 else None
    result.supplier_raw = result.supplier
    result.supplier_is_explicit = (
        next(iter(supplier_explicitness))
        if len(suppliers) == 1 and len(supplier_explicitness) == 1 else False
    )
    result.invoice_number = next(iter(invoice_numbers)) if len(invoice_numbers) == 1 else None
    result.invoice_date = next(iter(dates)) if len(dates) == 1 else None
    result.locale = next(iter(locales)) if len(locales) == 1 else None
    result.date_locale = document_date_locale(document_date_readings(result.lines), result.locale)
    return result


def from_tokens(tokens: list[Token], *, locale: str | None = None,
                raw_lines: dict[tuple[int, float], str] | None = None,
                literal_columns: bool = True) -> DocumentEvidence:
    lines = positioned_lines(tokens)
    if raw_lines:
        for line in lines:
            line.text = raw_lines.get((line.page, line.box[1]), line.text)
    if raw_lines is None:
        joined = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if (
                index + 1 < len(lines)
                and all(_header_matches(key, line.text) for key in ("description", "quantity", "total_ht"))
                and not _header_matches("unit_price", line.text)
            ):
                following = lines[index + 1]
                price = next((token for token in line.tokens if folded(token.text) == "prix"), None)
                if (
                    price is not None and following.page == line.page
                    and 0 <= following.box[1] - line.box[3] <= 2 * (price.box[3] - price.box[1])
                    and folded(following.text) == "unitaire"
                    and abs(following.box[0] - price.box[0]) <= 2 * (price.box[3] - price.box[1])
                ):
                    price_index = line.tokens.index(price)
                    combined = [
                        *line.tokens[:price_index + 1], *following.tokens,
                        *line.tokens[price_index + 1:],
                    ]
                    line = TextLine(" ".join(token.text for token in combined), combined,
                                    line.page, union_box(combined))
                    index += 1
            joined.append(line)
            index += 1
        lines = joined
    text = "\n".join(line.text for line in lines)
    if locale is None:
        locale = "fr-FR" if re.search(r"\bEUR\b|€|\bFACTURE\b", text, re.I) else (
            "en-GB" if re.search(r"\bGBP\b|£", text) else
            "en-US" if re.search(r"\bUSD\b|\$", text) else None
        )
    evidence = DocumentEvidence(text, locale, tokens, lines)
    date_readings = document_date_readings(lines)
    evidence.date_locale = document_date_locale(date_readings, locale)
    evidence.invoice_number = invoice_number_from_text(text)
    columns: dict[str, int] = {}
    starts: dict[str, float] = {}
    contextual_ocr_header = False
    header_page = -1
    suppliers: dict[str, str] = {}
    dates: set[str] = set()
    for line in lines:
        normalized = folded(line.text)
        evidence.numbers.extend(number_tokens(line.text, line, locale))
        supplier = re.search(r"(?:supplier|fournisseur)\s*:\s*(.+?)(?:\s+(?:bill to|factur[ée] [aà])\s*:|$)", line.text, re.I)
        if supplier:
            name = supplier.group(1).strip()
            if name and supplier_is_usable(name):
                key = re.sub(r"\b(?:ltd|limited)\.?$", "ltd", folded(name))
                key = re.sub(r"\bs[. ]*a[. ]*s\.?$", "sas", key)
                suppliers[key] = name
        amounts = number_tokens(line.text, line, locale)
        label = normalized_label(line.text)
        if (
            re.match(r"^(?:frais\s+de\s+livraison|shipping(?:\s+(?:charges?|cost))?)\b", label)
            and len(amounts) <= 1
        ):
            # A summary charge supplies no quantity or item-column evidence.
            columns, starts = {}, {}
            contextual_ocr_header = False
            continue
        if re.match(r"^(?:remise|discount|rabais)\b.*:", normalized) and not re.search(
            r"inclu|appliqu|already applied", normalized,
        ):
            money = [number for number in amounts if "%" not in number.raw]
            if len(money) == 1:
                evidence.discounts.append(money[0])
                continue
        tax_label = TAX_LINE_LABEL.match(label)
        total_field = None
        if TOTAL_HT_LABEL.match(label):
            total_field = "total_ht"
        elif TOTAL_TTC_LABEL.match(label):
            total_field = "total_ttc"
        unqualified_total = bool(
            UNQUALIFIED_TOTAL_LABEL.match(label) and not TOTAL_QUALIFIER.search(label)
        )
        if tax_label or total_field or unqualified_total:
            columns, starts = {}, {}
            contextual_ocr_header = False
            if tax_label and amounts:
                rate = next((n for n in amounts if "%" in n.raw), None)
                money = [n for n in amounts if "%" not in n.raw]
                if money and not is_identity_number(money[-1].raw):
                    tax = Tax(rate, money[-1], money[0] if len(money) > 1 else None)
                    if tax not in evidence.taxes:
                        evidence.taxes.append(tax)
            if total_field and amounts:
                evidence.totals.setdefault(total_field, []).append(amounts[-1])
            elif unqualified_total:
                money = [number for number in amounts if "%" not in number.raw]
                if len(money) == 1:
                    evidence.totals.setdefault("total_candidate", []).append(money[0])
            continue
        if literal_columns and "|" in line.text:
            inline = line.text.split("|")
            inline_values = {}
            for cell in inline[1:]:
                label, separator, content = cell.partition(":")
                if not separator:
                    continue
                for key in ("quantity", "unit_price", "total_ht"):
                    if _header_matches(key, label):
                        inline_values[key] = number_tokens(content, line, locale)
            if len(inline_values) >= 2:
                evidence.rows.append(TableRow(inline[0].strip(), inline_values, line))
                continue
        contextual_starts = _contextual_table_header_starts(line)
        contextual_columns = (
            _contextual_literal_header_columns(line)
            if literal_columns and "|" in line.text else {}
        )
        matches = {key: _header_matches(key, line.text) for key in HEADERS}
        if not matches["total_ht"]:
            matches["total_ht"] = is_line_total_header(line.text)
        for key in contextual_starts:
            matches[key] = True
        for key in contextual_columns:
            matches[key] = True
        partial_header = (
            all(matches[key] for key in ("description", "quantity", "unit_price"))
            and re.search(OPTIONAL_HEADERS["gross_amount"], normalized)
        )
        if all(matches.values()) or partial_header:
            required = {"description", "quantity", "unit_price"} if partial_header else set(HEADERS)
            if literal_columns and "|" in line.text:
                cells = line.text.split("|")
                columns = {}
                for key in HEADERS:
                    indexes = [
                        index for index, cell in enumerate(cells)
                        if _header_matches(key, cell)
                        or key == "total_ht" and is_line_total_header(cell)
                    ]
                    if key in contextual_columns:
                        indexes = [contextual_columns[key]]
                    if len(indexes) == 1:
                        columns[key] = indexes[0]
                for key, pattern in OPTIONAL_HEADERS.items():
                    indexes = [
                        index for index, cell in enumerate(cells)
                        if re.search(pattern, folded(cell))
                    ]
                    if len(indexes) == 1:
                        columns[key] = indexes[0]
                starts = {}
                valid_header = required.issubset(columns) and len(
                    set(columns.values())
                ) == len(columns)
            else:
                columns = {}
                starts = {}
                for key, pattern in HEADERS.items():
                    if key in contextual_starts:
                        starts[key] = contextual_starts[key]
                        continue
                    for index in range(len(line.tokens)):
                        suffix = " ".join(token.text for token in line.tokens[index:])
                        if _header_matches(key, suffix, at_start=True) or (
                            key == "total_ht" and is_line_total_header(suffix, at_start=True)
                        ):
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
                valid_header = (
                    required.issubset(starts)
                    and len(set(starts.values())) == len(starts)
                )
            if not valid_header:
                columns, starts = {}, {}
                contextual_ocr_header = False
                continue
            header_page = line.page
            contextual_ocr_header = (
                not literal_columns and bool(contextual_starts or contextual_columns)
            )
            continue
        if line.page != header_page or not (columns or starts):
            continue
        cell_texts: dict[str, str] = {}
        cell_lines: dict[str, TextLine] = {}
        if columns and literal_columns and "|" in line.text:
            cells = line.text.split("|")
            if len(cells) > max(columns.values()):
                cell_texts = {key: cells[index].strip() for key, index in columns.items()}
        elif starts:
            ordered = sorted(starts, key=starts.__getitem__)
            for index, key in enumerate(ordered):
                left = starts[key] - 4
                right = starts[ordered[index + 1]] - 4 if index + 1 < len(ordered) else float("inf")
                if key == "description" and index > 0 and ordered[index - 1] == "quantity":
                    quantity_tokens = [
                        token for token in line.tokens
                        if starts["quantity"] - 4 <= token.box[0] < left
                        and number_tokens(token.text, TextLine(token.text, [token], token.page, token.box), locale)
                    ]
                    if len(quantity_tokens) == 1:
                        # A centred description header can begin after the actual text.
                        # A unique printed quantity bounds the description on its left.
                        left = quantity_tokens[0].box[2] + 1
                cell_tokens = [
                    token for token in line.tokens
                    if left <= (token.box[0] + token.box[2]) / 2 < right
                ]
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
        description_cell = cell_texts.get("description", "")
        if description_cell.strip() and has_replacement_character(description_cell):
            evidence.issues.append(
                "Colonne description corrompue (U+FFFD) : "
                f"caractère non récupérable pour {line.text!r}."
            )
            continue
        if description_cell.strip() and not description_is_usable(description_cell):
            # A number where the description column is expected proves the header and the
            # data rows do not agree: keep the whole line unverified instead of guessing.
            evidence.issues.append(
                "Colonne description inexploitable (contenu numérique) : "
                f"alignement en-tête/ligne non recoupé pour {line.text!r}."
            )
            continue
        description_line = cell_lines.get("description", line)
        if contextual_ocr_header:
            required_cells = ("quantity", "unit_price", "total_ht")
            complete_cells = all(
                values[key]
                and cell_lines.get(key)
                and all(
                    token.confidence >= OCR_CONFIG.fast_token_confidence
                    for token in cell_lines[key].tokens
                )
                for key in required_cells
            )
            confident_description = (
                bool(description_cell.strip())
                and description_is_usable(description_cell)
                and bool(description_line.tokens)
                and all(
                    token.confidence >= OCR_CONFIG.fast_token_confidence
                    for token in description_line.tokens
                )
            )
            if not complete_cells or not confident_description:
                evidence.issues.append(
                    "Ligne de tableau contextuel ignorée : description ou cellules "
                    "quantité/prix/sous-total incomplètes ou peu fiables."
                )
                continue
        if sum(bool(values[key]) for key in ("quantity", "unit_price", "total_ht")) >= 2:
            low_confidence = any(t.confidence < 60 for t in line.tokens)
            if low_confidence:
                evidence.issues.append(f"Ligne OCR à faible confiance : {line.text}")
            evidence.rows.append(TableRow(
                cell_texts.get("description", ""), values, line,
                all(token.confidence >= 60 for token in description_line.tokens),
            ))
        elif evidence.rows and starts and cell_texts.get("description") and not any(values.values()):
            last = evidence.rows[-1]
            description_start = starts.get("description")
            description_text = cell_texts["description"]
            if (
                line.page == last.line.page
                and description_start is not None
                and line.tokens
                and 0 <= line.box[1] - last.line.box[3] <= 3 * max(
                    line.box[3] - line.box[1], last.line.box[3] - last.line.box[1],
                )
                and abs(min(token.box[0] for token in line.tokens) - description_start) <= 8
                and all(token.confidence >= 60 for token in line.tokens)
                and description_is_usable(description_text)
                and not has_replacement_character(description_text)
                and not CONTINUATION_EXCLUDED.search(folded(description_text))
            ):
                last.secondary_description = " ".join(
                    filter(None, (last.secondary_description, description_text))
                )
    for name in _issuer_names(lines):
        suppliers.setdefault(folded(name), name)
    for kind, raw, _ in date_readings:
        if kind != "invoice":
            continue
        parsed, style = parse_document_date(raw, evidence.date_locale)
        if parsed is not None:
            dates.add(parsed.isoformat())
        elif style:
            evidence.issues.append("invoice_date : date documentaire ambiguë ou invalide.")
    if len(suppliers) == 1:
        supplier = next(iter(suppliers.values()))
        evidence.supplier_raw = supplier
        evidence.supplier_is_explicit = True
        if supplier.upper() in {"ERROR", "INVALID"} or re.search(r"\b(?:LId|L1d|LtI|Lt1)\.?$", supplier):
            evidence.issues.append("supplier_name : non vérifié — nom ou suffixe OCR ambigu, aucune substitution.")
        else:
            evidence.supplier = re.sub(r"\b(?:Ltd|Limited)\.?$", "Ltd", " ".join(supplier.split()), flags=re.IGNORECASE)
    elif suppliers:
        evidence.issues.append("supplier_name : plusieurs fournisseurs incompatibles dans le document.")
    else:
        office = _registered_office_supplier(lines)
        if office is not None:
            evidence.supplier_raw = office.name
            evidence.supplier = office.name
            evidence.supplier_source = office.line
        else:
            implicit = _implicit_supplier_name(lines)
            if implicit:
                evidence.supplier_raw = implicit
                evidence.supplier = implicit
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
