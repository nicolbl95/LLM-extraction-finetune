"""Document anomalies only: no model prediction is an input to these checks."""

import re
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP, localcontext
from difflib import SequenceMatcher
from itertools import combinations

from invoice_evidence import DocumentEvidence, MONTHS, folded

CENT = Decimal(".01")
TOLERANCE = Decimal(".02")
# Calibrated against synthetic reference identifiers, not a legal validation service.
IDENTIFIER_SEVERITIES = {
    "siret": "information", "fr_vat": "erreur", "iban": "erreur",
    "uk_vat": "erreur", "ein": "erreur",
}
IDENTIFIER_CALIBRATION = {
    "siret": {"checked": 20, "failed": 18}, "fr_vat": {"checked": 20, "failed": 0},
    "iban": {"checked": 20, "failed": 0}, "uk_vat": {"checked": 20, "failed": 0},
    "ein": {"checked": 20, "failed": 0},
}


@dataclass(frozen=True)
class Flag:
    code: str
    severity: str
    location: str
    printed: str
    expected: str
    difference: str
    reason: str


@dataclass
class Validation:
    flags: list[Flag] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    def export(self) -> dict:
        return {"flags": [asdict(flag) for flag in self.flags], "passed": self.passed}


def rounded(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = max(50, len(value.as_tuple().digits) + abs(value.adjusted()) + 4)
        return value.quantize(CENT, rounding=ROUND_HALF_UP)


def unique_value(numbers: list) -> Decimal | None:
    values = {number.value for number in numbers}
    return next(iter(values)) if len(values) == 1 else None


def luhn(number: str) -> bool:
    if not number.isdigit():
        return False
    digits = [int(char) for char in number]
    return sum((digit * 2 - 9 if digit >= 5 else digit * 2) if index % 2 else digit
               for index, digit in enumerate(reversed(digits))) % 10 == 0


def valid_siret(value: str) -> bool:
    return len(value) == 14 and value.isdigit() and luhn(value)


def valid_fr_vat(vat: str, siret: str | None) -> bool:
    match = re.fullmatch(r"FR(\d{2})(\d{9})", vat)
    return bool(match and (siret is None or match[2] == siret[:9])
                and int(match[1]) == (12 + 3 * (int(match[2]) % 97)) % 97)


def valid_iban(value: str) -> bool:
    value = re.sub(r"\s+", "", value).upper()
    lengths = {"FR": 27, "GB": 22, "DE": 22, "ES": 24, "IT": 27, "BE": 16, "NL": 18}
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", value):
        return False
    if value[:2] in lengths and len(value) != lengths[value[:2]]:
        return False
    remainder = 0
    for char in value[4:] + value[:4]:
        for digit in str(ord(char) - 55) if char.isalpha() else char:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def valid_uk_vat(value: str) -> bool:
    return bool(re.fullmatch(r"GB(?:\d{9}|\d{12}|GD\d{3}|HA\d{3})", value))


def valid_ein(value: str) -> bool:
    return bool(re.fullmatch(r"\d{2}-\d{7}", value))


def identifier_checks(text: str) -> list[tuple[str, str, bool]]:
    checks = []
    sirets = re.findall(r"\bSIRET\s*:?\s*([0-9 ]{14,20})", text, re.I)
    sirets = [value.replace(" ", "").strip() for value in sirets]
    for value in sirets:
        checks.append(("siret", value, valid_siret(value)))
    for value in re.findall(r"\b(?:TVA|VAT ID)\s*:?\s*(FR[A-Z0-9]{2}\d{9})\b", text, re.I):
        checks.append(("fr_vat", value, valid_fr_vat(value, sirets[0] if len(sirets) == 1 else None)))
    for value in re.findall(r"\bIBAN\s*:?\s*([A-Z]{2}\d{2}[ A-Z0-9]+?)(?=\s*\||\s*;|\s+BIC\b|\n|$)", text):
        checks.append(("iban", value.strip(), valid_iban(value)))
    for value in re.findall(r"\bVAT ID\s*:?\s*([A-Z0-9]+)", text, re.I):
        if not value.upper().startswith("FR"):
            checks.append(("uk_vat", value, valid_uk_vat(value)))
    for value in re.findall(r"\bEIN\s*:?\s*([0-9-]+)", text, re.I):
        checks.append(("ein", value, valid_ein(value)))
    return checks


def parse_date(text: str, locale: str | None) -> tuple[date | None, str | None]:
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    local = re.search(r"\b(\d{1,2})([./])(\d{1,2})\2(\d{4})\b", text)
    normalized = folded(text)
    day_first = re.search(r"\b(\d{1,2}) ([a-z]+) (\d{4})\b", normalized)
    month_first = re.search(r"\b([a-z]+) (\d{1,2}),? (\d{4})\b", normalized)
    parts = None
    style = None
    if iso:
        parts, style = tuple(map(int, iso.groups())), "AAAA-MM-JJ"
    elif local:
        first, separator, second, year = local.groups()
        day, month = (second, first) if locale == "en-US" else (first, second)
        parts, style = (int(year), int(month), int(day)), f"numérique ({separator})"
    elif day_first and day_first[2] in MONTHS:
        parts, style = (int(day_first[3]), MONTHS[day_first[2]], int(day_first[1])), "mois en lettres"
    elif month_first and month_first[1] in MONTHS:
        parts, style = (int(month_first[3]), MONTHS[month_first[1]], int(month_first[2])), "mois en lettres"
    if parts is None:
        return None, None
    try:
        return date(*parts), style
    except ValueError:
        return None, style


def canonical_name(name: str) -> str:
    name = folded(name)
    name = re.sub(r"\b(?:ltd|limited)\.?\b", "ltd", name)
    name = re.sub(r"\bs[. ]*a[. ]*s\.?\b", "sas", name)
    return re.sub(r"[^\w ]", "", name).strip()


def document_validation(evidence: DocumentEvidence, *, today: date | None = None) -> Validation:
    result = Validation()
    if not evidence.available or evidence.quality_issues or (
        evidence.source_kind in {"ocr", "mixed"} and not evidence.ocr_cross_checked
    ):
        return result
    today = today or date.today()
    certain_lines = [line for line in evidence.lines if all(token.confidence >= 60 for token in line.tokens)]
    evidence = replace(evidence, text="\n".join(line.text for line in certain_lines))

    def issue(code, severity, location, printed, expected, difference, reason):
        result.flags.append(Flag(code, severity, location, str(printed), str(expected), str(difference), reason))

    printed_rows = []
    bad_lines = []
    signatures = {}
    row_taxes: dict[Decimal, list[Decimal]] = {}
    for index, row in enumerate(evidence.rows):
        values = {key: unique_value(row.cells.get(key, [])) for key in ("quantity", "unit_price", "total_ht")}
        certain = all(token.confidence >= 60 for token in row.line.tokens)
        if not certain or any(value is None for value in values.values()):
            continue
        quantity, price, amount = (values[key] for key in ("quantity", "unit_price", "total_ht"))
        expected = rounded(quantity * price)
        printed_rows.append(amount)
        location = f"Ligne {index + 1} « {row.description} »"
        if abs(expected - amount) > TOLERANCE:
            bad_lines.append(index + 1)
            issue("line_arithmetic", "erreur", location, f"{quantity} × {price} ; montant {amount}",
                  expected, abs(expected - amount), "Le produit des valeurs imprimées ne correspond pas au montant imprimé.")
        else:
            result.passed.append(f"Ligne {index + 1} : quantité × prix net = montant imprimé.")
        list_price = unique_value(row.cells.get("list_price", []))
        discount = unique_value(row.cells.get("discount", []))
        if list_price is not None and discount is not None:
            expected_price = rounded(list_price * (1 - discount / 100))
            if abs(expected_price - price) > TOLERANCE:
                issue("net_discount", "erreur", location, f"catalogue {list_price}, remise {discount} %, prix net {price}",
                      expected_price, abs(expected_price - price), "Le prix net imprimé ne correspond pas à la remise imprimée.")
            else:
                result.passed.append(f"Ligne {index + 1} : remise incluse dans le prix net ; pas de seconde application.")
        row_rate = unique_value(row.cells.get("tax_rate", []))
        row_tax = unique_value(row.cells.get("tax_amount", []))
        if row_rate is not None and row_tax is not None:
            expected_tax = rounded(amount * row_rate / 100)
            if abs(expected_tax - row_tax) > TOLERANCE:
                issue("line_tax", "erreur", location, f"base {amount}, taux {row_rate} %, taxe {row_tax}",
                      expected_tax, abs(expected_tax - row_tax), "Les trois valeurs fiscales sont imprimées mais incohérentes.")
            row_taxes.setdefault(row_rate, []).append(row_tax)
        signature = (canonical_name(row.description), quantity, price, amount)
        if signature in signatures:
            issue("duplicate_line", "avertissement", location, row.line.text,
                  "vérifier si la répétition est intentionnelle", "", f"Identique à la ligne {signatures[signature]}.")
        else:
            signatures[signature] = index + 1
    certain_boxes = {(line.page, line.box) for line in certain_lines}
    ht = unique_value([n for n in evidence.totals.get("total_ht", []) if (n.page, n.box) in certain_boxes])
    ttc = unique_value([n for n in evidence.totals.get("total_ttc", []) if (n.page, n.box) in certain_boxes])
    evidence = replace(evidence, taxes=[tax for tax in evidence.taxes if (tax.amount.page, tax.amount.box) in certain_boxes])
    if ht is not None and printed_rows and len(printed_rows) == len(evidence.rows):
        discounts = [
            number for number in evidence.discounts if (number.page, number.box) in certain_boxes
        ]
        line_sum = sum(printed_rows, Decimal(0)) - sum((abs(number.value) for number in discounts), Decimal(0))
        if abs(line_sum - ht) > TOLERANCE:
            consequence = (
                "Écart cohérent avec l'anomalie des lignes " + ", ".join(map(str, bad_lines))
                if bad_lines else "La somme des montants imprimés diffère du sous-total imprimé."
            )
            issue("subtotal", "erreur", "Sous-total", ht, line_sum, abs(line_sum - ht), consequence)
        else:
            result.passed.append(
                "Sous-total : somme des montants imprimés"
                + (" moins remise séparée imprimée" if discounts else "") + " = HT."
            )
    taxes = {tax.rate.value if tax.rate else None: tax.amount.value for tax in evidence.taxes}
    unambiguous_taxes = len({(tax.rate.value if tax.rate else None, tax.amount.value) for tax in evidence.taxes}) == len(taxes)
    if ht is not None and ttc is not None and evidence.taxes and unambiguous_taxes:
        total_tax = sum(taxes.values(), Decimal(0))
        if abs(ht + total_tax - ttc) > TOLERANCE:
            issue("tax_total", "erreur", "Total TTC", ttc, ht + total_tax, abs(ht + total_tax - ttc),
                  f"HT imprimé {ht} + TVA imprimée {total_tax} ne correspond pas au TTC imprimé.")
        else:
            result.passed.append(f"HT + taxes lues ({total_tax}) = TTC ; taux : {', '.join(str(rate) + ' %' for rate in taxes)}.")
    for rate, tax in taxes.items():
        if rate is None or not printed_rows or len(printed_rows) != len(evidence.rows):
            continue
        if sum(map(len, row_taxes.values())) == len(evidence.rows) and rate in row_taxes:
            expected_rows = sum(row_taxes[rate], Decimal(0))
            if abs(expected_rows - tax) > TOLERANCE:
                issue("tax_rate_sum", "erreur", f"Taxe à {rate} %", tax, expected_rows, abs(expected_rows - tax),
                      "La somme des taxes de ligne imprimées à ce taux diffère du récapitulatif.")
        bases = {
            item.base.value for item in evidence.taxes
            if item.rate and item.rate.value == rate and item.base is not None
        }
        if len(bases) == 1:
            base = next(iter(bases))
            expected_tax = rounded(base * rate / 100)
            if abs(expected_tax - tax) > TOLERANCE:
                issue("tax_printed_basis", "erreur", f"Taxe à {rate} %", f"base {base}, taxe {tax}",
                      expected_tax, abs(expected_tax - tax), "La base et le taux imprimés ne correspondent pas à la taxe imprimée.")
            else:
                result.passed.append(f"Taxe {rate} % : base imprimée {base} × taux = {tax}.")
        elif rate == 0:
            if abs(tax) > TOLERANCE:
                issue("tax_rate", "erreur", "Taxe à 0 %", tax, 0, abs(tax), "Une taxe au taux imprimé de 0 % doit être nulle.")
            else:
                result.passed.append("Taxe 0 % : montant nul.")
        elif len(taxes) == 1 and ht is not None:
            per_line = sum((rounded(amount * rate / 100) for amount in printed_rows), Decimal(0))
            on_base = rounded(ht * rate / 100)
            if min(abs(tax - per_line), abs(tax - on_base)) > TOLERANCE:
                issue("tax_rate", "erreur", f"Taxe à {rate} %", tax, f"{on_base} (base) ou {per_line} (lignes)",
                      min(abs(tax - per_line), abs(tax - on_base)), "Le taux et les bases imprimées ne justifient pas la taxe lue.")
            else:
                result.passed.append(f"Taxe {rate} % : cohérente avec la base ou l'arrondi par ligne.")
        elif len(printed_rows) <= 16:
            subsets = []
            for count in range(len(printed_rows) + 1):
                for indexes in combinations(range(len(printed_rows)), count):
                    base = sum((printed_rows[index] for index in indexes), Decimal(0))
                    per_line = sum((rounded(printed_rows[index] * rate / 100) for index in indexes), Decimal(0))
                    if min(abs(rounded(base * rate / 100) - tax), abs(per_line - tax)) <= TOLERANCE:
                        subsets.append(indexes)
            if len(subsets) == 1:
                result.passed.append(f"Taxe {rate} % : base identifiée sur les lignes {', '.join(str(i + 1) for i in subsets[0])}.")
            elif not subsets:
                issue("tax_basis", "avertissement", f"Taxe à {rate} %", tax, "base taxable à vérifier", "",
                      "Aucun regroupement des montants imprimés ne justifie ce taux ; ventilation nécessaire.")
    if re.search(r"discount already applied|remise.*(?:incluse|appliqu)", evidence.text, re.I):
        result.passed.append("Remise annoncée déjà incluse dans les prix nets : aucune seconde application.")
    _names(evidence, issue)
    _dates(evidence, result, issue, today)
    identifier_labels = {"siret": "SIRET", "fr_vat": "TVA FR", "iban": "IBAN", "uk_vat": "VAT UK", "ein": "EIN US"}
    for kind, value, valid in identifier_checks(evidence.text):
        if valid:
            result.passed.append(f"Identifiant {identifier_labels[kind]} : contrôle de structure ou clé réussi.")
        else:
            issue(f"identifier_{kind}", IDENTIFIER_SEVERITIES[kind], identifier_labels[kind], value,
                  "structure ou clé de contrôle valide", "",
                  "Contrôle informatif : ce contrôle échoue sur des identifiants synthétiques de référence."
                  if IDENTIFIER_SEVERITIES[kind] == "information" else
                  "La structure ou la clé imprimée ne satisfait pas le contrôle ; identifiant à vérifier.")
    _country(evidence, result, issue)
    if evidence.source_kind == "consensus":
        result.flags = []
    result.flags.sort(key=lambda flag: {"erreur": 0, "avertissement": 1, "information": 2}[flag.severity])
    return result


def _names(evidence, issue):
    text = evidence.text
    for label, pattern in (
        ("fournisseur", r"(?:Supplier|Fournisseur)\s*:\s*(.+?)(?=\s+(?:Bill to|Factur[ée] [aà])\s*:|$)"),
        ("client", r"(?:Bill to|Factur[ée] [aà])\s*:\s*(.+)$"),
    ):
        names = []
        for line in text.replace(" ; ", "\n").splitlines():
            match = re.search(pattern, line, re.I)
            if match and match[1].strip() not in names:
                names.append(match[1].strip())
        for left, right in combinations(names, 2):
            if canonical_name(left) == canonical_name(right):
                issue("name_typography", "information", f"Nom du {label}", f"« {left} » / « {right} »",
                      "formes équivalentes", "", "Casse, accents, ponctuation ou forme Ltd/Limited/SAS harmonisables ; pas une erreur.")
            else:
                diff = SequenceMatcher(None, left, right)
                changes = [f"« {left[a:b]} » → « {right[c:d]} »" for tag, a, b, c, d in diff.get_opcodes() if tag != "equal"]
                issue("name_variant", "avertissement", f"Nom du {label}", f"« {left} » / « {right} »",
                      "graphie à confirmer", "; ".join(changes), "Le même rôle apparaît avec des noms différents dans le document.")


def _dates(evidence, result, issue, today):
    dates = {}
    styles = set()
    for line in evidence.text.replace(" ; ", "\n").splitlines():
        normalized = folded(line)
        kind = "facture" if re.search(r"invoice date|date de facture|^date\s*:", normalized) else (
            "échéance" if re.search(r"due date|echeance", normalized) else None
        )
        if kind is None:
            continue
        value, style = parse_date(line, evidence.locale)
        if style:
            styles.add(style)
        if value:
            dates[kind] = value
            if value.year < 1990 or value > today + timedelta(days=366):
                issue("date_plausibility", "avertissement", f"Date de {kind}", value,
                      "date plausible à confirmer", "", "Date très ancienne ou distante de plus d'un an dans le futur.")
        else:
            issue("date_invalid", "avertissement", f"Date de {kind}", line, "date calendaire valide", "",
                  "Date illisible ou impossible ; aucune interprétation supposée.")
    if len(styles) > 1:
        issue("date_formats", "information", "Dates", ", ".join(sorted(styles)), "format uniforme conseillé", "",
              "Plusieurs formats imprimés dans le même document.")
    if "facture" in dates and "échéance" in dates:
        invoice, due = dates["facture"], dates["échéance"]
        terms = re.search(r"\bNET\s*(\d+)\b|(?:sous|within)\s+(\d+)\s+(?:jours|days)", evidence.text, re.I)
        if due < invoice:
            issue("due_before_invoice", "avertissement", "Échéance", due, f"au moins {invoice}", (invoice - due).days,
                  "L'échéance précède la date de facture.")
        elif terms:
            days = int(next(group for group in terms.groups() if group))
            if days > 3650:
                issue("payment_terms", "avertissement", "Conditions de paiement", terms.group(), "délai à confirmer", "",
                      "Délai inhabituellement long ; calcul non appliqué.")
            else:
                expected = invoice + timedelta(days=days)
                if due != expected:
                    issue("payment_terms", "avertissement", "Échéance", due, expected, abs((due - expected).days),
                          f"Date de facture {invoice} + {days} jours ne correspond pas à l'échéance imprimée.")
                else:
                    result.passed.append(f"Échéance : {invoice} + {days} jours = {due}.")
        else:
            result.passed.append("Échéance postérieure ou égale à la date de facture.")


def _country(evidence, result, issue):
    text = evidence.text
    countries = set(re.findall(r"\b(FR|GB|US)-\d{4}-\d+\b", text))
    codes = set(re.findall(r"\b(EUR|GBP|USD)\b", text))
    symbols = {symbol for symbol in ("€", "£", "$") if symbol in text}
    mapping = {"FR": ("EUR", "€"), "GB": ("GBP", "£"), "US": ("USD", "$")}
    if len(countries) > 1:
        issue("country_prefixes", "avertissement", "Numéro de facture", ", ".join(sorted(countries)),
              "préfixe de pays à confirmer", "", "Plusieurs préfixes de pays dans le document.")
    if len(countries) == 1:
        country = next(iter(countries))
        currency, symbol = mapping[country]
        if codes - {currency} or symbols - {symbol}:
            issue("country_currency", "avertissement", "Pays / devise", f"{country}, {sorted(codes)}, {sorted(symbols)}",
                  currency, "", "Le préfixe et les devises imprimées diffèrent ; facturation internationale possible.")
        elif currency in codes or symbol in symbols:
            result.passed.append(f"Devise {currency} cohérente avec le préfixe {country}.")
        vocab = folded(text)
        unexpected = country == "US" and bool(re.search(r"\b(?:tva|ttc|vat)\b", vocab)) or (
            country == "FR" and "sales tax" in vocab
        )
        if unexpected:
            issue("country_vocabulary", "information", "Vocabulaire fiscal", country, "vocabulaire local à confirmer", "",
                  "Vocabulaire fiscal mixte ; facture bilingue possible.")
        plausible = {"FR": {Decimal(0), Decimal("2.1"), Decimal("5.5"), Decimal(10), Decimal(20)},
                     "GB": {Decimal(0), Decimal(5), Decimal(20)}}
        for tax in evidence.taxes:
            if tax.rate is None:
                continue
            rate = tax.rate.value
            suspect = rate not in plausible[country] if country in plausible else not (0 <= rate <= 15)
            if suspect:
                issue("country_tax_rate", "avertissement", "Taux fiscal", f"{country} : {rate} %",
                      "taux applicable à confirmer", "", "Taux inhabituel pour le pays ; aucune règle fiscale supposée certaine.")


def render_report(evidence: DocumentEvidence, validation: Validation, statuses: dict[str, str],
                  technical_errors: list[str], *, available: bool = True, insufficient: bool = False,
                  suggestions: dict | None = None) -> str:
    insufficient = insufficient or bool(evidence.quality_issues) or (
        evidence.source_kind in {"ocr", "mixed"} and not evidence.ocr_cross_checked
    )
    identifier = re.search(r"\b(?:FR|GB|US)-\d{4}-\d+\b", evidence.text)
    title = identifier.group() if identifier else "document téléversé"
    currency = {"fr-FR": "EUR", "en-GB": "GBP", "en-US": "USD"}.get(evidence.locale, "devise non identifiée")
    verified = sum(value in {"valide", "récupéré (corroboré par OCR)"} for value in statuses.values())
    anomalies = [flag for flag in validation.flags if flag.severity != "information"]
    lines = [f"## Analyse de la facture {title} ({currency})"]
    doubtful = sum(value in {"neutralisé (non fiable)", "non vérifié"} for value in statuses.values())
    if statuses and doubtful / len(statuses) > .5:
        lines.append(
            "**Image de qualité insuffisante : renvoyez un scan plus net "
            "(300 dpi, bien droit, sans flou)**"
        )
    if insufficient:
        lines.append("**Preuves OCR insuffisantes : extraction non recoupée.**")
    elif not available:
        lines.append("**Preuves OCR indisponibles : extraction non recoupée.**")
    elif anomalies:
        lines.append(f"**{len(anomalies)} anomalie(s) détectée(s) dans la facture.**")
    elif not verified:
        lines.append("**Analyse partielle : preuves insuffisantes pour conclure.**")
    else:
        lines.append("**Aucune anomalie détectée dans les contrôles réalisables.**")
    lines.append(f"{verified} champs sur {len(statuses)} vérifiés dans le document.")
    lines.append("### Anomalies dans la facture")
    if not validation.flags:
        lines.append("Aucune anomalie documentaire détectée." if available and not insufficient else "Aucune conclusion documentaire possible sans preuves suffisantes.")
    for flag in validation.flags:
        lines.append(
            f"- **{flag.severity.capitalize()} — {flag.location}** : imprimé « {flag.printed} » ; "
            f"attendu : {flag.expected}. " + (f"Écart : {flag.difference}. " if flag.difference else "") + flag.reason
        )
    lines.append("### Contrôles réussis")
    lines.extend(f"- {message}" for message in validation.passed)
    if not validation.passed:
        lines.append("Aucun contrôle documentaire concluant avec les preuves disponibles.")
    recovered = sum(value == "récupéré (corroboré par OCR)" for value in statuses.values())
    neutralized = sum(value == "neutralisé (non fiable)" for value in statuses.values())
    unverified = sum(value == "non vérifié" for value in statuses.values())
    lines.append(
        f"\nLecture : {recovered} valeur(s) récupérée(s), {neutralized} neutralisée(s), "
        f"{unverified} non vérifiée(s). Détails de lecture dans « Détails techniques »."
    )
    if doubtful:
        lines.append("**Valeur non recoupée : à vérifier manuellement.** Les lectures ci-dessous ne font pas partie du JSON standardisé.")
    if suggestions:
        lines.append("### Lectures du modèle non fiables — hors JSON")
        for path, value in suggestions.items():
            escaped = json.dumps(value, ensure_ascii=False, default=str).replace("`", "\\u0060")
            lines.append(f"- `{path}` — lecture du modèle non vérifiée : `{escaped}`.")
    for error in technical_errors:
        if "Quota ZeroGPU" in error:
            lines.append(error[error.index("Quota ZeroGPU"):].split(" (", 1)[0])
        else:
            stage = "La seconde passe" if "seconde passe" in error else "La première passe"
            if "incomplète" in error:
                cause = "la réponse du modèle est incomplète"
            elif "module" in error.casefold() or "cuda" in error.casefold():
                cause = "le moteur d'inférence n'est pas disponible"
            elif "message non communiqué" in error:
                cause = "le service n'a pas communiqué la cause ; consultez les détails techniques"
            else:
                cause = "la réponse ou le service du modèle n'a pas permis une lecture exploitable"
            lines.append(f"{stage} a échoué : {cause}. Repli sur les preuves documentaires disponibles.")
    return "\n\n".join(lines)
