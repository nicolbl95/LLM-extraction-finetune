import re
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Annotated

from pydantic import (
    AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field,
    ValidationError, ValidationInfo, model_validator,
)

MONEY_TOLERANCE = Decimal("0.02")


def parse_invoice_number(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("Un booléen n'est pas un montant ou une quantité.")
    if isinstance(value, str):
        value = value.strip()
        if "," in value:
            if not re.fullmatch(r"[+-]?\d+,\d+", value) or re.fullmatch(r"[+-]?[1-9]\d{0,2},\d{3}", value):
                raise ValueError("Nombre ambigu : séparateur décimal ou de milliers indéterminé.")
            return value.replace(",", ".")
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value):
            raise ValueError("Nombre invalide : chiffres et séparateur décimal explicite requis.")
    return value


def require_text(value: str) -> str:
    if not value.strip() or value.strip().upper() in {"ERROR", "INVALID"}:
        raise ValueError("Texte vide ou explicitement invalide.")
    return value


def normalize_supplier(value: str) -> str:
    value = " ".join(require_text(value).split())
    if re.search(r"\b(?:LId|L1d|LtI|Lt1)\.?$", value):
        raise ValueError("Suffixe fournisseur OCR ambigu ; aucune substitution automatique.")
    return re.sub(r"\bLtd\.?$", "Ltd", value, flags=re.IGNORECASE)


InvoiceNumber = Annotated[float, BeforeValidator(parse_invoice_number)]
InvoiceText = Annotated[str, AfterValidator(require_text)]
SupplierName = Annotated[str, AfterValidator(normalize_supplier)]


def check_invoice_total(totals: list[float], total_ht: float) -> None:
    expected = sum((Decimal(str(total)) for total in totals), Decimal(0))
    actual = Decimal(str(total_ht))
    difference = abs(expected - actual)
    if difference > MONEY_TOLERANCE:
        raise ValueError(
            f"Incohérence de facture : somme des total_ht des lignes = {expected}, "
            f"mais total_ht global attendu = {actual}. Écart absolu : {difference}. "
            f"Tolérance : {MONEY_TOLERANCE}."
        )


class LineItem(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    description: InvoiceText
    quantity: InvoiceNumber
    unit_price: InvoiceNumber
    total_ht: InvoiceNumber

    def check_arithmetic(self) -> None:
        try:
            expected = (Decimal(str(self.quantity)) * Decimal(str(self.unit_price))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )
        except InvalidOperation as exc:
            raise ValueError("Montant de ligne non representable au centime.") from exc
        actual = Decimal(str(self.total_ht))
        difference = abs(expected - actual)
        if difference > MONEY_TOLERANCE:
            raise ValueError(
                f"Incohérence sur la ligne '{self.description}' : "
                f"quantity ({self.quantity}) * unit_price ({self.unit_price}) = {expected}, "
                f"mais total_ht indiqué = {actual}. Écart absolu : {difference}. "
                f"Tolérance : {MONEY_TOLERANCE}."
            )

    @model_validator(mode="after")
    def validate_line_total(self, info: ValidationInfo) -> "LineItem":
        if not (isinstance(info.context, dict) and info.context.get("validate_arithmetic") is False):
            self.check_arithmetic()
        return self


class Invoice(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    supplier_name: SupplierName
    invoice_date: date
    total_ht: InvoiceNumber
    total_ttc: InvoiceNumber
    line_items: list[LineItem] = Field(default_factory=list)

    def check_arithmetic(self) -> None:
        check_invoice_total([item.total_ht for item in self.line_items], self.total_ht)

    @model_validator(mode="after")
    def validate_invoice_total(self, info: ValidationInfo) -> "Invoice":
        if not (isinstance(info.context, dict) and info.context.get("validate_arithmetic") is False):
            self.check_arithmetic()
        return self

    def arithmetic_warnings(self) -> list[str]:
        warnings = []
        for index, item in enumerate(self.line_items, start=1):
            try:
                item.check_arithmetic()
            except ValueError as exc:
                warnings.append(
                    f"Ligne {index} : {exc} "
                    f"Sous-total HT de la facture : {self.total_ht}."
                )
        try:
            self.check_arithmetic()
        except ValueError as exc:
            warnings.append(f"Total HT de la facture : {exc}")
        return warnings


class LineItemResult(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    description: InvoiceText | None = None
    quantity: InvoiceNumber | None = None
    unit_price: InvoiceNumber | None = None
    total_ht: InvoiceNumber | None = None


class InvoiceResult(BaseModel):
    """Partial export only; never used to accept an invoice in strict scoring."""

    model_config = ConfigDict(allow_inf_nan=False)

    supplier_name: SupplierName | None = None
    invoice_date: date | None = None
    total_ht: InvoiceNumber | None = None
    total_ttc: InvoiceNumber | None = None
    line_items: list[LineItemResult | None] | None = None


def safe_invoice_payload(source: dict, warnings: list[str]) -> dict:
    cleaned = deepcopy(source)
    cleaned.setdefault("line_items", [])
    try:
        Invoice.model_validate(cleaned, context={"validate_arithmetic": False})
    except ValidationError as exc:
        for error in exc.errors(include_url=False, include_context=False):
            location = error["loc"]
            path = ".".join(map(str, location))
            warnings.append(
                f"{path} : {error['msg']} Valeur {error.get('input')!r} remplacée par null."
            )
            target = cleaned
            for part in location[:-1]:
                target = target[part]
            target[location[-1]] = None
    result = InvoiceResult.model_validate(cleaned)
    if result.supplier_name is not None and result.supplier_name != source.get("supplier_name"):
        warnings.append(
            f"supplier_name : normalisation typographique de {source.get('supplier_name')!r} "
            f"en {result.supplier_name!r}, sans substitution OCR."
        )

    def unset(item: BaseModel, field: str, path: str, reason: str) -> None:
        value = getattr(item, field)
        if value is not None:
            warnings.append(f"{path} : valeur {value!r} remplacée par null ; {reason}.")
            setattr(item, field, None)

    totals = [
        item.total_ht for item in result.line_items or []
        if item is not None and item.total_ht is not None
    ]
    global_uncertain = result.total_ht is None
    if result.line_items is not None and len(totals) == len(result.line_items) and result.total_ht is not None:
        try:
            check_invoice_total(totals, result.total_ht)
        except ValueError as exc:
            warnings.append(str(exc))
            global_uncertain = True

    incomplete_lines = result.line_items is None
    for index, item in enumerate(result.line_items or []):
        path = f"line_items.{index}"
        if item is None:
            incomplete_lines = True
            continue
        original = source.get("line_items", [])[index]
        for field in ("quantity", "unit_price", "total_ht"):
            value = getattr(item, field)
            if isinstance(original, dict) and isinstance(original.get(field), str) and value is not None:
                warnings.append(
                    f"{path}.{field} : nombre normalisé de {original[field]!r} en {value}."
                )
        invalid = item.quantity is None or item.unit_price is None or item.total_ht is None
        if not invalid:
            try:
                LineItem.model_validate({
                    **item.model_dump(), "description": item.description or f"Ligne {index + 1}",
                })
            except ValidationError as exc:
                warnings.append(f"Ligne {index + 1} : {exc}")
                invalid = True
        if invalid:
            incomplete_lines = True
            for field in ("quantity", "unit_price", "total_ht"):
                unset(item, field, f"{path}.{field}", "opérande fautif impossible à isoler")
        elif global_uncertain:
            unset(item, "total_ht", f"{path}.total_ht", "écart global non attribuable à une ligne certaine")
    if global_uncertain or incomplete_lines:
        unset(result, "total_ht", "total_ht", "total global non vérifiable avec les lignes disponibles")
    for field in ("total_ht", "total_ttc"):
        value = getattr(result, field)
        if isinstance(source.get(field), str) and value is not None:
            warnings.append(f"{field} : nombre normalisé de {source[field]!r} en {value}.")
    return result.model_dump(mode="json")
