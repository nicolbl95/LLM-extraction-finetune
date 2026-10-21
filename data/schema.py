from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MONEY_TOLERANCE = Decimal("0.02")


class LineItem(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    description: str
    quantity: float
    unit_price: float
    total_ht: float

    @model_validator(mode="after")
    def validate_line_total(self) -> Self:
        try:
            expected = (Decimal(str(self.quantity)) * Decimal(str(self.unit_price))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP,
            )
        except InvalidOperation as exc:
            raise ValueError("Montant de ligne non representable au centime.") from exc
        actual = Decimal(str(self.total_ht))
        if abs(expected - actual) > MONEY_TOLERANCE:
            raise ValueError(
                f"Incoherence de ligne : quantity * unit_price = {expected}, "
                f"mais total_ht = {actual} (tolerance : {MONEY_TOLERANCE})."
            )
        return self


class Invoice(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    supplier_name: str
    invoice_date: date
    total_ht: float
    total_ttc: float
    line_items: list[LineItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_invoice_total(self) -> Self:
        expected = sum((Decimal(str(item.total_ht)) for item in self.line_items), Decimal(0))
        actual = Decimal(str(self.total_ht))
        if abs(expected - actual) > MONEY_TOLERANCE:
            raise ValueError(
                f"Incoherence de facture : somme des total_ht des lignes = {expected}, "
                f"mais total_ht global = {actual} (tolerance : {MONEY_TOLERANCE})."
            )
        return self
