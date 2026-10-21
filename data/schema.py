from datetime import date

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    total_ht: float


class Invoice(BaseModel):
    supplier_name: str
    invoice_date: date
    total_ht: float
    total_ttc: float
    line_items: list[LineItem] = Field(default_factory=list)
