"""Original invoice compositions sharing their content with the text dataset."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pymupdf
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

STYLES = (
    ("#202020", "#34495e", "#525252"),
    ("#444444", "#64748b", "#475569"),
    ("#3155a5", "#2563a6", "#4338a0"),
    ("#9a3412", "#92400e", "#7f1d1d"),
    ("#087caa", "#0369a1", "#155e75"),
    ("#345b50", "#3f6255", "#365b63"),
    ("#373353", "#4c3866", "#1e4664"),
    ("#934858", "#7c455d", "#8c4f3f"),
    ("#28734a", "#27655d", "#3b6a30"),
)


@dataclass
class Document:
    title: str
    seller: list[str]
    customer: list[str]
    metadata: list[str]
    headers: list[str]
    rows: list[list[str]]
    description_column: int
    totals: list[tuple[str, str]]
    payment: list[str]
    legal: list[str]
    template: int
    font: str
    font_size: float
    accent: str
    margin: int
    gap: int

    def text_sections(self) -> list[str]:
        parties = ["\n".join(self.seller), "\n".join(self.customer)]
        if self.template in (3, 5):
            parties.reverse()
        table = "\n".join(" | ".join(row) for row in [self.headers, *self.rows])
        totals = "\n".join(f"{label} : {amount}" for label, amount in self.totals)
        payment = "\n".join(self.payment)
        sections = [self.title, *parties, "\n".join(self.metadata)]
        if self.template == 4:
            sections.append(payment)
        if self.template == 5:
            sections.extend([totals, table])
        else:
            sections.extend([table, totals])
        if self.template != 4:
            sections.append(payment)
        sections.append("\n".join(self.legal))
        return sections


def render_png(document: Document, path: Path) -> None:
    """Draw a single A4 page, failing explicitly if any content would overflow."""
    buffer = BytesIO()
    width, height = 595, 842
    canvas = Canvas(buffer, pagesize=(width, height), invariant=1)
    margin = document.margin
    available = width - 2 * margin
    accent = colors.HexColor(document.accent)
    style = ParagraphStyle(
        "invoice", fontName=document.font, fontSize=document.font_size,
        leading=document.font_size + 3, textColor=colors.HexColor("#161616"),
    )
    y = height - margin

    def paragraph(text: str, x: float, top: float, space: float, size: float | None = None) -> float:
        paragraph_style = style if size is None else ParagraphStyle(
            "heading", parent=style, fontSize=size, leading=size + 3,
        )
        content = Paragraph(escape(text), paragraph_style)
        _, content_height = content.wrap(space, height)
        if top - content_height < margin:
            raise ValueError(f"Invoice content overflows page: {path.name}")
        content.drawOn(canvas, x, top - content_height)
        return top - content_height

    def block(lines: list[str], x: float, top: float, space: float) -> float:
        for line in lines:
            top = paragraph(line, x, top, space) - 2
        return top

    if document.template in (0, 4, 8):
        canvas.setFillColor(accent)
        canvas.rect(0, height - 12, width, 12, fill=1, stroke=0)
    elif document.template in (5, 7):
        canvas.setFillColor(accent)
        canvas.rect(0, 0, 12, height, fill=1, stroke=0)
    if document.template == 3:
        y = paragraph(document.title, margin + available * 0.15, y, available * 0.85, 20)
    else:
        y = paragraph(document.title, margin, y, available, 16 if document.template == 1 else 22)
    y -= document.gap
    left, right = document.seller, document.customer
    if document.template in (3, 5):
        left, right = right, left
    left_bottom = block(left, margin, y, available * 0.48)
    right_bottom = block(right, margin + available * 0.53, y, available * 0.47)
    y = min(left_bottom, right_bottom) - document.gap
    y = block(document.metadata, margin, y, available) - document.gap

    def payment(top: float) -> float:
        return block(document.payment, margin, top, available) - document.gap

    def totals(top: float) -> float:
        x = margin if document.template == 5 else margin + available * 0.46
        space = available * 0.54
        canvas.setStrokeColor(accent)
        canvas.line(x, top, x + space, top)
        top -= 6
        for label, value in document.totals:
            top = paragraph(f"{label} : {value}", x, top, space) - 3
        return top - document.gap

    if document.template == 4:
        y = payment(y)
    if document.template == 5:
        y = totals(y)

    count = len(document.headers)
    widths = [available * 0.68 / (count - 1)] * count
    widths[document.description_column] = available * 0.32
    cell_style = ParagraphStyle("cell", parent=style, fontSize=document.font_size - 0.5, leading=document.font_size + 2)
    table = Table(
        [[Paragraph(escape(value), cell_style) for value in row] for row in [document.headers, *document.rows]],
        colWidths=widths, hAlign="LEFT",
    )
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, 0), 1, accent),
    ]
    if document.template in (0, 2, 4, 8):
        commands.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9edf2")))
    if document.template in (0, 1, 2, 7, 8):
        commands.append(("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#cccccc")))
    if document.template in (4, 6):
        commands.append(("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f4f6")]))
    table.setStyle(TableStyle(commands))
    _, table_height = table.wrap(available, height)
    if y - table_height < margin:
        raise ValueError(f"Invoice table overflows page: {path.name}")
    table.drawOn(canvas, margin, y - table_height)
    y -= table_height + document.gap
    if document.template != 5:
        y = totals(y)
    if document.template != 4:
        y = payment(y)
    block(document.legal, margin, y, available)
    canvas.showPage()
    canvas.save()
    path.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(stream=buffer.getvalue(), filetype="pdf") as pdf:
        pdf[0].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(str(path))
