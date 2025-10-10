"""HTML to PDF rendering helpers for reports."""

from __future__ import annotations

import io
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Optional, Tuple

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ...utils.logging import get_logger
from ..models.report import ClinicalReport

logger = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
)


class _HTMLStripper(HTMLParser):
    """Simple HTML to text converter used for the ReportLab fallback."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:  # pragma: no cover - simple utility
        self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)



def render_report_html(report: ClinicalReport) -> str:
    """Render the report Jinja2 template to an HTML string."""

    template = _env.get_template("report.html.j2")
    return template.render(report=report)


def render_report_pdf(report: ClinicalReport) -> Tuple[str, bytes]:
    """Render a :class:`ClinicalReport` to PDF bytes."""

    html = render_report_html(report)
    filename = build_report_filename(report)

    pdf_bytes = _render_with_weasyprint(html)
    if pdf_bytes is None:
        pdf_bytes = _render_with_reportlab(html)

    if pdf_bytes is None:
        raise RuntimeError("PDF rendering requires WeasyPrint oder ReportLab.")

    logger.info("Generated report PDF", extra={"extra_fields": {"filename": filename, "size": len(pdf_bytes)}})
    return filename, pdf_bytes


def build_report_filename(report: ClinicalReport) -> str:
    """Generate a deterministic filename for the PDF output."""

    patient = re.sub(r"[^A-Za-z0-9]+", "_", report.patient_name).strip("_") or "patient"
    return f"Behandlungsbericht_{patient}_{report.report_date.isoformat()}.pdf"




def _render_with_weasyprint(html: str) -> Optional[bytes]:
    """Render HTML with WeasyPrint if installed."""

    try:  # pragma: no cover - import depends on optional dependency
        from weasyprint import HTML  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        return HTML(string=html).write_pdf()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("WeasyPrint rendering failed: %s", exc)
        return None


def _render_with_reportlab(html: str) -> Optional[bytes]:
    """Render HTML using a text-only ReportLab fallback."""

    try:  # pragma: no cover - optional dependency import
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        logger.warning("ReportLab not available for PDF fallback")
        return None

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text = pdf.beginText(40, height - 60)
    text.setLeading(14)

    plain_text = _strip_html(html)
    for line in plain_text.splitlines():
        if not line.strip():
            text.textLine("")
            continue
        text.textLine(line.strip())

    pdf.drawText(text)
    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return buffer.read()


def _strip_html(html: str) -> str:
    """Remove HTML tags to produce a text-only representation."""

    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()
