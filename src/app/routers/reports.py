"""REST router for clinical report operations."""

from __future__ import annotations

import io
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from ...utils.logging import get_logger
from ..config.report_defaults import get_report_defaults
from ..models.report import (
    ClinicalReport,
    ClinicalReportDraft,
    ReportDefaults,
    ReportSaveResponse,
    ReportSuggestionRequest,
)
from ..services.pdf_renderer import build_report_filename, render_report_pdf
from ..services.report_extractor import report_extractor_service

router = APIRouter(prefix="/api/reports", tags=["reports"])
logger = get_logger(__name__)
_STORAGE_DIR = Path("storage/reports")


@router.get("/defaults", response_model=ReportDefaults)
async def read_report_defaults() -> ReportDefaults:
    """Return default header fields for the report form."""

    return get_report_defaults()


@router.post("/suggest", response_model=ClinicalReportDraft)
async def suggest_report(payload: ReportSuggestionRequest) -> ClinicalReportDraft:
    """Generate a report draft by analysing the transcript with the LLM."""

    return await report_extractor_service.suggest_report(payload)


@router.post("/pdf")
async def generate_report_pdf(report: ClinicalReport) -> StreamingResponse:
    """Render the submitted report as a downloadable PDF."""

    try:
        filename, pdf_bytes = render_report_pdf(report)
    except RuntimeError as exc:
        logger.error("PDF generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PDF-Erstellung ist nicht verfügbar.",
        ) from exc

    headers = {
        "Content-Disposition": f"attachment; filename=\"{filename}\"",
    }
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)


@router.post("/save", response_model=ReportSaveResponse)
async def save_report(report: ClinicalReport) -> ReportSaveResponse:
    """Persist the report payload to disk as JSON for later retrieval."""

    _STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(build_report_filename(report)).with_suffix(".json")
    path = _STORAGE_DIR / filename

    try:
        path.write_text(report.model_dump_json(indent=2, exclude_none=True, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        logger.error("Failed to persist report: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bericht konnte nicht gespeichert werden.",
        ) from exc

    logger.info("Stored clinical report", extra={"extra_fields": {"path": str(path)}})
    return ReportSaveResponse(success=True, path=str(path))
