"""Pydantic models for physiotherapy treatment reports."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class InsuranceType(str, Enum):
    """German health insurance categories."""

    PRIVAT = "PRIVAT"
    GESETZLICH = "GESETZLICH"
    UNKLAR = "UNKLAR"


class TreatmentOutcome(str, Enum):
    """Therapy outcome categories with synonym mapping."""

    BESCHWERDEFREI = "BESCHWERDEFREI"
    LINDERUNG = "LINDERUNG"
    KEINE_BESSERUNG = "KEINE_BESSERUNG"
    UNKLAR = "UNKLAR"

    @classmethod
    def from_text(cls, value: str | None) -> "TreatmentOutcome":
        """Map free-text descriptions to the canonical enum value."""

        if not value:
            return cls.UNKLAR

        normalized = value.strip().lower()
        if normalized in {"schmerzfrei", "beschwerdefrei", "keine beschwerden"}:
            return cls.BESCHWERDEFREI
        if normalized in {"besser", "verbesserung", "linderung"}:
            return cls.LINDERUNG
        if normalized in {"unverändert", "keine änderung", "keine besserung"}:
            return cls.KEINE_BESSERUNG
        return cls.UNKLAR


class ReportDefaults(BaseModel):
    """Default header values provided by configuration."""

    doctor_name: str
    patient_name: str
    patient_dob: str
    prescription_date: str
    treatment_date_from: str
    treatment_date_to: str
    physiotherapist_name: str
    report_city: str = Field(..., description="Fallback city for the report")


class ClinicalReportDraft(BaseModel):
    """Report draft payload returned by the suggestion endpoint."""

    doctor_name: str
    patient_name: str
    patient_dob: str
    prescription_date: str
    treatment_date_from: str
    treatment_date_to: str
    physiotherapist_name: str
    report_city: str
    report_date: date
    insurance_type: InsuranceType
    diagnoses: List[str] = Field(default_factory=list)
    prescribed_therapy_type: str
    patient_problem_statement: str
    treatment_outcome: TreatmentOutcome
    therapy_status_note: str
    follow_up_recommendation: str

    @field_validator("diagnoses", mode="before")
    @classmethod
    def ensure_list(cls, value: str | List[str]) -> List[str]:
        """Convert string inputs to a list of diagnoses."""

        if isinstance(value, list):
            return [item for item in value if item]
        if isinstance(value, str):
            return [part.strip() for part in value.split("\n") if part.strip()]
        return []


class ClinicalReport(ClinicalReportDraft):
    """Complete report payload submitted for saving or PDF generation."""

    transcript: Optional[str] = Field(
        default=None,
        description="Optional transcript snapshot used for auditing.",
    )


class ReportSuggestionRequest(BaseModel):
    """Request body for report suggestions."""

    transcript: Optional[str] = None
    transcripts: Optional[List[str]] = None
    accumulated_transcriptions: Optional[List[str]] = None

    def combined_text(self) -> str:
        """Combine all transcript sources into a single string."""

        segments: List[str] = []
        if self.transcript:
            segments.append(self.transcript)
        if self.transcripts:
            segments.extend(self.transcripts)
        if self.accumulated_transcriptions:
            segments.extend(self.accumulated_transcriptions)

        combined = "\n".join(
            segment.strip() for segment in segments if segment and segment.strip()
        )
        return combined.strip()

    @field_validator("transcript", "transcripts", "accumulated_transcriptions")
    @classmethod
    def empty_to_none(cls, value):
        """Normalize empty collections/strings to ``None``."""

        if value in ("", [], ()):  # pragma: no cover - defensive
            return None
        return value


class ReportSaveResponse(BaseModel):
    """Response returned after persisting a report."""

    success: bool
    path: Optional[str] = None


class PdfRenderResult(BaseModel):
    """Metadata returned after PDF generation."""

    filename: str
    size_bytes: int
