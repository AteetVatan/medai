"""LLM-powered extraction of clinical report fields."""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, Optional

from fastapi import HTTPException, status

from ...services.llm_service import llm_service
from ...utils.logging import get_logger
from ..config.report_defaults import get_report_defaults
from ..models.report import (
    ClinicalReportDraft,
    InsuranceType,
    ReportSuggestionRequest,
    TreatmentOutcome,
)

logger = get_logger(__name__)


class ReportExtractorService:
    """Service that orchestrates LLM extraction with deterministic post-processing."""

    def __init__(self, llm_client=llm_service):
        self._llm_client = llm_client
        self._defaults = get_report_defaults()

    async def suggest_report(self, payload: ReportSuggestionRequest) -> ClinicalReportDraft:
        """Generate a report draft from the supplied transcript payload."""

        transcript_text = payload.combined_text()
        if not transcript_text:
            logger.warning("Report suggestion requested without transcript content")
            return self._empty_draft()

        prompt = self._build_prompt(transcript_text)
        try:
            response = await self._llm_client._call_mistral(prompt, max_tokens=800)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("LLM call for report extraction failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Die Behandlungsauswertung ist vorübergehend nicht verfügbar.",
            ) from exc

        structured = self._parse_response(response)
        return self._build_draft_from_payload(structured)

    def _build_prompt(self, transcript: str) -> list[dict[str, str]]:
        """Create the deterministic prompt for the Mistral model."""

        system_message = (
            "Du bist ein deutscher Physiotherapeut und schreibst strukturierte "
            "Behandlungsberichte. Antworte ausschließlich mit JSON, ohne Erklärtext."
        )
        instruction = {
            "type": "object",
            "properties": {
                "report_city": {"type": "string"},
                "report_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                "insurance_type": {"type": "string"},
                "diagnoses": {"type": "array", "items": {"type": "string"}},
                "prescribed_therapy_type": {"type": "string"},
                "patient_problem_statement": {"type": "string"},
                "treatment_outcome": {"type": "string"},
                "therapy_status_note": {"type": "string"},
                "follow_up_recommendation": {"type": "string"},
            },
            "required": [
                "insurance_type",
                "prescribed_therapy_type",
                "patient_problem_statement",
                "treatment_outcome",
                "therapy_status_note",
                "follow_up_recommendation",
            ],
        }
        user_message = (
            "Extrahiere die folgenden Felder aus dem Transkript. Verwende die Werte "
            "PRIVAT, GESETZLICH oder UNKLAR für insurance_type. Für treatment_outcome "
            "nutze BESCHWERDEFREI, LINDERUNG, KEINE_BESSERUNG oder UNKLAR. Gib JSON "
            "im folgenden Schema zurück: "
            f"{json.dumps(instruction)}\n\nTRANSKRIPT:\n{transcript}"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract JSON payload from the LLM response."""

        content = response.get("content", "")
        if not content:
            logger.warning("Empty content returned from LLM report extraction")
            return {}

        json_start = content.find("{")
        json_end = content.rfind("}")
        if json_start == -1 or json_end == -1:
            logger.error("LLM response missing JSON body: %s", content)
            return {}

        try:
            return json.loads(content[json_start : json_end + 1])
        except json.JSONDecodeError as exc:
            logger.error("Failed to decode LLM JSON payload: %s", exc)
            return {}

    def _build_draft_from_payload(self, data: Dict[str, Any]) -> ClinicalReportDraft:
        """Combine defaults and extracted data into a report draft model."""

        report_city = self._coalesce_city(data.get("report_city"))
        report_date = self._parse_date(data.get("report_date"))
        insurance = self._parse_insurance(data.get("insurance_type"))
        diagnoses = data.get("diagnoses", [])

        draft = ClinicalReportDraft(
            doctor_name=self._defaults.doctor_name,
            patient_name=self._defaults.patient_name,
            patient_dob=self._defaults.patient_dob,
            prescription_date=self._defaults.prescription_date,
            treatment_date_from=self._defaults.treatment_date_from,
            treatment_date_to=self._defaults.treatment_date_to,
            physiotherapist_name=self._defaults.physiotherapist_name,
            report_city=report_city,
            report_date=report_date,
            insurance_type=insurance,
            diagnoses=diagnoses,
            prescribed_therapy_type=data.get("prescribed_therapy_type", ""),
            patient_problem_statement=data.get("patient_problem_statement", ""),
            treatment_outcome=self._parse_outcome(data.get("treatment_outcome")),
            therapy_status_note=data.get("therapy_status_note", ""),
            follow_up_recommendation=data.get("follow_up_recommendation", ""),
        )
        return draft

    def _empty_draft(self) -> ClinicalReportDraft:
        """Return a draft seeded only with default header values."""

        today = date.today()
        return ClinicalReportDraft(
            doctor_name=self._defaults.doctor_name,
            patient_name=self._defaults.patient_name,
            patient_dob=self._defaults.patient_dob,
            prescription_date=self._defaults.prescription_date,
            treatment_date_from=self._defaults.treatment_date_from,
            treatment_date_to=self._defaults.treatment_date_to,
            physiotherapist_name=self._defaults.physiotherapist_name,
            report_city=self._defaults.report_city,
            report_date=today,
            insurance_type=InsuranceType.UNKLAR,
            diagnoses=[],
            prescribed_therapy_type="",
            patient_problem_statement="",
            treatment_outcome=TreatmentOutcome.UNKLAR,
            therapy_status_note="",
            follow_up_recommendation="",
        )

    def _coalesce_city(self, candidate: Optional[str]) -> str:
        """Return a cleaned city name with default fallback."""

        if candidate and candidate.strip():
            return candidate.strip()
        return self._defaults.report_city

    def _parse_date(self, value: Optional[str]) -> date:
        """Convert various date representations to a :class:`date`."""

        if not value:
            return date.today()

        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
            try:
                return datetime.strptime(value.strip(), fmt).date()
            except ValueError:
                continue
        logger.warning("Could not parse report_date '%s', defaulting to today", value)
        return date.today()

    def _parse_insurance(self, value: Optional[str]) -> InsuranceType:
        """Normalize insurance type strings to the enum."""

        if not value:
            return InsuranceType.UNKLAR
        upper = value.strip().upper()
        if upper in InsuranceType.__members__:
            return InsuranceType[upper]
        return InsuranceType.UNKLAR

    def _parse_outcome(self, value: Optional[str]) -> TreatmentOutcome:
        """Normalize outcome text using synonym mapping."""

        if not value:
            return TreatmentOutcome.UNKLAR

        upper = value.strip().upper()
        if upper in TreatmentOutcome.__members__:
            return TreatmentOutcome[upper]
        return TreatmentOutcome.from_text(value)


report_extractor_service = ReportExtractorService()
