import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import asyncio
from datetime import date
import json

import pytest

from src.app.config.report_defaults import get_report_defaults
from src.app.models.report import (
    ClinicalReportDraft,
    InsuranceType,
    ReportSuggestionRequest,
    TreatmentOutcome,
)
from src.app.services.report_extractor import ReportExtractorService


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def _call_mistral(self, messages, max_tokens=800):
        self.calls.append((messages, max_tokens))
        return {"content": json.dumps(self.payload)}


@pytest.mark.asyncio
async def test_suggest_report_parses_llm_payload():
    payload = {
        "report_city": "Düsseldorf",
        "report_date": "2025-01-15",
        "insurance_type": "GESETZLICH",
        "diagnoses": ["Multifaktorielle Gangstörung", "PNP beginnend"],
        "prescribed_therapy_type": "KG Erwachsene ZNS",
        "patient_problem_statement": "Gangunsicherheit beim Treppensteigen.",
        "treatment_outcome": "schmerzfrei",
        "therapy_status_note": "Patient kann wieder ohne Stock gehen.",
        "follow_up_recommendation": "Weiterbehandlung mit Krankengymnastik."
    }
    service = ReportExtractorService(llm_client=FakeLLM(payload))
    request = ReportSuggestionRequest(transcript="Patient berichtet über Schmerzen, ist nun schmerzfrei.")

    result: ClinicalReportDraft = await service.suggest_report(request)

    assert result.report_city == "Düsseldorf"
    assert result.report_date == date(2025, 1, 15)
    assert result.insurance_type == InsuranceType.GESETZLICH
    assert result.treatment_outcome == TreatmentOutcome.BESCHWERDEFREI
    assert result.diagnoses == payload["diagnoses"]
    assert result.prescribed_therapy_type == payload["prescribed_therapy_type"]


@pytest.mark.asyncio
async def test_suggest_report_uses_defaults_when_missing_data():
    service = ReportExtractorService(llm_client=FakeLLM({"treatment_outcome": "unverändert"}))
    request = ReportSuggestionRequest(transcript="Keine Änderung feststellbar.")

    today = date.today()
    result = await service.suggest_report(request)
    defaults = get_report_defaults()

    assert result.report_city == defaults.report_city
    assert result.insurance_type == InsuranceType.UNKLAR
    assert result.treatment_outcome == TreatmentOutcome.KEINE_BESSERUNG
    assert result.report_date == today


@pytest.mark.asyncio
async def test_empty_transcript_returns_default_draft():
    service = ReportExtractorService(llm_client=FakeLLM({}))
    request = ReportSuggestionRequest()

    result = await service.suggest_report(request)
    defaults = get_report_defaults()

    assert result.doctor_name == defaults.doctor_name
    assert result.insurance_type == InsuranceType.UNKLAR
    assert result.diagnoses == []
