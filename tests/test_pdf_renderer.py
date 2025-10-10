import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datetime import date

from src.app.models.report import ClinicalReport, InsuranceType, TreatmentOutcome
from src.app.services.pdf_renderer import render_report_pdf


def build_sample_report() -> ClinicalReport:
    return ClinicalReport(
        doctor_name="Dr. Max Mustermann",
        patient_name="Max Patient",
        patient_dob="1970-01-01",
        prescription_date="2025-01-01",
        treatment_date_from="2025-01-05",
        treatment_date_to="2025-01-20",
        physiotherapist_name="Sotirios Dimitriou",
        report_city="Essen",
        report_date=date(2025, 1, 15),
        insurance_type=InsuranceType.GESETZLICH,
        diagnoses=["Multifaktorielle Gangstörung", "PNP beginnend"],
        prescribed_therapy_type="KG Erwachsene ZNS",
        patient_problem_statement="Patient klagt über Gangunsicherheit beim Treppensteigen.",
        treatment_outcome=TreatmentOutcome.LINDERUNG,
        therapy_status_note="Verbesserte Belastbarkeit der unteren Extremitäten.",
        follow_up_recommendation="Weiterbehandlung mit Krankengymnastik.",
        transcript="Patient berichtet über deutliche Verbesserung nach Therapie."
    )


def test_render_report_pdf_produces_bytes():
    report = build_sample_report()

    filename, content = render_report_pdf(report)

    assert filename.endswith('.pdf')
    assert isinstance(content, (bytes, bytearray))
    assert len(content) > 100
