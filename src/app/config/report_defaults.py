"""Configuration defaults for clinical report headers."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from ..models.report import ReportDefaults


class ReportDefaultsSettings(BaseSettings):
    """Environment-backed defaults for report header fields."""

    doctor_name: str = Field("Dr. Max Mustermann", env="REPORT_DOCTOR_NAME")
    patient_name: str = Field("Max Patient", env="REPORT_PATIENT_NAME")
    patient_dob: str = Field("1970-01-01", env="REPORT_PATIENT_DOB")
    prescription_date: str = Field("2025-01-01", env="REPORT_PRESCRIPTION_DATE")
    treatment_date_from: str = Field("2025-01-05", env="REPORT_TREATMENT_FROM")
    treatment_date_to: str = Field("2025-01-20", env="REPORT_TREATMENT_TO")
    physiotherapist_name: str = Field("Anita Bahmani", env="REPORT_PHYSIOTHERAPIST_NAME")
    report_city: str = Field("Essen", env="REPORT_CITY")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


@lru_cache(maxsize=1)
def get_report_defaults() -> ReportDefaults:
    """Return cached report defaults as a Pydantic model."""

    settings = ReportDefaultsSettings()
    return ReportDefaults(**settings.model_dump())


def get_default_value(field: str, fallback: Optional[str] = None) -> str:
    """Helper to fetch a single default value."""

    defaults = get_report_defaults()
    value = getattr(defaults, field)
    if value:
        return value
    if fallback is not None:
        return fallback
    raise AttributeError(f"Unknown report default field '{field}'")
