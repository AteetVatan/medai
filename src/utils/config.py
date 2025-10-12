"""
Configuration management for medAI MVP.
Handles API keys, latency thresholds, and environment settings.
"""

from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class Settings(PydanticBaseSettings):
    """Application settings with environment variable support."""

    debug: bool = Field(False, env="DEBUG")
    reload: bool = Field(False, env="RELOAD")

    # NER Microservice Configuration
    ner_microservice_base_url: str = Field(
        "https://medainer-production.up.railway.app/", env="NER_MICROSERVICE_BASE_URL"
    )

    # API Keys
    together_api_key: Optional[str] = Field(None, env="TOGETHER_API_KEY")
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_key: str = Field(..., env="SUPABASE_KEY")
    supabase_password: str = Field(..., env="SUPABASE_PASSWORD")

    # Model Configuration
    mistral_api_key: Optional[str] = Field(None, env="MISTRAL_API_KEY")
    openrouter_api_key: Optional[str] = Field(None, env="OPENROUTER_API_KEY")

    # Latency Thresholds (ms)
    stt_partial_threshold: int = Field(300, env="STT_PARTIAL_THRESHOLD")
    stt_final_threshold: int = Field(2000, env="STT_FINAL_THRESHOLD")
    llm_summary_threshold: int = Field(1800, env="LLM_SUMMARY_THRESHOLD")
    translation_threshold: int = Field(1000, env="TRANSLATION_THRESHOLD")

    # Model Endpoints
    mistral_endpoint: str = Field(
        "https://api.mistral.ai/v1/chat/completions", env="MISTRAL_ENDPOINT"
    )
    openrouter_endpoint: str = Field(
        "https://openrouter.ai/api/v1/chat/completions", env="OPENROUTER_ENDPOINT"
    )

    # Fallback Models
    fallback_llm_model: str = Field(
        "microsoft/Phi-3-mini-4k-instruct", env="FALLBACK_LLM_MODEL"
    )
    fallback_llm_provider: str = Field("openrouter", env="FALLBACK_LLM_PROVIDER")

    # Audio Processing
    # Note Check if this is needed
    audio_chunk_duration: float = Field(1.5, env="AUDIO_CHUNK_DURATION")
    audio_sample_rate: int = Field(16000, env="AUDIO_SAMPLE_RATE")
    audio_format: str = Field("wav", env="AUDIO_FORMAT")

    # Security & Compliance
    enable_pii_stripping: bool = Field(True, env="ENABLE_PII_STRIPPING")
    audit_log_retention_days: int = Field(90, env="AUDIT_LOG_RETENTION_DAYS")

    # Performance Tuning
    # Note Check if this is needed
    max_concurrent_requests: int = Field(10, env="MAX_CONCURRENT_REQUESTS")

    request_timeout: int = Field(30, env="REQUEST_TIMEOUT")

    # Note Check if this is needed
    enable_caching: bool = Field(True, env="ENABLE_CACHING")
    cache_ttl: int = Field(3600, env="CACHE_TTL")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_structured_logging: bool = Field(True, env="ENABLE_STRUCTURED_LOGGING")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


class ModelConfig:
    """Model-specific configuration constants."""

    # Temperature settings (always 0.0 for deterministic output)
    TEMPERATURE = 0.0
    TOP_P = 1.0

    # Token limits
    MAX_TOKENS_LLM = 2048
    MAX_TOKENS_TRANSLATION = 1024

    # Model-specific settings
    TOGETHER_WHISPER_MODEL = "openai/whisper-large-v3"
    MISTRAL_MODEL_OLD = "mistralai/Mistral-7B-Instruct-v0.2"
    MISTRAL_MODEL = "open-mixtral-8x7b"

    # NER Models - now using microservice architecture
    # SPACY_MODEL_DE = "de_core_news_sm"  # Removed - using NER microservice
    # MEDICAL_NER_MODEL = "en_core_med7_lg"  # Removed - using NER microservice

    # Supported languages for NLLB
    SUPPORTED_LANGUAGES = {
        "de": "German",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
    }


class LatencyConfig:
    """Latency monitoring and alerting configuration."""

    # Critical thresholds (ms)
    CRITICAL_STT_LATENCY = 2000
    CRITICAL_LLM_LATENCY = 3000
    CRITICAL_TRANSLATION_LATENCY = 1500

    # Warning thresholds (ms)
    WARNING_STT_LATENCY = 1000
    WARNING_LLM_LATENCY = 1500
    WARNING_TRANSLATION_LATENCY = 800

    # Fallback triggers
    FALLBACK_STT_LATENCY = 2500
    FALLBACK_LLM_LATENCY = 2500
    FALLBACK_TRANSLATION_LATENCY = 1200
