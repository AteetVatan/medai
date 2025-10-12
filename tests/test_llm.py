"""
Unit tests for LLM service.
Tests Mistral 7B and fallback model functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import json

from src.services.llm_service import LLMService, llm_service
from src.utils.config import ModelConfig


class TestLLMService:
    """Test cases for LLM service."""

    @pytest.fixture
    def llm_service_instance(self):
        """Create LLM service instance for testing."""
        return LLMService()

    @pytest.fixture
    def sample_transcript(self):
        """Sample transcript for testing."""
        return "Der Patient klagt über Kopfschmerzen und Schwindel. Er hat Diabetes und nimmt Metformin."

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            {"text": "Kopfschmerzen", "label": "SYMPTOM", "icd_code": "R51"},
            {"text": "Schwindel", "label": "SYMPTOM", "icd_code": "R42"},
            {"text": "Diabetes", "label": "DIAGNOSIS", "icd_code": "E11"},
            {"text": "Metformin", "label": "MEDICATION", "icd_code": "A10BA02"},
        ]

    @pytest.mark.asyncio
    async def test_generate_clinical_summary_success_mistral(
        self, llm_service_instance, sample_transcript, sample_entities
    ):
        """Test successful clinical summary generation with Mistral."""
        # Mock successful Mistral response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Strukturierte klinische Zusammenfassung..."}}
            ],
            "usage": {"total_tokens": 150},
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(
            llm_service_instance.mistral_client, "post", return_value=mock_response
        ):
            result = await llm_service_instance.generate_clinical_summary(
                transcript=sample_transcript,
                entities=sample_entities,
                task_type="intake_summary",
                user_id="test_user",
            )

            assert "content" in result
            assert result["model"] == "mistral-7b"
            assert result["provider"] == "mistral"
            assert result["fallback_used"] is False
            assert "usage" in result

    @pytest.mark.asyncio
    async def test_generate_clinical_summary_fallback_openrouter(
        self, llm_service_instance, sample_transcript, sample_entities
    ):
        """Test fallback to OpenRouter when Mistral fails."""
        # Mock Mistral failure
        mistral_mock_response = MagicMock()
        mistral_mock_response.raise_for_status.side_effect = Exception("Rate limited")

        # Mock successful OpenRouter response
        openrouter_mock_response = MagicMock()
        openrouter_mock_response.json.return_value = {
            "choices": [
                {"message": {"content": "Clinical summary from fallback model..."}}
            ],
            "usage": {"total_tokens": 120},
        }
        openrouter_mock_response.raise_for_status.return_value = None

        with patch.object(
            llm_service_instance.mistral_client,
            "post",
            return_value=mistral_mock_response,
        ), patch.object(
            llm_service_instance.openrouter_client,
            "post",
            return_value=openrouter_mock_response,
        ):

            result = await llm_service_instance.generate_clinical_summary(
                transcript=sample_transcript,
                entities=sample_entities,
                task_type="intake_summary",
                user_id="test_user",
            )

            assert "content" in result
            assert result["fallback_used"] is True
            assert result["provider"] == "openrouter"

    @pytest.mark.asyncio
    async def test_generate_clinical_summary_all_providers_fail(
        self, llm_service_instance, sample_transcript, sample_entities
    ):
        """Test behavior when all providers fail."""
        # Mock all providers failing
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Service unavailable")

        with patch.object(
            llm_service_instance.mistral_client, "post", return_value=mock_response
        ), patch.object(
            llm_service_instance.openrouter_client, "post", return_value=mock_response
        ), patch.object(
            llm_service_instance.together_client, "post", return_value=mock_response
        ):

            with pytest.raises(Exception, match="All LLM providers unavailable"):
                await llm_service_instance.generate_clinical_summary(
                    transcript=sample_transcript,
                    entities=sample_entities,
                    user_id="test_user",
                )

    @pytest.mark.asyncio
    async def test_generate_structured_notes_success(self, llm_service_instance):
        """Test structured notes generation."""
        clinical_summary = """
        HAUPTBESCHWERDEN: Kopfschmerzen und Schwindel
        AKTUELLE SYMPTOME: Übelkeit, Müdigkeit
        MEDIKAMENTE: Metformin, Aspirin
        """

        # Mock successful OpenRouter response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"hauptbeschwerden": ["Kopfschmerzen", "Schwindel"], "aktuelle_symptome": ["Übelkeit", "Müdigkeit"], "medikamente": ["Metformin", "Aspirin"]}'
                    }
                }
            ],
            "usage": {"total_tokens": 100},
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(
            llm_service_instance.openrouter_client, "post", return_value=mock_response
        ):
            result = await llm_service_instance.generate_structured_notes(
                clinical_summary=clinical_summary, format_type="json"
            )

            assert "structured_notes" in result
            assert result["format"] == "json"
            assert result["fallback_used"] is True
            assert isinstance(result["structured_notes"], dict)

    @pytest.mark.asyncio
    async def test_generate_structured_notes_json_parsing_failure(
        self, llm_service_instance
    ):
        """Test structured notes generation with JSON parsing failure."""
        clinical_summary = "Some clinical summary"

        # Mock response with invalid JSON
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Invalid JSON response"}}],
            "usage": {"total_tokens": 50},
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(
            llm_service_instance.openrouter_client, "post", return_value=mock_response
        ):
            result = await llm_service_instance.generate_structured_notes(
                clinical_summary=clinical_summary, format_type="json"
            )

            assert "structured_notes" in result
            assert result["structured_notes"]["structured"] is False
            assert "raw_summary" in result["structured_notes"]

    def test_strip_pii_enabled(self, llm_service_instance):
        """Test PII stripping when enabled."""
        text_with_pii = "Patient Max Mustermann (geb. 15.03.1980) wohnt in 12345 Berlin. Email: max@example.com"

        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.enable_pii_stripping = True

            stripped_text = llm_service_instance._strip_pii(text_with_pii)

            assert "[NAME]" in stripped_text
            assert "[DATUM]" in stripped_text
            assert "[PLZ]" in stripped_text
            assert "[EMAIL]" in stripped_text
            assert "Max Mustermann" not in stripped_text
            assert "max@example.com" not in stripped_text

    def test_strip_pii_disabled(self, llm_service_instance):
        """Test PII stripping when disabled."""
        text_with_pii = "Patient Max Mustermann (geb. 15.03.1980) wohnt in 12345 Berlin. Email: max@example.com"

        with patch("src.services.llm_service.settings") as mock_settings:
            mock_settings.enable_pii_stripping = False

            stripped_text = llm_service_instance._strip_pii(text_with_pii)

            assert stripped_text == text_with_pii  # Should be unchanged

    def test_load_system_prompts(self, llm_service_instance):
        """Test system prompts loading."""
        prompts = llm_service_instance._system_prompts

        assert "intake_summary" in prompts
        assert "assessment" in prompts
        assert "treatment_plan" in prompts

        # Check that prompts contain expected structure
        intake_prompt = prompts["intake_summary"]
        assert "HAUPTBESCHWERDEN" in intake_prompt
        assert "AKTUELLE SYMPTOME" in intake_prompt
        assert "MEDIKAMENTE" in intake_prompt

    def test_format_entities_for_llm(self, llm_service_instance):
        """Test entity formatting for LLM context."""
        entities = [
            {
                "text": "Kopfschmerzen",
                "label": "SYMPTOM",
                "icd_code": "R51",
                "category": "symptom",
            },
            {
                "text": "Diabetes",
                "label": "DIAGNOSIS",
                "icd_code": "E11",
                "category": "diagnosis",
            },
        ]

        formatted = llm_service_instance._format_entities_for_llm(entities)

        assert "Kopfschmerzen" in formatted
        assert "R51" in formatted
        assert "Diabetes" in formatted
        assert "E11" in formatted
        assert "- " in formatted  # Bullet points

    def test_format_entities_empty(self, llm_service_instance):
        """Test entity formatting with empty list."""
        formatted = llm_service_instance._format_entities_for_llm([])
        assert "Keine medizinischen Entitäten extrahiert" in formatted

    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_service_instance):
        """Test successful health check."""
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(
            llm_service_instance.mistral_client, "post", return_value=mock_response
        ), patch.object(
            llm_service_instance.openrouter_client, "post", return_value=mock_response
        ):

            health = await llm_service_instance.health_check()

            assert health["service"] == "llm"
            assert health["status"] == "healthy"
            assert "providers" in health
            assert "mistral" in health["providers"]
            assert "openrouter" in health["providers"]

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_service_instance):
        """Test health check with provider failures."""
        # Mock failed responses
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Service unavailable")

        with patch.object(
            llm_service_instance.mistral_client, "post", return_value=mock_response
        ), patch.object(
            llm_service_instance.openrouter_client, "post", return_value=mock_response
        ):

            health = await llm_service_instance.health_check()

            assert health["service"] == "llm"
            assert health["status"] == "unhealthy"
            assert health["providers"]["mistral"]["status"] == "unhealthy"
            assert health["providers"]["openrouter"]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_temperature_zero_enforcement(
        self, llm_service_instance, sample_transcript, sample_entities
    ):
        """Test that temperature is always set to 0.0."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_response.raise_for_status.return_value = None

        with patch.object(
            llm_service_instance.mistral_client, "post", return_value=mock_response
        ) as mock_post:
            await llm_service_instance.generate_clinical_summary(
                transcript=sample_transcript,
                entities=sample_entities,
                user_id="test_user",
            )

            # Check that temperature is set to 0.0
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            assert payload["temperature"] == ModelConfig.TEMPERATURE
            assert payload["temperature"] == 0.0
            assert payload["top_p"] == ModelConfig.TOP_P


@pytest.mark.asyncio
async def test_global_llm_service():
    """Test global LLM service instance."""
    assert llm_service is not None
    assert isinstance(llm_service, LLMService)


@pytest.mark.asyncio
async def test_llm_service_context_manager():
    """Test LLM service as context manager."""
    async with LLMService() as service:
        assert service is not None
        assert hasattr(service, "mistral_client")
        assert hasattr(service, "openrouter_client")
        assert hasattr(service, "together_client")


def test_model_config_constants():
    """Test ModelConfig constants."""
    assert ModelConfig.TEMPERATURE == 0.0
    assert ModelConfig.TOP_P == 1.0
    assert ModelConfig.MAX_TOKENS_LLM == 2048
    assert ModelConfig.MISTRAL_MODEL == "mistralai/Mistral-7B-Instruct-v0.2"
