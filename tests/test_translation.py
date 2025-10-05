"""
Unit tests for translation service.
Tests Google Translator functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from src.services.translation_service import TranslationService, translation_service
from src.utils.config import ModelConfig


class TestTranslationService:
    """Test cases for translation service."""
    
    @pytest.fixture
    def translation_service_instance(self):
        """Create translation service instance for testing."""
        return TranslationService()
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "Der Patient klagt über Kopfschmerzen und Schwindel."
    
    @pytest.fixture
    def sample_clinical_notes(self):
        """Sample clinical notes for testing."""
        return {
            "hauptbeschwerden": ["Kopfschmerzen", "Schwindel"],
            "aktuelle_symptome": ["Übelkeit", "Müdigkeit"],
            "medikamente": ["Metformin", "Aspirin"],
            "diagnose_verdacht": ["Migräne", "Vertigo"]
        }
    
    
    def test_get_google_language_code(self, translation_service_instance):
        """Test Google Translator language code mapping."""
        assert translation_service_instance._get_google_language_code("de") == "de"
        assert translation_service_instance._get_google_language_code("en") == "en"
        assert translation_service_instance._get_google_language_code("fr") == "fr"
        assert translation_service_instance._get_google_language_code("unknown") == "en"
    
    
    @pytest.mark.asyncio
    async def test_translate_text_google_success(self, translation_service_instance, sample_text):
        """Test successful translation with Google Translator."""
        # Mock Google Translator
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.return_value = "The patient complains of headache and dizziness."
            mock_gt_class.return_value = mock_gt
            
            result = await translation_service_instance.translate_text(
                text=sample_text,
                source_lang="de",
                target_lang="en"
            )
            
            assert result["translated_text"] == "The patient complains of headache and dizziness."
            assert result["source_language"] == "de"
            assert result["target_language"] == "en"
            assert result["model"] == "google-translator"
            assert result["provider"] == "google"
            assert result["fallback_used"] is False
    
    @pytest.mark.asyncio
    async def test_translate_text_google_failure(self, translation_service_instance, sample_text):
        """Test Google Translator failure."""
        # Mock Google Translator failure
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.side_effect = Exception("Translation failed")
            mock_gt_class.return_value = mock_gt
            
            with pytest.raises(Exception, match="Translation failed"):
                await translation_service_instance.translate_text(
                    text=sample_text,
                    source_lang="de",
                    target_lang="en"
                )
    
    
    @pytest.mark.asyncio
    async def test_translate_text_empty_text(self, translation_service_instance):
        """Test translation with empty text."""
        result = await translation_service_instance.translate_text(
            text="",
            source_lang="de",
            target_lang="en"
        )
        
        assert result["translated_text"] == ""
        assert result["source_language"] == "de"
        assert result["target_language"] == "en"
        assert result["model"] == "google-translator"
        assert result["provider"] == "google"
        assert result["fallback_used"] is False
    
    @pytest.mark.asyncio
    async def test_translate_batch(self, translation_service_instance):
        """Test batch translation."""
        texts = [
            "Der Patient hat Kopfschmerzen.",
            "Er nimmt Metformin.",
            "Keine Allergien bekannt."
        ]
        
        # Mock Google Translator
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.return_value = "The patient has a headache."
            mock_gt_class.return_value = mock_gt
            
            results = await translation_service_instance.translate_batch(
                texts=texts,
                source_lang="de",
                target_lang="en"
            )
            
            assert len(results) == 3
            for result in results:
                assert result["translated_text"] == "The patient has a headache."
                assert result["fallback_used"] is False
    
    @pytest.mark.asyncio
    async def test_translate_clinical_notes_success(self, translation_service_instance, sample_clinical_notes):
        """Test clinical notes translation."""
        # Mock Google Translator
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.return_value = "Headaches, dizziness"
            mock_gt_class.return_value = mock_gt
            
            result = await translation_service_instance.translate_clinical_notes(
                clinical_notes=sample_clinical_notes,
                target_lang="en"
            )
            
            assert "translated_notes" in result
            assert result["source_language"] == "de"
            assert result["target_language"] == "en"
            assert result["translation_applied"] is True
    
    @pytest.mark.asyncio
    async def test_translate_clinical_notes_failure(self, translation_service_instance, sample_clinical_notes):
        """Test clinical notes translation failure."""
        # Mock Google Translator failure
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.side_effect = Exception("Translation failed")
            mock_gt_class.return_value = mock_gt
            
            result = await translation_service_instance.translate_clinical_notes(
                clinical_notes=sample_clinical_notes,
                target_lang="en"
            )
            
            assert result["translated_notes"] == sample_clinical_notes
            assert result["translation_applied"] is False
            assert "error" in result
    
    def test_get_supported_languages(self, translation_service_instance):
        """Test supported languages retrieval."""
        languages = translation_service_instance.get_supported_languages()
        
        assert "google" in languages
        assert "de" in languages["google"]
        assert "en" in languages["google"]
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, translation_service_instance):
        """Test successful health check."""
        # Mock Google Translator
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.return_value = "Test"
            mock_gt_class.return_value = mock_gt
            health = await translation_service_instance.health_check()
            
            assert health["service"] == "translation"
            assert health["status"] == "healthy"
            assert "providers" in health
            assert health["providers"]["google"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, translation_service_instance):
        """Test health check with failures."""
        # Mock Google Translator failure
        with patch('src.services.translation_service.GoogleTranslator') as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.side_effect = Exception("Service unavailable")
            mock_gt_class.return_value = mock_gt
            health = await translation_service_instance.health_check()
            
            assert health["service"] == "translation"
            assert health["status"] == "unhealthy"
            assert health["providers"]["google"]["status"] == "unhealthy"
    


@pytest.mark.asyncio
async def test_global_translation_service():
    """Test global translation service instance."""
    assert translation_service is not None
    assert isinstance(translation_service, TranslationService)


@pytest.mark.asyncio
async def test_translation_service_context_manager():
    """Test translation service as context manager."""
    async with TranslationService() as service:
        assert service is not None
        assert hasattr(service, 'translate_text')


