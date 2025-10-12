"""
Translation service for medAI MVP.
Uses Google Translator for all translation needs.
NEVER uses LLM for translation - only dedicated translation models.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
import json

from deep_translator import GoogleTranslator

from src.utils.config import settings, ModelConfig
from src.utils.logging import get_logger, get_latency_logger, monitor_latency
from src.utils.cache import cached, cache_key, get_cache_stats

logger = get_logger(__name__)
latency_logger = get_latency_logger()


class TranslationService:
    """Translation service using Google Translator."""

    def __init__(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # No cleanup needed for Google Translator
        pass

    def _get_google_language_code(self, language: str) -> str:
        """Get Google Translator language code for given language."""
        language_mapping = {
            "de": "de",
            "en": "en",
            "fr": "fr",
            "es": "es",
            "it": "it",
            "pt": "pt",
            "ru": "ru",
            "zh": "zh",
            "ja": "ja",
            "ko": "ko",
            "ar": "ar",
            "hi": "hi",
            "tr": "tr",
            "pl": "pl",
            "nl": "nl",
            "sv": "sv",
            "da": "da",
            "no": "no",
            "fi": "fi",
            "cs": "cs",
            "hu": "hu",
            "ro": "ro",
            "bg": "bg",
            "hr": "hr",
            "sk": "sk",
            "sl": "sl",
            "et": "et",
            "lv": "lv",
            "lt": "lt",
            "el": "el",
        }
        return language_mapping.get(language.lower(), "en")

    @cached("translation_google", ttl=3600)  # Cache for 1 hour
    @monitor_latency("translation_google", "google-translator")
    async def translate_text(
        self, text: str, source_lang: str = "de", target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate text using Google Translator with latency monitoring.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dict with translation results
        """
        if not text.strip():
            return {
                "translated_text": "",
                "source_language": source_lang,
                "target_language": target_lang,
                "model": "google-translator",
                "provider": "google",
                "fallback_used": False,
                "threshold_exceeded": False,
            }

        start_time = time.time()

        try:
            # Get language codes
            src_code = self._get_google_language_code(source_lang)
            tgt_code = self._get_google_language_code(target_lang)

            # Use Google Translator
            gt = GoogleTranslator(source=src_code, target=tgt_code)
            translated_text = gt.translate(text)

            # Check latency against threshold
            latency_ms = (time.time() - start_time) * 1000
            threshold_exceeded = latency_ms > settings.translation_threshold

            if threshold_exceeded:
                logger.warning(
                    f"Translation latency {latency_ms:.1f}ms exceeds threshold {settings.translation_threshold}ms"
                )

            return {
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "model": "google-translator",
                "provider": "google",
                "fallback_used": False,
                "threshold_exceeded": threshold_exceeded,
                "latency_ms": latency_ms,
            }

        except Exception as e:
            logger.error(f"Google Translator failed: {e}")
            raise Exception(f"Translation failed: {e}")

    async def translate_batch(
        self, texts: List[str], source_lang: str = "de", target_lang: str = "en"
    ) -> List[Dict[str, Any]]:
        """Translate multiple texts in batch."""
        tasks = [self.translate_text(text, source_lang, target_lang) for text in texts]
        return await asyncio.gather(*tasks)

    async def translate_clinical_notes(
        self, clinical_notes: Dict[str, Any], target_lang: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate structured clinical notes.

        Args:
            clinical_notes: Structured clinical notes
            target_lang: Target language code

        Returns:
            Dict with translated notes
        """
        try:
            translated_notes = {}

            # Translate text fields
            text_fields = [
                "hauptbeschwerden",
                "aktuelle_symptome",
                "medizinische_vorgeschichte",
                "medikamente",
                "allergien",
                "befunde",
                "diagnose_verdacht",
                "behandlungsplan",
                "naechste_schritte",
            ]

            for field in text_fields:
                if field in clinical_notes:
                    value = clinical_notes[field]

                    if isinstance(value, list):
                        # Translate list items
                        translated_items = []
                        for item in value:
                            if isinstance(item, str):
                                result = await self.translate_text(
                                    item, "de", target_lang
                                )
                                translated_items.append(result["translated_text"])
                            else:
                                translated_items.append(item)
                        translated_notes[field] = translated_items

                    elif isinstance(value, str):
                        # Translate string
                        result = await self.translate_text(value, "de", target_lang)
                        translated_notes[field] = result["translated_text"]

                    else:
                        # Keep as-is for non-text fields
                        translated_notes[field] = value

            # Copy non-text fields
            for key, value in clinical_notes.items():
                if key not in text_fields:
                    translated_notes[key] = value

            return {
                "translated_notes": translated_notes,
                "source_language": "de",
                "target_language": target_lang,
                "translation_applied": True,
            }

        except Exception as e:
            logger.error(f"Clinical notes translation failed: {e}")
            return {
                "translated_notes": clinical_notes,
                "source_language": "de",
                "target_language": target_lang,
                "translation_applied": False,
                "error": str(e),
            }

    def get_supported_languages(self) -> Dict[str, List[str]]:
        """Get supported languages for Google Translator."""
        return {
            "google": [
                "de",
                "en",
                "fr",
                "es",
                "it",
                "pt",
                "ru",
                "zh",
                "ja",
                "ko",
                "ar",
                "hi",
                "tr",
                "pl",
                "nl",
                "sv",
                "da",
                "no",
                "fi",
                "cs",
                "hu",
                "ro",
                "bg",
                "hr",
                "sk",
                "sl",
                "et",
                "lv",
                "lt",
                "el",
            ]
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check translation service health."""
        health_status = {
            "service": "translation",
            "status": "healthy",
            "providers": {},
            "timestamp": time.time(),
        }

        # Test Google Translator
        try:
            start_time = time.time()
            await self.translate_text("Test", "de", "en")
            google_latency = (time.time() - start_time) * 1000
            health_status["providers"]["google"] = {
                "status": "healthy",
                "latency_ms": google_latency,
            }
        except Exception as e:
            health_status["providers"]["google"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "unhealthy"

        return health_status

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "total_translations": self.performance_stats["total_translations"],
            "successful_translations": self.performance_stats[
                "successful_translations"
            ],
            "failed_translations": self.performance_stats["failed_translations"],
            "avg_latency": self.performance_stats["avg_latency"],
            "provider": "google",
        }

        # Add cache statistics if caching is enabled
        if settings.enable_caching:
            stats["cache_stats"] = get_cache_stats()

        return stats


# Global translation service instance
translation_service = TranslationService()
