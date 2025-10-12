"""
Test caching functionality across services.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

from src.utils.cache import CacheManager, cached, get_cache_stats, clear_cache
from src.services.stt_service import STTService
from src.services.llm_service import LLMService
from src.services.translation_service import TranslationService
from src.utils.config import settings


class TestCacheManager:
    """Test cache manager functionality."""

    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = CacheManager()

        # Test set and get
        cache.set("test_key", "test_value", ttl=60)
        assert cache.get("test_key") == "test_value"

        # Test non-existent key
        assert cache.get("non_existent") is None

        # Test delete
        cache.delete("test_key")
        assert cache.get("test_key") is None

    def test_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        cache = CacheManager()

        # Set with very short TTL
        cache.set("expire_key", "expire_value", ttl=0.1)
        assert cache.get("expire_key") == "expire_value"

        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("expire_key") is None

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = CacheManager()

        key1 = cache._make_key("prefix", "arg1", "arg2", kwarg1="value1")
        key2 = cache._make_key("prefix", "arg1", "arg2", kwarg1="value1")
        key3 = cache._make_key("prefix", "arg1", "arg2", kwarg1="value2")

        # Same arguments should generate same key
        assert key1 == key2
        # Different arguments should generate different keys
        assert key1 != key3

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = CacheManager()

        # Add some entries
        cache.set("key1", "value1", ttl=60)
        cache.set("key2", "value2", ttl=60)

        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["enabled"] == settings.enable_caching


class TestCachingDecorator:
    """Test caching decorator functionality."""

    @pytest.mark.asyncio
    async def test_async_caching_decorator(self):
        """Test async caching decorator."""
        call_count = 0

        @cached("test_async", ttl=60)
        async def async_function(value):
            nonlocal call_count
            call_count += 1
            return f"result_{value}"

        # First call should execute function
        result1 = await async_function("test")
        assert result1 == "result_test"
        assert call_count == 1

        # Second call should use cache
        result2 = await async_function("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment

        # Different arguments should execute function
        result3 = await async_function("different")
        assert result3 == "result_different"
        assert call_count == 2

    def test_sync_caching_decorator(self):
        """Test sync caching decorator."""
        call_count = 0

        @cached("test_sync", ttl=60)
        def sync_function(value):
            nonlocal call_count
            call_count += 1
            return f"result_{value}"

        # First call should execute function
        result1 = sync_function("test")
        assert result1 == "result_test"
        assert call_count == 1

        # Second call should use cache
        result2 = sync_function("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment


class TestServiceCaching:
    """Test caching integration in services."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings with caching enabled."""
        with patch("src.utils.config.settings") as mock_settings:
            mock_settings.enable_caching = True
            mock_settings.cache_ttl = 3600
            yield mock_settings

    @pytest.mark.asyncio
    async def test_stt_service_caching(self, mock_settings):
        """Test STT service caching."""
        service = STTService()
        service.fw_model = MagicMock()
        service.together_client = MagicMock()

        # Mock transcription methods
        with patch.object(service, "_transcribe_with_whisper") as mock_whisper:
            mock_whisper.return_value = {
                "text": "test transcription",
                "model": "faster_whisper",
                "latency": 0.5,
                "confidence": 0.95,
            }

            # First call should execute transcription
            result1 = await service._transcribe_with_whisper(b"audio_data")
            assert result1["text"] == "test transcription"
            assert mock_whisper.call_count == 1

            # Second call should use cache
            result2 = await service._transcribe_with_whisper(b"audio_data")
            assert result2["text"] == "test transcription"
            assert mock_whisper.call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_llm_service_caching(self, mock_settings):
        """Test LLM service caching."""
        service = LLMService()
        service.mistral_client = MagicMock()

        # Mock LLM call
        with patch.object(service, "_call_mistral") as mock_mistral:
            mock_mistral.return_value = {
                "content": "test response",
                "usage": {"total_tokens": 100},
                "model": "mistral-7b",
            }

            messages = [{"role": "user", "content": "test prompt"}]

            # First call should execute LLM
            result1 = await service._call_mistral(messages)
            assert result1["content"] == "test response"
            assert mock_mistral.call_count == 1

            # Second call should use cache
            result2 = await service._call_mistral(messages)
            assert result2["content"] == "test response"
            assert mock_mistral.call_count == 1  # Should not increment

    @pytest.mark.asyncio
    async def test_translation_service_caching(self, mock_settings):
        """Test translation service caching."""
        service = TranslationService()

        # Mock Google Translator
        with patch(
            "src.services.translation_service.GoogleTranslator"
        ) as mock_gt_class:
            mock_gt = MagicMock()
            mock_gt.translate.return_value = "translated text"
            mock_gt_class.return_value = mock_gt

            # First call should execute translation
            result1 = await service.translate_text("test text", "de", "en")
            assert result1["translated_text"] == "translated text"
            assert mock_gt.translate.call_count == 1

            # Second call should use cache
            result2 = await service.translate_text("test text", "de", "en")
            assert result2["translated_text"] == "translated text"
            assert mock_gt.translate.call_count == 1  # Should not increment

    def test_cache_disabled(self):
        """Test that caching is disabled when setting is False."""
        with patch("src.utils.config.settings") as mock_settings:
            mock_settings.enable_caching = False

            cache = CacheManager()
            cache.set("test_key", "test_value")
            assert cache.get("test_key") is None  # Should not cache when disabled

    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        cache = CacheManager()

        # Add some entries with different TTLs
        cache.set("key1", "value1", ttl=0.1)  # Will expire
        cache.set("key2", "value2", ttl=60)  # Will not expire

        # Wait for first key to expire
        time.sleep(0.2)

        # Cleanup should remove expired entries
        cache.cleanup_expired()

        assert cache.get("key1") is None  # Should be removed
        assert cache.get("key2") == "value2"  # Should still be there

    def test_global_cache_functions(self):
        """Test global cache utility functions."""
        # Test clear cache
        clear_cache()

        # Test get cache stats
        stats = get_cache_stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "enabled" in stats
