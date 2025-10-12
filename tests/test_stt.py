"""
Unit tests for STT service.
Tests Faster-Whisper and Together AI fallback functionality.
"""

import pytest
import asyncio
import io
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.services.stt_service import STTService, stt_service
from src.utils.config import settings


class TestSTTService:
    """Test cases for STT service."""

    @pytest.fixture
    def stt_service_instance(self):
        """Create STT service instance for testing."""
        return STTService()

    @pytest.fixture
    def sample_audio_data(self):
        """Sample audio data for testing."""
        # Create a small WAV file in memory
        audio = io.BytesIO()
        # WAV header + minimal audio data
        audio.write(
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
        )
        audio.seek(0)
        return audio.getvalue()

    @pytest.mark.asyncio
    async def test_transcribe_audio_success_faster_whisper(
        self, stt_service_instance, sample_audio_data
    ):
        """Test successful transcription with Faster-Whisper."""
        # Mock successful Faster-Whisper response
        mock_segments = [
            MagicMock(text="Der Patient klagt über Kopfschmerzen und Schwindel.")
        ]
        mock_info = MagicMock()

        with patch.object(stt_service_instance, "fw_model") as mock_model:
            mock_model.transcribe.return_value = (mock_segments, mock_info)
            result = await stt_service_instance.transcribe_audio(
                sample_audio_data, "wav"
            )

            assert (
                result["text"] == "Der Patient klagt über Kopfschmerzen und Schwindel."
            )
            assert result["confidence"] == 1.0
            assert result["model"] == "faster_whisper"
            assert result["provider"] == "faster_whisper"
            assert result["fallback_used"] is False

    @pytest.mark.asyncio
    async def test_transcribe_audio_fallback_to_together(
        self, stt_service_instance, sample_audio_data
    ):
        """Test fallback to Together AI when Faster-Whisper fails."""
        # Mock Faster-Whisper failure
        with patch.object(
            stt_service_instance,
            "_transcribe_with_whisper",
            side_effect=Exception("Faster-Whisper failed"),
        ):
            # Mock successful Together AI response
            mock_response = MagicMock()
            mock_response.text = "Patient complains of headache and dizziness."

            with patch.object(
                stt_service_instance.together_client.audio,
                "transcribe",
                return_value=mock_response,
            ):
                result = await stt_service_instance.transcribe_audio(
                    sample_audio_data, "wav"
                )

                assert "Patient complains" in result["text"]
                assert result["model"] == "together_whisper"
                assert result["provider"] == "together"
                assert result["fallback_used"] is True

    @pytest.mark.asyncio
    async def test_transcribe_audio_webm_conversion(self, stt_service_instance):
        """Test WebM to WAV conversion."""
        # Mock WebM audio data
        webm_data = b"fake_webm_data"

        # Mock pydub conversion
        with patch("src.services.stt_service.AudioSegment") as mock_audio:
            mock_audio_instance = MagicMock()
            mock_audio_instance.set_frame_rate.return_value = mock_audio_instance
            mock_audio_instance.set_channels.return_value = mock_audio_instance
            mock_audio_instance.set_sample_width.return_value = mock_audio_instance
            mock_audio_instance.export.return_value = None
            mock_audio.from_file.return_value = mock_audio_instance

            # Mock successful transcription
            mock_segments = [MagicMock(text="Test transcription")]
            mock_info = MagicMock()

            with patch.object(stt_service_instance, "fw_model") as mock_model:
                mock_model.transcribe.return_value = (mock_segments, mock_info)
                result = await stt_service_instance.transcribe_audio(webm_data, "webm")

                assert result["text"] == "Test transcription"
                # Verify conversion was called
                mock_audio.from_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_audio_validation(self, stt_service_instance):
        """Test audio validation."""
        # Test empty audio
        with pytest.raises(ValueError, match="Audio too short"):
            await stt_service_instance.transcribe_audio(
                b"", "wav", duration_seconds=0.05
            )

        # Test audio too long
        long_audio = b"x" * 1000
        with pytest.raises(ValueError, match="Audio too long"):
            await stt_service_instance.transcribe_audio(
                long_audio, "wav", duration_seconds=400
            )

    @pytest.mark.asyncio
    async def test_transcribe_streaming(self, stt_service_instance, sample_audio_data):
        """Test streaming transcription."""
        # Mock successful transcription
        mock_segments = [MagicMock(text="Streaming transcription")]
        mock_info = MagicMock()

        with patch.object(stt_service_instance, "fw_model") as mock_model:
            mock_model.transcribe.return_value = (mock_segments, mock_info)

            # Create async generator for audio chunks
            async def audio_chunks():
                yield sample_audio_data[:100]
                yield sample_audio_data[100:]

            results = []
            async for result in stt_service_instance.transcribe_streaming(
                audio_chunks()
            ):
                results.append(result)

            assert len(results) > 0
            assert any(r["type"] == "final" for r in results)

    @pytest.mark.asyncio
    async def test_health_check(self, stt_service_instance):
        """Test health check functionality."""
        # Mock successful responses
        mock_segments = [MagicMock(text="Test")]
        mock_info = MagicMock()

        mock_together_response = MagicMock()
        mock_together_response.text = "Test"

        with patch.object(stt_service_instance, "fw_model") as mock_model, patch.object(
            stt_service_instance.together_client.audio,
            "transcribe",
            return_value=mock_together_response,
        ):
            mock_model.transcribe.return_value = (mock_segments, mock_info)

            health = await stt_service_instance.health_check()

            assert health["service"] == "stt"
            assert health["status"] == "healthy"
            assert "providers" in health
            assert "faster_whisper" in health["providers"]
            assert "together" in health["providers"]

    @pytest.mark.asyncio
    async def test_latency_threshold_trigger(
        self, stt_service_instance, sample_audio_data
    ):
        """Test that latency thresholds trigger fallback."""

        # Mock slow Faster-Whisper response
        async def slow_fw_transcribe(*args, **kwargs):
            await asyncio.sleep(3)  # Simulate slow response
            mock_segments = [MagicMock(text="Slow response")]
            mock_info = MagicMock()
            return (mock_segments, mock_info)

        # Mock fast Together AI response
        together_mock_response = MagicMock()
        together_mock_response.text = "Fast response"

        with patch.object(
            stt_service_instance,
            "_transcribe_with_whisper",
            side_effect=slow_fw_transcribe,
        ), patch.object(
            stt_service_instance.together_client.audio,
            "transcribe",
            return_value=together_mock_response,
        ):

            result = await stt_service_instance.transcribe_audio(
                sample_audio_data, "wav"
            )

            # Should fallback to Together AI due to slow Faster-Whisper response
            assert result["provider"] == "together"
            assert result["fallback_used"] is True


@pytest.mark.asyncio
async def test_global_stt_service():
    """Test global STT service instance."""
    # Test that global service can be imported and used
    assert stt_service is not None
    assert isinstance(stt_service, STTService)


@pytest.mark.asyncio
async def test_stt_service_context_manager():
    """Test STT service as context manager."""
    async with STTService() as service:
        assert service is not None
        assert hasattr(service, "fw_model")
        assert hasattr(service, "together_client")
