"""
Speech-to-Text service for medAI MVP.

This module provides STT functionality using Faster-Whisper as primary
and OpenAI Whisper as fallback, with streaming audio support.
"""
import os
import io
import asyncio
import time
from typing import Dict, Any, List, Optional, AsyncGenerator
import aiohttp
from together import Together
from faster_whisper import WhisperModel
import wave
from imageio_ffmpeg import get_ffmpeg_exe
import soundfile as sf
import torch
import numpy as np
import pydub
from pydub import AudioSegment

from ..utils.config import settings, ModelConfig, LatencyConfig
from ..utils.logging import get_logger, get_latency_logger, monitor_latency
from ..utils.cache import cached, cache_key, get_cache_stats
from ..utils.audio import audio_processor

logger = get_logger(__name__)
latency_logger = get_latency_logger()


class STTService:
    """Speech-to-Text service with streaming support."""
    
    def __init__(self):
        self.fw_model = None
        self.together_client = None
        self.models_warmed_up = False
        self.performance_stats = {
            "total_transcriptions": 0,
            "successful_transcriptions": 0,
            "failed_transcriptions": 0,
            "avg_latency": 0.0,
            "latencies": []
        }
        
        #to ensure any library (like Hugging Face) that calls ffmpeg without a full path will still find it.
        # Ensure ffmpeg is discoverable by libs expecting it on PATH
        try:
            ffmpeg_path = get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_path)
            path_parts = os.environ.get("PATH", "").split(os.pathsep)
            if ffmpeg_dir not in path_parts:
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"FFmpeg available at: {ffmpeg_path}")
        except Exception as e:
            logger.warning(f"Could not initialize ffmpeg from imageio-ffmpeg: {e}")
        
    async def warm_up_models(self) -> None:
        """Warm up STT models for optimal performance."""
        try:
            logger.info("Warming up STT models...")
            
            # Dynamic hardware detection
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"  # GPU-friendly
                model_size = "medium"     # use larger model for accuracy on GPU
                logger.info("CUDA detected — using GPU with float16 and medium model")
            else:
                device = "cpu"
                compute_type = "int8"     # fastest for CPU
                model_size = "small"      # small model for speed on CPU
                logger.info("No GPU detected — using CPU with int8 and small model")

            # Load Faster-Whisper model
            try:
                self.fw_model = WhisperModel(
                    model_size,   # dynamic based on hardware
                    device=device,
                    compute_type=compute_type,
                )
                logger.info(f"Faster-Whisper model loaded ({model_size}, {device}, {compute_type})")
            except Exception as e:
                logger.warning(f"Failed to load Faster-Whisper model: {e}")
                
            # Warm up Together AI client
            if hasattr(settings, 'together_api_key') and settings.together_api_key:
                try:
                    self.together_client = Together(api_key=settings.together_api_key)
                    logger.info("Together AI client initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Together AI client: {str(e)}")
            
            self.models_warmed_up = True
            logger.info("STT models warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to warm up STT models: {str(e)}")
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources if needed
        pass
    
    async def _process_audio_for_stt(self, audio_data: bytes, target_format: str = "wav") -> bytes:
        """Process audio to standard format for STT."""
        try:
            logger.info("Processing audio for STT")
            
           
            #for testing
            # audio_filename = f"audio_data_1.{target_format}"
            # audio_path = os.path.join("audio_data", audio_filename)
            
            # Save audio data as file
            # os.makedirs("audio_data", exist_ok=True)            
            # with open(audio_path, "wb") as f:
            #     f.write(audio_data)
                
            #for testing read all audio data    
            #with open(audio_path, "rb") as f:
            #    audio_data = f.read()
                
            
            return await audio_processor.process_audio_for_stt(audio_data, target_format)
            
            #ateet
 
            
            # # Load audio with pydub
            # audio = AudioSegment.from_file(io.BytesIO(audio_data))
            
            # # Convert to standard format
            # audio = audio.set_frame_rate(16000)  # 16kHz
            # audio = audio.set_channels(1)        # Mono
            # audio = audio.set_sample_width(2)    # 16-bit
            
            # # Export to target format
            # output_buffer = io.BytesIO()
            # audio.export(output_buffer, format=target_format)
            # return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise ValueError(f"Failed to process audio: {e}")
    
    def _validate_audio(self, audio_data: bytes, duration_seconds: float) -> bool:
        """Validate audio data meets requirements."""
        # Check duration limits
        if duration_seconds > 300:  # 5 minutes max
            raise ValueError("Audio too long (max 5 minutes)")
        
        if duration_seconds < 0.1:  # 100ms min
            raise ValueError("Audio too short (min 100ms)")
        
        # Check file size (50MB max)
        if len(audio_data) > 50 * 1024 * 1024:
            raise ValueError("Audio file too large (max 50MB)")
        
        return True
    
    @cached("stt_faster_whisper", ttl=3600)  # Cache for 1 hour
    @monitor_latency("stt_faster_whisper", "faster-whisper")
    async def _transcribe_with_whisper(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio using Faster-Whisper (local)."""
        start_time = time.time()
        try:
            if not self.fw_model:
                raise Exception("Faster-Whisper model not loaded")

            # Process audio to standard mono/16k WAV
            processed_audio = await self._process_audio_for_stt(audio_data, "wav")

            # Decode WAV -> float32 mono [-1, 1]
            with wave.open(io.BytesIO(processed_audio), "rb") as w:
                sr = w.getframerate()
                ch = w.getnchannels()
                sw = w.getsampwidth()
                frames = w.readframes(w.getnframes())
            if sw != 2:
                raise ValueError(f"Expected 16-bit WAV, got {sw*8}-bit")
            arr = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)

            # Transcribe with Faster-Whisper
            segments, info = self.fw_model.transcribe(
                arr,
                beam_size=5,
                language="de",  # German for medical context
                task="transcribe"
            )
            text = " ".join(seg.text for seg in segments).strip()

            latency = time.time() - start_time
            logger.debug(f"Faster-Whisper transcription completed in {latency:.3f}s")

            return {
                "text": text,
                "model": "faster_whisper",
                "latency": latency,
                "confidence": 1.0,  # FW doesn't return confidence
                "provider": "faster_whisper"
            }
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Faster-Whisper transcription failed: {e}")
            raise
    
    @cached("stt_together", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("stt_together", "whisper-together")
    async def _transcribe_with_together(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio using Together AI Whisper."""
        start_time = time.time()
        
        try:
            if not self.together_client:
                raise Exception("Together AI client not initialized")
            
            # Process audio for Together AI
            processed_audio = await self._process_audio_for_stt(audio_data, "wav")
            
            # Transcribe using Together AI
            file_obj = io.BytesIO(processed_audio)
            file_obj.name = "audio.wav"  # Together AI SDK reads filename from file-like

            # response = self.together_client.audio.transcribe(
            #     model=ModelConfig.TOGETHER_WHISPER_MODEL,
            #     file=file_obj,
            #     language="de",  # German for medical context
            #     response_format="json",
            #     timestamp_granularities="segment"
            # )
            
           
            response = self.together_client.audio.transcriptions.create(
                file=file_obj,
                model=ModelConfig.TOGETHER_WHISPER_MODEL,
                language="de",                      # German medical context
                response_format="json",
                timestamp_granularities="segment",
            )
          
            
            #  # Transcribe using Together AI translations
            # file_obj = io.BytesIO(processed_audio)
            # file_obj.name = "audio.wav"  # Together AI SDK reads filename from file-like
            
            # try:
            #     response_2 = self.together_client.audio.translations.create(
            #         file=file_obj,
            #         model=ModelConfig.TOGETHER_WHISPER_MODEL,
            #         language="de",                      # German medical context
            #         response_format="json",
            #         timestamp_granularities="segment",
            #     )
            # except Exception as e:
            #     pass
            
            
            
            transcription = (getattr(response, "text", "") or "").strip()
            latency = time.time() - start_time
            
            logger.debug(f"Together AI transcription completed in {latency:.3f}s")
            
            return {
                "text": transcription,
                "model": "together_whisper",
                "latency": latency,
                "confidence": 1.0,  # Together AI doesn't provide confidence scores
                "provider": "together"
            }
            
        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"Together AI transcription failed: {e}")
            raise
    
    @monitor_latency("stt_transcribe", "stt")
    async def transcribe_audio(
        self, 
        audio_data: bytes, 
        audio_format: str = "webm",
        duration_seconds: Optional[float] = None,
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio using primary or fallback STT service.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Input audio format (webm, wav, mp3, etc.)
            duration_seconds: Audio duration for validation
            use_fallback: Whether to use fallback service
            
        Returns:
            Dict with transcription results
        """
        
        start_time = time.time()
        
        try:
            self.performance_stats["total_transcriptions"] += 1
            
            # Validate audio
            if duration_seconds:
                self._validate_audio(audio_data, duration_seconds)
            
            # Try primary service first (unless fallback is explicitly requested)    
            if not use_fallback and self.together_client:
                try:
                    result = await self._transcribe_with_together(audio_data)
                    latency_ms = result["latency"] * 1000
                    
                    # Check if fallback also exceeds threshold
                    if latency_ms > settings.stt_final_threshold:
                        logger.warning(f"Fallback STT latency {latency_ms:.1f}ms exceeds threshold {settings.stt_final_threshold}ms")
                        result["threshold_exceeded"] = True
                    else:
                        result["threshold_exceeded"] = False
                    
                    self._update_stats(result["latency"], True)
                    result["fallback_used"] = True
                    return result
                except Exception as e:
                    logger.error(f"Fallback STT also failed: {str(e)}")
                    self._update_stats(time.time() - start_time, False)
                    #return {"error": f"STT failed: {str(e)}"}
            else:
                error_msg = "No STT services available"
                self._update_stats(time.time() - start_time, False)
                #return {"error": error_msg}
            
             # Fallback path (or forced fallback or primary failed)
            if self.fw_model:
                try:
                    result = await self._transcribe_with_whisper(audio_data)
                    latency_ms = result["latency"] * 1000
                    
                    # Check if latency exceeds threshold and trigger fallback
                    if latency_ms > settings.stt_final_threshold:
                        logger.warning(f"Primary STT latency {latency_ms:.1f}ms exceeds threshold {settings.stt_final_threshold}ms, trying fallback")
                        use_fallback = True
                    else:
                        self._update_stats(result["latency"], True)
                        result["fallback_used"] = False
                        result["threshold_exceeded"] = False
                        return result
                        
                except Exception as e:
                    logger.warning(f"Primary STT failed, trying fallback: {str(e)}")
            
            
            
          
        except Exception as e:
            latency = time.time() - start_time
            self._update_stats(latency, False)
            logger.error(f"Transcription failed: {str(e)}")
            return {"error": f"Transcription failed: {str(e)}"}
    
    async def transcribe_chunked_audio(self, audio_chunks: List[bytes]) -> str:
        """
        Transcribe audio from multiple chunks for streaming support.
        
        Args:
            audio_chunks: List of audio chunk bytes
            
        Returns:
            Combined transcription text
        """
        try:
            logger.debug(f"Transcribing {len(audio_chunks)} audio chunks")
            
            if not audio_chunks:
                return ""
            
            # Combine chunks if they're small
            if len(audio_chunks) == 1:
                result = await self.transcribe_audio(audio_chunks[0])
                return result.get("text", "") if "error" not in result else ""
            
            # For multiple chunks, process them efficiently
            combined_audio = b''.join(audio_chunks)
            
            # Transcribe combined audio
            result = await self.transcribe_audio(combined_audio)
            
            if "error" in result:
                logger.error(f"Chunked transcription failed: {result['error']}")
                return ""
            
            transcription = result["text"]
            logger.debug(f"Chunked transcription completed: '{transcription[:50]}...'")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Chunked transcription failed: {e}")
            return ""
    
    async def transcribe_streaming_audio(self, audio_stream: asyncio.StreamReader) -> str:
        """
        Transcribe real-time audio stream for ultra-low latency.
        
        Args:
            audio_stream: Async stream reader for audio data
            
        Returns:
            Transcription text
        """
        try:
            logger.debug("Starting streaming audio transcription")
            
            # Process stream chunks
            audio_chunks = []
            while True:
                try:
                    chunk = await asyncio.wait_for(audio_stream.read(1024), timeout=1.0)
                    if not chunk:
                        break
                    audio_chunks.append(chunk)
                    
                    # Process in batches for optimal performance
                    if len(audio_chunks) >= 5:  # Process every 5 chunks
                        break
                except asyncio.TimeoutError:
                    break
            
            if not audio_chunks:
                return ""
            
            # Transcribe accumulated chunks
            return await self.transcribe_chunked_audio(audio_chunks)
            
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}")
            return ""
    
    async def transcribe_streaming(
        self, 
        audio_chunks: AsyncGenerator[bytes, None],
        chunk_duration: float = 1.5
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream transcription results as audio chunks arrive.
        
        Args:
            audio_chunks: Async generator of audio chunks
            chunk_duration: Duration of each chunk in seconds
            
        Yields:
            Dict with partial or final transcription results
        """
        accumulated_audio = io.BytesIO()
        last_transcription = ""
        
        async for chunk in audio_chunks:
            try:
                # Accumulate audio data
                accumulated_audio.write(chunk)
                accumulated_audio.seek(0)
                
                # Check if we have enough audio for transcription
                audio_data = accumulated_audio.getvalue()
                if len(audio_data) < 16000:  # Less than 1 second at 16kHz
                    continue
                
                # Transcribe accumulated audio
                try:
                    result = await self.transcribe_audio(audio_data, "webm")
                    
                    # Check if transcription has changed significantly
                    current_text = result.get("text", "")
                    if current_text != last_transcription and len(current_text) > len(last_transcription):
                        yield {
                            "type": "partial",
                            "text": current_text,
                            "confidence": result.get("confidence", 1.0),
                            "model": result.get("model", "unknown"),
                            "timestamp": time.time()
                        }
                        last_transcription = current_text
                        
                except Exception as e:
                    logger.warning(f"Transcription failed for chunk: {e}")
                    continue
                    
            except Exception as e:
                logger.error(f"Streaming transcription error: {e}")
                yield {
                    "type": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }
        
        # Final transcription
        try:
            final_audio = accumulated_audio.getvalue()
            if len(final_audio) > 0:
                result = await self.transcribe_audio(final_audio, "webm")
                
                yield {
                    "type": "final",
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 1.0),
                    "model": result.get("model", "unknown"),
                    "fallback_used": result.get("fallback_used", False),
                    "timestamp": time.time()
                }
        except Exception as e:
            logger.error(f"Final transcription failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _update_stats(self, latency: float, success: bool) -> None:
        """Update performance statistics."""
        self.performance_stats["latencies"].append(latency)
        
        if success:
            self.performance_stats["successful_transcriptions"] += 1
        else:
            self.performance_stats["failed_transcriptions"] += 1
        
        # Keep only last 100 latencies
        if len(self.performance_stats["latencies"]) > 100:
            self.performance_stats["latencies"] = self.performance_stats["latencies"][-100:]
        
        # Update average latency
        if self.performance_stats["latencies"]:
            self.performance_stats["avg_latency"] = sum(self.performance_stats["latencies"]) / len(self.performance_stats["latencies"])
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "total_transcriptions": self.performance_stats["total_transcriptions"],
            "successful_transcriptions": self.performance_stats["successful_transcriptions"],
            "failed_transcriptions": self.performance_stats["failed_transcriptions"],
            "avg_latency": self.performance_stats["avg_latency"],
            "models_warmed_up": self.models_warmed_up,
            "primary_model": "faster_whisper" if self.fw_model else "none",
            "fallback_model": "together_whisper" if self.together_client else "none"
        }
        
        # Add cache statistics if caching is enabled
        if settings.enable_caching:
            stats["cache_stats"] = get_cache_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Check STT service health and latency."""
        health_status = {
            "service": "stt",
            "status": "healthy",
            "providers": {},
            "timestamp": time.time()
        }
        
        # try:
        #     audio_filename = f"audio_data.wav"
        #     audio_path = os.path.join("audio_data", audio_filename)
        #     with open(audio_path, "rb") as f:
        #         test_audio = f.read()
        # except Exception as e:
        #     test_audio = b"\x1aE" * 1600  # 0.1 seconds of silence
        #     logger.error(f"Failed to read audio data: {e}")
        #     raise
        
        
        
        def generate_silent_wav(duration_sec=0.1, sr=16000):
            samples = np.zeros(int(duration_sec * sr), dtype=np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())
            buf.seek(0)
            return buf.read()
        
        return
    
        # Test Faster-Whisper
        test_audio = generate_silent_wav(duration_sec=0.1, sr=1600)
        try:
            start_time = time.time()
            # Send minimal test request           
            await self._transcribe_with_whisper(test_audio)
            fw_latency = (time.time() - start_time) * 1000
            health_status["providers"]["faster_whisper"] = {
                "status": "healthy",
                "latency_ms": fw_latency
            }
        except Exception as e:
            health_status["providers"]["faster_whisper"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Test Together AI
        try:
            start_time = time.time()
            #test_audio = b"\x00" * 1600
            await self._transcribe_with_together(test_audio)
            together_latency = (time.time() - start_time) * 1000
            health_status["providers"]["together"] = {
                "status": "healthy",
                "latency_ms": together_latency
            }
        except Exception as e:
            health_status["providers"]["together"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        if not any(p["status"] == "healthy" for p in health_status["providers"].values()):
            health_status["status"] = "unhealthy"
        
        return health_status


# Global STT service instance
stt_service = STTService()
