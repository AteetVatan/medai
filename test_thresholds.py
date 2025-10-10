#!/usr/bin/env python3
"""
Test script to verify latency thresholds are working correctly.
"""

import asyncio
import time
from src.services.stt_service import stt_service
from src.services.llm_service import llm_service
from src.services.translation_service import translation_service
from src.utils.config import settings

async def test_stt_thresholds():
    """Test STT service thresholds."""
    print("Testing STT Service Thresholds")
    print(f"   STT Final Threshold: {settings.stt_final_threshold}ms")
    
    # Test with small audio (should be fast)
    small_audio = b"\x00" * 1600  # 0.1 seconds of silence
    
    try:
        result = await stt_service.transcribe_audio(small_audio, "wav")
        if "threshold_exceeded" in result:
            print(f"   [OK] Threshold monitoring: {result['threshold_exceeded']}")
        if "latency_ms" in result:
            print(f"   [OK] Latency: {result['latency_ms']:.1f}ms")
    except Exception as e:
        print(f"   [WARN]  STT test failed: {e}")

async def test_llm_thresholds():
    """Test LLM service thresholds."""
    print("\n Testing LLM Service Thresholds")
    print(f"   LLM Summary Threshold: {settings.llm_summary_threshold}ms")
    
    messages = [
        {"role": "user", "content": "Test message for threshold testing"}
    ]
    
    try:
        result = await llm_service.generate_text(messages)
        if "threshold_exceeded" in result:
            print(f"   [OK] Threshold monitoring: {result['threshold_exceeded']}")
        if "latency_ms" in result:
            print(f"   [OK] Latency: {result['latency_ms']:.1f}ms")
    except Exception as e:
        print(f"   [WARN]  LLM test failed: {e}")

async def test_translation_thresholds():
    """Test Translation service thresholds."""
    print("\n Testing Translation Service Thresholds")
    print(f"   Translation Threshold: {settings.translation_threshold}ms")
    
    try:
        result = await translation_service.translate_text("Test message", "de", "en")
        if "threshold_exceeded" in result:
            print(f"   [OK] Threshold monitoring: {result['threshold_exceeded']}")
        if "latency_ms" in result:
            print(f"   [OK] Latency: {result['latency_ms']:.1f}ms")
    except Exception as e:
        print(f"   [WARN]  Translation test failed: {e}")

async def main():
    """Run all threshold tests."""
    print(" Testing Latency Thresholds Implementation")
    print("=" * 50)
    
    # Test all services
    #await test_stt_thresholds()
    await test_llm_thresholds()
    await test_translation_thresholds()
    
    print("\n[OK] Threshold testing completed!")
    print("\n Current Thresholds:")
    print(f"   STT Final: {settings.stt_final_threshold}ms")
    print(f"   STT Partial: {settings.stt_partial_threshold}ms")
    print(f"   LLM Summary: {settings.llm_summary_threshold}ms")
    print(f"   Translation: {settings.translation_threshold}ms")

if __name__ == "__main__":
    asyncio.run(main())
