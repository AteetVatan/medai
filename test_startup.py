#!/usr/bin/env python3
"""
Test startup script to verify medAI MVP components.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_imports():
    """Test that all modules can be imported."""
    print(" Testing imports...")
    
    try:
        from utils.config import settings
        print("[OK] Config imported")
        
        from utils.logging import get_logger, get_latency_logger
        print("[OK] Logging imported")
        
        from services.stt_service import stt_service
        print("[OK] STT service imported")
        
        from services.ner_service import ner_service
        print("[OK] NER service imported")
        
        from services.llm_service import llm_service
        print("[OK] LLM service imported")
        
        from services.translation_service import translation_service
        print("[OK] Translation service imported")
        
        from services.storage_service import storage_service
        print("[OK] Storage service imported")
        
        from agents.clinical_intake_agent import clinical_intake_agent
        print("[OK] Clinical agent imported")
        
        from api.main import app
        print("[OK] FastAPI app imported")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

async def test_health_checks():
    """Test health checks for all services."""
    print("\n Testing health checks...")
    
    try:
        # Warm up services first
        print("🔥 Warming up services...")
        await clinical_intake_agent.warm_up_services()
        
        # Test STT service health
        stt_health = await stt_service.health_check()
        print(f"[OK] STT health: {stt_health['status']}")
        
        # Test NER service health
        ner_health = await ner_service.health_check()
        print(f"[OK] NER health: {ner_health['status']}")
        
        # Test LLM service health
        llm_health = await llm_service.health_check()
        print(f"[OK] LLM health: {llm_health['status']}")
        
        # Test Translation service health
        translation_health = await translation_service.health_check()
        print(f"[OK] Translation health: {translation_health['status']}")
        
        # Test Storage service health
        storage_health = await storage_service.health_check()
        print(f"[OK] Storage health: {storage_health['status']}")
        
        # Test Clinical agent health
        agent_health = await clinical_intake_agent.health_check()
        print(f"[OK] Agent health: {agent_health['status']}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Health check failed: {e}")
        return False

async def main():
    """Main test function."""
    print(" medAI MVP Startup Test")
    print("=" * 50)
    
    # Test imports
    if not await test_imports():
        print("\n💥 Import test failed!")
        return False
    
    # Test health checks
    if not await test_health_checks():
        print("\n💥 Health check test failed!")
        return False
    
    print("\n All tests passed! medAI MVP is ready to start.")
    print("\nTo start the server:")
    print("  python start_dev.py")
    print("\nOr with uvicorn directly:")
    print("  uvicorn src.api.main:app --reload")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
