#!/usr/bin/env python3
"""
Test script for NER microservice integration.
"""

import asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.services.ner_service import MedicalNERService, EntityModel


async def test_ner_microservice():
    """Test the NER microservice integration."""
    print("Testing NER Microservice Integration")
    print("=" * 50)

    # Initialize the service
    ner_service = MedicalNERService()

    try:
        # Test single text extraction
        print("\n1. Testing single text extraction:")
        sample_text = "mld beide beine 60mins, übungen mit theraband, stehen übungen, atemen übungen"
        print(f"Input: {sample_text}")

        entities = await ner_service.extract_entities(sample_text)
        print(f"Found {len(entities)} entities:")

        for entity in entities:
            print(
                f"  • {entity.text} ({entity.label}) - Confidence: {entity.confidence}"
            )
            if entity.source_model:
                print(f"    Source: {entity.source_model}")

        # Test batch extraction
        print("\n2. Testing batch extraction:")
        batch_texts = ["WTT LWS und BWS", "Mobilisation of schulter gelenk"]
        print(f"Input texts: {batch_texts}")

        batch_results = await ner_service.extract_entities_batch(batch_texts)
        print(f"Found {len(batch_results)} result sets:")

        for i, entities in enumerate(batch_results):
            print(f"  Text {i+1}: {len(entities)} entities")
            for entity in entities:
                print(f"    • {entity.text} ({entity.label})")

        # Test health check
        print("\n3. Testing health check:")
        health = await ner_service.health_check()
        print(f"Health status: {health}")

        # Test statistics
        print("\n4. Testing entity statistics:")
        if entities:
            stats = ner_service.get_entity_statistics(entities)
            print(f"Statistics: {stats}")

        # Test display formatting
        print("\n5. Testing display formatting:")
        display = ner_service.format_entities_for_display(entities)
        print(f"Formatted output:\n{display}")

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        await ner_service.close()


if __name__ == "__main__":
    asyncio.run(test_ner_microservice())
