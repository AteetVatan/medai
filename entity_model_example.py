#!/usr/bin/env python3
"""
Example demonstrating the EntityModel usage and features.
"""

import sys
import os
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.services.ner_service import EntityModel

def demonstrate_entity_model():
    """Demonstrate EntityModel features."""
    print("EntityModel Demonstration")
    print("=" * 50)
    
    # Example microservice response data
    microservice_response = {
        "text": "mld",
        "label": "MISC",
        "start": 0,
        "end": 3,
        "confidence": 0.8,
        "normalized_text": "Mld",
        "icd_code": None,
        "icd_description": None,
        "source_model": "spacy",
        "category": None
    }
    
    print("\n1. Creating EntityModel from microservice response:")
    print(f"Input data: {microservice_response}")
    
    # Create EntityModel from dictionary
    entity = EntityModel.from_dict(microservice_response)
    print(f"Created entity: {entity}")
    
    # Demonstrate string representation
    print(f"\nString representation: {str(entity)}")
    
    # Demonstrate to_dict conversion
    print(f"\n2. Converting back to dictionary:")
    entity_dict = entity.to_dict()
    print(f"Entity as dict: {entity_dict}")
    
    # Demonstrate direct creation
    print(f"\n3. Creating EntityModel directly:")
    direct_entity = EntityModel(
        text="theraband",
        label="ORGANIZATION",
        start=36,
        end=45,
        confidence=0.8,
        normalized_text="Theraband",
        icd_code=None,
        icd_description=None,
        source_model="spacy",
        category=None
    )
    print(f"Direct entity: {direct_entity}")
    
    # Demonstrate with ICD code
    print(f"\n4. Entity with ICD code:")
    icd_entity = EntityModel(
        text="LWS-Syndrom",
        label="DIAGNOSIS",
        start=0,
        end=11,
        confidence=0.9,
        normalized_text="lws-syndrom",
        icd_code="M53",
        icd_description="Zervikalsyndrom",
        source_model="patterns",
        category="diagnosis"
    )
    print(f"ICD entity: {icd_entity}")
    print(f"ICD entity dict: {icd_entity.to_dict()}")
    
    # Demonstrate batch processing
    print(f"\n5. Batch processing example:")
    entities_data = [
        {
            "text": "mld",
            "label": "MISC",
            "start": 0,
            "end": 3,
            "confidence": 0.8,
            "normalized_text": "Mld",
            "icd_code": None,
            "icd_description": None,
            "source_model": "spacy",
            "category": None
        },
        {
            "text": "beide",
            "label": "MED_DRUG",
            "start": 4,
            "end": 9,
            "confidence": 0.7,
            "normalized_text": "Beide",
            "icd_code": None,
            "icd_description": None,
            "source_model": "patterns",
            "category": None
        },
        {
            "text": "theraband",
            "label": "ORGANIZATION",
            "start": 36,
            "end": 45,
            "confidence": 0.8,
            "normalized_text": "Theraband",
            "icd_code": None,
            "icd_description": None,
            "source_model": "spacy",
            "category": None
        }
    ]
    
    entities = [EntityModel.from_dict(data) for data in entities_data]
    print(f"Created {len(entities)} entities:")
    for i, entity in enumerate(entities):
        print(f"  {i+1}. {entity}")
    
    # Demonstrate statistics
    print(f"\n6. Entity statistics:")
    from src.services.ner_service import MedicalNERService
    ner_service = MedicalNERService()
    
    stats = ner_service.get_entity_statistics(entities)
    print(f"Statistics: {stats}")
    
    # Demonstrate display formatting
    print(f"\n7. Display formatting:")
    display = ner_service.format_entities_for_display(entities)
    print(f"Formatted display:\n{display}")
    
    print(f"\n✅ EntityModel demonstration completed!")

if __name__ == "__main__":
    demonstrate_entity_model()
