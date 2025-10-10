# EntityModel Documentation

## Purpose
The `EntityModel` is a comprehensive data class that represents medical entities extracted from text, designed to match the microservice response format exactly.

## Design
The `EntityModel` provides a structured way to handle entity data with:
- **Exact microservice compatibility**: Matches the JSON response format from the NER microservice
- **Type safety**: Full type hints for all fields
- **Conversion methods**: Easy conversion between dict and EntityModel
- **String representation**: Human-readable entity display
- **Backward compatibility**: Alias for `MedicalEntity` maintained

## Code Structure

### Core Fields
```python
@dataclass
class EntityModel:
    text: str                    # The extracted text
    label: str                   # Entity label/type
    start: int                    # Start position in text
    end: int                     # End position in text
    confidence: float            # Extraction confidence (0.0-1.0)
    normalized_text: Optional[str] = None      # Normalized version
    icd_code: Optional[str] = None             # ICD-10/ICD-11 code
    icd_description: Optional[str] = None      # ICD description
    source_model: Optional[str] = None         # Model that extracted it
    category: Optional[str] = None            # Entity category
```

### Key Methods

#### `from_dict(data: Dict[str, Any]) -> EntityModel`
Creates EntityModel from microservice response dictionary.
```python
entity_data = {
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
entity = EntityModel.from_dict(entity_data)
```

#### `to_dict() -> Dict[str, Any]`
Converts EntityModel back to dictionary format.
```python
entity_dict = entity.to_dict()
```

#### `__str__() -> str`
Provides human-readable string representation.
```python
print(entity)  # "mld (MISC) conf:0.80 src:spacy"
```

## Usage Examples

### Basic Entity Creation
```python
from src.services.ner_service import EntityModel

# From microservice response
entity = EntityModel.from_dict(microservice_response)

# Direct creation
entity = EntityModel(
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
```

### Batch Processing
```python
# Process multiple entities from microservice
entities_data = response.get("entities", [])
entities = [EntityModel.from_dict(data) for data in entities_data]

# Convert back to dictionaries for API responses
entity_dicts = [entity.to_dict() for entity in entities]
```

### Integration with NER Service
```python
from src.services.ner_service import MedicalNERService

ner_service = MedicalNERService()
entities = await ner_service.extract_entities("mld beide beine 60mins")

# All methods now use EntityModel
stats = ner_service.get_entity_statistics(entities)
display = ner_service.format_entities_for_display(entities)
```

## Microservice Response Format

The EntityModel matches this exact JSON structure:
```json
{
  "text": "mld",
  "label": "MISC",
  "start": 0,
  "end": 3,
  "confidence": 0.8,
  "normalized_text": "Mld",
  "icd_code": null,
  "icd_description": null,
  "source_model": "spacy",
  "category": null
}
```

## Field Descriptions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `text` | str | The actual text that was extracted | "mld" |
| `label` | str | The entity type/label | "MISC", "DIAGNOSIS", "MED_DRUG" |
| `start` | int | Character start position in original text | 0 |
| `end` | int | Character end position in original text | 3 |
| `confidence` | float | Extraction confidence (0.0-1.0) | 0.8 |
| `normalized_text` | Optional[str] | Normalized version of the text | "Mld" |
| `icd_code` | Optional[str] | ICD-10/ICD-11 code if applicable | "M53" |
| `icd_description` | Optional[str] | ICD description if available | "Zervikalsyndrom" |
| `source_model` | Optional[str] | Model that extracted the entity | "spacy", "patterns" |
| `category` | Optional[str] | Entity category | "diagnosis", "symptom" |

## Edge Cases Handled

1. **Missing fields**: All optional fields default to `None`
2. **Type conversion**: Safe handling of different data types
3. **Empty responses**: Graceful handling of empty entity lists
4. **Invalid data**: Robust error handling in `from_dict()`

## Backward Compatibility

The `MedicalEntity` alias is maintained for backward compatibility:
```python
# Both work identically
from src.services.ner_service import EntityModel, MedicalEntity
entity1 = EntityModel.from_dict(data)
entity2 = MedicalEntity.from_dict(data)  # Same as above
```

## Testing

Run the example script to see EntityModel in action:
```bash
python entity_model_example.py
```

This demonstrates:
- Creating entities from microservice responses
- Direct entity creation
- Dictionary conversion
- String representation
- Batch processing
- Statistics and display formatting

## Suggestions for Production

1. **Validation**: Add field validation for confidence scores (0.0-1.0)
2. **Serialization**: Consider adding JSON serialization methods
3. **Comparison**: Add `__eq__` method for entity comparison
4. **Filtering**: Add methods to filter entities by label/category
5. **Metrics**: Add methods to calculate entity overlap and coverage
