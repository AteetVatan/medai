from dataclasses import dataclass
from typing import Dict, Any, Optional



@dataclass
class EntityModel:
    """Entity model matching microservice response format exactly."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_text: Optional[str] = None
    icd_code: Optional[str] = None
    icd_description: Optional[str] = None
    source_model: Optional[str] = None
    category: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityModel':
        """Create EntityModel from dictionary (microservice response)."""
        return cls(
            text=data.get("text", ""),
            label=data.get("label", ""),
            start=data.get("start", 0),
            end=data.get("end", 0),
            confidence=data.get("confidence", 0.0),
            normalized_text=data.get("normalized_text"),
            icd_code=data.get("icd_code"),
            icd_description=data.get("icd_description"),
            source_model=data.get("source_model"),
            category=data.get("category")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EntityModel to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "normalized_text": self.normalized_text,
            "icd_code": self.icd_code,
            "icd_description": self.icd_description,
            "source_model": self.source_model,
            "category": self.category
        }
    
    def __str__(self) -> str:
        """String representation of the entity."""
        parts = [f"{self.text} ({self.label})"]
        if self.confidence:
            parts.append(f"conf:{self.confidence:.2f}")
        if self.source_model:
            parts.append(f"src:{self.source_model}")
        if self.icd_code:
            parts.append(f"ICD:{self.icd_code}")
        return " ".join(parts)
