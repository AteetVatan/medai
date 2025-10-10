"""
Medical Named Entity Recognition service for medAI MVP.
Uses external microservice for NER processing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

import httpx

from src.utils import settings
from ..utils.logging import get_logger, get_latency_logger, monitor_latency
from src.models import EntityModel

logger = get_logger(__name__)
latency_logger = get_latency_logger()

# Backward compatibility alias
MedicalEntity = EntityModel


class NERMicroserviceClient:
    """HTTP client for NER microservice."""
    
    def __init__(self, base_url: str = settings.ner_microservice_base_url):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """Call the /extract endpoint."""
        try:
            response = await self.client.post(
                f"{self.base_url}/extract",
                json={"text": text},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling NER microservice: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling NER microservice: {e}")
            raise

    async def extract_entities_batch(self, texts: List[str]) -> Dict[str, Any]:
        """Call the /extract_batch endpoint."""
        try:
            response = await self.client.post(
                f"{self.base_url}/extract_batch",
                json={"texts": texts},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling NER microservice batch: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling NER microservice batch: {e}")
            raise


class MedicalNERService:
    """Medical Named Entity Recognition service using microservice."""
    
    def __init__(self, microservice_url: str = settings.ner_microservice_base_url):
        self.client = NERMicroserviceClient(microservice_url)
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.close()
    
    def _convert_microservice_entity(self, entity_data: Dict[str, Any]) -> EntityModel:
        """Convert microservice entity response to EntityModel."""
        return EntityModel.from_dict(entity_data)
    
    # ---------- extraction ----------
    @monitor_latency("ner_extract", "microservice")
    async def extract_entities(self, text: str) -> List[EntityModel]:
        """
        Extract medical entities from text using microservice.
        """
        try:
            response = await self.client.extract_entities(text)
            entities = []
            
            for entity_data in response.get("entities", []):
                entity = self._convert_microservice_entity(entity_data)
                entities.append(entity)
            
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise
    
    # ---------- batch extraction ----------
    @monitor_latency("ner_extract_batch", "microservice")
    async def extract_entities_batch(self, texts: List[str]) -> List[List[EntityModel]]:
        """Extract entities from multiple texts in batch using microservice."""
        try:
            response = await self.client.extract_entities_batch(texts)
            results = []
            
            for text_entities in response.get("results", []):
                entities = []
                for entity_data in text_entities:
                    entity = self._convert_microservice_entity(entity_data)
                    entities.append(entity)
                results.append(entities)
            
            return results

        except Exception as e:
            logger.error(f"Batch entity extraction failed: {e}")
            raise
    
    # ---------- stats / display ----------
    def get_entity_statistics(self, entities: List[EntityModel]) -> Dict[str, Any]:
        stats = {
            "total_entities": len(entities),
            "by_category": {},
            "by_label": {},
            "by_source_model": {},
            "with_icd_codes": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }
        for e in entities:
            cat = e.category or "unknown"
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            stats["by_label"][e.label] = stats["by_label"].get(e.label, 0) + 1
            
            if e.source_model:
                stats["by_source_model"][e.source_model] = stats["by_source_model"].get(e.source_model, 0) + 1
            
            if e.icd_code:
                stats["with_icd_codes"] += 1
            if e.confidence > 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif e.confidence > 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
        return stats
    
    def format_entities_for_display(self, entities: List[EntityModel]) -> str:
        if not entities:
            return "Keine medizinischen Entitäten gefunden."
        lines = []
        for e in entities:
            line = f"• {e.text} ({e.label})"
            if e.confidence:
                line += f" – Confidence: {e.confidence:.2f}"
            if e.icd_code:
                line += f" – ICD: {e.icd_code}"
            if e.icd_description:
                line += f" – {e.icd_description}"
            if e.source_model:
                line += f" – Source: {e.source_model}"
            lines.append(line)
        return "\n".join(lines)
    
    # ---------- health ----------
    async def health_check(self) -> Dict[str, Any]:
        try:
            sample = "Patient mit LWS-Syndrom, Gangschulung empfohlen. Paracetamol 500 mg 2× täglich."
            ents = await self.extract_entities(sample)
            return {
                "service": "ner",
                "status": "healthy",
                "microservice_url": self.client.base_url,
                "test_entities_found": len(ents),
                "test_processing_time": "measured by microservice",
            }
        except Exception as e:
            return {"service": "ner", "status": "unhealthy", "error": str(e)}

# Global NER service instance
ner_service = MedicalNERService()
