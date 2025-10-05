"""
LLM service for medAI MVP.
Supports Mistral 7B via online APIs with fallback to cost-effective models.
Temperature is always 0.0 for deterministic output.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
import re

import httpx

from ..utils.config import settings, ModelConfig, LatencyConfig
from ..utils.logging import get_logger, get_latency_logger, monitor_latency
from ..utils.cache import cached, cache_key, get_cache_stats

logger = get_logger(__name__)
latency_logger = get_latency_logger()


class LLMService:
    """LLM service with Mistral 7B primary and fallback support."""
    
    def __init__(self):
        self.mistral_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers={"Authorization": f"Bearer {settings.mistral_api_key}"}
        )
        self.openrouter_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"}
        )
        self.together_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers={"Authorization": f"Bearer {settings.together_api_key}"}
        )
        self._fallback_triggered = False
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "latencies": []
        }
        self._system_prompts = self._load_system_prompts()
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.mistral_client.aclose()
        await self.openrouter_client.aclose()
        await self.together_client.aclose()
    
    def _load_system_prompts(self) -> Dict[str, str]:
        """Load system prompts for different clinical tasks."""
        return {
            "intake_summary": """Du bist ein erfahrener medizinischer Assistent. Deine Aufgabe ist es, klinische Aufnahmegespräche zu strukturieren und zusammenzufassen.

WICHTIGE REGELN:
- Verwende NUR medizinische Fachsprache
- Strukturiere die Informationen klar und präzise
- Verwende KEINE persönlichen Identifikationsmerkmale (PII)
- Fokussiere auf medizinisch relevante Informationen
- Verwende deutsche medizinische Terminologie

Struktur:
1. HAUPTBESCHWERDEN
2. AKTUELLE SYMPTOME
3. MEDIZINISCHE VORGESCHICHTE
4. MEDIKAMENTE
5. ALLERGIEN
6. SOZIALE ANGELEGENHEITEN
7. BEFUNDE
8. DIAGNOSE/VERDACHT
9. BEHANDLUNGSPLAN
10. NÄCHSTE SCHRITTE""",

            "assessment": """Du bist ein klinischer Psychologe. Erstelle eine strukturierte psychologische Einschätzung basierend auf dem Gespräch.

Struktur:
1. PRÄSENTIERENDE PROBLEME
2. PSYCHOLOGISCHE SYMPTOME
3. RISIKOFAKTOREN
4. SCHUTZFAKTOREN
5. DIFFERENTIALDIAGNOSE
6. BEHANDLUNGSEMPFEHLUNGEN
7. PROGNOSE""",

            "treatment_plan": """Du bist ein Therapeut. Erstelle einen strukturierten Behandlungsplan.

Struktur:
1. BEHANDLUNGSZIELE
2. THERAPEUTISCHE INTERVENTIONEN
3. ZEITRAHMEN
4. MESSBARE ZIELE
5. HOMEWORK/ÜBUNGEN
6. NÄCHSTE TERMINE"""
        }
    
    def _strip_pii(self, text: str) -> str:
        """Strip personally identifiable information from text."""
        if not settings.enable_pii_stripping:
            return text
        
        # Common German PII patterns
        pii_patterns = [
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]'),  # Names
            (r'\b\d{1,2}\.\d{1,2}\.\d{4}\b', '[DATUM]'),  # Dates
            (r'\b\d{5}\b', '[PLZ]'),  # Postal codes
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
            (r'\b\d{3,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b', '[TELEFON]'),  # Phone
            (r'\b[A-Za-z0-9]{8,}\b', '[ID]'),  # IDs
        ]
        
        stripped_text = text
        for pattern, replacement in pii_patterns:
            stripped_text = re.sub(pattern, replacement, stripped_text)
        
        return stripped_text
    
    @cached("llm_mistral", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("llm_mistral", "mistral-7b")
    async def _call_mistral(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> Dict[str, Any]:
        """Call Mistral 7B API."""
        try:
            payload = {
                "model": ModelConfig.MISTRAL_MODEL,
                "messages": messages,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P,
                "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
                "stream": False
            }
            
            response = await self.mistral_client.post(
                settings.mistral_endpoint,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": "mistral-7b",
                "provider": "mistral",
                "usage": result.get("usage", {}),
                "fallback_used": False
            }
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Mistral API rate limited, triggering fallback")
                raise Exception("Rate limited - triggering fallback")
            elif e.response.status_code >= 500:
                logger.warning(f"Mistral API error {e.response.status_code}, triggering fallback")
                raise Exception("API error - triggering fallback")
            else:
                logger.error(f"Mistral API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Mistral API call failed: {e}")
            raise
    
    @cached("llm_openrouter", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("llm_openrouter", "phi-3-mini")
    async def _call_openrouter_fallback(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> Dict[str, Any]:
        """Call OpenRouter fallback model."""
        try:
            payload = {
                "model": settings.fallback_llm_model,
                "messages": messages,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P,
                "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
                "stream": False
            }
            
            response = await self.openrouter_client.post(
                settings.openrouter_endpoint,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": settings.fallback_llm_model,
                "provider": "openrouter",
                "usage": result.get("usage", {}),
                "fallback_used": True
            }
            
        except Exception as e:
            logger.error(f"OpenRouter fallback failed: {e}")
            raise
    
    @cached("llm_together", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("llm_together", "gemini-flash")
    async def _call_together_fallback(self, messages: List[Dict[str, str]], max_tokens: int = 2048) -> Dict[str, Any]:
        """Call Together AI fallback model."""
        try:
            payload = {
                "model": "google/gemini-flash-1.5",
                "messages": messages,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P,
                "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
                "stream": False
            }
            
            response = await self.together_client.post(
                "https://api.together.xyz/v1/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            
            return {
                "content": result["choices"][0]["message"]["content"],
                "model": "gemini-flash-1.5",
                "provider": "together",
                "usage": result.get("usage", {}),
                "fallback_used": True
            }
            
        except Exception as e:
            logger.error(f"Together AI fallback failed: {e}")
            raise
    
    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text using primary or fallback LLM service with latency thresholds.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            use_fallback: Whether to use fallback service
            
        Returns:
            Dict with generation results
        """
        start_time = time.time()
        
        try:
            # Try primary service first (unless fallback is explicitly requested)
            if not use_fallback and self.mistral_client:
                try:
                    result = await self._call_mistral(messages, max_tokens)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Check if latency exceeds threshold and trigger fallback
                    if latency_ms > settings.llm_summary_threshold:
                        logger.warning(f"Primary LLM latency {latency_ms:.1f}ms exceeds threshold {settings.llm_summary_threshold}ms, trying fallback")
                        use_fallback = True
                    else:
                        result["threshold_exceeded"] = False
                        return result
                        
                except Exception as e:
                    logger.warning(f"Primary LLM failed, trying fallback: {str(e)}")
                    use_fallback = True
            
            # Try OpenRouter fallback
            if use_fallback and self.openrouter_client:
                try:
                    result = await self._call_openrouter(messages, max_tokens)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if latency_ms > settings.llm_summary_threshold:
                        logger.warning(f"OpenRouter LLM latency {latency_ms:.1f}ms exceeds threshold {settings.llm_summary_threshold}ms, trying final fallback")
                        use_fallback = True
                    else:
                        result["threshold_exceeded"] = False
                        return result
                        
                except Exception as e:
                    logger.warning(f"OpenRouter fallback failed, trying final fallback: {str(e)}")
            
            # Try Together AI final fallback
            if self.together_client:
                try:
                    result = await self._call_together(messages, max_tokens)
                    latency_ms = (time.time() - start_time) * 1000
                    
                    if latency_ms > settings.llm_summary_threshold:
                        logger.warning(f"Together AI LLM latency {latency_ms:.1f}ms exceeds threshold {settings.llm_summary_threshold}ms")
                        result["threshold_exceeded"] = True
                    else:
                        result["threshold_exceeded"] = False
                    
                    return result
                except Exception as e:
                    logger.error(f"All LLM services failed: {str(e)}")
                    raise Exception(f"LLM generation failed: {str(e)}")
            else:
                raise Exception("No LLM services available")
                
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    async def generate_clinical_summary(
        self,
        transcript: str,
        entities: List[Dict[str, Any]],
        task_type: str = "intake_summary",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate structured clinical summary from transcript and entities.
        
        Args:
            transcript: Speech-to-text transcript
            entities: Extracted medical entities
            task_type: Type of clinical task (intake_summary, assessment, treatment_plan)
            user_id: User ID for compliance logging
            
        Returns:
            Dict with structured clinical summary
        """
        try:
            # Strip PII from transcript
            clean_transcript = self._strip_pii(transcript)
            
            # Prepare messages
            system_prompt = self._system_prompts.get(task_type, self._system_prompts["intake_summary"])
            
            # Format entities for context
            entity_context = self._format_entities_for_llm(entities)
            
            user_message = f"""Transkript des Patientengesprächs:

{clean_transcript}

Extrahierte medizinische Entitäten:
{entity_context}

Bitte erstelle eine strukturierte klinische Zusammenfassung basierend auf dem obigen Transkript und den extrahierten Entitäten."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Try Mistral first (unless fallback already triggered)
            if not self._fallback_triggered:
                try:
                    result = await self._call_mistral(messages)
                    self._log_llm_interaction("mistral-7b", len(user_message), len(result["content"]), user_id)
                    return result
                except Exception as e:
                    if "fallback" in str(e).lower():
                        logger.warning("Mistral failed, switching to fallback")
                        self._fallback_triggered = True
                    else:
                        raise
            
            # Try OpenRouter fallback
            try:
                result = await self._call_openrouter_fallback(messages)
                self._log_llm_interaction(settings.fallback_llm_model, len(user_message), len(result["content"]), user_id)
                return result
            except Exception as e:
                logger.warning(f"OpenRouter fallback failed: {e}")
            
            # Try Together AI as last resort
            try:
                result = await self._call_together_fallback(messages)
                self._log_llm_interaction("gemini-flash-1.5", len(user_message), len(result["content"]), user_id)
                return result
            except Exception as e:
                logger.error(f"All LLM providers failed: {e}")
                raise Exception("All LLM providers unavailable")
                
        except Exception as e:
            logger.error(f"Clinical summary generation failed: {e}")
            raise
    
    def _format_entities_for_llm(self, entities: List[Dict[str, Any]]) -> str:
        """Format extracted entities for LLM context."""
        if not entities:
            return "Keine medizinischen Entitäten extrahiert."
        
        formatted_entities = []
        for entity in entities:
            line = f"- {entity.get('text', '')} ({entity.get('label', '')})"
            if entity.get('icd_code'):
                line += f" [ICD: {entity['icd_code']}]"
            if entity.get('category'):
                line += f" [Kategorie: {entity['category']}]"
            formatted_entities.append(line)
        
        return "\n".join(formatted_entities)
    
    def _log_llm_interaction(self, model: str, prompt_length: int, response_length: int, user_id: Optional[str]):
        """Log LLM interaction for compliance."""
        from ..utils.logging import get_compliance_logger
        compliance_logger = get_compliance_logger()
        
        compliance_logger.log_llm_interaction(
            request_id=f"llm_{int(time.time())}",
            model=model,
            prompt_length=prompt_length,
            response_length=response_length,
            user_id=user_id or "unknown",
            pii_stripped=settings.enable_pii_stripping
        )
    
    async def generate_structured_notes(
        self,
        clinical_summary: str,
        format_type: str = "json"
    ) -> Dict[str, Any]:
        """
        Generate structured notes from clinical summary.
        
        Args:
            clinical_summary: Raw clinical summary text
            format_type: Output format (json, xml, markdown)
            
        Returns:
            Dict with structured notes
        """
        try:
            system_prompt = """Du bist ein medizinischer Assistent. Konvertiere die klinische Zusammenfassung in strukturierte Notizen.

WICHTIGE REGELN:
- Verwende JSON-Format
- Strukturiere alle Informationen hierarchisch
- Verwende deutsche medizinische Terminologie
- Stelle sicher, dass alle Felder korrekt ausgefüllt sind

JSON-Struktur:
{
  "hauptbeschwerden": [],
  "aktuelle_symptome": [],
  "medizinische_vorgeschichte": [],
  "medikamente": [],
  "allergien": [],
  "soziale_angelegenheiten": {},
  "befunde": {},
  "diagnose_verdacht": [],
  "behandlungsplan": {},
  "naechste_schritte": []
}"""

            user_message = f"""Konvertiere diese klinische Zusammenfassung in strukturierte JSON-Notizen:

{clinical_summary}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Use fallback model for structured output (more reliable)
            try:
                result = await self._call_openrouter_fallback(messages, max_tokens=1024)
                
                # Try to parse JSON from response
                content = result["content"]
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    structured_notes = json.loads(json_str)
                else:
                    # Fallback to simple structure
                    structured_notes = {
                        "raw_summary": content,
                        "structured": False
                    }
                
                return {
                    "structured_notes": structured_notes,
                    "format": format_type,
                    "model": result["model"],
                    "fallback_used": result["fallback_used"]
                }
                
            except Exception as e:
                logger.error(f"Structured notes generation failed: {e}")
                return {
                    "structured_notes": {"raw_summary": clinical_summary, "structured": False},
                    "format": format_type,
                    "error": str(e)
                }
                
        except Exception as e:
            logger.error(f"Structured notes generation failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check LLM service health."""
        health_status = {
            "service": "llm",
            "status": "healthy",
            "providers": {},
            "timestamp": time.time()
        }
        
        # Test Mistral
        try:
            start_time = time.time()
            test_messages = [
                {"role": "system", "content": "Du bist ein medizinischer Assistent."},
                {"role": "user", "content": "Test"}
            ]
            await self._call_mistral(test_messages, max_tokens=10)
            mistral_latency = (time.time() - start_time) * 1000
            health_status["providers"]["mistral"] = {
                "status": "healthy",
                "latency_ms": mistral_latency
            }
        except Exception as e:
            health_status["providers"]["mistral"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Test OpenRouter fallback
        try:
            start_time = time.time()
            test_messages = [
                {"role": "system", "content": "Du bist ein medizinischer Assistent."},
                {"role": "user", "content": "Test"}
            ]
            await self._call_openrouter_fallback(test_messages, max_tokens=10)
            openrouter_latency = (time.time() - start_time) * 1000
            health_status["providers"]["openrouter"] = {
                "status": "healthy",
                "latency_ms": openrouter_latency
            }
        except Exception as e:
            health_status["providers"]["openrouter"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        if not any(p["status"] == "healthy" for p in health_status["providers"].values()):
            health_status["status"] = "unhealthy"
        
        return health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            "total_requests": self.performance_stats["total_requests"],
            "successful_requests": self.performance_stats["successful_requests"],
            "failed_requests": self.performance_stats["failed_requests"],
            "avg_latency": self.performance_stats["avg_latency"],
            "models_warmed_up": self.models_warmed_up,
            "primary_model": "mistral" if self.mistral_client else "none",
            "fallback_models": {
                "openrouter": "available" if self.openrouter_client else "unavailable",
                "together": "available" if self.together_client else "unavailable"
            }
        }
        
        # Add cache statistics if caching is enabled
        if settings.enable_caching:
            stats["cache_stats"] = get_cache_stats()
        
        return stats


# Global LLM service instance
llm_service = LLMService()
