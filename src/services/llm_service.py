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

from src.utils.config import settings, ModelConfig, LatencyConfig
from src.utils.logging import get_logger, get_latency_logger, monitor_latency
from src.utils.cache import cached, cache_key, get_cache_stats

logger = get_logger(__name__)
latency_logger = get_latency_logger()


class LLMService:
    """LLM service with Mistral 7B primary and fallback support."""

    def __init__(self):
        self.mistral_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers={"Authorization": f"Bearer {settings.mistral_api_key}"},
        )
        self.openrouter_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
        )
        self.together_client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            headers={"Authorization": f"Bearer {settings.together_api_key}"},
        )
        self._fallback_triggered = False
        self.performance_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_latency": 0.0,
            "latencies": [],
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
            "physio_text_cleaner": """
Du bist ein erfahrener deutscher Physiotherapeut mit klinischer Dokumentationserfahrung.
Deine Aufgabe ist es, mehrere unklare oder fehlerhafte Transkripte aus Sprachaufnahmen zu einem
einheitlichen, sauberen und fachlich korrekten physiotherapeutischen Text zu bereinigen und kurz zusammenzufassen.

REGELN:
- Fasse alle übermittelten Teiltranskripte (Text- oder Bildliste) zu einem zusammenhängenden Therapiesatz zusammen.
- Bereinige Grammatik, Ausdruck und medizinische Terminologie.
- Verwende ausschließlich korrekte physiotherapeutische Fachsprache und übliche Abkürzungen (z. B. MLD, KG, MT, PNF, BWS, LWS, HWS, Mobilisation, Kräftigung, Dehnung, Stabilisation).
- Ersetze unklare oder falsch erkannte Wörter durch plausible physiotherapeutische Begriffe.
- Fasse die therapeutischen Inhalte präzise zusammen: Welche Technik, welche Körperregion, welches Ziel?
- Gib **nur den bereinigten und zusammengefassten Satz oder kurzen Absatz** zurück – keine Aufzählungen, keine Erklärungen, keine Zusatztexte.

ZIEL:
Ein kurzer, professioneller physiotherapeutischer Behandlungsvermerk im Stil:  
„MLD an den Beinen, Kräftigung, WTT-Technik mit Langerbeugelsohle, Mobilisation der Schulter beidseits, PNF für die rechte Schulter und Übungen mit dem Theraband.“
""",
            "intake_summary": """Du bist ein erfahrener Physiotherapeut. Deine Aufgabe ist es, physiotherapeutische Aufnahmegespräche zu strukturieren und zusammenzufassen.

WICHTIGE REGELN:
- Verwende NUR physiotherapeutische Fachsprache
- Strukturiere die Informationen klar und präzise
- Verwende KEINE persönlichen Identifikationsmerkmale (PII)
- Fokussiere auf bewegungs- und funktionsrelevante Informationen
- Verwende deutsche physiotherapeutische Terminologie

Struktur:
1. HAUPTBESCHWERDEN
2. SCHMERZANALYSE
3. BEWEGUNGSEINSCHRÄNKUNGEN
4. FUNKTIONELLE EINSCHRÄNKUNGEN
5. MEDIZINISCHE VORGESCHICHTE
6. MEDIKAMENTE
7. ALLERGIEN
8. BEFUNDE
9. THERAPIEZIELE
10. BEHANDLUNGSPLAN""",
            "assessment": """Du bist ein erfahrener Physiotherapeut. Erstelle eine strukturierte physiotherapeutische Einschätzung basierend auf dem Gespräch.

Struktur:
1. HAUPTBESCHWERDEN
2. BEWEGUNGSEINSCHRÄNKUNGEN
3. SCHMERZANALYSE
4. FUNKTIONELLE EINSCHRÄNKUNGEN
5. BEFUNDE
6. BEHANDLUNGSPLAN
7. THERAPIEZIELE""",
            "treatment_plan": """Du bist ein erfahrener Physiotherapeut. Erstelle einen strukturierten physiotherapeutischen Behandlungsplan.

Struktur:
1. THERAPIEZIELE
2. MANUELLE THERAPIE
3. BEWEGUNGSTHERAPIE
4. ÜBUNGSPROGRAMM
5. HILFSMITTEL
6. ZEITRAHMEN
7. NÄCHSTE TERMINE""",
        }

    def _strip_pii(self, text: str) -> str:
        """Strip personally identifiable information (PII) from German medical text."""
        if not settings.enable_pii_stripping:
            return text

        # German PII patterns — conservative & context-aware
        pii_patterns = [
            # Names (First + Last with capital initials)
            (r"\b[A-ZÄÖÜ][a-zäöüß]+ [A-ZÄÖÜ][a-zäöüß]+\b", "[NAME]"),
            # Dates: DD.MM.YYYY or DD.MM.YY
            (r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b", "[DATUM]"),
            # Postal codes (PLZ)
            (r"\b\d{5}\b", "[PLZ]"),
            # Email addresses
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]"),
            # Phone numbers (+49 or local formats)
            (r"\b(?:\+49|0)\s?\d{2,4}[\s/-]?\d{3,4}[\s/-]?\d{3,4}\b", "[TELEFON]"),
            # German Personal ID or Insurance number (alphanumeric + digits)
            (r"\b[A-Z]{1,2}\d{8,10}\b", "[ID]"),
            # IBAN numbers
            (r"\bDE\d{20}\b", "[IBAN]"),
            # Street addresses (e.g., Musterstraße 12)
            (r"\b[A-ZÄÖÜ][a-zäöüß]+straße \d{1,3}\b", "[ADRESSE]"),
        ]

        stripped_text = text
        for pattern, replacement in pii_patterns:
            stripped_text = re.sub(pattern, replacement, stripped_text)

        return stripped_text

    @cached("llm_mistral", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("llm_mistral", "mistral-7b")
    async def _call_mistral(
        self, messages: List[Dict[str, str]], max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Call Mistral 7B API."""
        try:

            # payload = {
            #     "model": ModelConfig.MISTRAL_MODEL,
            #     "messages": messages,
            #     "temperature": ModelConfig.TEMPERATURE,
            #     "top_p": ModelConfig.TOP_P,
            #     "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
            #     "stream": False
            # }

            # response = await self.mistral_client.post(
            #     settings.mistral_endpoint,
            #     json=payload
            # )
            # response.raise_for_status()

            # result = response.json()

            # return {
            #     "content": result["choices"][0]["message"]["content"],
            #     "model": "mistral-7b",
            #     "provider": "mistral",
            #     "usage": result.get("usage", {}),
            #     "fallback_used": False
            # }

            payload = {
                "model": ModelConfig.MISTRAL_MODEL,
                "messages": messages,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P,
                "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
                "stream": False,
            }

            # payload = {
            #     "model": ModelConfig.MISTRAL_MODEL,
            #     "messages": [
            #         {"role": "user", "content": "Hello"}
            #     ],
            #     "temperature": 0.0,
            #     "top_p": 1.0,
            #     "max_tokens": 32,
            #     "stream": False
            # }

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(settings.request_timeout),
                headers={"Authorization": f"Bearer {settings.mistral_api_key}"},
            ) as client:
                response = await client.post(settings.mistral_endpoint, json=payload)
                response.raise_for_status()
                result = response.json()
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "model": "mistral-7b",
                    "provider": "mistral",
                    "usage": result.get("usage", {}),
                    "fallback_used": False,
                }

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Mistral API rate limited, triggering fallback")
                raise Exception("Rate limited - triggering fallback")
            elif e.response.status_code >= 500:
                logger.warning(
                    f"Mistral API error {e.response.status_code}, triggering fallback"
                )
                raise Exception("API error - triggering fallback")
            else:
                logger.error(f"Mistral API error: {e}")
                raise
        except Exception as e:
            logger.error(f"Mistral API call failed: {e}")
            raise

    @cached("llm_openrouter", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("llm_openrouter", "phi-3-mini")
    async def _call_openrouter_fallback(
        self, messages: List[Dict[str, str]], max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Call OpenRouter fallback model."""
        try:
            payload = {
                "model": settings.fallback_llm_model,
                "messages": messages,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P,
                "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
                "stream": False,
            }

            response = await self.openrouter_client.post(
                settings.openrouter_endpoint, json=payload
            )
            response.raise_for_status()

            result = response.json()

            return {
                "content": result["choices"][0]["message"]["content"],
                "model": settings.fallback_llm_model,
                "provider": "openrouter",
                "usage": result.get("usage", {}),
                "fallback_used": True,
            }

        except Exception as e:
            logger.error(f"OpenRouter fallback failed: {e}")
            raise

    @cached("llm_together", ttl=1800)  # Cache for 30 minutes
    @monitor_latency("llm_together", "gemini-flash")
    async def _call_together_fallback(
        self, messages: List[Dict[str, str]], max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Call Together AI fallback model."""
        try:
            payload = {
                "model": "google/gemini-flash-1.5",
                "messages": messages,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P,
                "max_tokens": min(max_tokens, ModelConfig.MAX_TOKENS_LLM),
                "stream": False,
            }

            response = await self.together_client.post(
                "https://api.together.xyz/v1/chat/completions", json=payload
            )
            response.raise_for_status()

            result = response.json()

            return {
                "content": result["choices"][0]["message"]["content"],
                "model": "gemini-flash-1.5",
                "provider": "together",
                "usage": result.get("usage", {}),
                "fallback_used": True,
            }

        except Exception as e:
            logger.error(f"Together AI fallback failed: {e}")
            raise

    async def generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        use_fallback: bool = False,
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
                        logger.warning(
                            f"Primary LLM latency {latency_ms:.1f}ms exceeds threshold {settings.llm_summary_threshold}ms, trying fallback"
                        )
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
                        logger.warning(
                            f"OpenRouter LLM latency {latency_ms:.1f}ms exceeds threshold {settings.llm_summary_threshold}ms, trying final fallback"
                        )
                        use_fallback = True
                    else:
                        result["threshold_exceeded"] = False
                        return result

                except Exception as e:
                    logger.warning(
                        f"OpenRouter fallback failed, trying final fallback: {str(e)}"
                    )

            # Try Together AI final fallback
            if self.together_client:
                try:
                    result = await self._call_together(messages, max_tokens)
                    latency_ms = (time.time() - start_time) * 1000

                    if latency_ms > settings.llm_summary_threshold:
                        logger.warning(
                            f"Together AI LLM latency {latency_ms:.1f}ms exceeds threshold {settings.llm_summary_threshold}ms"
                        )
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

    async def clean_transcript(
        self,
        transcripts: list[str],
        task_type: str = "physio_text_cleaner",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Clean and standardize a physiotherapy transcript using Mistral only.

        Args:
            transcript: Raw or unstructured speech-to-text transcript
            task_type: Task type to select the correct system prompt
            user_id: Optional user ID for audit logging

        Returns:
            Dict with cleaned and standardized physiotherapy transcript
        """
        try:
            # Step 1: Remove personal identifiable information (PII)
            clean_transcripts = []
            for transcript in transcripts:
                clean_transcript = self._strip_pii(transcript)
                clean_transcripts.append(clean_transcript)

            combined_text = " ".join(clean_transcripts)

            # Step 2: Load system prompt (use physio_text_cleaner as default)
            system_prompt = self._system_prompts.get(
                task_type, self._system_prompts["physio_text_cleaner"]
            )

            # Step 3: Build user instruction (no entity context)
            user_message = f"""
            Hier sind mehrere unklare oder fehlerhafte Teiltranskripte:

            {combined_text}

            Bitte bereinige und fasse sie zu einem korrekten physiotherapeutischen Therapietext zusammen.
            """

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Step 4: Call Mistral (primary and only model)
            result = await self._call_mistral(messages)

            # Step 5: Log model interaction (optional audit)
            self._log_llm_interaction(
                "mistral-7b", len(user_message), len(result.get("content", "")), user_id
            )

            return result
        except Exception as e:
            logger.error(f"Transcript cleaning failed: {e}")
            raise Exception(f"Transcript cleaning failed: {e}")

    async def generate_clinical_summary(
        self,
        transcript: str,
        entities: List[Dict[str, Any]],
        task_type: str = "intake_summary",
        user_id: Optional[str] = None,
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
            # clean_transcript = self._strip_pii(transcript)

            # Prepare messages
            system_prompt = self._system_prompts.get(
                task_type, self._system_prompts["intake_summary"]
            )

            # Format entities for context
            entity_context = self._format_entities_for_llm(entities)

            user_message = f"""Transkript des Patientengesprächs:

{transcript}

Extrahierte medizinische Entitäten:
{entity_context}

Bitte erstelle eine strukturierte klinische Zusammenfassung basierend auf dem obigen Transkript und den extrahierten Entitäten."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Try Mistral first (unless fallback already triggered)
            if not self._fallback_triggered:
                try:
                    result = await self._call_mistral(messages)
                    self._log_llm_interaction(
                        "mistral-7b", len(user_message), len(result["content"]), user_id
                    )
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
                self._log_llm_interaction(
                    settings.fallback_llm_model,
                    len(user_message),
                    len(result["content"]),
                    user_id,
                )
                return result
            except Exception as e:
                logger.warning(f"OpenRouter fallback failed: {e}")

            # Try Together AI as last resort
            try:
                result = await self._call_together_fallback(messages)
                self._log_llm_interaction(
                    "gemini-flash-1.5",
                    len(user_message),
                    len(result["content"]),
                    user_id,
                )
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
            if entity.get("icd_code"):
                line += f" [ICD: {entity['icd_code']}]"
            if entity.get("category"):
                line += f" [Kategorie: {entity['category']}]"
            formatted_entities.append(line)

        return "\n".join(formatted_entities)

    def _log_llm_interaction(
        self,
        model: str,
        prompt_length: int,
        response_length: int,
        user_id: Optional[str],
    ):
        """Log LLM interaction for compliance."""
        from src.utils.logging import get_compliance_logger

        compliance_logger = get_compliance_logger()

        compliance_logger.log_llm_interaction(
            request_id=f"llm_{int(time.time())}",
            model=model,
            prompt_length=prompt_length,
            response_length=response_length,
            user_id=user_id or "unknown",
            pii_stripped=settings.enable_pii_stripping,
        )

    async def generate_structured_notes(
        self, clinical_summary: str, format_type: str = "json"
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
            system_prompt = """Du bist ein erfahrener Physiotherapeut. Konvertiere die physiotherapeutische Zusammenfassung in strukturierte Notizen.

WICHTIGE REGELN:
- Verwende JSON-Format
- Strukturiere alle Informationen hierarchisch
- Verwende deutsche physiotherapeutische Terminologie
- Stelle sicher, dass alle Felder korrekt ausgefüllt sind

JSON-Struktur:
{
  "hauptbeschwerden": [],
  "schmerzanalyse": {},
  "bewegungseinschraenkungen": [],
  "funktionelle_einschraenkungen": [],
  "medizinische_vorgeschichte": [],
  "medikamente": [],
  "allergien": [],
  "befunde": {},
  "therapieziele": [],
  "behandlungsplan": {},
  "naechste_schritte": []
}"""

            user_message = f"""Konvertiere diese physiotherapeutische Zusammenfassung in strukturierte JSON-Notizen:

{clinical_summary}"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Use fallback model for structured output (more reliable)
            try:
                result = await self._call_openrouter_fallback(messages, max_tokens=1024)

                # Try to parse JSON from response
                content = result["content"]
                json_start = content.find("{")
                json_end = content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    structured_notes = json.loads(json_str)
                else:
                    # Fallback to simple structure
                    structured_notes = {"raw_summary": content, "structured": False}

                return {
                    "structured_notes": structured_notes,
                    "format": format_type,
                    "model": result["model"],
                    "fallback_used": result["fallback_used"],
                }

            except Exception as e:
                logger.error(f"Structured notes generation failed: {e}")
                return {
                    "structured_notes": {
                        "raw_summary": clinical_summary,
                        "structured": False,
                    },
                    "format": format_type,
                    "error": str(e),
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
            "timestamp": time.time(),
        }
        return health_status
        # Test Mistral
        try:
            start_time = time.time()
            test_messages = [
                {
                    "role": "system",
                    "content": "Du bist ein erfahrener Physiotherapeut.",
                },
                {"role": "user", "content": "Test"},
            ]
            await self._call_mistral(test_messages, max_tokens=10)
            mistral_latency = (time.time() - start_time) * 1000
            health_status["providers"]["mistral"] = {
                "status": "healthy",
                "latency_ms": mistral_latency,
            }
        except Exception as e:
            health_status["providers"]["mistral"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Test OpenRouter fallback
        try:
            start_time = time.time()
            test_messages = [
                {
                    "role": "system",
                    "content": "Du bist ein erfahrener Physiotherapeut.",
                },
                {"role": "user", "content": "Test"},
            ]
            await self._call_openrouter_fallback(test_messages, max_tokens=10)
            openrouter_latency = (time.time() - start_time) * 1000
            health_status["providers"]["openrouter"] = {
                "status": "healthy",
                "latency_ms": openrouter_latency,
            }
        except Exception as e:
            health_status["providers"]["openrouter"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Determine overall status
        if not any(
            p["status"] == "healthy" for p in health_status["providers"].values()
        ):
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
                "together": "available" if self.together_client else "unavailable",
            },
        }

        # Add cache statistics if caching is enabled
        if settings.enable_caching:
            stats["cache_stats"] = get_cache_stats()

        return stats


# Global LLM service instance
llm_service = LLMService()
