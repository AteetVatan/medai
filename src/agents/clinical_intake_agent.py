"""
Clinical Intake Agent for medAI MVP.
Orchestrates the complete pipeline: STT → NER → LLM → Translation → Storage.
Uses Agno framework for agent-based architecture.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from agno import Agent, Tool, Message
from agno.tools import ToolResult

from ..services.stt_service import stt_service
from ..services.ner_service import ner_service
from ..services.llm_service import llm_service
from ..services.translation_service import translation_service
from ..services.storage_service import storage_service
from ..utils.config import settings, LatencyConfig
from ..utils.logging import get_logger, get_latency_logger, RequestContext

logger = get_logger(__name__)
latency_logger = get_latency_logger()


@dataclass
class ClinicalIntakeResult:
    """Result of clinical intake processing."""
    encounter_id: str
    audio_record_id: str
    transcription: str
    entities: List[Dict[str, Any]]
    clinical_summary: str
    structured_notes: Dict[str, Any]
    translated_notes: Optional[Dict[str, Any]]
    processing_time_ms: float
    success: bool
    errors: List[str]


class STTTool(Tool):
    """Tool for speech-to-text processing."""
    
    def __init__(self):
        super().__init__(
            name="speech_to_text",
            description="Convert audio to text using Whisper with fallback support"
        )
    
    async def execute(self, audio_data: bytes, audio_format: str = "webm", **kwargs) -> ToolResult:
        """Execute speech-to-text processing."""
        try:
            start_time = time.time()
            
            result = await stt_service.transcribe_audio(
                audio_data=audio_data,
                audio_format=audio_format,
                duration_seconds=kwargs.get("duration_seconds")
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={
                    "transcription": result["text"],
                    "confidence": result["confidence"],
                    "model": result["model"],
                    "fallback_used": result.get("fallback_used", False),
                    "processing_time_ms": processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"STT tool failed: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


class NERTool(Tool):
    """Tool for medical named entity recognition."""
    
    def __init__(self):
        super().__init__(
            name="medical_ner",
            description="Extract medical entities from text using spaCy and ICD dictionaries"
        )
    
    async def execute(self, text: str, **kwargs) -> ToolResult:
        """Execute medical entity extraction."""
        try:
            start_time = time.time()
            
            entities = await ner_service.extract_entities(text)
            
            # Convert to dict format for serialization
            entity_dicts = []
            for entity in entities:
                entity_dicts.append({
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence,
                    "icd_code": entity.icd_code,
                    "icd_description": entity.icd_description,
                    "normalized_text": entity.normalized_text,
                    "category": entity.category
                })
            
            processing_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={
                    "entities": entity_dicts,
                    "count": len(entity_dicts),
                    "processing_time_ms": processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"NER tool failed: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


class LLMTool(Tool):
    """Tool for LLM-based clinical summarization."""
    
    def __init__(self):
        super().__init__(
            name="clinical_llm",
            description="Generate clinical summaries using Mistral 7B with fallback support"
        )
    
    async def execute(self, transcript: str, entities: List[Dict[str, Any]], task_type: str = "intake_summary", **kwargs) -> ToolResult:
        """Execute clinical summarization."""
        try:
            start_time = time.time()
            
            # Generate clinical summary
            summary_result = await llm_service.generate_clinical_summary(
                transcript=transcript,
                entities=entities,
                task_type=task_type,
                user_id=kwargs.get("user_id")
            )
            
            # Generate structured notes
            structured_result = await llm_service.generate_structured_notes(
                clinical_summary=summary_result["content"]
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={
                    "clinical_summary": summary_result["content"],
                    "structured_notes": structured_result["structured_notes"],
                    "model": summary_result["model"],
                    "fallback_used": summary_result.get("fallback_used", False),
                    "processing_time_ms": processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"LLM tool failed: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


class TranslationTool(Tool):
    """Tool for clinical notes translation."""
    
    def __init__(self):
        super().__init__(
            name="clinical_translation",
            description="Translate clinical notes using NLLB with DeepL fallback"
        )
    
    async def execute(self, clinical_notes: Dict[str, Any], target_lang: str = "en", **kwargs) -> ToolResult:
        """Execute clinical notes translation."""
        try:
            start_time = time.time()
            
            result = await translation_service.translate_clinical_notes(
                clinical_notes=clinical_notes,
                target_lang=target_lang
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={
                    "translated_notes": result["translated_notes"],
                    "source_language": result["source_language"],
                    "target_language": result["target_language"],
                    "translation_applied": result["translation_applied"],
                    "processing_time_ms": processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Translation tool failed: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


class StorageTool(Tool):
    """Tool for data storage operations."""
    
    def __init__(self):
        super().__init__(
            name="clinical_storage",
            description="Store clinical data in Supabase with audit logging"
        )
    
    async def execute(self, operation: str, data: Dict[str, Any], **kwargs) -> ToolResult:
        """Execute storage operations."""
        try:
            start_time = time.time()
            
            if operation == "save_audio":
                result = await storage_service.save_audio_record(
                    encounter_id=data["encounter_id"],
                    organization_id=data["organization_id"],
                    audio_data=data["audio_data"],
                    audio_metadata=data["audio_metadata"],
                    user_id=data["user_id"]
                )
            elif operation == "save_entities":
                result = await storage_service.save_medical_entities(
                    encounter_id=data["encounter_id"],
                    audio_record_id=data["audio_record_id"],
                    organization_id=data["organization_id"],
                    entities=data["entities"],
                    user_id=data["user_id"]
                )
            elif operation == "save_notes":
                result = await storage_service.save_clinical_notes(
                    encounter_id=data["encounter_id"],
                    organization_id=data["organization_id"],
                    notes_data=data["notes_data"],
                    user_id=data["user_id"]
                )
            else:
                raise ValueError(f"Unknown storage operation: {operation}")
            
            processing_time = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data={
                    "result": result,
                    "operation": operation,
                    "processing_time_ms": processing_time
                }
            )
            
        except Exception as e:
            logger.error(f"Storage tool failed: {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )


class ClinicalIntakeAgent(Agent):
    """Main clinical intake agent orchestrating the complete pipeline."""
    
    def __init__(self):
        super().__init__(
            name="ClinicalIntakeAgent",
            description="Orchestrates clinical intake processing pipeline"
        )
        
        # Initialize tools
        self.stt_tool = STTTool()
        self.ner_tool = NERTool()
        self.llm_tool = LLMTool()
        self.translation_tool = TranslationTool()
        self.storage_tool = StorageTool()
        
        # Add tools to agent
        self.add_tool(self.stt_tool)
        self.add_tool(self.ner_tool)
        self.add_tool(self.llm_tool)
        self.add_tool(self.translation_tool)
        self.add_tool(self.storage_tool)
    
    async def process_clinical_intake(
        self,
        audio_data: bytes,
        encounter_id: str,
        organization_id: str,
        user_id: str,
        audio_format: str = "webm",
        duration_seconds: Optional[float] = None,
        translate_to: Optional[str] = None,
        task_type: str = "intake_summary"
    ) -> ClinicalIntakeResult:
        """
        Process complete clinical intake pipeline.
        
        Args:
            audio_data: Raw audio bytes
            encounter_id: Clinical encounter ID
            organization_id: Organization ID
            user_id: User ID for audit logging
            audio_format: Audio format (webm, wav, etc.)
            duration_seconds: Audio duration
            translate_to: Target language for translation
            task_type: Type of clinical task
            
        Returns:
            ClinicalIntakeResult with complete processing results
        """
        start_time = time.time()
        errors = []
        
        try:
            with RequestContext(request_id=f"intake_{encounter_id}", user_id=user_id):
                logger.info(f"Starting clinical intake processing for encounter {encounter_id}")
                
                # Step 1: Speech-to-Text
                logger.info("Step 1: Speech-to-Text processing")
                stt_result = await self.stt_tool.execute(
                    audio_data=audio_data,
                    audio_format=audio_format,
                    duration_seconds=duration_seconds
                )
                
                if not stt_result.success:
                    raise Exception(f"STT failed: {stt_result.error}")
                
                transcription = stt_result.data["transcription"]
                logger.info(f"Transcription completed: {len(transcription)} characters")
                
                # Step 2: Medical Entity Recognition
                logger.info("Step 2: Medical entity extraction")
                ner_result = await self.ner_tool.execute(text=transcription)
                
                if not ner_result.success:
                    logger.warning(f"NER failed: {ner_result.error}")
                    entities = []
                    errors.append(f"NER failed: {ner_result.error}")
                else:
                    entities = ner_result.data["entities"]
                    logger.info(f"Extracted {len(entities)} medical entities")
                
                # Step 3: LLM Clinical Summarization
                logger.info("Step 3: Clinical summarization")
                llm_result = await self.llm_tool.execute(
                    transcript=transcription,
                    entities=entities,
                    task_type=task_type,
                    user_id=user_id
                )
                
                if not llm_result.success:
                    raise Exception(f"LLM processing failed: {llm_result.error}")
                
                clinical_summary = llm_result.data["clinical_summary"]
                structured_notes = llm_result.data["structured_notes"]
                logger.info("Clinical summarization completed")
                
                # Step 4: Translation (if requested)
                translated_notes = None
                if translate_to and translate_to != "de":
                    logger.info(f"Step 4: Translation to {translate_to}")
                    translation_result = await self.translation_tool.execute(
                        clinical_notes=structured_notes,
                        target_lang=translate_to
                    )
                    
                    if translation_result.success:
                        translated_notes = translation_result.data["translated_notes"]
                        logger.info("Translation completed")
                    else:
                        logger.warning(f"Translation failed: {translation_result.error}")
                        errors.append(f"Translation failed: {translation_result.error}")
                
                # Step 5: Storage Operations
                logger.info("Step 5: Storing results")
                
                # Save audio record
                audio_metadata = {
                    "duration_seconds": duration_seconds,
                    "format": audio_format,
                    "transcription_text": transcription,
                    "transcription_confidence": stt_result.data["confidence"],
                    "processing_status": "completed"
                }
                
                audio_storage_result = await self.storage_tool.execute(
                    operation="save_audio",
                    data={
                        "encounter_id": encounter_id,
                        "organization_id": organization_id,
                        "audio_data": audio_data,
                        "audio_metadata": audio_metadata,
                        "user_id": user_id
                    }
                )
                
                if not audio_storage_result.success:
                    raise Exception(f"Audio storage failed: {audio_storage_result.error}")
                
                audio_record_id = audio_storage_result.data["result"]["id"]
                logger.info(f"Audio record saved: {audio_record_id}")
                
                # Save medical entities
                if entities:
                    entities_storage_result = await self.storage_tool.execute(
                        operation="save_entities",
                        data={
                            "encounter_id": encounter_id,
                            "audio_record_id": audio_record_id,
                            "organization_id": organization_id,
                            "entities": entities,
                            "user_id": user_id
                        }
                    )
                    
                    if not entities_storage_result.success:
                        logger.warning(f"Entities storage failed: {entities_storage_result.error}")
                        errors.append(f"Entities storage failed: {entities_storage_result.error}")
                
                # Save clinical notes
                notes_data = {
                    "note_type": task_type,
                    "structured_notes": structured_notes,
                    "raw_text": clinical_summary,
                    "language": "de",
                    "translated_content": translated_notes,
                    "translated_language": translate_to if translated_notes else None,
                    "model_used": llm_result.data["model"],
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "confidence_score": stt_result.data["confidence"]
                }
                
                notes_storage_result = await self.storage_tool.execute(
                    operation="save_notes",
                    data={
                        "encounter_id": encounter_id,
                        "organization_id": organization_id,
                        "notes_data": notes_data,
                        "user_id": user_id
                    }
                )
                
                if not notes_storage_result.success:
                    logger.warning(f"Notes storage failed: {notes_storage_result.error}")
                    errors.append(f"Notes storage failed: {notes_storage_result.error}")
                
                # Calculate total processing time
                total_processing_time = (time.time() - start_time) * 1000
                
                logger.info(f"Clinical intake processing completed in {total_processing_time:.2f}ms")
                
                return ClinicalIntakeResult(
                    encounter_id=encounter_id,
                    audio_record_id=audio_record_id,
                    transcription=transcription,
                    entities=entities,
                    clinical_summary=clinical_summary,
                    structured_notes=structured_notes,
                    translated_notes=translated_notes,
                    processing_time_ms=total_processing_time,
                    success=True,
                    errors=errors
                )
                
        except Exception as e:
            total_processing_time = (time.time() - start_time) * 1000
            error_msg = f"Clinical intake processing failed: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return ClinicalIntakeResult(
                encounter_id=encounter_id,
                audio_record_id="",
                transcription="",
                entities=[],
                clinical_summary="",
                structured_notes={},
                translated_notes=None,
                processing_time_ms=total_processing_time,
                success=False,
                errors=errors
            )
    
    async def process_streaming_intake(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        encounter_id: str,
        organization_id: str,
        user_id: str,
        chunk_duration: float = 1.5
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process streaming clinical intake with real-time updates.
        
        Args:
            audio_chunks: Async generator of audio chunks
            encounter_id: Clinical encounter ID
            organization_id: Organization ID
            user_id: User ID for audit logging
            chunk_duration: Duration of each audio chunk
            
        Yields:
            Dict with processing updates and results
        """
        accumulated_audio = b""
        last_transcription = ""
        
        try:
            with RequestContext(request_id=f"streaming_{encounter_id}", user_id=user_id):
                logger.info(f"Starting streaming clinical intake for encounter {encounter_id}")
                
                async for chunk in audio_chunks:
                    accumulated_audio += chunk
                    
                    # Process chunk for partial transcription
                    try:
                        # Convert accumulated audio to WAV
                        from ..services.stt_service import stt_service
                        wav_data = stt_service._convert_webm_to_wav(accumulated_audio)
                        
                        # Get partial transcription
                        stt_result = await self.stt_tool.execute(
                            audio_data=wav_data,
                            audio_format="wav"
                        )
                        
                        if stt_result.success:
                            current_transcription = stt_result.data["transcription"]
                            
                            # Only yield if transcription has changed significantly
                            if current_transcription != last_transcription and len(current_transcription) > len(last_transcription):
                                yield {
                                    "type": "partial_transcription",
                                    "transcription": current_transcription,
                                    "confidence": stt_result.data["confidence"],
                                    "timestamp": time.time()
                                }
                                last_transcription = current_transcription
                    
                    except Exception as e:
                        logger.warning(f"Partial transcription failed: {e}")
                        yield {
                            "type": "error",
                            "error": f"Partial transcription failed: {e}",
                            "timestamp": time.time()
                        }
                
                # Final processing
                if accumulated_audio:
                    logger.info("Processing final accumulated audio")
                    final_result = await self.process_clinical_intake(
                        audio_data=accumulated_audio,
                        encounter_id=encounter_id,
                        organization_id=organization_id,
                        user_id=user_id,
                        audio_format="wav"
                    )
                    
                    yield {
                        "type": "final_result",
                        "result": final_result,
                        "timestamp": time.time()
                    }
                
        except Exception as e:
            logger.error(f"Streaming intake failed: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def warm_up_services(self) -> None:
        """Warm up all services for optimal performance."""
        try:
            logger.info("Warming up clinical intake services...")
            
            # Warm up STT service models
            if hasattr(stt_service, 'warm_up_models'):
                await stt_service.warm_up_models()
            
            logger.info("All services warmed up successfully")
            
        except Exception as e:
            logger.error(f"Failed to warm up services: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent and all tools health."""
        health_status = {
            "agent": "ClinicalIntakeAgent",
            "status": "healthy",
            "tools": {},
            "timestamp": time.time()
        }
        
        # Check each tool
        tools = [
            ("stt", stt_service),
            ("ner", ner_service),
            ("llm", llm_service),
            ("translation", translation_service),
            ("storage", storage_service)
        ]
        
        for tool_name, service in tools:
            try:
                tool_health = await service.health_check()
                health_status["tools"][tool_name] = tool_health
            except Exception as e:
                health_status["tools"][tool_name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Determine overall status
        if not all(tool.get("status") == "healthy" for tool in health_status["tools"].values()):
            health_status["status"] = "degraded"
        
        return health_status


# Global clinical intake agent instance
clinical_intake_agent = ClinicalIntakeAgent()
