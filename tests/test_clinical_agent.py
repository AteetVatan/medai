"""
Unit tests for Clinical Intake Agent.
Tests the complete pipeline orchestration.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio
import io

from src.agents.clinical_intake_agent import (
    ClinicalIntakeAgent, 
    clinical_intake_agent,
    ClinicalIntakeResult,
    STTTool,
    NERTool,
    LLMTool,
    TranslationTool,
    StorageTool
)
from src.services.stt_service import stt_service
from src.services.ner_service import ner_service
from src.services.llm_service import llm_service
from src.services.translation_service import translation_service
from src.services.storage_service import storage_service


class TestClinicalIntakeAgent:
    """Test cases for Clinical Intake Agent."""
    
    @pytest.fixture
    def agent_instance(self):
        """Create agent instance for testing."""
        return ClinicalIntakeAgent()
    
    @pytest.fixture
    def sample_audio_data(self):
        """Sample audio data for testing."""
        return b"fake_audio_data"
    
    @pytest.fixture
    def sample_transcript(self):
        """Sample transcript for testing."""
        return "Der Patient klagt über Kopfschmerzen und Schwindel."
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            {"text": "Kopfschmerzen", "label": "SYMPTOM", "icd_code": "R51"},
            {"text": "Schwindel", "label": "SYMPTOM", "icd_code": "R42"}
        ]
    
    @pytest.fixture
    def sample_clinical_summary(self):
        """Sample clinical summary for testing."""
        return "Strukturierte klinische Zusammenfassung mit Hauptbeschwerden und Symptomen."
    
    @pytest.fixture
    def sample_structured_notes(self):
        """Sample structured notes for testing."""
        return {
            "hauptbeschwerden": ["Kopfschmerzen", "Schwindel"],
            "aktuelle_symptome": ["Übelkeit"],
            "medikamente": ["Aspirin"]
        }
    
    def test_agent_initialization(self, agent_instance):
        """Test agent initialization with tools."""
        assert agent_instance.name == "ClinicalIntakeAgent"
        assert len(agent_instance.tools) == 5
        
        tool_names = [tool.name for tool in agent_instance.tools]
        assert "speech_to_text" in tool_names
        assert "medical_ner" in tool_names
        assert "clinical_llm" in tool_names
        assert "clinical_translation" in tool_names
        assert "clinical_storage" in tool_names
    
    @pytest.mark.asyncio
    async def test_process_clinical_intake_success(self, agent_instance, sample_audio_data):
        """Test successful clinical intake processing."""
        # Mock all service responses
        with patch.object(stt_service, 'transcribe_audio') as mock_stt, \
             patch.object(ner_service, 'extract_entities') as mock_ner, \
             patch.object(llm_service, 'generate_clinical_summary') as mock_llm, \
             patch.object(llm_service, 'generate_structured_notes') as mock_structured, \
             patch.object(storage_service, 'save_audio_record') as mock_save_audio, \
             patch.object(storage_service, 'save_medical_entities') as mock_save_entities, \
             patch.object(storage_service, 'save_clinical_notes') as mock_save_notes:
            
            # Configure mocks
            mock_stt.return_value = {
                "text": "Der Patient klagt über Kopfschmerzen.",
                "confidence": 0.95,
                "model": "whisper-hf",
                "fallback_used": False
            }
            
            mock_ner.return_value = [
                MagicMock(text="Kopfschmerzen", label="SYMPTOM", start=15, end=27, 
                         confidence=0.9, icd_code="R51", icd_description="Kopfschmerz",
                         normalized_text="kopfschmerzen", category="symptom")
            ]
            
            mock_llm.return_value = {
                "content": "Klinische Zusammenfassung...",
                "model": "mistral-7b",
                "fallback_used": False
            }
            
            mock_structured.return_value = {
                "structured_notes": {"hauptbeschwerden": ["Kopfschmerzen"]},
                "format": "json",
                "fallback_used": False
            }
            
            mock_save_audio.return_value = {"id": "audio_123"}
            mock_save_entities.return_value = [{"id": "entity_123"}]
            mock_save_notes.return_value = {"id": "notes_123"}
            
            # Process clinical intake
            result = await agent_instance.process_clinical_intake(
                audio_data=sample_audio_data,
                encounter_id="encounter_123",
                organization_id="org_123",
                user_id="user_123",
                audio_format="wav"
            )
            
            # Verify result
            assert isinstance(result, ClinicalIntakeResult)
            assert result.success is True
            assert result.encounter_id == "encounter_123"
            assert result.audio_record_id == "audio_123"
            assert "Kopfschmerzen" in result.transcription
            assert len(result.entities) == 1
            assert "Klinische Zusammenfassung" in result.clinical_summary
            assert result.structured_notes["hauptbeschwerden"] == ["Kopfschmerzen"]
            assert result.processing_time_ms > 0
            assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_process_clinical_intake_stt_failure(self, agent_instance, sample_audio_data):
        """Test clinical intake processing with STT failure."""
        with patch.object(stt_service, 'transcribe_audio') as mock_stt:
            mock_stt.side_effect = Exception("STT service unavailable")
            
            result = await agent_instance.process_clinical_intake(
                audio_data=sample_audio_data,
                encounter_id="encounter_123",
                organization_id="org_123",
                user_id="user_123"
            )
            
            assert result.success is False
            assert "STT service unavailable" in result.errors[0]
            assert result.transcription == ""
    
    @pytest.mark.asyncio
    async def test_process_clinical_intake_ner_failure(self, agent_instance, sample_audio_data):
        """Test clinical intake processing with NER failure."""
        with patch.object(stt_service, 'transcribe_audio') as mock_stt, \
             patch.object(ner_service, 'extract_entities') as mock_ner, \
             patch.object(llm_service, 'generate_clinical_summary') as mock_llm, \
             patch.object(llm_service, 'generate_structured_notes') as mock_structured, \
             patch.object(storage_service, 'save_audio_record') as mock_save_audio, \
             patch.object(storage_service, 'save_clinical_notes') as mock_save_notes:
            
            # Configure mocks
            mock_stt.return_value = {"text": "Test transcript", "confidence": 0.95, "model": "whisper-hf", "fallback_used": False}
            mock_ner.side_effect = Exception("NER service unavailable")
            mock_llm.return_value = {"content": "Summary", "model": "mistral-7b", "fallback_used": False}
            mock_structured.return_value = {"structured_notes": {}, "format": "json", "fallback_used": False}
            mock_save_audio.return_value = {"id": "audio_123"}
            mock_save_notes.return_value = {"id": "notes_123"}
            
            result = await agent_instance.process_clinical_intake(
                audio_data=sample_audio_data,
                encounter_id="encounter_123",
                organization_id="org_123",
                user_id="user_123"
            )
            
            assert result.success is True  # Should continue despite NER failure
            assert "NER service unavailable" in result.errors[0]
            assert len(result.entities) == 0
    
    @pytest.mark.asyncio
    async def test_process_clinical_intake_with_translation(self, agent_instance, sample_audio_data):
        """Test clinical intake processing with translation."""
        with patch.object(stt_service, 'transcribe_audio') as mock_stt, \
             patch.object(ner_service, 'extract_entities') as mock_ner, \
             patch.object(llm_service, 'generate_clinical_summary') as mock_llm, \
             patch.object(llm_service, 'generate_structured_notes') as mock_structured, \
             patch.object(translation_service, 'translate_clinical_notes') as mock_translate, \
             patch.object(storage_service, 'save_audio_record') as mock_save_audio, \
             patch.object(storage_service, 'save_medical_entities') as mock_save_entities, \
             patch.object(storage_service, 'save_clinical_notes') as mock_save_notes:
            
            # Configure mocks
            mock_stt.return_value = {"text": "Test transcript", "confidence": 0.95, "model": "whisper-hf", "fallback_used": False}
            mock_ner.return_value = []
            mock_llm.return_value = {"content": "Summary", "model": "mistral-7b", "fallback_used": False}
            mock_structured.return_value = {"structured_notes": {"hauptbeschwerden": ["Test"]}, "format": "json", "fallback_used": False}
            mock_translate.return_value = {
                "translated_notes": {"main_complaints": ["Test"]},
                "source_language": "de",
                "target_language": "en",
                "translation_applied": True
            }
            mock_save_audio.return_value = {"id": "audio_123"}
            mock_save_entities.return_value = []
            mock_save_notes.return_value = {"id": "notes_123"}
            
            result = await agent_instance.process_clinical_intake(
                audio_data=sample_audio_data,
                encounter_id="encounter_123",
                organization_id="org_123",
                user_id="user_123",
                translate_to="en"
            )
            
            assert result.success is True
            assert result.translated_notes is not None
            assert result.translated_notes["main_complaints"] == ["Test"]
    
    @pytest.mark.asyncio
    async def test_process_streaming_intake(self, agent_instance):
        """Test streaming clinical intake processing."""
        # Create async generator for audio chunks
        async def audio_chunks():
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"
        
        with patch.object(agent_instance, 'process_clinical_intake') as mock_process:
            mock_process.return_value = ClinicalIntakeResult(
                encounter_id="encounter_123",
                audio_record_id="audio_123",
                transcription="Final transcript",
                entities=[],
                clinical_summary="Summary",
                structured_notes={},
                translated_notes=None,
                processing_time_ms=1000,
                success=True,
                errors=[]
            )
            
            results = []
            async for result in agent_instance.process_streaming_intake(
                audio_chunks=audio_chunks(),
                encounter_id="encounter_123",
                organization_id="org_123",
                user_id="user_123"
            ):
                results.append(result)
            
            # Should have at least one result
            assert len(results) > 0
            # Should have final result
            final_results = [r for r in results if r["type"] == "final_result"]
            assert len(final_results) == 1


class TestSTTTool:
    """Test cases for STT Tool."""
    
    @pytest.fixture
    def stt_tool(self):
        """Create STT tool instance."""
        return STTTool()
    
    @pytest.mark.asyncio
    async def test_stt_tool_execute_success(self, stt_tool):
        """Test successful STT tool execution."""
        with patch.object(stt_service, 'transcribe_audio') as mock_transcribe:
            mock_transcribe.return_value = {
                "text": "Test transcription",
                "confidence": 0.95,
                "model": "whisper-hf",
                "fallback_used": False
            }
            
            result = await stt_tool.execute(
                audio_data=b"fake_audio",
                audio_format="wav",
                duration_seconds=2.0
            )
            
            assert result.success is True
            assert result.data["transcription"] == "Test transcription"
            assert result.data["confidence"] == 0.95
            assert result.data["model"] == "whisper-hf"
            assert result.data["fallback_used"] is False
    
    @pytest.mark.asyncio
    async def test_stt_tool_execute_failure(self, stt_tool):
        """Test STT tool execution failure."""
        with patch.object(stt_service, 'transcribe_audio') as mock_transcribe:
            mock_transcribe.side_effect = Exception("STT service error")
            
            result = await stt_tool.execute(
                audio_data=b"fake_audio",
                audio_format="wav"
            )
            
            assert result.success is False
            assert "STT service error" in result.error


class TestNERTool:
    """Test cases for NER Tool."""
    
    @pytest.fixture
    def ner_tool(self):
        """Create NER tool instance."""
        return NERTool()
    
    @pytest.mark.asyncio
    async def test_ner_tool_execute_success(self, ner_tool):
        """Test successful NER tool execution."""
        with patch.object(ner_service, 'extract_entities') as mock_extract:
            mock_entity = MagicMock()
            mock_entity.text = "Kopfschmerzen"
            mock_entity.label = "SYMPTOM"
            mock_entity.start = 0
            mock_entity.end = 12
            mock_entity.confidence = 0.9
            mock_entity.icd_code = "R51"
            mock_entity.icd_description = "Kopfschmerz"
            mock_entity.normalized_text = "kopfschmerzen"
            mock_entity.category = "symptom"
            
            mock_extract.return_value = [mock_entity]
            
            result = await ner_tool.execute(text="Der Patient hat Kopfschmerzen.")
            
            assert result.success is True
            assert len(result.data["entities"]) == 1
            assert result.data["entities"][0]["text"] == "Kopfschmerzen"
            assert result.data["count"] == 1
    
    @pytest.mark.asyncio
    async def test_ner_tool_execute_failure(self, ner_tool):
        """Test NER tool execution failure."""
        with patch.object(ner_service, 'extract_entities') as mock_extract:
            mock_extract.side_effect = Exception("NER service error")
            
            result = await ner_tool.execute(text="Test text")
            
            assert result.success is False
            assert "NER service error" in result.error


class TestLLMTool:
    """Test cases for LLM Tool."""
    
    @pytest.fixture
    def llm_tool(self):
        """Create LLM tool instance."""
        return LLMTool()
    
    @pytest.mark.asyncio
    async def test_llm_tool_execute_success(self, llm_tool):
        """Test successful LLM tool execution."""
        with patch.object(llm_service, 'generate_clinical_summary') as mock_summary, \
             patch.object(llm_service, 'generate_structured_notes') as mock_structured:
            
            mock_summary.return_value = {
                "content": "Clinical summary",
                "model": "mistral-7b",
                "fallback_used": False
            }
            
            mock_structured.return_value = {
                "structured_notes": {"hauptbeschwerden": ["Test"]},
                "format": "json",
                "fallback_used": False
            }
            
            result = await llm_tool.execute(
                transcript="Test transcript",
                entities=[{"text": "Test", "label": "SYMPTOM"}],
                task_type="intake_summary",
                user_id="test_user"
            )
            
            assert result.success is True
            assert result.data["clinical_summary"] == "Clinical summary"
            assert result.data["structured_notes"]["hauptbeschwerden"] == ["Test"]
            assert result.data["model"] == "mistral-7b"
            assert result.data["fallback_used"] is False
    
    @pytest.mark.asyncio
    async def test_llm_tool_execute_failure(self, llm_tool):
        """Test LLM tool execution failure."""
        with patch.object(llm_service, 'generate_clinical_summary') as mock_summary:
            mock_summary.side_effect = Exception("LLM service error")
            
            result = await llm_tool.execute(
                transcript="Test transcript",
                entities=[],
                user_id="test_user"
            )
            
            assert result.success is False
            assert "LLM service error" in result.error


class TestTranslationTool:
    """Test cases for Translation Tool."""
    
    @pytest.fixture
    def translation_tool(self):
        """Create translation tool instance."""
        return TranslationTool()
    
    @pytest.mark.asyncio
    async def test_translation_tool_execute_success(self, translation_tool):
        """Test successful translation tool execution."""
        with patch.object(translation_service, 'translate_clinical_notes') as mock_translate:
            mock_translate.return_value = {
                "translated_notes": {"main_complaints": ["Headache"]},
                "source_language": "de",
                "target_language": "en",
                "translation_applied": True
            }
            
            result = await translation_tool.execute(
                clinical_notes={"hauptbeschwerden": ["Kopfschmerzen"]},
                target_lang="en"
            )
            
            assert result.success is True
            assert result.data["translated_notes"]["main_complaints"] == ["Headache"]
            assert result.data["source_language"] == "de"
            assert result.data["target_language"] == "en"
            assert result.data["translation_applied"] is True
    
    @pytest.mark.asyncio
    async def test_translation_tool_execute_failure(self, translation_tool):
        """Test translation tool execution failure."""
        with patch.object(translation_service, 'translate_clinical_notes') as mock_translate:
            mock_translate.side_effect = Exception("Translation service error")
            
            result = await translation_tool.execute(
                clinical_notes={"test": "data"},
                target_lang="en"
            )
            
            assert result.success is False
            assert "Translation service error" in result.error


class TestStorageTool:
    """Test cases for Storage Tool."""
    
    @pytest.fixture
    def storage_tool(self):
        """Create storage tool instance."""
        return StorageTool()
    
    @pytest.mark.asyncio
    async def test_storage_tool_execute_save_audio(self, storage_tool):
        """Test storage tool for saving audio."""
        with patch.object(storage_service, 'save_audio_record') as mock_save:
            mock_save.return_value = {"id": "audio_123"}
            
            result = await storage_tool.execute(
                operation="save_audio",
                data={
                    "encounter_id": "encounter_123",
                    "organization_id": "org_123",
                    "audio_data": b"fake_audio",
                    "audio_metadata": {"duration": 2.0},
                    "user_id": "user_123"
                }
            )
            
            assert result.success is True
            assert result.data["result"]["id"] == "audio_123"
            assert result.data["operation"] == "save_audio"
    
    @pytest.mark.asyncio
    async def test_storage_tool_execute_unknown_operation(self, storage_tool):
        """Test storage tool with unknown operation."""
        result = await storage_tool.execute(
            operation="unknown_operation",
            data={}
        )
        
        assert result.success is False
        assert "Unknown storage operation" in result.error


@pytest.mark.asyncio
async def test_global_clinical_intake_agent():
    """Test global clinical intake agent instance."""
    assert clinical_intake_agent is not None
    assert isinstance(clinical_intake_agent, ClinicalIntakeAgent)


@pytest.mark.asyncio
async def test_clinical_intake_result_dataclass():
    """Test ClinicalIntakeResult dataclass."""
    result = ClinicalIntakeResult(
        encounter_id="test_123",
        audio_record_id="audio_123",
        transcription="Test transcript",
        entities=[],
        clinical_summary="Test summary",
        structured_notes={},
        translated_notes=None,
        processing_time_ms=1000.0,
        success=True,
        errors=[]
    )
    
    assert result.encounter_id == "test_123"
    assert result.success is True
    assert result.processing_time_ms == 1000.0
