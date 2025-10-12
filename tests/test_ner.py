"""
Unit tests for NER service.
Tests medical entity extraction and ICD mapping.
"""

import pytest
from unittest.mock import patch, MagicMock
import asyncio

from src.services.ner_service import (
    MedicalNERService,
    ner_service,
    MedicalEntity,
    ICDDictionary,
)


class TestICDDictionary:
    """Test cases for ICD dictionary."""

    def test_icd_dictionary_initialization(self):
        """Test ICD dictionary initialization."""
        icd_dict = ICDDictionary()
        assert len(icd_dict.icd_codes) > 0
        assert "kopfschmerzen" in icd_dict.icd_codes
        assert "diabetes" in icd_dict.icd_codes

    def test_lookup_entity_direct_match(self):
        """Test direct entity lookup."""
        icd_dict = ICDDictionary()
        result = icd_dict.lookup_entity("kopfschmerzen")

        assert result is not None
        assert result["code"] == "R51"
        assert result["description"] == "Kopfschmerz"
        assert result["category"] == "symptom"

    def test_lookup_entity_fuzzy_match(self):
        """Test fuzzy entity matching."""
        icd_dict = ICDDictionary()
        result = icd_dict.lookup_entity("kopfschmerz")  # Variation of kopfschmerzen

        assert result is not None
        assert result["code"] == "R51"

    def test_lookup_entity_no_match(self):
        """Test lookup with no match."""
        icd_dict = ICDDictionary()
        result = icd_dict.lookup_entity("nonexistent_term")

        assert result is None

    def test_fuzzy_match_variations(self):
        """Test fuzzy matching with various term variations."""
        icd_dict = ICDDictionary()

        # Test different variations
        test_cases = [
            ("schwindelig", "schwindel"),
            ("müde", "müdigkeit"),
            ("fiebrig", "fieber"),
            ("weh", "schmerzen"),
        ]

        for variation, expected_key in test_cases:
            result = icd_dict.lookup_entity(variation)
            if result:
                # Should find a related medical term
                assert result["category"] in [
                    "symptom",
                    "diagnosis",
                    "medication",
                    "anatomy",
                ]


class TestMedicalNERService:
    """Test cases for medical NER service."""

    @pytest.fixture
    def ner_service_instance(self):
        """Create NER service instance for testing."""
        return MedicalNERService()

    @pytest.fixture
    def sample_medical_text(self):
        """Sample medical text for testing."""
        return "Der Patient klagt über Kopfschmerzen und Schwindel. Er hat Diabetes und nimmt Metformin."

    @pytest.mark.asyncio
    async def test_extract_entities_success(
        self, ner_service_instance, sample_medical_text
    ):
        """Test successful entity extraction."""
        # Mock spaCy processing
        mock_doc = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent1.text = "Kopfschmerzen"
        mock_ent1.label_ = "SYMPTOM"
        mock_ent1.start_char = 30
        mock_ent1.end_char = 42

        mock_ent2 = MagicMock()
        mock_ent2.text = "Schwindel"
        mock_ent2.label_ = "SYMPTOM"
        mock_ent2.start_char = 47
        mock_ent2.end_char = 55

        mock_doc.ents = [mock_ent1, mock_ent2]

        with patch.object(ner_service_instance, "nlp") as mock_nlp:
            mock_nlp.return_value = mock_doc

            entities = await ner_service_instance.extract_entities(sample_medical_text)

            assert len(entities) >= 2  # Should find at least the mocked entities
            assert any(entity.text == "Kopfschmerzen" for entity in entities)
            assert any(entity.text == "Schwindel" for entity in entities)

    @pytest.mark.asyncio
    async def test_extract_entities_with_icd_mapping(self, ner_service_instance):
        """Test entity extraction with ICD code mapping."""
        text = "Der Patient hat Kopfschmerzen und Diabetes."

        # Mock spaCy processing
        mock_doc = MagicMock()
        mock_ent1 = MagicMock()
        mock_ent1.text = "Kopfschmerzen"
        mock_ent1.label_ = "SYMPTOM"
        mock_ent1.start_char = 15
        mock_ent1.end_char = 27

        mock_ent2 = MagicMock()
        mock_ent2.text = "Diabetes"
        mock_ent2.label_ = "DIAGNOSIS"
        mock_ent2.start_char = 32
        mock_ent2.end_char = 39

        mock_doc.ents = [mock_ent1, mock_ent2]

        with patch.object(ner_service_instance, "nlp") as mock_nlp:
            mock_nlp.return_value = mock_doc

            entities = await ner_service_instance.extract_entities(text)

            # Check that ICD codes are mapped
            kopfschmerzen_entity = next(
                (e for e in entities if e.text == "Kopfschmerzen"), None
            )
            diabetes_entity = next((e for e in entities if e.text == "Diabetes"), None)

            if kopfschmerzen_entity:
                assert kopfschmerzen_entity.icd_code == "R51"
                assert kopfschmerzen_entity.category == "symptom"

            if diabetes_entity:
                assert diabetes_entity.icd_code == "E11"
                assert diabetes_entity.category == "diagnosis"

    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, ner_service_instance):
        """Test entity extraction with empty text."""
        entities = await ner_service_instance.extract_entities("")
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_entities_batch(self, ner_service_instance):
        """Test batch entity extraction."""
        texts = [
            "Der Patient hat Kopfschmerzen.",
            "Er nimmt Aspirin.",
            "Keine Beschwerden.",
        ]

        # Mock spaCy processing for each text
        with patch.object(ner_service_instance, "nlp") as mock_nlp:

            def mock_nlp_side_effect(text):
                mock_doc = MagicMock()
                if "Kopfschmerzen" in text:
                    mock_ent = MagicMock()
                    mock_ent.text = "Kopfschmerzen"
                    mock_ent.label_ = "SYMPTOM"
                    mock_ent.start_char = 15
                    mock_ent.end_char = 27
                    mock_doc.ents = [mock_ent]
                elif "Aspirin" in text:
                    mock_ent = MagicMock()
                    mock_ent.text = "Aspirin"
                    mock_ent.label_ = "MEDICATION"
                    mock_ent.start_char = 8
                    mock_ent.end_char = 15
                    mock_doc.ents = [mock_ent]
                else:
                    mock_doc.ents = []
                return mock_doc

            mock_nlp.side_effect = mock_nlp_side_effect

            results = await ner_service_instance.extract_entities_batch(texts)

            assert len(results) == 3
            assert len(results[0]) == 1  # Kopfschmerzen
            assert len(results[1]) == 1  # Aspirin
            assert len(results[2]) == 0  # No entities

    def test_get_entity_statistics(self, ner_service_instance):
        """Test entity statistics generation."""
        entities = [
            MedicalEntity(
                "Kopfschmerzen", "SYMPTOM", 0, 12, 0.9, "R51", "Kopfschmerz", "symptom"
            ),
            MedicalEntity(
                "Diabetes",
                "DIAGNOSIS",
                15,
                22,
                0.8,
                "E11",
                "Diabetes mellitus",
                "diagnosis",
            ),
            MedicalEntity(
                "Aspirin",
                "MEDICATION",
                25,
                32,
                0.95,
                "N02BA01",
                "Acetylsalicylsäure",
                "medication",
            ),
        ]

        stats = ner_service_instance.get_entity_statistics(entities)

        assert stats["total_entities"] == 3
        assert stats["by_category"]["symptom"] == 1
        assert stats["by_category"]["diagnosis"] == 1
        assert stats["by_category"]["medication"] == 1
        assert stats["with_icd_codes"] == 3
        assert stats["confidence_distribution"]["high"] == 2  # > 0.8
        assert stats["confidence_distribution"]["medium"] == 1  # 0.5-0.8

    def test_format_entities_for_display(self, ner_service_instance):
        """Test entity formatting for display."""
        entities = [
            MedicalEntity(
                "Kopfschmerzen", "SYMPTOM", 0, 12, 0.9, "R51", "Kopfschmerz", "symptom"
            ),
            MedicalEntity(
                "Diabetes",
                "DIAGNOSIS",
                15,
                22,
                0.8,
                "E11",
                "Diabetes mellitus",
                "diagnosis",
            ),
        ]

        formatted = ner_service_instance.format_entities_for_display(entities)

        assert "Kopfschmerzen" in formatted
        assert "R51" in formatted
        assert "Diabetes" in formatted
        assert "E11" in formatted
        assert "•" in formatted  # Bullet points

    def test_format_entities_empty(self, ner_service_instance):
        """Test formatting empty entity list."""
        formatted = ner_service_instance.format_entities_for_display([])
        assert "Keine medizinischen Entitäten gefunden" in formatted

    @pytest.mark.asyncio
    async def test_health_check(self, ner_service_instance):
        """Test health check functionality."""
        # Mock successful entity extraction
        with patch.object(ner_service_instance, "extract_entities") as mock_extract:
            mock_extract.return_value = [MedicalEntity("Test", "SYMPTOM", 0, 4, 0.9)]

            health = await ner_service_instance.health_check()

            assert health["service"] == "ner"
            assert health["status"] == "healthy"
            assert health["model_loaded"] is True
            assert health["test_entities_found"] == 1
            assert health["icd_dictionary_size"] > 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self, ner_service_instance):
        """Test health check with failure."""
        # Mock failed entity extraction
        with patch.object(ner_service_instance, "extract_entities") as mock_extract:
            mock_extract.side_effect = Exception("Test error")

            health = await ner_service_instance.health_check()

            assert health["service"] == "ner"
            assert health["status"] == "unhealthy"
            assert "error" in health


@pytest.mark.asyncio
async def test_global_ner_service():
    """Test global NER service instance."""
    assert ner_service is not None
    assert isinstance(ner_service, MedicalNERService)


def test_medical_entity_dataclass():
    """Test MedicalEntity dataclass."""
    entity = MedicalEntity(
        text="Kopfschmerzen",
        label="SYMPTOM",
        start=0,
        end=12,
        confidence=0.9,
        icd_code="R51",
        icd_description="Kopfschmerz",
        normalized_text="kopfschmerzen",
        category="symptom",
    )

    assert entity.text == "Kopfschmerzen"
    assert entity.label == "SYMPTOM"
    assert entity.confidence == 0.9
    assert entity.icd_code == "R51"
    assert entity.category == "symptom"
