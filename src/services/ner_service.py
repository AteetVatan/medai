"""
Medical Named Entity Recognition service for medAI MVP.
Uses spaCy with German clinical models and ICD-10/ICD-11 dictionaries.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

import spacy
from spacy import displacy
from spacy.pipeline import EntityRuler

import httpx

from ..utils.config import settings, ModelConfig
from ..utils.logging import get_logger, get_latency_logger, monitor_latency

logger = get_logger(__name__)
latency_logger = get_latency_logger()


@dataclass
class MedicalEntity:
    """Medical entity extracted from text."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    icd_code: Optional[str] = None
    icd_description: Optional[str] = None
    normalized_text: Optional[str] = None
    category: Optional[str] = None


class ICDDictionary:
    """ICD-10/ICD-11 dictionary for medical entity mapping."""
    
    def __init__(self):
        self.icd_codes = {}
        self._load_icd_dictionary()
    
    def _load_icd_dictionary(self):
        """Load ICD-10/ICD-11 codes and descriptions."""
        # In production, this would load from a comprehensive database
        # For MVP, we'll use a subset of common German medical terms
        self.icd_codes = {
            # Common symptoms
            "kopfschmerzen": {"code": "R51", "description": "Kopfschmerz", "category": "symptom"},
            "schwindel": {"code": "R42", "description": "Schwindel und Gangunsicherheit", "category": "symptom"},
            "müdigkeit": {"code": "R53", "description": "Unwohlsein und Ermüdung", "category": "symptom"},
            "fieber": {"code": "R50", "description": "Fieber", "category": "symptom"},
            "schmerzen": {"code": "R52", "description": "Schmerz", "category": "symptom"},
            "übelkeit": {"code": "R11", "description": "Übelkeit und Erbrechen", "category": "symptom"},
            
            # Common diagnoses
            "hypertension": {"code": "I10", "description": "Essentielle Hypertonie", "category": "diagnosis"},
            "diabetes": {"code": "E11", "description": "Diabetes mellitus, Typ 2", "category": "diagnosis"},
            "depression": {"code": "F32", "description": "Depressive Episode", "category": "diagnosis"},
            "angina": {"code": "I20", "description": "Angina pectoris", "category": "diagnosis"},
            "asthma": {"code": "J45", "description": "Asthma bronchiale", "category": "diagnosis"},
            
            # Body parts/anatomy
            "kopf": {"code": "S01", "description": "Verletzung des Kopfes", "category": "anatomy"},
            "rücken": {"code": "M54", "description": "Rückenschmerzen", "category": "anatomy"},
            "herz": {"code": "I25", "description": "Chronische ischämische Herzkrankheit", "category": "anatomy"},
            "lunge": {"code": "J44", "description": "Sonstige chronische obstruktive Lungenkrankheit", "category": "anatomy"},
            "magen": {"code": "K25", "description": "Ulcus ventriculi", "category": "anatomy"},          
                        
            # Medications
            "aspirin": {"code": "N02BA01", "description": "Acetylsalicylsäure", "category": "medication"},
            "paracetamol": {"code": "N02BE01", "description": "Paracetamol", "category": "medication"},
            "metformin": {"code": "A10BA02", "description": "Metformin", "category": "medication"},
            "lisinopril": {"code": "C09AA03", "description": "Lisinopril", "category": "medication"},
            "ibuprofen": {"code": "M01AE01", "description": "Ibuprofen", "category": "medication"},
            "diclofenac": {"code": "M01AB05", "description": "Diclofenac", "category": "medication"},
            
            # Common rehab diagnoses (M-Kapitel = Muskel-Skelett)
            "rückenschmerzen": {"code": "M54", "description": "Rückenschmerzen/Dorsalgie", "category": "diagnosis"},
            "ischias": {"code": "M54", "description": "Rückenschmerzen/Ischialgie", "category": "diagnosis"},
            "hws-syndrom": {"code": "M53", "description": "Zervikalsyndrom", "category": "diagnosis"},
            "bws-syndrom": {"code": "M53", "description": "Thorakalsyndrom", "category": "diagnosis"},
            "lws-syndrom": {"code": "M53", "description": "Lumbalsyndrom", "category": "diagnosis"},
            "gonarthrose": {"code": "M17", "description": "Gonarthrose (Kniearthrose)", "category": "diagnosis"},
            "coxarthrose": {"code": "M16", "description": "Coxarthrose (Hüftarthrose)", "category": "diagnosis"},
            "tendinitis": {"code": "M77", "description": "Sonstige Enthesiopathien/Tendinitis", "category": "diagnosis"},
            "bandscheibenvorfall": {"code": "M51", "description": "Bandscheibenschäden", "category": "diagnosis"},
        }
    
    def lookup_entity(self, text: str) -> Optional[Dict[str, str]]:
        """Look up entity in ICD dictionary."""
        text_lower = text.lower().strip()
        
        # Direct lookup
        if text_lower in self.icd_codes:
            return self.icd_codes[text_lower]
        
        # Fuzzy matching for common variations
        for key, value in self.icd_codes.items():
            if self._fuzzy_match(text_lower, key):
                return value
        
        return None
    
    def _fuzzy_match(self, text: str, key: str) -> bool:
        """Simple fuzzy matching for medical terms."""
        # Check if key is contained in text or vice versa
        if key in text or text in key:
            return True
        
        # Check for common medical term variations
        variations = {
            "kopfschmerzen": ["kopfschmerz", "kopfweh", "kopfschmerz"],
            "schwindel": ["schwindelig", "schwindelgefühl"],
            "müdigkeit": ["müde", "erschöpfung", "ermüdung"],
            "fieber": ["temperatur", "fiebrig"],
            "schmerzen": ["schmerz", "weh", "wehweh"],
            "übelkeit": ["übel", "brechreiz"],
        }
        
        if key in variations:
            for variation in variations[key]:
                if variation in text:
                    return True
        
        return False


class MedicalNERService:
    """Medical Named Entity Recognition service."""
    
    def __init__(self):       
        self.nlp_base = None           # de_core_news_md
        self.nlp_med = None            # de_GERNERMED (optional)
        self.icd_dict = ICDDictionary()
        self._load_models()
    
    def _load_models(self):
        """Load spaCy models for German medical text processing."""
         # Base German model (tokens/POS/lemma/parser + generic NER)
        try:
            self.nlp_base = spacy.load("de_core_news_md")
            logger.info("Loaded spaCy model: de_core_news_md")
        except OSError as e:
            logger.error("Model 'de_core_news_md' not found. Install with: python -m spacy download de_core_news_md")
            raise

        # Optional: medication NER (GERNERMED). If missing, we continue gracefully.
        try:
            self.nlp_med = spacy.load("de_GERNERMED")
            logger.info("Loaded spaCy model: de_GERNERMED (medication NER)")
        except Exception:
            self.nlp_med = None
            logger.warning("Optional model 'de_GERNERMED' not available. "
                           "Install via pip URL from the official repo to enable German medication NER.")

        
        # Add custom medical entity patterns
        self._add_medical_patterns()
        self._add_physio_ruler()
    
    def _add_medical_patterns(self):
        """Add custom patterns for medical entities."""
        from spacy.matcher import Matcher
        
        matcher = Matcher(self.nlp_base.vocab)
        
        # Patterns for common medical terms
        patterns = [
            # Symptoms
            [{"LOWER": {"IN": ["kopfschmerzen", "kopfschmerz", "kopfweh"]}}],
            [{"LOWER": {"IN": ["schwindel", "schwindelig", "schwindelgefühl"]}}],
            [{"LOWER": {"IN": ["müdigkeit", "müde", "erschöpfung"]}}],
            [{"LOWER": {"IN": ["fieber", "temperatur", "fiebrig"]}}],
            [{"LOWER": {"IN": ["schmerzen", "schmerz", "weh"]}}],
            [{"LOWER": {"IN": ["übelkeit", "übel", "brechreiz"]}}],
            
            # Diagnoses
            [{"LOWER": {"IN": ["hypertension", "hochdruck", "bluthochdruck"]}}],
            [{"LOWER": {"IN": ["diabetes", "zuckerkrankheit", "diabetes mellitus"]}}],
            [{"LOWER": {"IN": ["depression", "depressiv", "deprimiert"]}}],
            [{"LOWER": {"IN": ["angina", "angina pectoris", "herzschmerz"]}}],
            [{"LOWER": {"IN": ["asthma", "asthma bronchiale", "atemnot"]}}],
            
            # Body parts
            [{"LOWER": {"IN": ["kopf", "haupt", "schädel"]}}],
            [{"LOWER": {"IN": ["rücken", "wirbelsäule", "spine"]}}],
            [{"LOWER": {"IN": ["herz", "kardial", "kardio"]}}],
            [{"LOWER": {"IN": ["lunge", "pulmonal", "respiratorisch"]}}],
            [{"LOWER": {"IN": ["magen", "gastro", "gastrointestinal"]}}],
            
            # Medications
            [{"LOWER": {"IN": ["aspirin", "acetylsalicylsäure", "asa"]}}],
            [{"LOWER": {"IN": ["paracetamol", "acetaminophen"]}}],
            [{"LOWER": {"IN": ["metformin", "glucophage"]}}],
            [{"LOWER": {"IN": ["lisinopril", "ace-hemmer"]}}],
        ]
        
        # Add patterns to matcher
        for i, pattern in enumerate(patterns):
            matcher.add(f"MEDICAL_TERM_{i}", [pattern])
        
        # Add matcher to pipeline
        self.nlp_base.add_pipe("matcher", config={"matcher": matcher}, last=True)
        
    def _add_physio_ruler(self):
        """Add an EntityRuler with common physiotherapy/rehab terms & abbreviations."""
        ruler: EntityRuler = self.nlp_base.add_pipe("entity_ruler", name="physio_ruler", config={"overwrite_ents": False})

        # Terms aligned with typical outpatient/inpatient rehab documentation (not OPS codes, but close wording)
        physio_terms = [
            # treatments
            ("Krankengymnastik", "PHYSIO_TREATMENT"),
            ("KG", "PHYSIO_TREATMENT"),
            ("Manuelle Therapie", "PHYSIO_TREATMENT"),
            ("MT", "PHYSIO_TREATMENT"),
            ("Krankengymnastik am Gerät", "PHYSIO_TREATMENT"),
            ("KGG", "PHYSIO_TREATMENT"),
            ("Manuelle Lymphdrainage", "PHYSIO_TREATMENT"),
            ("MLD", "PHYSIO_TREATMENT"),
            ("Atemtherapie", "PHYSIO_TREATMENT"),
            ("Gangschule", "PHYSIO_TREATMENT"),
            ("Gangschulung", "PHYSIO_TREATMENT"),
            ("Triggerpunktbehandlung", "PHYSIO_TREATMENT"),
            ("Fango", "PHYSIO_TREATMENT"),
            ("Wärmetherapie", "PHYSIO_TREATMENT"),
            ("Kryotherapie", "PHYSIO_TREATMENT"),
            ("Elektrotherapie", "PHYSIO_TREATMENT"),
            ("TENS", "PHYSIO_TREATMENT"),
            ("Dehnung", "PHYSIO_TREATMENT"),
            ("Kräftigung", "PHYSIO_TREATMENT"),
            ("Gelenkmobilisation", "PHYSIO_TREATMENT"),
            # anatomy/regions shorthand often seen in notes
            ("HWS", "ANATOMY"),
            ("BWS", "ANATOMY"),
            ("LWS", "ANATOMY"),
            ("ISG", "ANATOMY"),
            ("Schulter", "ANATOMY"),
            ("Knie", "ANATOMY"),
            ("Hüfte", "ANATOMY"),
            ("Sprunggelenk", "ANATOMY"),
            # common complaints/goals
            ("Beweglichkeit", "FUNCTION"),
            ("ROM", "FUNCTION"),
            ("Belastbarkeit", "FUNCTION"),
            ("Gangbild", "FUNCTION"),
            ("Schmerzreduktion", "GOAL"),
            ("Aufbau", "GOAL"),
            ("Stabilisation", "GOAL"),
        ]

        patterns = [{"label": label, "pattern": term} for (term, label) in physio_terms]
        ruler.add_patterns(patterns)
    
   
    
     # ---------- label mapping ----------
    @staticmethod
    def _map_spacy_label(label: str) -> Optional[str]:
        """
        Map spaCy labels to medAI-normalized categories.
        - From base model: PERSON/ORG/GPE/DATE/etc.
        - From GERNERMED: Drug/Strength/Route/Form/Dosage/Frequency/Duration
        - From physio ruler: PHYSIO_TREATMENT/ANATOMY/FUNCTION/GOAL
        """
        base_map = {
            # generic
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "DATE": "DATE",
            "TIME": "TIME",
            "PERCENT": "MEASUREMENT",
            "MONEY": "MEASUREMENT",
            "QUANTITY": "MEASUREMENT",
            "CARDINAL": "NUMBER",
            "ORDINAL": "NUMBER",
            # physio ruler
            "PHYSIO_TREATMENT": "PHYSIO_TREATMENT",
            "ANATOMY": "ANATOMY",
            "FUNCTION": "FUNCTION",
            "GOAL": "GOAL",
            # GERNERMED (med7-style)
            "Drug": "MED_DRUG",
            "Strength": "MED_STRENGTH",
            "Route": "MED_ROUTE",
            "Form": "MED_FORM",
            "Dosage": "MED_DOSAGE",
            "Frequency": "MED_FREQUENCY",
            "Duration": "MED_DURATION",
        }
        return base_map.get(label)
    
    # ---------- entity helpers ----------
    @staticmethod
    def _deduplicate_entities(entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove duplicates (same span & text)."""
        seen = set()
        uniq: List[MedicalEntity] = []
        for e in entities:
            key = (e.start, e.end, e.text.lower())
            if key not in seen:
                seen.add(key)
                uniq.append(e)
        return uniq
    
    # ---------- extraction ----------
    @monitor_latency("ner_extract", "spacy")
    async def extract_entities(self, text: str) -> List[MedicalEntity]:
        """
        Extract medical entities from text (German).
        Strategy:
          1) Run base pipeline (de_core_news_md) generic NER + physio ruler.
          2) If available, run de_GERNERMED  add medication structure.
          3) Merge, map labels, try ICD/ATC lookup.
        """
        try:
            entities: List[MedicalEntity] = []

            # 1) Base German model
            doc_base = self.nlp_base(text)
            for ent in doc_base.ents:
                mapped = self._map_spacy_label(ent.label_)
                if mapped:
                    icd = self.icd_dict.lookup_entity(ent.text)
                    entities.append(MedicalEntity(
                        text=ent.text,
                        label=mapped,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.80,
                        icd_code=icd.get("code") if icd else None,
                        icd_description=icd.get("description") if icd else None,
                        normalized_text=ent.text.lower(),
                        category=icd.get("category") if icd else None
                    ))

            # 2) Medication model (optional)
            if self.nlp_med is not None:
                doc_med = self.nlp_med(text)
                for ent in doc_med.ents:
                    mapped = self._map_spacy_label(ent.label_)
                    if mapped:
                        icd = self.icd_dict.lookup_entity(ent.text)
                        entities.append(MedicalEntity(
                            text=ent.text,
                            label=mapped,
                            start=ent.start_char,
                            end=ent.end_char,
                            confidence=0.85,  # medication spans from dedicated model
                            icd_code=icd.get("code") if icd else None,
                            icd_description=icd.get("description") if icd else None,
                            normalized_text=ent.text.lower(),
                            category=icd.get("category") if icd else None
                        ))

            # 3) Clean up
            entities = self._deduplicate_entities(entities)
            entities.sort(key=lambda x: x.start)
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            raise
    
    # ---------- batch extraction ----------
    async def extract_entities_batch(self, texts: List[str]) -> List[List[MedicalEntity]]:
        """Extract entities from multiple texts in batch."""
        tasks = [self.extract_entities(text) for text in texts]
        return await asyncio.gather(*tasks)
    
    # ---------- stats / display ----------
    def get_entity_statistics(self, entities: List[MedicalEntity]) -> Dict[str, Any]:
        stats = {
            "total_entities": len(entities),
            "by_category": {},
            "by_label": {},
            "with_icd_codes": 0,
            "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        }
        for e in entities:
            cat = e.category or "unknown"
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            stats["by_label"][e.label] = stats["by_label"].get(e.label, 0) + 1
            if e.icd_code:
                stats["with_icd_codes"] += 1
            if e.confidence > 0.8:
                stats["confidence_distribution"]["high"] += 1
            elif e.confidence > 0.5:
                stats["confidence_distribution"]["medium"] += 1
            else:
                stats["confidence_distribution"]["low"] += 1
        return stats
    
    def format_entities_for_display(self, entities: List[MedicalEntity]) -> str:
        if not entities:
            return "Keine medizinischen Entitäten gefunden."
        lines = []
        for e in entities:
            line = f"• {e.text} ({e.label})"
            if e.icd_code:
                line += f" – ICD: {e.icd_code}"
            if e.icd_description:
                line += f" – {e.icd_description}"
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
                "base_model": "de_core_news_md",
                "med_model": bool(self.nlp_med),
                "test_entities_found": len(ents),
                "icd_dictionary_size": len(self.icd_dict.icd_codes),
            }
        except Exception as e:
            return {"service": "ner", "status": "unhealthy", "error": str(e)}

# Global NER service instance
ner_service = MedicalNERService()
