"""
Storage service for medAI MVP.
Handles Supabase Postgres operations and S3 file storage with signed URLs.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import uuid
import json

import httpx
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

from ..utils.config import settings
from ..utils.logging import get_logger, get_latency_logger, get_compliance_logger, monitor_latency

logger = get_logger(__name__)
latency_logger = get_latency_logger()
compliance_logger = get_compliance_logger()


class StorageService:
    """Storage service for Supabase Postgres and S3 operations."""
    
    def __init__(self):
        self.supabase: Optional[Client] = None
        self._initialize_supabase()
    
    def _initialize_supabase(self):
        """Initialize Supabase client."""
        try:
            options = ClientOptions(
                auto_refresh_token=True,
                persist_session=True
            )
            
            self.supabase = create_client(
                settings.supabase_url,
                settings.supabase_key,
                options=options
            )
            logger.info("Supabase client initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    @monitor_latency("storage_create_patient", "supabase")
    async def create_patient(
        self,
        organization_id: str,
        patient_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Create a new patient record."""
        try:
            patient_record = {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "patient_number": patient_data.get("patient_number"),
                "first_name": patient_data.get("first_name"),
                "last_name": patient_data.get("last_name"),
                "date_of_birth": patient_data.get("date_of_birth"),
                "gender": patient_data.get("gender"),
                "contact_info": patient_data.get("contact_info", {}),
                "medical_history": patient_data.get("medical_history", {}),
                "preferences": patient_data.get("preferences", {}),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("patients").insert(patient_record).execute()
            
            if result.data:
                compliance_logger.log_data_access(
                    resource_type="patient",
                    resource_id=patient_record["id"],
                    user_id=user_id,
                    operation="create",
                    success=True
                )
                return result.data[0]
            else:
                raise Exception("Failed to create patient record")
                
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="patient",
                resource_id="unknown",
                user_id=user_id,
                operation="create",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to create patient: {e}")
            raise
    
    @monitor_latency("storage_create_encounter", "supabase")
    async def create_encounter(
        self,
        organization_id: str,
        encounter_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Create a new clinical encounter."""
        try:
            encounter_record = {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "patient_id": encounter_data.get("patient_id"),
                "therapist_id": encounter_data.get("therapist_id"),
                "referral_id": encounter_data.get("referral_id"),
                "encounter_type": encounter_data.get("encounter_type", "intake"),
                "status": encounter_data.get("status", "active"),
                "scheduled_at": encounter_data.get("scheduled_at"),
                "started_at": encounter_data.get("started_at"),
                "notes": encounter_data.get("notes", {}),
                "metadata": encounter_data.get("metadata", {}),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("encounters").insert(encounter_record).execute()
            
            if result.data:
                compliance_logger.log_data_access(
                    resource_type="encounter",
                    resource_id=encounter_record["id"],
                    user_id=user_id,
                    operation="create",
                    success=True
                )
                return result.data[0]
            else:
                raise Exception("Failed to create encounter record")
                
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="encounter",
                resource_id="unknown",
                user_id=user_id,
                operation="create",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to create encounter: {e}")
            raise
    
    @monitor_latency("storage_save_audio", "supabase")
    async def save_audio_record(
        self,
        encounter_id: str,
        organization_id: str,
        audio_data: bytes,
        audio_metadata: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Save audio record to S3 and create database record."""
        try:
            # Generate unique file path
            audio_id = str(uuid.uuid4())
            file_path = f"audio/{organization_id}/{encounter_id}/{audio_id}.wav"
            
            # Upload to Supabase Storage (S3)
            upload_result = self.supabase.storage.from_("audio").upload(
                file_path,
                audio_data,
                file_options={"content-type": "audio/wav"}
            )
            
            if upload_result.get("error"):
                raise Exception(f"Failed to upload audio: {upload_result['error']}")
            
            # Create database record
            audio_record = {
                "id": audio_id,
                "encounter_id": encounter_id,
                "organization_id": organization_id,
                "file_path": file_path,
                "file_size_bytes": len(audio_data),
                "duration_seconds": audio_metadata.get("duration_seconds"),
                "audio_format": audio_metadata.get("format", "wav"),
                "sample_rate": audio_metadata.get("sample_rate", 16000),
                "channels": audio_metadata.get("channels", 1),
                "transcription_text": audio_metadata.get("transcription_text"),
                "transcription_confidence": audio_metadata.get("transcription_confidence"),
                "processing_status": audio_metadata.get("processing_status", "completed"),
                "metadata": audio_metadata.get("metadata", {}),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("audio_records").insert(audio_record).execute()
            
            if result.data:
                compliance_logger.log_audio_processing(
                    audio_id=audio_id,
                    duration_seconds=audio_metadata.get("duration_seconds", 0),
                    user_id=user_id,
                    operation="save",
                    success=True
                )
                return result.data[0]
            else:
                raise Exception("Failed to create audio record")
                
        except Exception as e:
            compliance_logger.log_audio_processing(
                audio_id="unknown",
                duration_seconds=0,
                user_id=user_id,
                operation="save",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to save audio record: {e}")
            raise
    
    @monitor_latency("storage_save_entities", "supabase")
    async def save_medical_entities(
        self,
        encounter_id: str,
        audio_record_id: str,
        organization_id: str,
        entities: List[Dict[str, Any]],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Save extracted medical entities."""
        try:
            entity_records = []
            for entity in entities:
                entity_record = {
                    "id": str(uuid.uuid4()),
                    "encounter_id": encounter_id,
                    "audio_record_id": audio_record_id,
                    "organization_id": organization_id,
                    "entity_type": entity.get("label"),
                    "entity_text": entity.get("text"),
                    "start_char": entity.get("start"),
                    "end_char": entity.get("end"),
                    "confidence": entity.get("confidence"),
                    "icd_code": entity.get("icd_code"),
                    "icd_description": entity.get("icd_description"),
                    "icd_category": entity.get("category"),
                    "normalized_text": entity.get("normalized_text"),
                    "metadata": entity.get("metadata", {}),
                    "created_at": datetime.utcnow().isoformat()
                }
                entity_records.append(entity_record)
            
            result = self.supabase.table("medical_entities").insert(entity_records).execute()
            
            if result.data:
                compliance_logger.log_data_access(
                    resource_type="medical_entities",
                    resource_id=encounter_id,
                    user_id=user_id,
                    operation="create",
                    success=True,
                    count=len(entities)
                )
                return result.data
            else:
                raise Exception("Failed to save medical entities")
                
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="medical_entities",
                resource_id=encounter_id,
                user_id=user_id,
                operation="create",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to save medical entities: {e}")
            raise
    
    @monitor_latency("storage_save_notes", "supabase")
    async def save_clinical_notes(
        self,
        encounter_id: str,
        organization_id: str,
        notes_data: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Save structured clinical notes."""
        try:
            notes_record = {
                "id": str(uuid.uuid4()),
                "encounter_id": encounter_id,
                "organization_id": organization_id,
                "note_type": notes_data.get("note_type", "intake_summary"),
                "content": notes_data.get("structured_notes", {}),
                "raw_text": notes_data.get("raw_text"),
                "language": notes_data.get("language", "de"),
                "translated_content": notes_data.get("translated_content"),
                "translated_language": notes_data.get("translated_language"),
                "model_used": notes_data.get("model_used"),
                "processing_time_ms": notes_data.get("processing_time_ms"),
                "confidence_score": notes_data.get("confidence_score"),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table("clinical_notes").insert(notes_record).execute()
            
            if result.data:
                compliance_logger.log_data_access(
                    resource_type="clinical_notes",
                    resource_id=notes_record["id"],
                    user_id=user_id,
                    operation="create",
                    success=True
                )
                return result.data[0]
            else:
                raise Exception("Failed to save clinical notes")
                
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="clinical_notes",
                resource_id="unknown",
                user_id=user_id,
                operation="create",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to save clinical notes: {e}")
            raise
    
    @monitor_latency("storage_get_signed_url", "supabase")
    async def get_signed_url(
        self,
        file_path: str,
        expires_in: int = 3600,
        user_id: str = "system"
    ) -> str:
        """Get signed URL for S3 file access."""
        try:
            # Create signed URL for file access
            signed_url = self.supabase.storage.from_("audio").create_signed_url(
                file_path,
                expires_in=expires_in
            )
            
            if signed_url.get("error"):
                raise Exception(f"Failed to create signed URL: {signed_url['error']}")
            
            compliance_logger.log_data_access(
                resource_type="audio_file",
                resource_id=file_path,
                user_id=user_id,
                operation="read",
                success=True
            )
            
            return signed_url["signedURL"]
            
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="audio_file",
                resource_id=file_path,
                user_id=user_id,
                operation="read",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to get signed URL: {e}")
            raise
    
    @monitor_latency("storage_get_encounter", "supabase")
    async def get_encounter(
        self,
        encounter_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get encounter with related data."""
        try:
            # Get encounter with patient and therapist info
            result = self.supabase.table("encounters").select("""
                *,
                patients:patient_id(*),
                users:therapist_id(*),
                audio_records(*),
                clinical_notes(*),
                medical_entities(*)
            """).eq("id", encounter_id).execute()
            
            if result.data:
                compliance_logger.log_data_access(
                    resource_type="encounter",
                    resource_id=encounter_id,
                    user_id=user_id,
                    operation="read",
                    success=True
                )
                return result.data[0]
            else:
                return None
                
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="encounter",
                resource_id=encounter_id,
                user_id=user_id,
                operation="read",
                success=False,
                error=str(e)
            )
            logger.error(f"Failed to get encounter: {e}")
            raise
    
    @monitor_latency("storage_search_patients", "supabase")
    async def search_patients(
        self,
        organization_id: str,
        query: str,
        user_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search patients by name or patient number."""
        try:
            # Use full-text search
            result = self.supabase.table("patients").select("*").eq(
                "organization_id", organization_id
            ).text_search(
                "first_name,last_name,patient_number", query
            ).limit(limit).execute()
            
            compliance_logger.log_data_access(
                resource_type="patients",
                resource_id="search",
                user_id=user_id,
                operation="search",
                success=True,
                query=query
            )
            
            return result.data or []
            
        except Exception as e:
            compliance_logger.log_data_access(
                resource_type="patients",
                resource_id="search",
                user_id=user_id,
                operation="search",
                success=False,
                error=str(e),
                query=query
            )
            logger.error(f"Failed to search patients: {e}")
            raise
    
    @monitor_latency("storage_audit_log", "supabase")
    async def log_audit_event(
        self,
        organization_id: str,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """Log audit event for compliance."""
        try:
            audit_record = {
                "id": str(uuid.uuid4()),
                "organization_id": organization_id,
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": details,
                "success": success,
                "error_message": error_message,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.table("audit_log").insert(audit_record).execute()
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check storage service health."""
        try:
            # Test database connection
            start_time = time.time()
            result = self.supabase.table("organizations").select("id").limit(1).execute()
            db_latency = (time.time() - start_time) * 1000
            
            # Test storage connection
            start_time = time.time()
            # Try to list storage buckets
            storage_result = self.supabase.storage.list_buckets()
            storage_latency = (time.time() - start_time) * 1000
            
            return {
                "service": "storage",
                "status": "healthy",
                "database": {
                    "status": "healthy",
                    "latency_ms": db_latency
                },
                "storage": {
                    "status": "healthy",
                    "latency_ms": storage_latency
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "service": "storage",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global storage service instance
storage_service = StorageService()
