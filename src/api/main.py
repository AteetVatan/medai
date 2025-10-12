"""
FastAPI main application for medAI MVP.
Provides REST endpoints for clinical intake and data management.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

from ..agents.clinical_intake_agent import clinical_intake_agent, ClinicalIntakeResult
from ..services.storage_service import storage_service
from ..app.routers.reports import router as reports_router
from ..utils.config import settings
from ..utils.logging import get_logger, get_compliance_logger, RequestContext
from ..utils.cache import get_cache_stats
from .ws import websocket_endpoint

logger = get_logger(__name__)
compliance_logger = get_compliance_logger()


# Pydantic models
class PatientCreate(BaseModel):
    patient_number: str = Field(..., description="Internal patient number")
    first_name: str = Field(..., description="Patient first name")
    last_name: str = Field(..., description="Patient last name")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    gender: Optional[str] = Field(None, description="Patient gender")
    contact_info: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Contact information"
    )
    medical_history: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Medical history"
    )
    preferences: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Patient preferences"
    )


class EncounterCreate(BaseModel):
    patient_id: str = Field(..., description="Patient ID")
    therapist_id: str = Field(..., description="Therapist ID")
    referral_id: Optional[str] = Field(None, description="Referral ID")
    encounter_type: str = Field(default="intake", description="Type of encounter")
    scheduled_at: Optional[str] = Field(None, description="Scheduled time (ISO format)")
    notes: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Initial notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ClinicalIntakeRequest(BaseModel):
    encounter_id: str = Field(..., description="Clinical encounter ID")
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="webm", description="Audio format")
    duration_seconds: Optional[float] = Field(
        None, description="Audio duration in seconds"
    )
    translate_to: Optional[str] = Field(
        None, description="Target language for translation"
    )
    task_type: str = Field(
        default="intake_summary", description="Type of clinical task"
    )


class ClinicalIntakeResponse(BaseModel):
    success: bool
    encounter_id: str
    audio_record_id: str
    transcription: str
    entities: List[Dict[str, Any]]
    clinical_summary: str
    structured_notes: Dict[str, Any]
    translated_notes: Optional[Dict[str, Any]]
    processing_time_ms: float
    errors: List[str]


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    services: Dict[str, Any]
    cache_stats: Optional[Dict[str, Any]] = None
    version: str = "1.0.0"


# Create FastAPI app
app = FastAPI(
    title="medAI MVP API",
    description="Clinical speech and documentation backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(reports_router)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]  # Configure appropriately for production
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# WebSocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_route(
    websocket: WebSocket,
    session_id: str,
    user_id: str = "demo_user",
    organization_id: str = "demo_org",
):
    """WebSocket endpoint for real-time audio streaming."""
    await websocket_endpoint(websocket, session_id, user_id, organization_id)


# Root route to serve frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page."""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Frontend not found")


# Dependency for getting current user (simplified for MVP)
async def get_current_user() -> Dict[str, str]:
    """Get current user from request context."""
    # In production, this would extract user from JWT token
    return {"user_id": "demo_user", "organization_id": "demo_org"}


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and all services."""
    try:
        # Check agent health
        agent_health = await clinical_intake_agent.health_check()

        # Get cache statistics if caching is enabled
        cache_stats = None
        if settings.enable_caching:
            cache_stats = get_cache_stats()

        return HealthResponse(
            status="healthy" if agent_health["status"] == "healthy" else "degraded",
            timestamp=agent_health["timestamp"],
            services=agent_health["tools"],
            cache_stats=cache_stats,
            version="1.0.0",
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().timestamp(),
            services={},
            version="1.0.0",
        )


# Patient management endpoints
@app.post("/patients", status_code=status.HTTP_201_CREATED)
async def create_patient(
    patient: PatientCreate,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    """Create a new patient record."""
    try:
        with RequestContext(user_id=current_user["user_id"]):
            patient_data = patient.dict()
            result = await storage_service.create_patient(
                organization_id=current_user["organization_id"],
                patient_data=patient_data,
                user_id=current_user["user_id"],
            )

            # Log audit event
            background_tasks.add_task(
                storage_service.log_audit_event,
                organization_id=current_user["organization_id"],
                user_id=current_user["user_id"],
                action="create",
                resource_type="patient",
                resource_id=result["id"],
                details={"patient_number": patient.patient_number},
            )

            return result

    except Exception as e:
        logger.error(f"Failed to create patient: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create patient: {str(e)}",
        )


@app.get("/patients")
async def search_patients(
    q: str, limit: int = 20, current_user: Dict[str, str] = Depends(get_current_user)
):
    """Search patients by name or patient number."""
    try:
        with RequestContext(user_id=current_user["user_id"]):
            patients = await storage_service.search_patients(
                organization_id=current_user["organization_id"],
                query=q,
                user_id=current_user["user_id"],
                limit=limit,
            )

            return {"patients": patients}

    except Exception as e:
        logger.error(f"Failed to search patients: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search patients: {str(e)}",
        )


# Encounter management endpoints
@app.post("/encounters", status_code=status.HTTP_201_CREATED)
async def create_encounter(
    encounter: EncounterCreate,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    """Create a new clinical encounter."""
    try:
        with RequestContext(user_id=current_user["user_id"]):
            encounter_data = encounter.dict()
            result = await storage_service.create_encounter(
                organization_id=current_user["organization_id"],
                encounter_data=encounter_data,
                user_id=current_user["user_id"],
            )

            # Log audit event
            background_tasks.add_task(
                storage_service.log_audit_event,
                organization_id=current_user["organization_id"],
                user_id=current_user["user_id"],
                action="create",
                resource_type="encounter",
                resource_id=result["id"],
                details={"encounter_type": encounter.encounter_type},
            )

            return result

    except Exception as e:
        logger.error(f"Failed to create encounter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create encounter: {str(e)}",
        )


@app.get("/encounters/{encounter_id}")
async def get_encounter(
    encounter_id: str, current_user: Dict[str, str] = Depends(get_current_user)
):
    """Get encounter with related data."""
    try:
        with RequestContext(user_id=current_user["user_id"]):
            encounter = await storage_service.get_encounter(
                encounter_id=encounter_id, user_id=current_user["user_id"]
            )

            if not encounter:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Encounter not found"
                )

            return encounter

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get encounter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get encounter: {str(e)}",
        )


# Clinical intake processing endpoint
@app.post("/clinical-intake", response_model=ClinicalIntakeResponse)
async def process_clinical_intake(
    request: ClinicalIntakeRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    """Process clinical intake with audio transcription and analysis."""
    try:
        import base64

        with RequestContext(user_id=current_user["user_id"]):
            # Decode base64 audio data
            try:
                audio_data = base64.b64decode(request.audio_data)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 audio data: {str(e)}",
                )

            # Process clinical intake
            result = await clinical_intake_agent.process_clinical_intake(
                audio_data=audio_data,
                encounter_id=request.encounter_id,
                organization_id=current_user["organization_id"],
                user_id=current_user["user_id"],
                audio_format=request.audio_format,
                duration_seconds=request.duration_seconds,
                translate_to=request.translate_to,
                task_type=request.task_type,
            )

            # Log audit event
            background_tasks.add_task(
                storage_service.log_audit_event,
                organization_id=current_user["organization_id"],
                user_id=current_user["user_id"],
                action="process_audio",
                resource_type="encounter",
                resource_id=request.encounter_id,
                details={
                    "audio_format": request.audio_format,
                    "duration_seconds": request.duration_seconds,
                    "task_type": request.task_type,
                    "success": result.success,
                    "processing_time_ms": result.processing_time_ms,
                },
            )

            return ClinicalIntakeResponse(
                success=result.success,
                encounter_id=result.encounter_id,
                audio_record_id=result.audio_record_id,
                transcription=result.transcription,
                entities=result.entities,
                clinical_summary=result.clinical_summary,
                structured_notes=result.structured_notes,
                translated_notes=result.translated_notes,
                processing_time_ms=result.processing_time_ms,
                errors=result.errors,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clinical intake processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clinical intake processing failed: {str(e)}",
        )


# Audio file access endpoint
@app.get("/audio/{audio_record_id}/download")
async def get_audio_download_url(
    audio_record_id: str,
    expires_in: int = 3600,
    current_user: Dict[str, str] = Depends(get_current_user),
):
    """Get signed URL for audio file download."""
    try:
        with RequestContext(user_id=current_user["user_id"]):
            # Get audio record to find file path
            # This is simplified - in production, you'd query the database
            file_path = f"audio/{current_user['organization_id']}/{audio_record_id}.wav"

            signed_url = await storage_service.get_signed_url(
                file_path=file_path,
                expires_in=expires_in,
                user_id=current_user["user_id"],
            )

            return {"download_url": signed_url, "expires_in": expires_in}

    except Exception as e:
        logger.error(f"Failed to get audio download URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get audio download URL: {str(e)}",
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting medAI MVP API server")

    # Warm up services
    try:
        # Warm up clinical intake agent services
        await clinical_intake_agent.warm_up_services()

        # Test service health
        health = await clinical_intake_agent.health_check()
        logger.info(f"Services initialized: {health['status']}")
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down medAI MVP API server")


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
