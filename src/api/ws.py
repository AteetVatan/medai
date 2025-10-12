"""
WebSocket handlers for medAI MVP.
Provides real-time audio streaming and transcription updates.
"""

import asyncio
import json
import logging
import base64
from typing import Dict, Any, Optional, List
import uuid

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException
from fastapi import WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel

from ..agents.clinical_intake_agent import clinical_intake_agent
from ..utils.logging import get_logger, RequestContext

logger = get_logger(__name__)


class WebSocketMessage(BaseModel):
    """WebSocket message model."""

    type: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WebSocketManager:
    """Manages WebSocket connections and audio streaming."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_buffers: Dict[str, bytes] = {}
        self.partial_transcriptions: Dict[str, List[Dict[str, Any]]] = {}
        self.session_data: Dict[str, Dict[str, Any]] = {}

    async def connect(
        self, websocket: WebSocket, session_id: str, user_id: str, organization_id: str
    ):
        """Accept WebSocket connection and initialize session."""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.audio_buffers[session_id] = b""
        self.partial_transcriptions[session_id] = []
        self.session_data[session_id] = {
            "user_id": user_id,
            "organization_id": organization_id,
            "encounter_id": None,
            "started_at": None,
            "last_activity": None,
        }

        logger.info(f"WebSocket connected: {session_id} for user {user_id}")

        # Send welcome message
        await self.send_message(
            session_id,
            {
                "type": "connected",
                "data": {
                    "session_id": session_id,
                    "message": "WebSocket connected successfully",
                },
            },
        )

    def disconnect(self, session_id: str):
        """Remove WebSocket connection and cleanup."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.audio_buffers:
            del self.audio_buffers[session_id]
        if session_id in self.partial_transcriptions:
            del self.partial_transcriptions[session_id]
        if session_id in self.session_data:
            del self.session_data[session_id]

        logger.info(f"WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send message to specific WebSocket connection."""
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_text(json.dumps(message))
            except ConnectionClosed:
                self.disconnect(session_id)
            except Exception as e:
                logger.error(f"Failed to send message to {session_id}: {e}")
                self.disconnect(session_id)

    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all active connections."""
        for session_id in list(self.active_connections.keys()):
            await self.send_message(session_id, message)

    def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.session_data.get(session_id)

    def update_session_data(self, session_id: str, data: Dict[str, Any]):
        """Update session data."""
        if session_id in self.session_data:
            self.session_data[session_id].update(data)

    def add_audio_chunk(self, session_id: str, audio_chunk: bytes):
        """Add audio chunk to buffer."""
        if session_id in self.audio_buffers:
            self.audio_buffers[session_id] += audio_chunk

    def get_audio_buffer(self, session_id: str) -> bytes:
        """Get accumulated audio buffer."""
        return self.audio_buffers.get(session_id, b"")

    def clear_audio_buffer(self, session_id: str):
        """Clear audio buffer."""
        if session_id in self.audio_buffers:
            self.audio_buffers[session_id] = b""

    def add_partial_transcription(
        self, session_id: str, transcription_data: Dict[str, Any]
    ):
        """Add partial transcription result to buffer."""
        if session_id in self.partial_transcriptions:
            self.partial_transcriptions[session_id].append(transcription_data)

    def get_partial_transcriptions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all accumulated partial transcriptions."""
        return self.partial_transcriptions.get(session_id, [])

    def clear_partial_transcriptions(self, session_id: str):
        """Clear partial transcriptions buffer."""
        if session_id in self.partial_transcriptions:
            self.partial_transcriptions[session_id] = []


# Global WebSocket manager
ws_manager = WebSocketManager()


async def websocket_endpoint(
    websocket: WebSocket,
    session_id: str = None,
    user_id: str = "demo_user",
    organization_id: str = "demo_org",
):
    """Main WebSocket endpoint for audio streaming."""
    if not session_id:
        session_id = str(uuid.uuid4())

    await ws_manager.connect(websocket, session_id, user_id, organization_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)

            # Update last activity
            ws_manager.update_session_data(
                session_id, {"last_activity": asyncio.get_event_loop().time()}
            )

            # Handle different message types
            await handle_websocket_message(session_id, message)

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)
    except WebSocketException as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        ws_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket {session_id}: {e}")
        ws_manager.disconnect(session_id)


async def handle_websocket_message(session_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")
    data = message.get("data", {})

    try:
        if message_type == "start_session":
            await handle_start_session(session_id, data)

        elif message_type == "start_recording":
            await handle_start_recording(session_id, data)

        elif message_type == "audio_chunk":
            await handle_audio_chunk(session_id, data)

        elif message_type == "end_recording":
            await handle_end_recording(session_id, data)

        elif message_type == "end_session":
            await handle_end_session(session_id, data)

        elif message_type == "ping":
            await handle_ping(session_id)

        else:
            await ws_manager.send_message(
                session_id,
                {"type": "error", "error": f"Unknown message type: {message_type}"},
            )

    except Exception as e:
        logger.error(f"Error handling message for {session_id}: {e}")
        await ws_manager.send_message(
            session_id, {"type": "error", "error": f"Message handling failed: {str(e)}"}
        )


async def handle_start_session(session_id: str, data: Dict[str, Any]):
    """Handle session start."""
    encounter_id = data.get("encounter_id")
    if not encounter_id:
        await ws_manager.send_message(
            session_id, {"type": "error", "error": "encounter_id is required"}
        )
        return

    # Update session data
    ws_manager.update_session_data(
        session_id,
        {
            "encounter_id": encounter_id,
            "started_at": asyncio.get_event_loop().time(),
            "task_type": data.get("task_type", "intake_summary"),
            "translate_to": data.get("translate_to"),
        },
    )

    await ws_manager.send_message(
        session_id,
        {
            "type": "session_started",
            "data": {
                "session_id": session_id,
                "encounter_id": encounter_id,
                "message": "Session started, ready for audio",
            },
        },
    )


async def handle_start_recording(session_id: str, data: Dict[str, Any]):
    """Handle start of new recording within existing session."""
    try:
        # Verify session exists
        session_data = ws_manager.get_session_data(session_id)
        if not session_data or not session_data.get("encounter_id"):
            await ws_manager.send_message(
                session_id, {"type": "error", "error": "Session not started"}
            )
            return

        # Clear audio buffer and partial transcriptions for new recording
        ws_manager.clear_audio_buffer(session_id)
        ws_manager.clear_partial_transcriptions(session_id)

        logger.info(f"Started new recording in session {session_id}")

        await ws_manager.send_message(
            session_id,
            {
                "type": "recording_started",
                "data": {
                    "session_id": session_id,
                    "encounter_id": session_data["encounter_id"],
                    "message": "New recording started, ready for audio",
                },
            },
        )

    except Exception as e:
        logger.error(f"Error starting recording in session {session_id}: {e}")
        await ws_manager.send_message(
            session_id,
            {"type": "error", "error": f"Failed to start recording: {str(e)}"},
        )


async def handle_audio_chunk(session_id: str, data: Dict[str, Any]):
    """Handle audio chunk processing."""
    audio_data = data.get("audio_data")
    if not audio_data:
        await ws_manager.send_message(
            session_id, {"type": "error", "error": "audio_data is required"}
        )
        return

    try:
        # Decode base64 audio data
        audio_chunk = base64.b64decode(audio_data)

        # Add to buffer
        ws_manager.add_audio_chunk(session_id, audio_chunk)

        # Get session data
        session_data = ws_manager.get_session_data(session_id)
        if not session_data or not session_data.get("encounter_id"):
            await ws_manager.send_message(
                session_id, {"type": "error", "error": "Session not started"}
            )
            return

        # Process partial transcription if we have enough audio
        accumulated_audio = ws_manager.get_audio_buffer(session_id)
        logger.info(
            f"Audio chunk received: {len(audio_chunk)} bytes, total accumulated: {len(accumulated_audio)} bytes"
        )
        if (
            len(accumulated_audio) > 8000
        ):  # Reduced threshold for faster feedback (0.5 seconds)
            logger.info(
                f"Triggering partial transcription for session {session_id} - threshold met!"
            )
            try:
                await process_partial_transcription(session_id, accumulated_audio)
                logger.info(
                    f"Partial transcription completed successfully for session {session_id}"
                )
            except Exception as e:
                logger.error(
                    f"Partial transcription failed for session {session_id}: {e}"
                )
        else:
            logger.info(
                f"Not enough audio yet: {len(accumulated_audio)} bytes (need > 8000)"
            )

        # Send acknowledgment
        await ws_manager.send_message(
            session_id,
            {
                "type": "audio_received",
                "data": {
                    "chunk_size": len(audio_chunk),
                    "total_size": len(accumulated_audio),
                },
            },
        )

    except Exception as e:
        logger.error(f"Error processing audio chunk for {session_id}: {e}")
        await ws_manager.send_message(
            session_id, {"type": "error", "error": f"Audio processing failed: {str(e)}"}
        )


async def process_partial_transcription(
    session_id: str, audio_data: bytes, use_stt: bool = True
):
    """Process partial transcription for real-time updates."""
    try:

        if use_stt:
            session_data = ws_manager.get_session_data(session_id)
            user_id = session_data["user_id"]

            logger.info(
                f"Starting partial transcription for session {session_id}, audio size: {len(audio_data)} bytes"
            )

            # Convert to WAV format
            from ..services.stt_service import stt_service

            # wav_data = await stt_service._process_audio_for_stt(audio_data, "wav")

            # Get partial transcription with proper context
            with RequestContext(
                user_id=user_id, request_id=f"partial_transcription_{session_id}"
            ):
                logger.info(
                    f"Calling STT service for partial transcription, audio size: {len(audio_data)} bytes"
                )
                result = await stt_service.transcribe_audio(
                    audio_data, "webm"
                )  # Use webm format as sent from frontend
                logger.info(f"STT service returned result: {result}")

            logger.info(
                f"Partial transcription completed for session {session_id}: '{result['text'][:50]}...'"
            )

        else:
            result = {
                "text": "Processing...",
                "model": "together_whisper",
                "latency": 0.0,
                "confidence": 1.0,  # Together AI doesn't provide confidence scores
                "provider": "None",
            }

        # Store partial transcription data
        transcription_data = {
            "transcription": result["text"],
            "confidence": result["confidence"],
            "model": result["model"],
            "timestamp": asyncio.get_event_loop().time(),
            "audio_size": len(audio_data),
        }
        ws_manager.add_partial_transcription(session_id, transcription_data)

        # Send partial transcription
        await ws_manager.send_message(
            session_id, {"type": "partial_transcription", "data": transcription_data}
        )

        logger.info(
            f"Partial transcription stored and sent to frontend for session {session_id}"
        )

    except Exception as e:
        logger.warning(f"Partial transcription failed for {session_id}: {e}")
        # Don't send error for partial transcription failures


async def handle_end_recording(session_id: str, data: Dict[str, Any]):
    """Handle recording end and processing within session."""
    try:
        session_data = ws_manager.get_session_data(session_id)
        if not session_data or not session_data.get("encounter_id"):
            await ws_manager.send_message(
                session_id, {"type": "error", "error": "Session not started"}
            )
            return

        # Get accumulated audio and partial transcriptions
        accumulated_audio = ws_manager.get_audio_buffer(session_id)
        accumulated_transcriptions = ws_manager.get_partial_transcriptions(session_id)

        if not accumulated_audio:
            await ws_manager.send_message(
                session_id, {"type": "error", "error": "No audio data received"}
            )
            return

        logger.info(
            f"Recording in session {session_id} ending with {len(accumulated_transcriptions)} partial transcriptions"
        )

        # Send processing started message
        await ws_manager.send_message(
            session_id,
            {
                "type": "processing_started",
                "data": {
                    "message": "Processing audio and generating clinical summary..."
                },
            },
        )

        # Process complete clinical intake
        result = await clinical_intake_agent.process_clinical_intake(
            audio_data=accumulated_audio,
            encounter_id=session_data["encounter_id"],
            organization_id=session_data["organization_id"],
            user_id=session_data["user_id"],
            transcriptions=accumulated_transcriptions,
            audio_format="webm",
            translate_to=session_data.get("translate_to"),
            task_type=session_data.get("task_type", "intake_summary"),
        )

        # Send final results
        await ws_manager.send_message(
            session_id,
            {
                "type": "processing_completed",
                "data": {
                    "success": result.success,
                    "encounter_id": result.encounter_id,
                    "audio_record_id": result.audio_record_id,
                    "transcription": result.transcription,
                    "entities": result.entities,
                    "clinical_summary": result.clinical_summary,
                    "structured_notes": result.structured_notes,
                    "translated_notes": result.translated_notes,
                    "processing_time_ms": result.processing_time_ms,
                    "errors": result.errors,
                    "partial_transcriptions": accumulated_transcriptions,
                    "partial_transcription_count": len(accumulated_transcriptions),
                },
            },
        )

        # Clear buffers for next recording
        ws_manager.clear_audio_buffer(session_id)
        ws_manager.clear_partial_transcriptions(session_id)

    except Exception as e:
        logger.error(f"Error ending recording in session {session_id}: {e}")
        await ws_manager.send_message(
            session_id,
            {"type": "error", "error": f"Recording end processing failed: {str(e)}"},
        )


async def handle_end_session(session_id: str, data: Dict[str, Any]):
    """Handle session end and cleanup."""
    try:
        session_data = ws_manager.get_session_data(session_id)
        if not session_data:
            await ws_manager.send_message(
                session_id, {"type": "error", "error": "Session not found"}
            )
            return

        logger.info(f"Ending session {session_id}")

        # Send session ended message
        await ws_manager.send_message(
            session_id,
            {
                "type": "session_ended",
                "data": {
                    "session_id": session_id,
                    "message": "Session ended successfully",
                },
            },
        )

        # Clean up session resources
        ws_manager.clear_audio_buffer(session_id)
        ws_manager.clear_partial_transcriptions(session_id)

        # Disconnect the WebSocket
        ws_manager.disconnect(session_id)

    except Exception as e:
        logger.error(f"Error ending session {session_id}: {e}")
        await ws_manager.send_message(
            session_id, {"type": "error", "error": f"Session end failed: {str(e)}"}
        )
        # Still try to cleanup
        try:
            ws_manager.disconnect(session_id)
        except:
            pass


async def handle_ping(session_id: str):
    """Handle ping message."""
    await ws_manager.send_message(
        session_id,
        {"type": "pong", "data": {"timestamp": asyncio.get_event_loop().time()}},
    )


# WebSocket connection manager for FastAPI
class ConnectionManager:
    """FastAPI WebSocket connection manager."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


# FastAPI WebSocket endpoint
async def websocket_fastapi_endpoint(
    websocket: WebSocket,
    session_id: str = None,
    user_id: str = "demo_user",
    organization_id: str = "demo_org",
):
    """FastAPI WebSocket endpoint."""
    await manager.connect(websocket)

    if not session_id:
        session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle message
            await handle_websocket_message(session_id, message)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
