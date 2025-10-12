"""
Structured logging configuration for medAI MVP.
Provides request tracking, latency metrics, and compliance logging.
"""

import asyncio
import logging
import sys
import time
import uuid
from typing import Any, Dict, Optional
from contextvars import ContextVar
from datetime import datetime
import json

from src.utils.config import settings

# Request context for tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request context if available
        if request_id := request_id_var.get():
            log_entry["request_id"] = request_id
        if user_id := user_id_var.get():
            log_entry["user_id"] = user_id
        if session_id := session_id_var.get():
            log_entry["session_id"] = session_id

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class LatencyLogger:
    """Specialized logger for latency tracking and performance monitoring."""

    def __init__(self, name: str = "latency"):
        self.logger = logging.getLogger(name)

    def log_latency(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        model: Optional[str] = None,
        fallback_used: bool = False,
        **kwargs,
    ) -> None:
        """Log operation latency with context."""
        extra_fields = {
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "model": model,
            "fallback_used": fallback_used,
            **kwargs,
        }

        # Determine log level based on latency and threshold
        threshold_exceeded = kwargs.get("threshold_exceeded", False)
        if threshold_exceeded:
            level = logging.WARNING
        elif duration_ms > 3000:
            level = logging.ERROR
        elif duration_ms > 1500:
            level = logging.WARNING
        else:
            level = logging.INFO

        message = f"{operation} completed in {duration_ms:.2f}ms"
        if threshold_exceeded:
            message += " [THRESHOLD EXCEEDED]"
        if fallback_used:
            message += " [FALLBACK USED]"

        self.logger.log(level, message, extra={"extra_fields": extra_fields})

        # Log to performance metrics if enabled
        if settings.enable_structured_logging:
            self._log_performance_metrics(
                operation,
                duration_ms,
                success,
                model,
                fallback_used,
                threshold_exceeded,
            )

    def _log_performance_metrics(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        model: Optional[str],
        fallback_used: bool,
        threshold_exceeded: bool = False,
    ) -> None:
        """Log performance metrics for monitoring."""
        metrics = {
            "type": "performance_metric",
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "model": model,
            "fallback_used": fallback_used,
            "threshold_exceeded": threshold_exceeded,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # In production, this would send to monitoring system
        self.logger.info("Performance metric", extra={"extra_fields": metrics})


class ComplianceLogger:
    """Logger for compliance and audit trail requirements."""

    def __init__(self, name: str = "compliance"):
        self.logger = logging.getLogger(name)

    def log_audio_processing(
        self,
        audio_id: str,
        duration_seconds: float,
        user_id: str,
        operation: str,
        success: bool,
        **kwargs,
    ) -> None:
        """Log audio processing for compliance."""
        extra_fields = {
            "type": "audio_processing",
            "audio_id": audio_id,
            "duration_seconds": duration_seconds,
            "user_id": user_id,
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }

        self.logger.info(
            f"Audio processing: {operation}", extra={"extra_fields": extra_fields}
        )

    def log_llm_interaction(
        self,
        request_id: str,
        model: str,
        prompt_length: int,
        response_length: int,
        user_id: str,
        pii_stripped: bool = False,
        **kwargs,
    ) -> None:
        """Log LLM interactions for compliance."""
        extra_fields = {
            "type": "llm_interaction",
            "request_id": request_id,
            "model": model,
            "prompt_length": prompt_length,
            "response_length": response_length,
            "user_id": user_id,
            "pii_stripped": pii_stripped,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }

        self.logger.info(
            f"LLM interaction: {model}", extra={"extra_fields": extra_fields}
        )

    def log_data_access(
        self,
        resource_type: str,
        resource_id: str,
        user_id: str,
        operation: str,
        success: bool,
        **kwargs,
    ) -> None:
        """Log data access for audit trail."""
        extra_fields = {
            "type": "data_access",
            "resource_type": resource_type,
            "resource_id": resource_id,
            "user_id": user_id,
            "operation": operation,
            "success": success,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **kwargs,
        }

        self.logger.info(
            f"Data access: {operation} {resource_type}",
            extra={"extra_fields": extra_fields},
        )


def setup_logging() -> None:
    """Configure application logging."""
    # Create formatters
    if settings.enable_structured_logging:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler for compliance logs
    if settings.enable_structured_logging:
        file_handler = logging.FileHandler("compliance.log")
        file_handler.setFormatter(formatter)
        compliance_logger = logging.getLogger("compliance")
        compliance_logger.addHandler(file_handler)
        compliance_logger.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance with proper configuration."""
    return logging.getLogger(name)


def get_latency_logger() -> LatencyLogger:
    """Get latency logger instance."""
    return LatencyLogger()


def get_compliance_logger() -> ComplianceLogger:
    """Get compliance logger instance."""
    return ComplianceLogger()


# Context managers for request tracking
class RequestContext:
    """Context manager for request tracking."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
        self._tokens = []

    def __enter__(self):
        self._tokens.append(request_id_var.set(self.request_id))
        if self.user_id:
            self._tokens.append(user_id_var.set(self.user_id))
        if self.session_id:
            self._tokens.append(session_id_var.set(self.session_id))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore previous context variable values using the tokens
        for token in reversed(self._tokens):
            token.var.reset(token)


def _check_threshold(operation: str, duration_ms: float) -> bool:
    """Check if operation duration exceeds configured thresholds."""
    try:
        from src.utils.config import settings

        # Map operations to their thresholds
        if "stt" in operation.lower() or "whisper" in operation.lower():
            return duration_ms > settings.stt_final_threshold
        elif (
            "llm" in operation.lower()
            or "mistral" in operation.lower()
            or "phi" in operation.lower()
            or "gemini" in operation.lower()
        ):
            return duration_ms > settings.llm_summary_threshold
        elif (
            "translation" in operation.lower()
            or "google" in operation.lower()
            or "nllb" in operation.lower()
        ):
            return duration_ms > settings.translation_threshold
        else:
            return False
    except Exception:
        return False


# Performance monitoring decorator
def monitor_latency(operation: str, model: Optional[str] = None):
    """Decorator to monitor operation latency with threshold checking."""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            fallback_used = False
            threshold_exceeded = False

            try:
                result = await func(*args, **kwargs)

                # Check thresholds
                duration_ms = (time.time() - start_time) * 1000
                threshold_exceeded = _check_threshold(operation, duration_ms)

                # Add threshold info to result if it's a dict
                if isinstance(result, dict):
                    result["threshold_exceeded"] = threshold_exceeded
                    result["latency_ms"] = duration_ms

                return result
            except Exception as e:
                success = False
                # Check if it's a fallback scenario
                if "fallback" in str(e).lower():
                    fallback_used = True
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                latency_logger = get_latency_logger()
                latency_logger.log_latency(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success,
                    model=model,
                    fallback_used=fallback_used,
                    threshold_exceeded=threshold_exceeded,
                )

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            fallback_used = False
            threshold_exceeded = False

            try:
                result = func(*args, **kwargs)

                # Check thresholds
                duration_ms = (time.time() - start_time) * 1000
                threshold_exceeded = _check_threshold(operation, duration_ms)

                # Add threshold info to result if it's a dict
                if isinstance(result, dict):
                    result["threshold_exceeded"] = threshold_exceeded
                    result["latency_ms"] = duration_ms

                return result
            except Exception as e:
                success = False
                if "fallback" in str(e).lower():
                    fallback_used = True
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                latency_logger = get_latency_logger()
                latency_logger.log_latency(
                    operation=operation,
                    duration_ms=duration_ms,
                    success=success,
                    model=model,
                    fallback_used=fallback_used,
                    threshold_exceeded=threshold_exceeded,
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Initialize logging on module import
setup_logging()
