#!/usr/bin/env python3
"""
Development server startup script for medAI MVP.
Handles environment setup and service initialization.
"""

import os
import sys
import subprocess
import time
import asyncio
import logging
import argparse
from pathlib import Path
import urllib.request
import zipfile
import uvicorn
from src.utils.config import settings


def check_environment():
    """Check if environment is properly configured."""
    print(" Checking environment...")

    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("[ERROR] .env file not found!")
        print("   Please copy env.example to .env and configure your API keys.")
        return False

    # Check Python version
    if sys.version_info < (3, 12):
        print(f"[ERROR] Python 3.12+ required, found {sys.version}")
        return False

    print("[OK] Environment check passed")
    return True


def install_dependencies():
    """Install Python dependencies."""
    print(" Installing dependencies...")

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
        )
        print("[OK] Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False


def check_ner_microservice():
    """
    Check if NER microservice is available and healthy.
    Returns True if microservice is accessible.
    """
    print("Checking NER microservice availability...")

    try:
        import httpx
        import asyncio

        async def check_microservice():
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    response = await client.get(
                        f"{settings.ner_microservice_base_url}/health"
                    )
                    if response.status_code == 200:
                        print("✔ NER microservice is healthy")
                        return True
                    else:
                        print(
                            f"✗ NER microservice returned status {response.status_code}"
                        )
                        return False
                except httpx.ConnectError:
                    print("✗ NER microservice is not running on localhost:8000")
                    return False

        return asyncio.run(check_microservice())
    except ImportError:
        print("✗ httpx not available for microservice check")
        return False
    except Exception as e:
        print(f"✗ Error checking NER microservice: {e}")
        return False


async def run_service_async(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = True,
    log_level: str = "info",
):
    """Run FastAPI service safely within an existing event loop."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting medAI MVP Service on {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Reload: {reload}")
    logger.info(f"Log level: {log_level}")

    config_uvicorn = uvicorn.Config(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=reload,
    )
    server = uvicorn.Server(config_uvicorn)
    await server.serve()


async def run_production_service(
    host: str = "0.0.0.0", port: int = 8000, workers: int = 1, log_level: str = "info"
):
    """Run FastAPI service in production mode (no reload)."""
    return await run_service_async(
        host=host, port=port, workers=workers, reload=False, log_level=log_level
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="medAI MVP Development Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    parser.add_argument(
        "--production", action="store_true", help="Run in production mode (no reload)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip environment and microservice checks",
    )
    return parser.parse_args()


def main():
    """Main startup function."""
    args = parse_arguments()

    print(" medAI MVP Development Server")
    print("=" * 60)
    print(f"Mode: {'Production' if args.production else 'Development'}")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("-" * 60)

    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    # Check environment (unless skipped)
    if not args.skip_checks:
        # if not check_environment():
        #     sys.exit(1)

        # # Install dependencies
        # if not install_dependencies():
        #     sys.exit(1)

        # Check NER microservice availability
        check_ner_microservice()

    # Start server
    try:
        if args.production:
            # Production mode
            asyncio.run(
                run_production_service(
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    log_level=args.log_level,
                )
            )
        else:
            # Development mode
            asyncio.run(
                run_service_async(
                    host=args.host,
                    port=args.port,
                    workers=args.workers,
                    reload=True,
                    log_level=args.log_level,
                )
            )
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
