#!/usr/bin/env python3
"""
Development server startup script for medAI MVP.
Handles environment setup and service initialization.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import urllib.request
import zipfile

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
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("[OK] Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False

def download_spacy_models(model_dir: str = "models"):
    """
    Download/install spaCy models for the German physio/rehab MVP and optionally
    fetch the GERNERMEDpp model (GottBERT variant) and unpack it for manual loading.
    Returns True if at least the core German model works.
    """
    print("Installing spaCy models for MVP (German physiotherapy)...")

    # 1) German core pipeline
    cmd_core = [sys.executable, "-m", "spacy", "download", "de_core_news_md"]
    try:
        print(f"  Running: {' '.join(cmd_core)}")
        subprocess.run(cmd_core, check=True, capture_output=True, text=True)
        print("  [OK] Installed de_core_news_md")
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] Failed to download de_core_news_md:\n{e.stderr or e.stdout or 'No output'}")

    # 2) Download GERNERMEDpp zip to local model directory
    gernermedpp_link = "https://myweb.rz.uni-augsburg.de/~freijoha/GERNERMEDpp/GERNERMEDpp_GottBERT.zip"
    target_zip = os.path.join(model_dir, "GERNERMEDpp_GottBERT.zip")
    target_folder = os.path.join(model_dir, "GERNERMEDpp_GottBERT")
    try:
        os.makedirs(model_dir, exist_ok=True)
        print(f"  Downloading GERNERMEDpp model from {gernermedpp_link} …")
        urllib.request.urlretrieve(gernermedpp_link, target_zip)
        print(f"  [OK] Downloaded to {target_zip}")
    except Exception as e:
        print(f"  [WARN] Could not download GERNERMEDpp zip: {e}")

    # 3) Unzip the model if download succeeded
    if os.path.exists(target_zip):
        try:
            print(f"  Unzipping GERNERMEDpp model to {target_folder} …")
            with zipfile.ZipFile(target_zip, 'r') as zf:
                zf.extractall(target_folder)
            print("  [OK] Unzipped GERNERMEDpp")
        except Exception as e:
            print(f"  [WARN] Failed to extract GERNERMEDpp zip: {e}")

    # Optional sanity check: try loading spaCy German model and (if installed) GERNERMEDpp
    try:
        import spacy
        # Load German core pipeline
        nlp_core = spacy.load("de_core_news_md")
        print("✔ Loaded de_core_news_md")

        # Attempt to load GERNERMEDpp as spaCy pipeline (if it has spaCy format)
        # Assume that if there is a "pipeline" folder inside target_folder, it may be spaCy.
        if os.path.isdir(target_folder):
            try:
                nlp_pp = spacy.load(target_folder)
                print("✔ Loaded GERNERMEDpp spaCy pipeline")
            except Exception as med_err:
                print(f"[INFO] GERNERMEDpp is not a spaCy-serializable pipeline, or load failed: {med_err}")
        else:
            print("[INFO] GERNERMEDpp directory not present for spaCy load attempt")

        print("[OK] Verified spaCy core; GERNERMEDpp integration is optional")
        return True
    except Exception as e:
        print(f"[WARN] Load check failed: {e}")
        print("You may need to reinstall the core German model or adjust your spaCy versions.")
        return False

def start_server():
    """Start the development server."""
    print(" Starting medAI MVP development server...")
    print("   API will be available at: http://localhost:8000")
    print("   API docs at: http://localhost:8000/docs")
    print("   Press Ctrl+C to stop")
    print("-" * 60)
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"[ERROR] Server error: {e}")
        return False
    
    return True


def main():
    """Main startup function."""
    print(" medAI MVP Development Server")
    print("=" * 60)
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Download spaCy models
    download_spacy_models()
    
    # Start server
    if not start_server():
        sys.exit(1)


if __name__ == "__main__":
    main()
