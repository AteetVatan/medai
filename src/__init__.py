"""
medAI MVP - Clinical Speech & Documentation Platform
"""

__version__ = "1.0.0"
__author__ = "medAI Team"
__description__ = "Clinical speech and documentation backend with AI-powered processing"

# Import main modules for easy access
from src import agents
from src import api
from src import app
from src import models
from src import services
from src import utils

__all__ = [
    "agents",
    "api", 
    "app",
    "models",
    "services",
    "utils",
    "__version__",
    "__author__",
    "__description__",
]