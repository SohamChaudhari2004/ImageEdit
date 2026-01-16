"""
Configuration for Agentic Image Editor.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv(Path(__file__).parent / ".env")

# Project paths
PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Models (Groq-compatible)
LLM_MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

# FFmpeg
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFMPEG_TIMEOUT = 60

# Workflow
MAX_RETRIES = 3