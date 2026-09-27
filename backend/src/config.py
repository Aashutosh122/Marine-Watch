"""
Application-wide configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Resolve important paths relative to this file's location
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent          # .../backend/src/
_BACKEND_DIR = _SRC_DIR.parent                       # .../backend/


class Settings:
    # Where processed images and reports are saved
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", str(_BACKEND_DIR / "outputs"))

    # Path to the YOLOv8 weights file
    # Drop your `best.pt` here, or set the env-var to point elsewhere.
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", str(_SRC_DIR / "models" / "best.pt")
    )

    # Fallback model downloaded if best.pt is absent (yolov8n = tiny, fast)
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "yolov8n.pt")

    # Confidence threshold for keeping a detection (0.0 – 1.0)
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))

    # IoU threshold for non-maximum suppression
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.45"))

    # Maximum uploaded file size in megabytes
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "500"))

    # How many completed jobs to keep in memory before GC
    MAX_JOB_HISTORY: int = int(os.getenv("MAX_JOB_HISTORY", "50"))

    # Lee filter window size (must be odd)
    LEE_FILTER_SIZE: int = int(os.getenv("LEE_FILTER_SIZE", "7"))

    # Default towfish altitude (metres) used when telemetry is unavailable
    DEFAULT_ALTITUDE_M: float = float(os.getenv("DEFAULT_ALTITUDE_M", "3.0"))

    # Default slant range (metres) used as fallback
    DEFAULT_SLANT_RANGE_M: float = float(os.getenv("DEFAULT_SLANT_RANGE_M", "30.0"))


settings = Settings()
