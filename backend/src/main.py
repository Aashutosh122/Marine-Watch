"""
SIH Ghost Net / Marine Debris Detection — FastAPI Backend
=========================================================
Entry point. Mounts all routers and configures middleware.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api import router as api_router
from src.config import settings

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sonar_backend")


# ---------------------------------------------------------------------------
# Lifespan: startup / shutdown hooks
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the YOLO model once at startup so every request is fast."""
    logger.info("🚀  Starting SIH Sonar Debris Detection Backend …")
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    # Eagerly load YOLO so first request doesn't pay the cost
    from src.stages.stage3_ai import load_model
    load_model()
    logger.info("✅  YOLOv8 model loaded and ready.")

    yield

    logger.info("🛑  Shutting down — cleaning up resources.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SIH — Marine Debris SSS Detection API",
    description=(
        "End-to-end pipeline: XTF ingest → acoustic pre-processing → "
        "YOLOv8 inference → geospatial projection → anomaly report."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the React frontend running on any port during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the outputs directory so processed images are directly servable
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# Include API routes
app.include_router(api_router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Dev-server entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
