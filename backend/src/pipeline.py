"""
Pipeline Orchestrator
=====================
Coordinates all 5 stages sequentially for a single job.
Handles progress tracking and error propagation.
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from src.config import settings
from src.stages.stage1_ingest import ingest_xtf
from src.stages.stage2_preprocess import preprocess
from src.stages.stage3_ai import apply_noise_filter, run_inference
from src.stages.stage4_geospatial import geoproject_detections
from src.stages.stage5_report import generate_reports
from src.utils.image_utils import draw_detections, save_png

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------
_jobs: Dict[str, Dict[str, Any]] = {}   # job_id → job_record


def _new_job(filename: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id":        job_id,
        "filename":      filename,
        "status":        "pending",
        "progress_pct":  0,
        "current_stage": "queued",
        "created_at":    datetime.now(tz=timezone.utc).isoformat(),
        "result":        None,
        "error":         None,
    }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)


def list_jobs() -> list[Dict[str, Any]]:
    return list(_jobs.values())


def _update(job_id: str, **kwargs: Any) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------
ProgressFn = Callable[[int, str], None]


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    job_id: str,
    file_path: str | Path,
    base_url: str = "",
    progress_fn: Optional[ProgressFn] = None,
) -> Dict[str, Any]:
    """
    Execute the full 5-stage pipeline for a single XTF file.

    Args:
        job_id:      Unique job identifier (already registered in _jobs)
        file_path:   Path to the uploaded .xtf file
        base_url:    Server base URL for building artifact URLs
        progress_fn: Optional callback(pct, stage_name) for progress updates

    Returns:
        The final JSON report dict from Stage 5.
    """

    def _progress(pct: int, stage: str) -> None:
        _update(job_id, progress_pct=pct, current_stage=stage, status="processing")
        if progress_fn:
            progress_fn(pct, stage)
        logger.info("[%s] %d%%  %s", job_id[:8], pct, stage)

    # Setup output directory
    output_dir = Path(settings.OUTPUT_DIR) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        t0 = time.perf_counter()

        with open(file_path, 'rb') as f:
            header = f.read(8)
            
        is_standard_image = (
            header.startswith(b'\x89PNG\r\n\x1a\n') or 
            header.startswith(b'\xff\xd8') or 
            header.startswith(b'BM')
        )

        if is_standard_image:
            import cv2
            _progress(5, "Loading standard image file")
            processed = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if processed is None:
                raise ValueError(f"Could not read image file: {file_path}")
            telemetry = {}
            
            raw_png_path = output_dir / "raw.png"
            save_png(processed, raw_png_path)
            
            _progress(25, "Image loaded, skipping acoustic pre-processing")
        else:
            # ---------------------------------------------------------------
            # Stage 1 — Ingest
            # ---------------------------------------------------------------
            _progress(5, "Ingesting XTF file")
            raw_array, telemetry = ingest_xtf(file_path)

            # ---------------------------------------------------------------
            # Stage 2 — Pre-process
            # ---------------------------------------------------------------
            _progress(25, "Applying Lee filter & CLAHE contrast enhancement")
            processed = preprocess(raw_array)

            # Save the clean sonar image
            raw_png_path = output_dir / "raw.png"
            save_png(processed, raw_png_path)

        # ---------------------------------------------------------------
        # Stage 3 — AI Inference
        # ---------------------------------------------------------------
        _progress(50, "Running YOLOv8 inference")
        raw_detections = run_inference(processed)
        filtered_detections = apply_noise_filter(raw_detections)

        # ---------------------------------------------------------------
        # Stage 4 — Geospatial Projection
        # ---------------------------------------------------------------
        _progress(70, "Computing GPS coordinates and physical dimensions")
        enriched = geoproject_detections(filtered_detections, processed, telemetry)

        # ---------------------------------------------------------------
        # Save annotated image
        # ---------------------------------------------------------------
        _progress(85, "Rendering annotated image")
        annotated = draw_detections(processed, enriched)
        annotated_png_path = output_dir / "annotated.png"
        save_png(annotated, annotated_png_path)

        # ---------------------------------------------------------------
        # Stage 5 — Report
        # ---------------------------------------------------------------
        _progress(95, "Generating JSON and CSV reports")
        report = generate_reports(
            job_id=job_id,
            filename=Path(file_path).name,
            telemetry=telemetry,
            detections=enriched,
            output_dir=output_dir,
            base_url=base_url,
        )

        elapsed = round(time.perf_counter() - t0, 2)
        report["meta"]["processing_time_s"] = elapsed

        _update(
            job_id,
            status="done",
            progress_pct=100,
            current_stage="completed",
            result=report,
        )
        logger.info("Pipeline complete for job %s in %.2fs.", job_id, elapsed)
        return report

    except Exception as exc:
        logger.exception("Pipeline failed for job %s: %s", job_id, exc)
        _update(job_id, status="error", error=str(exc), current_stage="failed")
        raise


# ---------------------------------------------------------------------------
# Convenience factory: register + schedule
# ---------------------------------------------------------------------------
def create_job(filename: str) -> str:
    """Register a new job and return its ID."""
    job_id = _new_job(filename)
    return job_id
