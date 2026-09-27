"""
FastAPI router — all /api/v1/* endpoints.

Endpoints:
  GET  /health                     — system health + model status
  POST /upload                     — upload sonar file, kick off pipeline
  GET  /status/{job_id}            — poll job progress
  GET  /results/{job_id}           — full JSON report (done only)
  GET  /image/{job_id}/raw         — raw sonar PNG
  GET  /image/{job_id}/annotated   — annotated sonar PNG
  GET  /report/{job_id}/csv        — CSV download
  GET  /jobs                       — list all jobs
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.config import settings
from src.pipeline import create_job, get_job, list_jobs, run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()

# Maximum file size in bytes
_MAX_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@router.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Returns API health and model load status."""
    from src.stages.stage3_ai import _model
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_path": settings.MODEL_PATH,
        "output_dir": settings.OUTPUT_DIR,
    }


# ---------------------------------------------------------------------------
# Upload — trigger pipeline
# ---------------------------------------------------------------------------
@router.post("/upload", tags=["Pipeline"])
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Accept an XTF file upload and kick off the 5-stage processing pipeline
    in the background.  Returns a `job_id` that can be polled for status.

    Accepts:
      - `.xtf` files (real side-scan sonar data)
      - `.bin`, `.dat`, `.raw` — other binary sonar formats (parsed as XTF)
      - Any file for demo/testing (falls back to synthetic sonar)
    """
    # Size guard
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {settings.MAX_UPLOAD_MB} MB.",
        )

    # Persist the upload to disk
    job_id = create_job(file.filename or "unknown.xtf")
    upload_dir = Path(settings.OUTPUT_DIR) / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / (file.filename or "input.xtf")

    async with aiofiles.open(upload_path, "wb") as out_file:
        while chunk := await file.read(1024 * 1024):   # 1 MB chunks
            await out_file.write(chunk)

    logger.info("Saved upload for job %s → %s", job_id, upload_path)

    # Determine base URL for building artifact URLs
    base_url = str(request.base_url).rstrip("/")

    # Schedule the pipeline as a background task
    background_tasks.add_task(
        _run_pipeline_bg,
        job_id=job_id,
        file_path=str(upload_path),
        base_url=base_url,
    )

    return JSONResponse(
        status_code=202,
        content={
            "job_id":   job_id,
            "status":   "pending",
            "message":  "File received. Processing started in background.",
            "poll_url": f"{base_url}/api/v1/status/{job_id}",
        },
    )


def _run_pipeline_bg(job_id: str, file_path: str, base_url: str) -> None:
    """Wrapper for background task execution (must be sync for FastAPI BG tasks)."""
    try:
        run_pipeline(job_id=job_id, file_path=file_path, base_url=base_url)
    except Exception as exc:
        logger.error("Background pipeline error for job %s: %s", job_id, exc)


# ---------------------------------------------------------------------------
# Status polling
# ---------------------------------------------------------------------------
@router.get("/status/{job_id}", tags=["Pipeline"])
async def job_status(job_id: str) -> JSONResponse:
    """Poll the processing status of a job."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JSONResponse({
        "job_id":        job["job_id"],
        "status":        job["status"],
        "progress_pct":  job["progress_pct"],
        "current_stage": job["current_stage"],
        "error":         job.get("error"),
    })


# ---------------------------------------------------------------------------
# Full results (JSON report)
# ---------------------------------------------------------------------------
@router.get("/results/{job_id}", tags=["Pipeline"])
async def get_results(job_id: str) -> JSONResponse:
    """
    Return the full anomaly report for a completed job.
    Returns 202 if the job is still processing.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] == "processing" or job["status"] == "pending":
        return JSONResponse(
            status_code=202,
            content={
                "job_id":       job_id,
                "status":       job["status"],
                "progress_pct": job["progress_pct"],
                "message":      "Processing in progress — please poll again.",
            },
        )

    if job["status"] == "error":
        raise HTTPException(
            status_code=500,
            detail={"job_id": job_id, "error": job.get("error", "Unknown error")},
        )

    return JSONResponse(job["result"])


# ---------------------------------------------------------------------------
# Image endpoints
# ---------------------------------------------------------------------------
@router.get("/image/{job_id}/raw", tags=["Artifacts"])
async def get_raw_image(job_id: str) -> FileResponse:
    """Download the clean (pre-AI) sonar PNG."""
    return _serve_image(job_id, "raw.png")


@router.get("/image/{job_id}/annotated", tags=["Artifacts"])
async def get_annotated_image(job_id: str) -> FileResponse:
    """Download the annotated sonar PNG with bounding boxes."""
    return _serve_image(job_id, "annotated.png")


def _serve_image(job_id: str, filename: str) -> FileResponse:
    _assert_job_done(job_id)
    path = Path(settings.OUTPUT_DIR) / job_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not yet generated.")
    return FileResponse(str(path), media_type="image/png", filename=filename)


# ---------------------------------------------------------------------------
# CSV report download
# ---------------------------------------------------------------------------
@router.get("/report/{job_id}/csv", tags=["Artifacts"])
async def get_csv_report(job_id: str) -> FileResponse:
    """Download the anomaly report as a CSV file."""
    _assert_job_done(job_id)
    path = Path(settings.OUTPUT_DIR) / job_id / "report.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV report not found.")
    return FileResponse(
        str(path),
        media_type="text/csv",
        filename=f"anomaly_report_{job_id[:8]}.csv",
    )


# ---------------------------------------------------------------------------
# Jobs list (admin / debug)
# ---------------------------------------------------------------------------
@router.get("/jobs", tags=["System"])
async def list_all_jobs() -> JSONResponse:
    """Return a summary of all known jobs (most recent first)."""
    jobs = list_jobs()
    summary = [
        {
            "job_id":       j["job_id"],
            "filename":     j["filename"],
            "status":       j["status"],
            "progress_pct": j["progress_pct"],
            "created_at":   j["created_at"],
        }
        for j in reversed(jobs)
    ]
    return JSONResponse({"total": len(summary), "jobs": summary})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _assert_job_done(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] not in ("done", "error"):
        raise HTTPException(
            status_code=202,
            detail=f"Job '{job_id}' is still {job['status']} ({job['progress_pct']}%).",
        )
