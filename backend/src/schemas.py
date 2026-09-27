"""
Shared Pydantic schemas used across the API layer.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: int = Field(ge=0, le=100)
    current_stage: str
    message: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------
class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width_px: int
    height_px: int


class GeoLocation(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    depth_m: Optional[float] = None


class Detection(BaseModel):
    detection_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: BoundingBox
    estimated_height_m: Optional[float] = None
    estimated_width_m: Optional[float] = None
    location: Optional[GeoLocation] = None
    shadow_pixel_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Full pipeline result
# ---------------------------------------------------------------------------
class TelemetryData(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    heading_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    slant_range_m: Optional[float] = None
    ping_count: Optional[int] = None
    timestamp: Optional[str] = None


class PipelineResult(BaseModel):
    job_id: str
    filename: str
    processed_at: datetime
    model_version: str
    total_detections: int
    telemetry: TelemetryData
    detections: List[Detection]
    raw_image_url: Optional[str] = None
    annotated_image_url: Optional[str] = None
    report_csv_url: Optional[str] = None
