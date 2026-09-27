"""
Stage 5 — Anomaly Reporting
============================
Serializes the pipeline output into:
  - A structured JSON report (machine-readable, API response)
  - A CSV file (human-readable, download endpoint)

Both files are saved into the job's output directory.
"""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------
def build_json_report(
    job_id: str,
    filename: str,
    telemetry: Dict[str, Any],
    detections: List[Dict[str, Any]],
    model_version: str = "YOLOv8-custom",
    raw_image_url: str = "",
    annotated_image_url: str = "",
    csv_url: str = "",
) -> Dict[str, Any]:
    """
    Assemble a full JSON-serializable report dict.

    The report follows the structure specified in the problem statement:
    each detection includes label, confidence, bounding box, physical
    dimensions, and GPS coordinates.
    """
    report = {
        "meta": {
            "job_id":          job_id,
            "source_filename": filename,
            "processed_at":    datetime.now(tz=timezone.utc).isoformat(),
            "model_version":   model_version,
            "pipeline_stages": [
                "XTF Ingestion",
                "Acoustic Pre-processing (Lee Filter + CLAHE)",
                "AI Inference (YOLOv8)",
                "Geospatial Projection",
                "Report Generation",
            ],
            "total_detections": len(detections),
        },
        "telemetry": {
            "source_latitude":  telemetry.get("latitude"),
            "source_longitude": telemetry.get("longitude"),
            "heading_deg":      telemetry.get("heading_deg"),
            "altitude_m":       telemetry.get("altitude_m"),
            "slant_range_m":    telemetry.get("slant_range_m"),
            "ping_count":       telemetry.get("ping_count"),
            "timestamp":        telemetry.get("timestamp"),
            "synthetic_data":   telemetry.get("_synthetic", False),
        },
        "detections": [
            {
                "id":              d["detection_id"],
                "label":           d["label"],
                "confidence_pct":  round(d["confidence"] * 100, 1),
                "bounding_box": {
                    "x_min":     d["x_min"],
                    "y_min":     d["y_min"],
                    "x_max":     d["x_max"],
                    "y_max":     d["y_max"],
                    "width_px":  d["x_max"] - d["x_min"],
                    "height_px": d["y_max"] - d["y_min"],
                },
                "physical_dimensions": {
                    "height_m": d.get("estimated_height_m"),
                    "width_m":  d.get("estimated_width_m"),
                },
                "acoustic_shadow_pixels": d.get("shadow_pixel_count"),
                "across_track_m":         d.get("across_track_m"),
                "location": {
                    "latitude":  d.get("latitude"),
                    "longitude": d.get("longitude"),
                },
            }
            for d in detections
        ],
        "artifacts": {
            "raw_sonar_png":     raw_image_url,
            "annotated_png":     annotated_image_url,
            "csv_report":        csv_url,
        },
    }
    return report


def save_json_report(report: Dict[str, Any], output_dir: Path) -> Path:
    """Write the JSON report to disk and return its path."""
    path = output_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("JSON report saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------
_CSV_FIELDS = [
    "id", "label", "confidence_pct",
    "x_min", "y_min", "x_max", "y_max",
    "height_m", "width_m",
    "shadow_pixels", "across_track_m",
    "latitude", "longitude",
]


def save_csv_report(
    detections: List[Dict[str, Any]], job_id: str, output_dir: Path
) -> Path:
    """Write a flat CSV report and return its path."""
    path = output_dir / "report.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for d in detections:
            writer.writerow({
                "id":             d["detection_id"],
                "label":          d["label"],
                "confidence_pct": round(d["confidence"] * 100, 1),
                "x_min":          d["x_min"],
                "y_min":          d["y_min"],
                "x_max":          d["x_max"],
                "y_max":          d["y_max"],
                "height_m":       d.get("estimated_height_m", ""),
                "width_m":        d.get("estimated_width_m", ""),
                "shadow_pixels":  d.get("shadow_pixel_count", ""),
                "across_track_m": d.get("across_track_m", ""),
                "latitude":       d.get("latitude", ""),
                "longitude":      d.get("longitude", ""),
            })
    logger.info("CSV report saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_reports(
    job_id: str,
    filename: str,
    telemetry: Dict[str, Any],
    detections: List[Dict[str, Any]],
    output_dir: Path,
    model_version: str = "YOLOv8-custom",
    base_url: str = "",
) -> Dict[str, Any]:
    """
    Generate and persist both JSON and CSV reports.

    Returns the full JSON report dict.
    """
    logger.info("Stage 5 — generating reports for job %s.", job_id)

    raw_url        = f"{base_url}/outputs/{job_id}/raw.png"
    annotated_url  = f"{base_url}/outputs/{job_id}/annotated.png"
    csv_url        = f"{base_url}/api/v1/report/{job_id}/csv"

    report = build_json_report(
        job_id=job_id,
        filename=filename,
        telemetry=telemetry,
        detections=detections,
        model_version=model_version,
        raw_image_url=raw_url,
        annotated_image_url=annotated_url,
        csv_url=csv_url,
    )

    save_json_report(report, output_dir)
    save_csv_report(detections, job_id, output_dir)

    logger.info("Stage 5 complete — reports saved to %s.", output_dir)
    return report
