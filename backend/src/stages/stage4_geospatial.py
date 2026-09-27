"""
Stage 4 — Spatial Telemetry & Geospatial Projection
=====================================================
For every detection bounding box this stage:

  1. Counts shadow pixels (dark pixels inside the bbox) to estimate
     the object's acoustic shadow length (L_s).

  2. Applies the sonar geometry formula to compute the real-world
     height of the object in metres:
         H = (L_s × H_t) / across_track_distance

  3. Computes the across-track distance from the towfish to the object
     using the pixel position relative to nadir.

  4. Projects the towfish GPS coordinate outward by that distance,
     perpendicular to the vessel heading, to obtain the object's
     real-world Latitude / Longitude.

Telemetry fallbacks are used when the XTF file doesn't contain valid
navigation data (e.g. during demo / testing with synthetic images).
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import settings
from src.utils.geo_utils import (
    offset_coordinate,
    perpendicular_bearing,
    shadow_to_height,
)

logger = logging.getLogger(__name__)

# Shadow threshold: pixels darker than this fraction of max intensity
# are counted as "acoustic shadow"
_SHADOW_THRESHOLD_FRAC = 0.20


def _count_shadow_pixels(
    image: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    threshold_frac: float = _SHADOW_THRESHOLD_FRAC,
) -> int:
    """Count pixels inside the bbox that are below the shadow threshold."""
    roi = image[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return 0
    max_val = roi.max()
    if max_val == 0:
        return roi.size
    threshold = max_val * threshold_frac
    return int(np.sum(roi < threshold))


def _pixel_to_across_track_m(
    x_center_px: float,
    image_width_px: int,
    slant_range_m: float,
    towfish_altitude_m: float,
) -> float:
    """
    Convert the horizontal pixel position of a detection to a real-world
    across-track distance from the nadir (in metres).

    The SSS waterfall maps as:
        pixel 0           → far range (port side)
        pixel width // 2  → nadir (directly below towfish)
        pixel width - 1   → far range (starboard side)
    """
    nadir_px = image_width_px / 2.0
    frac_from_nadir = abs(x_center_px - nadir_px) / (nadir_px + 1e-6)

    # Max horizontal range from nadir = sqrt(slant_range² - altitude²)
    try:
        max_across_m = math.sqrt(max(slant_range_m ** 2 - towfish_altitude_m ** 2, 0.0))
    except ValueError:
        max_across_m = slant_range_m

    return frac_from_nadir * max_across_m


def _estimate_width_m(
    bbox_width_px: int,
    image_width_px: int,
    slant_range_m: float,
    towfish_altitude_m: float,
) -> float:
    """Estimate physical width of the detection in metres."""
    try:
        max_across_m = math.sqrt(max(slant_range_m ** 2 - towfish_altitude_m ** 2, 0.0))
    except ValueError:
        max_across_m = slant_range_m
    metres_per_pixel = (2 * max_across_m) / (image_width_px + 1e-6)
    return round(bbox_width_px * metres_per_pixel, 3)


def geoproject_detections(
    detections: List[Dict[str, Any]],
    processed_image: np.ndarray,
    telemetry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Enrich each detection dict with geospatial information.

    Added keys per detection:
      - shadow_pixel_count: int
      - estimated_height_m: float
      - estimated_width_m:  float
      - across_track_m:     float
      - latitude:           float | None
      - longitude:          float | None

    Args:
        detections:       Output list from Stage 3
        processed_image:  Grayscale uint8 array from Stage 2
        telemetry:        Dict from Stage 1

    Returns:
        Enriched detection list.
    """
    logger.info("Stage 4 — geo-projecting %d detections.", len(detections))

    # Pull telemetry values, falling back to configured defaults
    lat:        Optional[float] = telemetry.get("latitude")
    lon:        Optional[float] = telemetry.get("longitude")
    heading:    float           = float(telemetry.get("heading_deg") or 0.0)
    altitude_m: float           = float(telemetry.get("altitude_m") or settings.DEFAULT_ALTITUDE_M)
    slant_m:    float           = float(telemetry.get("slant_range_m") or settings.DEFAULT_SLANT_RANGE_M)

    image_h, image_w = processed_image.shape[:2]

    enriched: List[Dict[str, Any]] = []
    for det in detections:
        x_min, y_min = det["x_min"], det["y_min"]
        x_max, y_max = det["x_max"], det["y_max"]

        x_center = (x_min + x_max) / 2.0
        bbox_w_px = x_max - x_min
        bbox_h_px = y_max - y_min

        # ---- Shadow analysis ------------------------------------------------
        shadow_px = _count_shadow_pixels(
            processed_image, x_min, y_min, x_max, y_max
        )

        # ---- Height calculation ---------------------------------------------
        height_m = shadow_to_height(
            shadow_pixels=shadow_px,
            total_range_pixels=image_w,
            towfish_altitude_m=altitude_m,
            slant_range_m=slant_m,
        )

        # ---- Width estimation -----------------------------------------------
        width_m = _estimate_width_m(bbox_w_px, image_w, slant_m, altitude_m)

        # ---- Across-track distance ------------------------------------------
        across_track_m = _pixel_to_across_track_m(
            x_center, image_w, slant_m, altitude_m
        )

        # ---- GPS projection -------------------------------------------------
        obj_lat: Optional[float] = None
        obj_lon: Optional[float] = None

        if lat is not None and lon is not None:
            # Determine which side of nadir the detection is on
            side = "port" if x_center < image_w / 2 else "starboard"
            bearing = perpendicular_bearing(heading, side=side)
            obj_lat, obj_lon = offset_coordinate(lat, lon, bearing, across_track_m)
            obj_lat  = round(obj_lat, 7)
            obj_lon  = round(obj_lon, 7)

        enriched.append({
            **det,
            "shadow_pixel_count":  shadow_px,
            "estimated_height_m":  height_m,
            "estimated_width_m":   width_m,
            "across_track_m":      round(across_track_m, 2),
            "latitude":            obj_lat,
            "longitude":           obj_lon,
        })

    logger.info("Stage 4 complete — all detections geo-projected.")
    return enriched
