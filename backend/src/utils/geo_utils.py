"""
Geospatial helper utilities.

Implements:
  - Haversine distance between two lat/lon pairs
  - Bearing calculation between two points
  - GPS coordinate offset given a bearing and distance
  - Shadow-to-height formula: H = (L_s × H_t) / R
"""
from __future__ import annotations

import math
from typing import Tuple

EARTH_RADIUS_M = 6_371_000.0  # metres


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def bearing_to(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute initial bearing (degrees, 0 = North, clockwise) from point 1 → 2."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    x = math.sin(Δλ) * math.cos(φ2)
    y = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_coordinate(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> Tuple[float, float]:
    """
    Return a new (lat, lon) that is `distance_m` metres away from
    (lat, lon) in the direction `bearing_deg` (degrees, clockwise from North).
    Uses the spherical Earth approximation — accurate to <0.1% for distances
    up to ~100 km.
    """
    δ = distance_m / EARTH_RADIUS_M          # angular distance in radians
    θ = math.radians(bearing_deg)

    φ1 = math.radians(lat)
    λ1 = math.radians(lon)

    φ2 = math.asin(
        math.sin(φ1) * math.cos(δ) +
        math.cos(φ1) * math.sin(δ) * math.cos(θ)
    )
    λ2 = λ1 + math.atan2(
        math.sin(θ) * math.sin(δ) * math.cos(φ1),
        math.cos(δ) - math.sin(φ1) * math.sin(φ2),
    )
    return math.degrees(φ2), (math.degrees(λ2) + 540) % 360 - 180


def shadow_to_height(
    shadow_pixels: int,
    total_range_pixels: int,
    towfish_altitude_m: float,
    slant_range_m: float,
) -> float:
    """
    Estimate real-world object height from its acoustic shadow.

    Formula: H = (L_s / R_px) × R_m × H_t / sqrt(R_m² - H_t²)

    Where:
      L_s  = shadow length in pixels
      R_px = total across-track pixel count (maps to slant_range_m)
      R_m  = slant range in metres
      H_t  = towfish altitude in metres

    Returns height in metres, or 0 if geometry is degenerate.
    """
    if total_range_pixels <= 0 or slant_range_m <= 0:
        return 0.0

    metres_per_pixel = slant_range_m / total_range_pixels
    shadow_length_m = shadow_pixels * metres_per_pixel

    try:
        across_track_m = math.sqrt(max(slant_range_m ** 2 - towfish_altitude_m ** 2, 0.0))
        if across_track_m == 0:
            return 0.0
        height_m = (shadow_length_m * towfish_altitude_m) / across_track_m
    except (ValueError, ZeroDivisionError):
        height_m = 0.0

    return round(height_m, 3)


def perpendicular_bearing(heading_deg: float, side: str = "port") -> float:
    """
    Return the bearing perpendicular to the vessel heading.
    SSS emits beams to both port (left) and starboard (right).
    """
    offset = -90 if side == "port" else 90
    return (heading_deg + offset) % 360
