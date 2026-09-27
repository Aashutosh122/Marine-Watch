"""
Stage 1 — XTF Ingestion & Telemetry Extraction
===============================================
Reads a binary .xtf file using `pyxtf` and returns:
  - A 2D NumPy array (waterfall sonar image)
  - A telemetry dictionary with GPS, heading, altitude, and timestamps

If the file cannot be parsed (e.g. during testing with a dummy file),
a synthetic sonar waterfall is generated so the rest of the pipeline
can run end-to-end.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
TelemetryDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def ingest_xtf(file_path: str | Path) -> Tuple[np.ndarray, TelemetryDict]:
    """
    Parse an XTF file and return (waterfall_array, telemetry_dict).

    Falls back to a synthetic sonar image if pyxtf cannot parse the file.
    """
    file_path = Path(file_path)
    logger.info("Stage 1 — ingesting %s (%.1f MB)", file_path.name, file_path.stat().st_size / 1e6)

    try:
        return _parse_xtf(file_path)
    except Exception as exc:
        logger.warning("pyxtf parsing failed (%s) — generating synthetic sonar image.", exc)
        return _synthetic_sonar(file_path.name)


# ---------------------------------------------------------------------------
# Real XTF parser
# ---------------------------------------------------------------------------
def _parse_xtf(file_path: Path) -> Tuple[np.ndarray, TelemetryDict]:
    """Use pyxtf to parse a real XTF file."""
    try:
        import pyxtf
    except ImportError as exc:
        raise ImportError("pyxtf is not installed — run: pip install pyxtf") from exc

    (file_header, packets) = pyxtf.xtf_read(str(file_path))

    # -----------------------------------------------------------------------
    # Build waterfall image from sonar packets
    # -----------------------------------------------------------------------
    pings_port: list[np.ndarray] = []
    pings_stbd: list[np.ndarray] = []

    # Telemetry accumulators
    latitudes: list[float] = []
    longitudes: list[float] = []
    headings: list[float] = []
    altitudes: list[float] = []
    timestamps: list[str] = []

    for packet in packets.get(pyxtf.XtfPacketType.XtfHeaderSonar, []):
        for ping in packet:
            # Acoustic data — normalise each channel
            data = np.array(ping.data, dtype=np.float32)
            if ping.ping_channel == 0:          # port
                pings_port.append(data)
            else:                               # starboard
                pings_stbd.append(data)

            # Telemetry — prefer NMEA fields embedded in the ping header
            lat = getattr(ping, "SensorXcoord", None) or getattr(ping, "Latitude", None)
            lon = getattr(ping, "SensorYcoord", None) or getattr(ping, "Longitude", None)
            hdg = getattr(ping, "SensorHeading", None)
            alt = getattr(ping, "SensorPrimaryAltitude", None)
            ts  = getattr(ping, "TimeTag", None)

            if lat is not None:
                latitudes.append(float(lat))
            if lon is not None:
                longitudes.append(float(lon))
            if hdg is not None:
                headings.append(float(hdg))
            if alt is not None:
                altitudes.append(float(alt))
            if ts is not None:
                timestamps.append(str(ts))

    # Stack channels side-by-side (port reversed | starboard)
    waterfall = _stack_channels(pings_port, pings_stbd)

    telemetry = _summarise_telemetry(
        latitudes, longitudes, headings, altitudes, timestamps,
        ping_count=len(pings_port) + len(pings_stbd),
    )

    logger.info(
        "Stage 1 complete — waterfall shape %s, %d detectable pings.",
        waterfall.shape, telemetry.get("ping_count", 0),
    )
    return waterfall, telemetry


def _stack_channels(
    port: list[np.ndarray],
    stbd: list[np.ndarray],
) -> np.ndarray:
    """
    Combine port and starboard ping lists into a single 2D waterfall.
    Port channel is horizontally flipped (convention: nadir in centre).
    """
    if port and stbd:
        n_pings = min(len(port), len(stbd))
        max_samples = max(p.shape[0] for p in port[:n_pings] + stbd[:n_pings])

        def _pad(lst: list[np.ndarray]) -> np.ndarray:
            padded = [
                np.pad(p, (0, max_samples - len(p))) for p in lst[:n_pings]
            ]
            return np.stack(padded, axis=0)  # (n_pings, max_samples)

        p_arr = _pad(port)[:, ::-1]   # flip port so nadir is right-edge
        s_arr = _pad(stbd)
        return np.concatenate([p_arr, s_arr], axis=1)   # (n_pings, 2×samples)

    # Mono channel
    pings = port or stbd
    if not pings:
        raise ValueError("No sonar pings found in XTF file.")
    max_samples = max(p.shape[0] for p in pings)
    padded = [np.pad(p, (0, max_samples - len(p))) for p in pings]
    return np.stack(padded, axis=0)


def _summarise_telemetry(
    lats: list, lons: list, hdgs: list, alts: list, ts: list, ping_count: int
) -> TelemetryDict:
    def _avg(lst: list) -> Optional[float]:
        return round(float(np.mean(lst)), 6) if lst else None

    return {
        "latitude":       _avg(lats),
        "longitude":      _avg(lons),
        "heading_deg":    _avg(hdgs),
        "altitude_m":     _avg(alts),
        "slant_range_m":  None,          # computed from geometry if needed
        "ping_count":     ping_count,
        "timestamp":      ts[0] if ts else None,
    }


# ---------------------------------------------------------------------------
# Synthetic sonar fallback (for testing / demo)
# ---------------------------------------------------------------------------
def _synthetic_sonar(filename: str) -> Tuple[np.ndarray, TelemetryDict]:
    """
    Generate a realistic-looking synthetic SSS waterfall with:
      - Gaussian speckle noise (simulates underwater multi-path)
      - A dark nadir band
      - 2-3 bright anomaly blobs with trailing shadows (simulates debris)
    """
    rng = np.random.default_rng(42)

    height, width = 1024, 1500   # typical SSS waterfall dimensions
    image = rng.normal(loc=80, scale=25, size=(height, width)).astype(np.float32)

    # Nadir (centre) = very dark
    nadir_w = 30
    cx = width // 2
    image[:, cx - nadir_w : cx + nadir_w] = rng.normal(5, 3, (height, 2 * nadir_w))

    # Sediment texture — low-frequency variation
    for _ in range(6):
        cx_s = rng.integers(100, width - 100)
        cy_s = rng.integers(100, height - 100)
        sigma_x = rng.integers(60, 200)
        sigma_y = rng.integers(10, 40)
        xs = np.arange(width)
        ys = np.arange(height)
        XX, YY = np.meshgrid(xs, ys)
        blob = 30 * np.exp(-((XX - cx_s) ** 2 / (2 * sigma_x ** 2) + (YY - cy_s) ** 2 / (2 * sigma_y ** 2)))
        image += blob.astype(np.float32)

    # --- Debris anomalies ---------------------------------------------------
    anomalies = [
        {"cx": 380,  "cy": 200, "w": 35, "h": 20, "shadow_len": 55},
        {"cx": 1100, "cy": 600, "w": 45, "h": 25, "shadow_len": 70},
        {"cx": 700,  "cy": 820, "w": 20, "h": 12, "shadow_len": 40},
    ]
    for a in anomalies:
        # Bright highlight (acoustic reflection)
        x0, y0 = a["cx"] - a["w"] // 2, a["cy"] - a["h"] // 2
        x1, y1 = x0 + a["w"],           y0 + a["h"]
        image[y0:y1, x0:x1] = 220 + rng.normal(0, 5, (y1 - y0, x1 - x0))

        # Dark acoustic shadow directly to the right of the object
        sx0, sx1 = x1, x1 + a["shadow_len"]
        image[y0:y1, sx0:min(sx1, width)] = rng.normal(8, 3, (y1 - y0, min(sx1, width) - sx0))

    # Clip and return
    image = np.clip(image, 0, 255)

    telemetry: TelemetryDict = {
        "latitude":       12.9716,     # Bangalore, just as a demo coord
        "longitude":      77.5946,
        "heading_deg":    045.0,
        "altitude_m":     3.5,
        "slant_range_m":  30.0,
        "ping_count":     height,
        "timestamp":      "2024-01-15T08:30:00Z",
        "_synthetic":     True,
        "source_file":    filename,
    }

    logger.info("Stage 1 complete (synthetic) — waterfall shape %s.", image.shape)
    return image, telemetry
