"""
Image utility helpers — encoding, decoding, drawing.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def numpy_to_png_bytes(arr: np.ndarray) -> bytes:
    """Encode a NumPy (H, W) or (H, W, 3) array to PNG bytes."""
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if arr.ndim == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def save_png(arr: np.ndarray, path: str | Path) -> Path:
    """Save a NumPy array as a PNG file and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = numpy_to_png_bytes(arr)
    path.write_bytes(data)
    return path


def load_png(path: str | Path) -> np.ndarray:
    """Load a PNG as a BGR NumPy array."""
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def array_to_base64(arr: np.ndarray) -> str:
    """Return a base64-encoded PNG data-URI string."""
    return base64.b64encode(numpy_to_png_bytes(arr)).decode()


def draw_detections(
    image: np.ndarray,
    detections: list,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels on a copy of `image`.

    `detections` is expected to be a list of dicts with keys:
        label, confidence, x_min, y_min, x_max, y_max
    """
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for det in detections:
        x0, y0, x1, y1 = det["x_min"], det["y_min"], det["x_max"], det["y_max"]
        label = det.get("label", "debris")
        conf = det.get("confidence", 0.0)

        # Box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), color, thickness)

        # Label background
        text = f"{label}  {conf:.1%}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(
            annotated,
            (x0, max(0, y0 - th - baseline - 4)),
            (x0 + tw + 4, y0),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            text,
            (x0 + 2, max(th, y0 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def resize_for_inference(
    arr: np.ndarray, max_side: int = 1280
) -> Tuple[np.ndarray, float]:
    """
    Resize the image so its longest side is ≤ `max_side`.
    Returns (resized_array, scale_factor) so bboxes can be un-scaled later.
    """
    h, w = arr.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return arr, scale
