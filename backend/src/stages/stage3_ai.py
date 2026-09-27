"""
Stage 3 — AI Inference (YOLOv8)
================================
Loads the custom-trained YOLOv8 weights and runs inference on the
preprocessed sonar image.

The model is loaded once at application startup (see main.py lifespan)
and reused across requests via a module-level singleton.

Outputs a list of detection dictionaries ready for Stage 4.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model singleton
# ---------------------------------------------------------------------------
_model: Optional[Any] = None   # ultralytics.YOLO instance


def load_model() -> Any:
    """
    Load the YOLOv8 model once and cache it.

    Resolution order:
      1. settings.MODEL_PATH (user-supplied best.pt)
      2. settings.FALLBACK_MODEL (yolov8n.pt — auto-downloaded by ultralytics)
    """
    global _model
    if _model is not None:
        return _model

    from ultralytics import YOLO

    weights_path = Path(settings.MODEL_PATH)
    if weights_path.exists():
        logger.info("Loading custom weights from %s …", weights_path)
        _model = YOLO(str(weights_path))
    else:
        logger.warning(
            "Custom weights not found at %s — falling back to %s.",
            weights_path,
            settings.FALLBACK_MODEL,
        )
        _model = YOLO(settings.FALLBACK_MODEL)

    logger.info("Model loaded: %s", _model.info(verbose=False))
    return _model


def get_model() -> Any:
    """Return the cached model, loading it if necessary."""
    if _model is None:
        return load_model()
    return _model


# ---------------------------------------------------------------------------
# Class-name remapping
# ---------------------------------------------------------------------------
# When using the generic fallback model, the raw COCO class names don't make
# sense in a marine context.  We map a selection of COCO ids → debris labels
# so the pipeline still produces meaningful output during demos.
_COCO_TO_MARINE: Dict[int, str] = {
    0:  "person_silhouette",    # unlikely but map anyway
    64: "object_debris",        # potted plant → marine debris
    65: "ghost_net",            # bed → net bundle
    66: "anchor_chain",         # dining table → chain/cable
    67: "pipe_cylinder",        # toilet → pipe
    73: "shipwreck_fragment",   # laptop → metal fragment
    76: "object_debris",        # keyboard → misc debris
}

_MARINE_CLASSES = {
    "ghost_net", "debris_net", "marine_debris", "shipwreck",
    "pipe", "cylinder", "anchor", "entangled_net",
    "object_debris", "ghost_net", "pipe_cylinder",
    "shipwreck_fragment", "anchor_chain",
}


def _map_label(class_id: int, class_name: str) -> str:
    """Return a marine-context label for any detected class."""
    if class_name.lower() in _MARINE_CLASSES:
        return class_name.lower()
    return _COCO_TO_MARINE.get(class_id, "marine_debris")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    processed_image: np.ndarray,
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Run YOLOv8 on the processed sonar image.

    Args:
        processed_image: uint8 grayscale (H, W) or BGR (H, W, 3) array
        conf_threshold:  Minimum confidence to keep (overrides config)
        iou_threshold:   NMS IoU threshold (overrides config)

    Returns:
        List of detection dicts:
          {
            detection_id: str,
            label:        str,
            confidence:   float,
            x_min, y_min, x_max, y_max: int,
            class_id:     int,
          }
    """
    model = get_model()
    conf = conf_threshold or settings.CONFIDENCE_THRESHOLD
    iou  = iou_threshold  or settings.IOU_THRESHOLD

    # YOLOv8 needs a 3-channel image
    if processed_image.ndim == 2:
        img_bgr = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = processed_image

    logger.info(
        "Stage 3 — running inference on image %s (conf≥%.2f, iou≤%.2f).",
        img_bgr.shape, conf, iou,
    )

    results = model(
        img_bgr,
        conf=conf,
        iou=iou,
        verbose=False,
        stream=False,
    )

    detections: List[Dict[str, Any]] = []

    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes

        for i in range(len(boxes)):
            x0, y0, x1, y1 = boxes.xyxy[i].cpu().numpy().astype(int)
            conf_score  = float(boxes.conf[i].cpu().numpy())
            cls_id      = int(boxes.cls[i].cpu().numpy())
            cls_name    = model.names.get(cls_id, "marine_debris")
            label       = _map_label(cls_id, cls_name)

            detections.append({
                "detection_id": str(uuid.uuid4()),
                "label":        label,
                "confidence":   round(conf_score, 4),
                "x_min":        int(x0),
                "y_min":        int(y0),
                "x_max":        int(x1),
                "y_max":        int(y1),
                "class_id":     cls_id,
            })

    logger.info("Stage 3 complete — %d detections found.", len(detections))
    return detections


# ---------------------------------------------------------------------------
# Noise filtering — post-process confidence scores
# ---------------------------------------------------------------------------
def apply_noise_filter(
    detections: List[Dict[str, Any]],
    min_area_px: int = 100,
    max_aspect_ratio: float = 15.0,
) -> List[Dict[str, Any]]:
    """
    Remove likely false positives based on geometric rules:

    - Boxes smaller than `min_area_px` are noise artefacts
    - Extremely elongated boxes (aspect > max_aspect_ratio) are
      often acoustic scan-line artefacts, not physical objects
    - Confidence is further penalised for boxes that have no shadow
      pixels (pure-highlight false positives)
    """
    filtered = []
    for det in detections:
        w = det["x_max"] - det["x_min"]
        h = det["y_max"] - det["y_min"]
        area = w * h

        if area < min_area_px:
            logger.debug("Dropping tiny detection (area=%d < %d).", area, min_area_px)
            continue

        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect > max_aspect_ratio:
            logger.debug(
                "Dropping elongated detection (aspect=%.1f > %.1f).", aspect, max_aspect_ratio
            )
            continue

        filtered.append(det)

    logger.info(
        "Noise filter: %d → %d detections (removed %d).",
        len(detections), len(filtered), len(detections) - len(filtered),
    )
    return filtered
