# Backend API - Marine Debris Detection

This directory contains the Python-based backend for the Marine Debris Detection System. It is built using FastAPI and integrates computer vision models (such as YOLOv8) to process Side-Scan Sonar (SSS) imagery.

## Overview

The backend orchestrates a multi-stage pipeline:
1. **Ingestion:** Parses XTF binaries or direct imagery to extract 2D sonar waterfalls and telemetry.
2. **Preprocessing:** Applies noise reduction filters (e.g., Lee adaptive filter) and CLAHE for image enhancement.
3. **AI Inference:** Runs object detection using YOLOv8 to identify man-made anomalies.
4. **Geospatial Processing:** Calculates physical dimensions (shadow height) and maps bounding boxes to GPS coordinates.
5. **Reporting:** Generates structured JSON and CSV reports with detection metadata.

## Setup & Run

### Requirements
- Python 3.8+

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and update the necessary variables:
- `MODEL_PATH`: Path to your trained `best.pt` weights.
- `CONFIDENCE_THRESHOLD`: Minimum confidence threshold for AI detections.

### Running the Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API documentation (Swagger UI) will be available at `http://localhost:8000/docs`.
