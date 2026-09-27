// ============================================================
// Shared TypeScript types for the SIH Marine Debris Detection UI
// ============================================================

// ---- Backend API response types ----

export type JobStatus = 'pending' | 'processing' | 'done' | 'error';

export interface UploadResponse {
  job_id: string;
  status: JobStatus;
  message: string;
  poll_url: string;
}

export interface StatusResponse {
  job_id: string;
  status: JobStatus;
  progress_pct: number;
  current_stage: string;
  error?: string;
}

export interface BoundingBox {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  width_px: number;
  height_px: number;
}

export interface PhysicalDimensions {
  height_m: number | null;
  width_m: number | null;
}

export interface Location {
  latitude: number | null;
  longitude: number | null;
}

export interface Detection {
  id: string;
  label: string;
  confidence_pct: number;
  bounding_box: BoundingBox;
  physical_dimensions: PhysicalDimensions;
  acoustic_shadow_pixels: number | null;
  across_track_m: number | null;
  location: Location;
}

export interface Telemetry {
  source_latitude: number | null;
  source_longitude: number | null;
  heading_deg: number | null;
  altitude_m: number | null;
  slant_range_m: number | null;
  ping_count: number | null;
  timestamp: string | null;
  synthetic_data: boolean;
}

export interface Artifacts {
  raw_sonar_png: string;
  annotated_png: string;
  csv_report: string;
}

export interface ReportMeta {
  job_id: string;
  source_filename: string;
  processed_at: string;
  model_version: string;
  pipeline_stages: string[];
  total_detections: number;
  processing_time_s?: number;
}

export interface AnalysisResult {
  meta: ReportMeta;
  telemetry: Telemetry;
  detections: Detection[];
  artifacts: Artifacts;
}

export interface JobSummary {
  job_id: string;
  filename: string;
  status: JobStatus;
  progress_pct: number;
  created_at: string;
}

export interface JobsListResponse {
  total: number;
  jobs: JobSummary[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_path: string;
  output_dir: string;
}

// ---- UI state types ----

export type UploadState =
  | { phase: 'idle' }
  | { phase: 'uploading'; progress: number }
  | { phase: 'processing'; jobId: string; progress: number; stage: string }
  | { phase: 'done'; jobId: string; result: AnalysisResult }
  | { phase: 'error'; message: string };

export const PIPELINE_STAGES: { name: string; pct: number }[] = [
  { name: 'Ingesting sonar file', pct: 5 },
  { name: 'Applying Lee filter & CLAHE enhancement', pct: 25 },
  { name: 'Running YOLOv8 inference', pct: 50 },
  { name: 'Computing GPS coordinates', pct: 70 },
  { name: 'Rendering annotated image', pct: 85 },
  { name: 'Generating JSON & CSV reports', pct: 95 },
  { name: 'Complete', pct: 100 },
];
