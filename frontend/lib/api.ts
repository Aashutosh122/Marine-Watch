// ============================================================
// Backend API client — all communication with the FastAPI server
// ============================================================

import type {
  AnalysisResult,
  HealthResponse,
  JobsListResponse,
  StatusResponse,
  UploadResponse,
} from './types';

// ---- Configuration ----
// The backend runs on http://localhost:8000 by default.
// Override via NEXT_PUBLIC_API_URL environment variable.
const API_BASE =
  (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL) ||
  'http://localhost:8000';

const API_PREFIX = `${API_BASE}/api/v1`;

// ---- Helpers ----

async function apiFetch<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const url = `${API_PREFIX}${path}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      ...(options?.headers ?? {}),
    },
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore parse errors
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  return res.json() as Promise<T>;
}

// ---- Public API surface ----

/**
 * Check if the backend is reachable and the model is loaded.
 */
export async function healthCheck(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health');
}

/**
 * Upload a sonar file (any format — XTF or fallback synthetic).
 * Returns a job_id to poll for progress.
 */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_PREFIX}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    let detail = `Upload failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  return res.json() as Promise<UploadResponse>;
}

/**
 * Poll a job's current status and progress percentage.
 */
export async function pollStatus(jobId: string): Promise<StatusResponse> {
  return apiFetch<StatusResponse>(`/status/${jobId}`);
}

/**
 * Fetch the full analysis result once a job is done.
 * Returns 202 if still processing (caller should retry).
 */
export async function getResults(jobId: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_PREFIX}/results/${jobId}`);

  if (res.status === 202) {
    // Still processing — caller handles this
    const body = await res.json();
    throw new Error(`still_processing:${body.progress_pct ?? 0}`);
  }

  if (!res.ok) {
    let detail = `Results fetch failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // ignore
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }

  return res.json() as Promise<AnalysisResult>;
}

/**
 * Get the URL for an image artifact (raw or annotated sonar PNG).
 * These are served as static files directly from the backend.
 */
export function getImageUrl(
  jobId: string,
  type: 'raw' | 'annotated',
): string {
  return `${API_BASE}/outputs/${jobId}/${type}.png`;
}

/**
 * Get the direct URL to download the CSV report for a job.
 */
export function getCsvUrl(jobId: string): string {
  return `${API_PREFIX}/report/${jobId}/csv`;
}

/**
 * Trigger a browser download of the CSV report.
 */
export function downloadCsv(jobId: string, filename?: string): void {
  const a = document.createElement('a');
  a.href = getCsvUrl(jobId);
  a.download = filename ?? `anomaly_report_${jobId.slice(0, 8)}.csv`;
  a.click();
}

/**
 * List all past jobs (admin/debug view).
 */
export async function listJobs(): Promise<JobsListResponse> {
  return apiFetch<JobsListResponse>('/jobs');
}

/**
 * Poll a job until it completes, calling onProgress at each step.
 * Resolves with the final AnalysisResult on success.
 * Rejects on error or timeout.
 */
export async function waitForJob(
  jobId: string,
  onProgress: (pct: number, stage: string) => void,
  intervalMs = 1500,
  timeoutMs = 10 * 60 * 1000, // 10-minute timeout
): Promise<AnalysisResult> {
  const deadline = Date.now() + timeoutMs;

  return new Promise((resolve, reject) => {
    const tick = async () => {
      if (Date.now() > deadline) {
        reject(new Error('Timed out waiting for pipeline to complete.'));
        return;
      }

      try {
        const status = await pollStatus(jobId);

        onProgress(status.progress_pct, status.current_stage);

        if (status.status === 'error') {
          reject(new Error(status.error ?? 'Pipeline error'));
          return;
        }

        if (status.status === 'done') {
          const result = await getResults(jobId);
          resolve(result);
          return;
        }

        // Still pending/processing — schedule next poll
        setTimeout(() => void tick(), intervalMs);
      } catch (err) {
        reject(err);
      }
    };

    void tick();
  });
}
