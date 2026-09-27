'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Anchor,
  CheckCircle2,
  ChevronDown,
  Clock,
  Download,
  ExternalLink,
  FileJson,
  Gauge,
  Globe,
  Info,
  Layers,
  MapPin,
  Radar,
  Radio,
  RefreshCw,
  Satellite,
  Shield,
  Sonar,
  Target,
  Upload,
  Waves,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react';

import { downloadCsv, getCsvUrl, getImageUrl, healthCheck, listJobs, uploadFile, waitForJob } from '@/lib/api';
import type { AnalysisResult, Detection, JobSummary, UploadState } from '@/lib/types';
import { PIPELINE_STAGES } from '@/lib/types';

// ─── helpers ───────────────────────────────────────────────────────────────

function fmtPct(n: number) {
  return `${n.toFixed(1)}%`;
}
function fmtCoord(v: number | null | undefined, digits = 6) {
  if (v == null) return '—';
  return v.toFixed(digits);
}
function fmtMetres(v: number | null | undefined) {
  if (v == null) return '—';
  return `${v.toFixed(2)} m`;
}
function labelColor(label: string) {
  if (/ghost|net/i.test(label)) return '#f0b765';
  if (/ship|wreck/i.test(label)) return '#e05c6e';
  if (/pipe|cylinder/i.test(label)) return '#a78bfa';
  if (/anchor|chain/i.test(label)) return '#38bdf8';
  if (/debris|object/i.test(label)) return '#34d399';
  return '#56dac2';
}
function labelIcon(label: string) {
  if (/ghost|net/i.test(label)) return '🕸️';
  if (/ship|wreck/i.test(label)) return '🚢';
  if (/pipe|cylinder/i.test(label)) return '🔩';
  if (/anchor|chain/i.test(label)) return '⚓';
  return '🔍';
}

// ─── sub-components ────────────────────────────────────────────────────────

function OceanBg() {
  return (
    <div className="ocean-ambience" aria-hidden="true">
      <div className="ocean-rays" />
      <div className="ocean-current" />
      <div className="ocean-shimmer" />
    </div>
  );
}

function SonarRing({ active }: { active: boolean }) {
  return (
    <div className={`sonar-ring-wrap ${active ? 'active' : ''}`}>
      <div className="sonar-ring r1" />
      <div className="sonar-ring r2" />
      <div className="sonar-ring r3" />
      <div className="sonar-sweep" />
      <Radar size={38} className="sonar-center-icon" />
    </div>
  );
}

function HealthBadge({ online }: { online: boolean | null }) {
  if (online === null) return <span className="hbadge checking"><Radio size={12} />Checking…</span>;
  if (online) return <span className="hbadge online"><Wifi size={12} />Backend Online</span>;
  return <span className="hbadge offline"><WifiOff size={12} />Backend Offline</span>;
}

function PipelineProgress({ pct, stage }: { pct: number; stage: string }) {
  return (
    <div className="pipeline-wrap">
      <div className="pipeline-header">
        <span className="pipeline-stage-name"><Activity size={14} />{stage}</span>
        <span className="pipeline-pct">{pct}%</span>
      </div>
      <div className="pipeline-bar-track">
        <div className="pipeline-bar-fill" style={{ width: `${pct}%` }} />
        <div className="pipeline-bar-shimmer" style={{ left: `${pct - 10}%` }} />
      </div>
      <div className="pipeline-stages-row">
        {PIPELINE_STAGES.map((s) => (
          <div
            key={s.pct}
            className={`pipeline-dot ${pct >= s.pct ? 'done' : pct > s.pct - 15 ? 'active' : ''}`}
            title={s.name}
          />
        ))}
      </div>
    </div>
  );
}

function StatCard({ label, value, icon, accent }: { label: string; value: string | number; icon: React.ReactNode; accent?: string }) {
  return (
    <div className="stat-card" style={accent ? { borderLeftColor: accent } as React.CSSProperties : {}}>
      <div className="stat-icon">{icon}</div>
      <div className="stat-body">
        <span className="stat-label">{label}</span>
        <strong className="stat-value">{value}</strong>
      </div>
    </div>
  );
}

function UploadZone({
  onFile,
  disabled,
}: {
  onFile: (file: File) => void;
  disabled: boolean;
}) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDrag(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  return (
    <div
      id="upload-zone"
      className={`upload-zone ${drag ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={disabled ? undefined : handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      role="button"
      tabIndex={disabled ? -1 : 0}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }}
      aria-label="Upload sonar file"
    >
      <input
        ref={inputRef}
        type="file"
        hidden
        accept=".xtf,.bin,.dat,.raw,.png,.jpg,.jpeg,.webp"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) { onFile(f); e.target.value = ''; } }}
      />
      <div className="upload-zone-content">
        <Upload size={40} className="upload-icon" />
        <h3 className="upload-title">Drop your sonar file here</h3>
        <p className="upload-sub">Supports <code>.xtf</code>, <code>.bin</code>, <code>.dat</code>, <code>.raw</code> — or any image for demo mode</p>
        <div className="upload-btn">Browse files</div>
      </div>
      <div className="upload-corner tl" /><div className="upload-corner tr" />
      <div className="upload-corner bl" /><div className="upload-corner br" />
    </div>
  );
}

function DetectionBadge({ det, selected, onSelect }: { det: Detection; selected: boolean; onSelect: () => void }) {
  const color = labelColor(det.label);
  return (
    <button
      className={`det-badge ${selected ? 'selected' : ''}`}
      style={{ '--det-color': color } as React.CSSProperties}
      onClick={onSelect}
      aria-pressed={selected}
    >
      <span className="det-badge-icon">{labelIcon(det.label)}</span>
      <span className="det-badge-label">{det.label.replace(/_/g, ' ')}</span>
      <span className="det-badge-conf">{fmtPct(det.confidence_pct)}</span>
    </button>
  );
}

// ─── main page ─────────────────────────────────────────────────────────────

export default function Home() {
  const [state, setState] = useState<UploadState>({ phase: 'idle' });
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [threshold, setThreshold] = useState(25);
  const [selectedDet, setSelectedDet] = useState<Detection | null>(null);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [showJobs, setShowJobs] = useState(false);
  const [annotatedError, setAnnotatedError] = useState(false);
  const [rawTab, setRawTab] = useState<'annotated' | 'raw'>('annotated');

  // ── Backend health check on mount ──────────────────────────────────────
  useEffect(() => {
    healthCheck()
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  // ── Load jobs list ──────────────────────────────────────────────────────
  const refreshJobs = useCallback(() => {
    listJobs()
      .then((r) => setJobs(r.jobs))
      .catch(() => {/* ignore if offline */});
  }, []);

  useEffect(() => { refreshJobs(); }, [refreshJobs]);

  // ── Upload + pipeline flow ─────────────────────────────────────────────
  const handleFile = useCallback(async (file: File) => {
    setAnnotatedError(false);
    setSelectedDet(null);

    setState({ phase: 'uploading', progress: 0 });

    try {
      const { job_id } = await uploadFile(file);

      setState({ phase: 'processing', jobId: job_id, progress: 5, stage: 'Ingesting sonar file' });

      const result = await waitForJob(
        job_id,
        (pct, stage) => setState({ phase: 'processing', jobId: job_id, progress: pct, stage }),
      );

      setState({ phase: 'done', jobId: job_id, result });
      refreshJobs();
    } catch (err) {
      setState({ phase: 'error', message: (err as Error).message });
    }
  }, [refreshJobs]);

  // ── Derived state ──────────────────────────────────────────────────────
  const result = state.phase === 'done' ? state.result : null;
  const jobId = state.phase === 'done' || state.phase === 'processing' ? (state as { jobId: string }).jobId : null;

  const visibleDetections = result
    ? result.detections.filter((d) => d.confidence_pct >= threshold)
    : [];

  const imageUrl = jobId ? getImageUrl(jobId, rawTab) : null;

  // ─────────────────────────────────────────────────────────────────────
  return (
    <div className="ocean-app" id="marine-watch-app">
      <OceanBg />

      {/* ── HEADER ── */}
      <header id="app-header">
        <div className="brand">
          <Radar size={28} className="brand-icon" />
          <span className="brand-name">Marine<span className="brand-accent">Watch</span></span>
          <span className="brand-sub">Sonar Intelligence Platform</span>
        </div>
        <div className="header-right">
          <HealthBadge online={backendOnline} />
          <div className="header-badge">
            <Shield size={13} />SIH 2026
          </div>
          <div className="header-badge accent">
            <Zap size={13} />YOLOv8 + SSS
          </div>
        </div>
      </header>

      {/* ── MAIN ── */}
      <main className="dashboard" id="main-content">

        {/* ─ Hero strip ─ */}
        <section className="hero-strip" id="hero-section">
          <div className="hero-left">
            <div className="eyebrow"><Satellite size={13} />Ghost Net & Marine Debris Detection</div>
            <h1 className="hero-title">
              Scan the ocean floor.<br />
              <span className="hero-title-accent">Detect what hides in the dark.</span>
            </h1>
            <p className="hero-desc">
              Upload a Side-Scan Sonar file and let the AI pipeline detect ghost nets, shipwrecks,
              pipes, and man-made debris — with GPS-accurate geotagging and a full anomaly report.
            </p>
          </div>
          <div className="hero-right">
            <SonarRing active={state.phase === 'processing'} />
          </div>
        </section>

        {/* ─ Upload / Status ─ */}
        <section className="upload-section" id="upload-section">
          {(state.phase === 'idle' || state.phase === 'error') && (
            <>
              <UploadZone onFile={handleFile} disabled={false} />
              {state.phase === 'error' && (
                <div className="toast-error" role="alert">
                  <AlertTriangle size={16} />
                  {state.message}
                  <button className="toast-close" onClick={() => setState({ phase: 'idle' })}>✕</button>
                </div>
              )}
              {backendOnline === false && (
                <div className="toast-warn" role="status">
                  <WifiOff size={15} />
                  Backend offline — start with <code>python -m src.main</code> in the backend folder.
                </div>
              )}
            </>
          )}

          {state.phase === 'uploading' && (
            <div className="status-card" aria-live="polite">
              <div className="status-card-icon uploading"><Upload size={24} /></div>
              <div className="status-card-body">
                <div className="status-card-title">Uploading file…</div>
                <div className="status-card-sub">Sending your sonar data to the processing pipeline</div>
              </div>
              <div className="spinner" />
            </div>
          )}

          {state.phase === 'processing' && (
            <div className="status-card processing" aria-live="polite">
              <div className="status-card-icon processing"><Activity size={24} /></div>
              <div className="status-card-body full-width">
                <div className="status-card-title">Pipeline Running…</div>
                <PipelineProgress pct={state.progress} stage={state.stage} />
              </div>
            </div>
          )}

          {state.phase === 'done' && (
            <div className="status-card done" aria-live="polite">
              <div className="status-card-icon done"><CheckCircle2 size={24} /></div>
              <div className="status-card-body">
                <div className="status-card-title">Analysis Complete</div>
                <div className="status-card-sub">
                  {result?.meta.total_detections ?? 0} anomalies detected
                  {result?.meta.processing_time_s != null && ` in ${result.meta.processing_time_s}s`}
                </div>
              </div>
              <button
                className="ghost-btn"
                onClick={() => { setState({ phase: 'idle' }); setSelectedDet(null); }}
              >
                <RefreshCw size={15} />New scan
              </button>
            </div>
          )}
        </section>

        {/* ─ Stats row ─ */}
        {result && (
          <section className="stats-row" id="stats-section" aria-label="Survey statistics">
            <StatCard
              label="Total Detections"
              value={result.meta.total_detections}
              icon={<Target size={18} />}
              accent="#56dac2"
            />
            <StatCard
              label="Above Threshold"
              value={visibleDetections.length}
              icon={<Gauge size={18} />}
              accent="#f0b765"
            />
            <StatCard
              label="Model"
              value={result.meta.model_version}
              icon={<Layers size={18} />}
              accent="#a78bfa"
            />
            <StatCard
              label="Ping Count"
              value={result.telemetry.ping_count ?? '—'}
              icon={<Radio size={18} />}
              accent="#38bdf8"
            />
            <StatCard
              label="Survey Origin"
              value={
                result.telemetry.source_latitude != null
                  ? `${fmtCoord(result.telemetry.source_latitude, 4)}°, ${fmtCoord(result.telemetry.source_longitude, 4)}°`
                  : 'No GPS'
              }
              icon={<Globe size={18} />}
              accent="#34d399"
            />
            <StatCard
              label="Processing Time"
              value={result.meta.processing_time_s != null ? `${result.meta.processing_time_s}s` : '—'}
              icon={<Clock size={18} />}
              accent="#fb923c"
            />
          </section>
        )}

        {/* ─ Main workspace ─ */}
        {result && (
          <section className="workspace" id="workspace-section">

            {/* Left: sonar image viewer */}
            <div className="panel image-panel" id="sonar-image-panel">
              <div className="panel-head">
                <div className="panel-head-left">
                  <Waves size={16} />
                  <h2>Sonar Image</h2>
                  <span className="panel-filename">{result.meta.source_filename}</span>
                  {result.telemetry.synthetic_data && (
                    <span className="synthetic-badge">Synthetic Demo</span>
                  )}
                </div>
                <div className="panel-head-right">
                  <div className="tab-switcher">
                    <button
                      className={rawTab === 'annotated' ? 'active' : ''}
                      onClick={() => { setRawTab('annotated'); setAnnotatedError(false); }}
                    >Annotated</button>
                    <button
                      className={rawTab === 'raw' ? 'active' : ''}
                      onClick={() => setRawTab('raw')}
                    >Raw</button>
                  </div>
                </div>
              </div>

              <div className="image-viewport" id="sonar-viewport">
                {imageUrl && !annotatedError ? (
                  <div className="image-wrap" id="sonar-image-wrap">
                    <img
                      src={imageUrl}
                      alt={`${rawTab} sonar scan`}
                      onError={() => setAnnotatedError(true)}
                    />
                    {/* Overlay bounding boxes only on annotated tab */}
                    {rawTab === 'annotated' && visibleDetections.map((det) => {
                      // We need image natural dimensions — get from the img element
                      // Boxes are drawn as percentage overlays
                      const imgEl = document.getElementById('sonar-image-wrap')?.querySelector('img');
                      const iw = imgEl?.naturalWidth || 1;
                      const ih = imgEl?.naturalHeight || 1;
                      const bb = det.bounding_box;
                      const color = labelColor(det.label);
                      const isSelected = selectedDet?.id === det.id;
                      return (
                        <button
                          key={det.id}
                          className={`sonar-box ${isSelected ? 'active' : ''}`}
                          style={{
                            left: `${(bb.x_min / iw) * 100}%`,
                            top: `${(bb.y_min / ih) * 100}%`,
                            width: `${((bb.x_max - bb.x_min) / iw) * 100}%`,
                            height: `${((bb.y_max - bb.y_min) / ih) * 100}%`,
                            '--box-color': color,
                          } as React.CSSProperties}
                          onClick={() => setSelectedDet(isSelected ? null : det)}
                          aria-label={`Detection: ${det.label} at ${fmtPct(det.confidence_pct)} confidence`}
                        >
                          <span className="box-label">{labelIcon(det.label)} {fmtPct(det.confidence_pct)}</span>
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <div className="image-empty">
                    <Radar size={52} />
                    <p>{annotatedError ? 'Image not yet ready — try refreshing.' : 'Loading sonar image…'}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Right: contact details */}
            <div className="panel detail-panel" id="contact-detail-panel">
              <div className="panel-head">
                <div className="panel-head-left">
                  <MapPin size={16} />
                  <h2>Contact Details</h2>
                </div>
              </div>

              <div className="detail-body">
                {selectedDet ? (
                  <>
                    <div className="det-label-row">
                      <span className="det-icon-big">{labelIcon(selectedDet.label)}</span>
                      <div>
                        <div className="eyebrow">Detection ID</div>
                        <div className="det-id-short">{selectedDet.id.slice(0, 8).toUpperCase()}</div>
                      </div>
                    </div>

                    <h3 className="det-title" style={{ color: labelColor(selectedDet.label) }}>
                      {selectedDet.label.replace(/_/g, ' ')}
                    </h3>

                    <div className="confidence-ring-wrap">
                      <div
                        className="confidence-ring"
                        style={{
                          '--conf': selectedDet.confidence_pct / 100,
                          '--ring-color': labelColor(selectedDet.label),
                        } as React.CSSProperties}
                      >
                        <span className="confidence-val">{fmtPct(selectedDet.confidence_pct)}</span>
                        <span className="confidence-lbl">confidence</span>
                      </div>
                    </div>

                    <dl className="det-dl">
                      <dt><Layers size={13} />Bounding Box (px)</dt>
                      <dd className="mono">
                        ({selectedDet.bounding_box.x_min}, {selectedDet.bounding_box.y_min}) →
                        ({selectedDet.bounding_box.x_max}, {selectedDet.bounding_box.y_max})
                      </dd>

                      <dt><Gauge size={13} />Size (W × H)</dt>
                      <dd className="mono">
                        {selectedDet.bounding_box.width_px}px × {selectedDet.bounding_box.height_px}px
                      </dd>

                      <dt><Anchor size={13} />Physical Dims</dt>
                      <dd className="mono">
                        W: {fmtMetres(selectedDet.physical_dimensions.width_m)} ·
                        H: {fmtMetres(selectedDet.physical_dimensions.height_m)}
                      </dd>

                      <dt><Waves size={13} />Acoustic Shadow</dt>
                      <dd className="mono">{selectedDet.acoustic_shadow_pixels ?? '—'} px</dd>

                      <dt><Globe size={13} />Across-Track</dt>
                      <dd className="mono">{fmtMetres(selectedDet.across_track_m)}</dd>

                      <dt><MapPin size={13} />GPS Location</dt>
                      <dd className="mono">
                        {selectedDet.location.latitude != null
                          ? `${fmtCoord(selectedDet.location.latitude)}°N, ${fmtCoord(selectedDet.location.longitude)}°E`
                          : 'No GPS data'}
                      </dd>
                    </dl>

                    {selectedDet.location.latitude != null && (
                      <a
                        className="map-link"
                        href={`https://www.openstreetmap.org/?mlat=${selectedDet.location.latitude}&mlon=${selectedDet.location.longitude}#map=16/${selectedDet.location.latitude}/${selectedDet.location.longitude}`}
                        target="_blank"
                        rel="noreferrer"
                        id="osm-link"
                      >
                        <ExternalLink size={14} />View on OpenStreetMap
                      </a>
                    )}

                    <button className="close-det-btn" onClick={() => setSelectedDet(null)}>
                      Clear selection
                    </button>
                  </>
                ) : (
                  <div className="detail-empty">
                    <MapPin size={42} />
                    <p>Click a bounding box on the image, or select a row from the table below to inspect a detection.</p>
                  </div>
                )}
              </div>

              {/* Telemetry mini panel */}
              <div className="telemetry-mini">
                <div className="eyebrow"><Satellite size={12} />Survey Telemetry</div>
                <div className="telemetry-grid">
                  <span className="tele-k">Vessel Lat</span>
                  <span className="tele-v mono">{fmtCoord(result.telemetry.source_latitude)}</span>
                  <span className="tele-k">Vessel Lon</span>
                  <span className="tele-v mono">{fmtCoord(result.telemetry.source_longitude)}</span>
                  <span className="tele-k">Heading</span>
                  <span className="tele-v mono">{result.telemetry.heading_deg?.toFixed(1) ?? '—'}°</span>
                  <span className="tele-k">Altitude</span>
                  <span className="tele-v mono">{fmtMetres(result.telemetry.altitude_m)}</span>
                  <span className="tele-k">Slant Range</span>
                  <span className="tele-v mono">{fmtMetres(result.telemetry.slant_range_m)}</span>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* ─ Detection table ─ */}
        {result && (
          <section className="panel table-section" id="detection-table-section">
            <div className="panel-head">
              <div className="panel-head-left">
                <Target size={16} />
                <h2>Detection Review</h2>
                <span className="count-badge">{visibleDetections.length} shown</span>
              </div>
              <div className="panel-head-right table-controls">
                <label className="threshold-label" htmlFor="threshold-slider">
                  Min confidence: <strong>{threshold}%</strong>
                </label>
                <input
                  id="threshold-slider"
                  type="range"
                  min={0}
                  max={100}
                  value={threshold}
                  className="range-slider"
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
                <button
                  id="download-csv-btn"
                  className="action-btn"
                  onClick={() => downloadCsv(jobId!, result.meta.source_filename.replace(/\.\w+$/, ''))}
                  disabled={!jobId}
                >
                  <Download size={14} />Export CSV
                </button>
                <a
                  id="download-json-btn"
                  className="action-btn ghost"
                  href={`http://localhost:8000/api/v1/results/${jobId}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  <FileJson size={14} />Full JSON
                </a>
              </div>
            </div>

            {/* Detection badge row */}
            <div className="det-badges-row">
              {result.detections.map((det) => (
                <DetectionBadge
                  key={det.id}
                  det={det}
                  selected={selectedDet?.id === det.id}
                  onSelect={() => setSelectedDet(selectedDet?.id === det.id ? null : det)}
                />
              ))}
            </div>

            <div className="table-scroll">
              <table className="det-table" id="detections-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Type</th>
                    <th>Confidence</th>
                    <th>Box (px)</th>
                    <th>W × H (m)</th>
                    <th>Shadow (px)</th>
                    <th>Latitude</th>
                    <th>Longitude</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleDetections.map((det, i) => {
                    const color = labelColor(det.label);
                    const isSelected = selectedDet?.id === det.id;
                    return (
                      <tr
                        key={det.id}
                        className={isSelected ? 'row-selected' : ''}
                        onClick={() => setSelectedDet(isSelected ? null : det)}
                        style={{ '--row-color': color } as React.CSSProperties}
                      >
                        <td className="mono dim">{String(i + 1).padStart(2, '0')}</td>
                        <td>
                          <span className="label-chip" style={{ borderColor: color, color }}>
                            {labelIcon(det.label)} {det.label.replace(/_/g, ' ')}
                          </span>
                        </td>
                        <td>
                          <span className="conf-bar-wrap">
                            <span className="conf-bar-fill" style={{ width: `${det.confidence_pct}%`, background: color }} />
                            <span className="conf-bar-text">{fmtPct(det.confidence_pct)}</span>
                          </span>
                        </td>
                        <td className="mono dim small-text">
                          {det.bounding_box.x_min},{det.bounding_box.y_min}→{det.bounding_box.x_max},{det.bounding_box.y_max}
                        </td>
                        <td className="mono dim">
                          {fmtMetres(det.physical_dimensions.width_m)} × {fmtMetres(det.physical_dimensions.height_m)}
                        </td>
                        <td className="mono dim">{det.acoustic_shadow_pixels ?? '—'}</td>
                        <td className="mono dim">{fmtCoord(det.location.latitude)}</td>
                        <td className="mono dim">{fmtCoord(det.location.longitude)}</td>
                        <td>
                          <button
                            className="inspect-btn"
                            onClick={(e) => { e.stopPropagation(); setSelectedDet(isSelected ? null : det); }}
                          >
                            Inspect
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {visibleDetections.length === 0 && (
                <div className="table-empty">
                  <Info size={20} />
                  {result.detections.length === 0
                    ? 'No detections found in this scan.'
                    : `No detections above ${threshold}% — try lowering the threshold.`}
                </div>
              )}
            </div>
          </section>
        )}

        {/* ─ Job history ─ */}
        <section className="panel jobs-section" id="jobs-history-section">
          <button
            className="panel-head collapsible"
            onClick={() => setShowJobs((v) => !v)}
            aria-expanded={showJobs}
          >
            <div className="panel-head-left">
              <Clock size={16} />
              <h2>Job History</h2>
              <span className="count-badge">{jobs.length}</span>
            </div>
            <ChevronDown size={16} className={`chevron ${showJobs ? 'open' : ''}`} />
          </button>

          {showJobs && (
            <div className="jobs-list" id="jobs-list">
              {jobs.length === 0 ? (
                <div className="jobs-empty">No jobs yet — upload a file to begin.</div>
              ) : (
                jobs.map((job) => (
                  <div
                    key={job.job_id}
                    className={`job-row ${job.status}`}
                    role="listitem"
                  >
                    <div className="job-row-left">
                      <span className={`job-status-dot ${job.status}`} />
                      <span className="job-filename">{job.filename}</span>
                      <span className="job-status-text">{job.status}</span>
                    </div>
                    <div className="job-row-right">
                      <span className="job-time dim mono">{new Date(job.created_at).toLocaleTimeString()}</span>
                      {job.status === 'done' && (
                        <button
                          className="ghost-btn small"
                          onClick={() => downloadCsv(job.job_id, job.filename.replace(/\.\w+$/, ''))}
                        >
                          <Download size={12} />CSV
                        </button>
                      )}
                    </div>
                  </div>
                ))
              )}
              <button className="ghost-btn small refresh-btn" onClick={refreshJobs}>
                <RefreshCw size={13} />Refresh
              </button>
            </div>
          )}
        </section>

        {/* ─ Footer note ─ */}
        <p className="footer-note" id="footer-note">
          <Info size={13} />
          SIH 2026 Prototype — Marine Debris Detection via Side-Scan Sonar + YOLOv8.
          Trained on AI4Shipwrecks dataset. All GPS coordinates are computed from sonar geometry and may have survey-grade accuracy only.
          Files are processed server-side and not stored permanently.
        </p>
      </main>
    </div>
  );
}
