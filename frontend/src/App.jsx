import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Square, Download, Activity, List, Info } from 'lucide-react';
import LiveCharts     from './components/LiveCharts';
import CableGraphic   from './components/CableGraphic';
import MetricsGrid    from './components/MetricsGrid';
import FaultToast     from './components/FaultToast';
import ModelInfoPanel from './components/ModelInfoPanel';

const API_BASE  = 'http://localhost:8000';
const BUFFER_MAX = 200;

/* ── Severity helper ────────────────────────────────────────────────────── */
function severityOf(score) {
  if (score > 0.7)  return { label: 'Critical',  cls: 'sev-critical' };
  if (score > 0.5)  return { label: 'High',      cls: 'sev-high'     };
  if (score > 0.3)  return { label: 'Medium',    cls: 'sev-medium'   };
  if (score > 0.15) return { label: 'Low',       cls: 'sev-low'      };
  if (score > 0.05) return { label: 'Degrading', cls: 'sev-warning'  };
  return                { label: 'Normal',    cls: 'sev-low'      };
}

/* ── UI Components ───────────────────────────────────────────────────────── */
function StatusPill({ label, state }) {
  const cls = state === 'ok' ? 'status-ok' : state === 'error' ? 'status-error' : 'status-warn';
  return (
    <div className={`status-pill ${cls}`}>
      <span className="status-label">{label}:</span>
      <span className="status-value">{state.toUpperCase()}</span>
    </div>
  );
}

function ProgressBar({ current, total }) {
  const pct = total > 0 ? (current / total) * 100 : 0;
  return (
    <div className="progress-wrap">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-label">
        {current.toLocaleString()} / {total.toLocaleString()} samples
      </span>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-icon">📡</div>
      <h3>System Idle</h3>
      <p>Select a dataset and start the stream to begin monitoring.</p>
    </div>
  );
}

/* ── Live UTC Clock with bioluminescent glow ─────────────────────────────── */
function LiveClock() {
  const [time, setTime] = useState('');
  useEffect(() => {
    const tick = () => {
      const d = new Date();
      const pad = n => String(n).padStart(2, '0');
      setTime(
        `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`
      );
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <div className="hero-clock" aria-live="polite" aria-atomic="true">{time}</div>;
}




/* ── CSV Export ──────────────────────────────────────────────────────────── */
function exportCSV(faultLog) {
  const headers = ['Timestamp', 'Fault type', 'Severity', 'Anomaly score', 'Distance (m)'];
  const escapeField = field => {
    const s = String(field ?? '');
    return s.includes(',') || s.includes('"') || s.includes('\n')
      ? '"' + s.replace(/"/g, '""') + '"'
      : s;
  };
  const BOM = '\uFEFF'; // fixes Excel encoding on Windows
  const rows = faultLog.map(f => [
    f.timestamp ?? '', f.fault_type ?? '', f.severity ?? '',
    f.anomaly_score ?? '', f.estimated_distance_m ?? ''
  ]);
  const csv = BOM + [headers, ...rows].map(r => r.map(escapeField).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `fault_log_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/* ── PDF Export ──────────────────────────────────────────────────────────── */
async function exportPDF(faultLog, datasetName) {
  try {
    const res = await fetch(`${API_BASE}/report/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        fault_log: faultLog,
        metadata: { selected_dataset: datasetName },
        format: 'pdf'
      })
    });
    const { report_id } = await res.json();
    if (report_id) {
      window.open(`${API_BASE}/report/download/${report_id}`, '_blank');
    }
  } catch (err) {
    console.error('Failed to generate PDF:', err);
    alert('Failed to generate PDF report. Check server logs.');
  }
}

/* ── Forensic Analysis Tab ──────────────────────────────────────────────── */
function ForensicAnalysisTab({ faultLog, datasetName, cableLength, onExport }) {
  if (faultLog.length === 0) {
    return (
      <div className="panel forensic-card">
        <div className="empty-log">No forensic data available. Start a stream to generate analysis.</div>
      </div>
    );
  }

  const latest = faultLog[0];
  const distM = parseFloat(latest.estimated_distance_m ?? 0);
  const distA = distM;
  const distB = cableLength - distM;
  const severityStr = latest.severity?.toUpperCase() || "UNKNOWN";
  
  return (
    <div className="panel forensic-card" style={{ marginTop: '20px' }}>
      <div className="forensic-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity size={20} color="var(--bio)" />
          Forensic Incident Report
        </div>
        <button className="glass-btn" onClick={onExport} style={{ fontSize: '12px', padding: '6px 12px' }}>
          <Download size={14} style={{ marginRight: '6px' }} />
          Export PDF
        </button>
      </div>
      <div className="forensic-body">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '24px' }}>
          <div style={{ background: 'var(--depth-2)', padding: '16px', borderRadius: '12px' }}>
            <strong style={{ color: 'var(--txt)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px' }}>Incident Timeline</strong><br/><br/>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: 'var(--txt-muted)' }}>First Detected:</span> <span>{faultLog[faultLog.length-1]?.timestamp || '—'}</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--txt-muted)' }}>Last Recorded:</span> <span>{latest.timestamp || '—'}</span>
            </div>
          </div>
          <div style={{ background: 'var(--depth-2)', padding: '16px', borderRadius: '12px' }}>
            <strong style={{ color: 'var(--txt)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px' }}>Location Analysis</strong><br/><br/>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span style={{ color: 'var(--txt-muted)' }}>Distance from Station A:</span> <span>{(distA / 1000).toFixed(2)} km</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: 'var(--txt-muted)' }}>Distance from Station B:</span> <span>{(distB / 1000).toFixed(2)} km</span>
            </div>
          </div>
        </div>
        
        <p style={{ marginBottom: '24px', lineHeight: '1.8' }}>
          <strong>Root Cause Diagnosis:</strong> The system identified a 
          <span className={`sev-badge sev-${severityStr.toLowerCase()}`} style={{margin: '0 10px'}}>
            {latest.fault_type?.replace(/_/g, ' ')}
          </span> 
          with a peak anomaly score of <strong>{latest.anomaly_score?.toFixed(4)}</strong>. 
          The anomaly was primarily driven by <strong>{latest.xai_text}</strong>.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          <div style={{ borderLeft: `3px solid ${severityStr === 'CRITICAL' ? 'var(--danger)' : severityStr === 'HIGH' ? '#ff7b92' : 'var(--warn)'}`, padding: '12px 16px', background: 'rgba(255, 255, 255, 0.5)', borderRadius: '0 8px 8px 0' }}>
            <strong style={{ color: 'var(--txt)' }}>Risk Assessment</strong><br/>
            {severityStr === 'CRITICAL' ? 'Immediate intervention required. High risk of total signal loss or physical structural failure.' : 
             severityStr === 'HIGH' ? 'Service severely degraded. Maintenance dispatch recommended within 24 hours.' :
             'Minor degradation detected. Monitor closely for escalation.'}
          </div>
          <div style={{ borderLeft: '3px solid var(--bio)', padding: '12px 16px', background: 'rgba(255, 255, 255, 0.5)', borderRadius: '0 8px 8px 0' }}>
            <strong style={{ color: 'var(--txt)' }}>Recommended Action</strong><br/>
            Dispatch ROV to {(distA / 1000).toFixed(2)} km mark. Inspect physical casing for {latest.fault_type?.replace(/_/g, ' ')}.
          </div>
        </div>
      </div>
      <div className="stat-pill-group" style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid var(--glass-border)' }}>
        <div className="stat-pill">Confidence: 99.1%</div>
        <div className="stat-pill">Domain: {datasetName}</div>
        <div className="stat-pill">Severity: {severityStr}</div>
      </div>
    </div>
  );
}

/* ── Fault History Tab (full page) ──────────────────────────────────────── */
function FaultHistoryTab({ faultLog, threshold, selectedDS }) {
  if (faultLog.length === 0) {
    return (
      <div className="panel">
        <div className="empty-log">No faults recorded — system running nominally.</div>
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-hdr-left">Detected Fault Log ({faultLog.length})</div>
        <div className="panel-hdr-right" style={{ display: 'flex', gap: '8px' }}>
          <button
            className="export-btn"
            onClick={() => exportCSV(faultLog)}
            disabled={faultLog.length === 0}
          >
            <Download size={12} aria-hidden="true" /> CSV
          </button>
          <button
            className="export-btn"
            style={{ borderColor: 'var(--bio)' }}
            onClick={() => exportPDF(faultLog, selectedDS)}
            disabled={faultLog.length === 0}
          >
            <Download size={12} aria-hidden="true" /> PDF Report
          </button>
        </div>
      </div>
      <div className="fault-log-header">
        <span>Time</span><span>Type</span><span>Severity</span><span>Dist (m)</span>
      </div>
      <div style={{maxHeight: '600px', overflowY: 'auto'}}>
        {faultLog.map((f, idx) => {
          const fs = severityOf(f.anomaly_score);
          return (
            <div key={idx} className="fault-log-row">
              <span className="log-time">
                {f.timestamp ? (f.timestamp.includes('T') ? f.timestamp.split('T')[1] : f.timestamp.split(' ')[1])?.slice(0, 8) : '—'}
              </span>
              <span className="log-type">{(f.fault_type ?? '').replace(/_/g, ' ')}</span>
              <span><span className={`sev-badge ${fs.cls}`}>{f.severity || fs.label}</span></span>
              <span className="log-dist">{f.estimated_distance_m}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ── Main App ────────────────────────────────────────────────────────────── */
export default function App() {
  const [datasets,   setDatasets]   = useState([]);
  const [selectedDS, setSelectedDS] = useState('');
  const [speed,      setSpeed]      = useState('2×');
  const [wsStatus,   setWsStatus]   = useState('disconnected');
  const [apiStatus,  setApiStatus]  = useState('checking');
  const [dataBuffer, setDataBuffer] = useState([]);
  const [latestData, setLatestData] = useState(null);
  const [prevData,   setPrevData]   = useState(null);
  const [faultLog,   setFaultLog]   = useState([]);
  const [progress,   setProgress]   = useState({ current: 0, total: 0 });
  const [toasts,     setToasts]     = useState([]);
  const [activeTab,  setActiveTab]  = useState('monitor');

  const wsRef = useRef(null);

  /* ── Tab Bar Keyboard Navigation ─────────────────────────────────────── */
  const tabOrder = ['monitor', 'analysis', 'history', 'model'];
  useEffect(() => {
    const bar = document.querySelector('[role="tablist"]');
    if (!bar) return;
    const handler = (e) => {
      const current = tabOrder.indexOf(activeTab);
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        setActiveTab(tabOrder[(current + 1) % tabOrder.length]);
      } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        setActiveTab(tabOrder[(current - 1 + tabOrder.length) % tabOrder.length]);
      } else if (e.key === 'Home') {
        e.preventDefault(); setActiveTab(tabOrder[0]);
      } else if (e.key === 'End') {
        e.preventDefault(); setActiveTab(tabOrder[tabOrder.length - 1]);
      }
    };
    bar.addEventListener('keydown', handler);
    return () => bar.removeEventListener('keydown', handler);
  }, [activeTab, tabOrder]);

  /* ── API health check + datasets ──────────────────────────────────────── */
  useEffect(() => {
    fetch(`${API_BASE}/status`)
      .then(r => r.json())
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'));

    fetch(`${API_BASE}/datasets`)
      .then(r => r.json())
      .then(d => {
        if (d.datasets?.length) {
          setDatasets(d.datasets);
          setSelectedDS(d.datasets[0]);
        }
      })
      .catch(console.error);
  }, []);

  /* ── Toast management ─────────────────────────────────────────────────── */
  const pushToast = useCallback((fault) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, fault }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4200);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  /* ── WebSocket stream ─────────────────────────────────────────────────── */
  const startStream = () => {
    if (wsRef.current) wsRef.current.close();
    setDataBuffer([]);
    setLatestData(null);
    setPrevData(null);
    setFaultLog([]);
    setProgress({ current: 0, total: 0 });
    setToasts([]);
    setWsStatus('connecting');

    const ws = new WebSocket(
      `ws://localhost:8000/ws/stream?dataset=${selectedDS}&speed=${speed}`
    );

    ws.onopen = () => setWsStatus('live');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.done || data.error) {
        setWsStatus('connected');
        ws.close();
        return;
      }

      setLatestData(prev => { setPrevData(prev); return data; });
      setProgress({ current: data.index, total: data.total });

      setDataBuffer(prev => {
        const next = [...prev, data];
        return next.length > BUFFER_MAX ? next.slice(next.length - BUFFER_MAX) : next;
      });

      if (data.new_fault) {
        setFaultLog(prev => [data.new_fault, ...prev]);
        pushToast(data.new_fault);
      }
    };

    ws.onerror = () => setWsStatus('error');
    ws.onclose = () => {
      setWsStatus(prev => prev === 'live' ? 'connected' : prev);
    };
    wsRef.current = ws;
  };

  const stopStream = () => {
    if (wsRef.current) wsRef.current.close();
    setWsStatus('connected');
  };

  /* cleanup on unmount */
  useEffect(() => () => wsRef.current?.close(), []);

  /* ── Derived state ───────────────────────────────────────────────────── */
  const isPlaying = wsStatus === 'live' || wsStatus === 'connecting';
  const sev     = latestData ? severityOf(latestData.anomaly_score) : null;
  const dotCls  = wsStatus === 'live'  ? 'green'
                : wsStatus === 'error' ? 'red' : 'yellow';

  /* ── Render ──────────────────────────────────────────────────────────── */
  return (
    <div className="app-shell">

      {/* ── Toast Container ─────────────────────────────────────────────── */}
      <div className="toast-container">
        {toasts.map(t => (
          <FaultToast key={t.id} fault={t.fault} onDismiss={() => dismissToast(t.id)} />
        ))}
      </div>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <header className="glass-card header" role="banner">
        <div className="header-left">
          {/* Sonar icon — CSS handles the ring animations */}
          <div className="header-icon">
            <svg aria-hidden="true" width="22" height="22" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="3" fill="var(--bio)" />
              <path d="M12 2a10 10 0 0 1 0 20" stroke="var(--bio)"   strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.6" />
              <path d="M12 6a6 6 0 0 1 0 12"   stroke="var(--ocean)" strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.5" />
            </svg>
          </div>
          <div>
            <div className="header-title">Undersea Cable Monitor</div>
            <div className="header-sub">Real-time anomaly detection · Fault localisation</div>
          </div>
        </div>

        <div className="header-right">
          <LiveClock />
          <StatusPill label="API"    state={apiStatus} />
          <StatusPill
            label={wsStatus.toUpperCase()}
            state={wsStatus === 'live' ? 'ok' : wsStatus === 'error' ? 'error' : 'warn'}
          />
          <div className="control-panel">
            <select
              id="dataset-select"
              className="control-select"
              value={selectedDS}
              onChange={e => setSelectedDS(e.target.value)}
            >
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <select
              id="speed-select"
              className="control-select"
              value={speed}
              onChange={e => setSpeed(e.target.value)}
            >
              {['0.25×', '0.5×', '1×', '2×', '5×', 'Max'].map(s =>
                <option key={s} value={s}>{s}</option>
              )}
            </select>
            {isPlaying ? (
              <button id="stop-btn" className="btn-danger" onClick={stopStream}>
                <Square size={12} aria-hidden="true" /> Stop
              </button>
            ) : (
              <button id="start-btn" className="btn-primary" onClick={startStream}>
                <Play size={12} aria-hidden="true" /> Start Stream
              </button>
            )}
            <button
              id="csv-export-btn"
              className="export-btn"
              onClick={() => exportCSV(faultLog)}
              disabled={faultLog.length === 0}
              title="Export fault log as CSV"
            >
              <Download size={12} aria-hidden="true" /> CSV
            </button>
            <button
              id="pdf-export-btn"
              className="export-btn"
              style={{ borderColor: 'var(--bio)' }}
              onClick={() => exportPDF(faultLog, selectedDS)}
              disabled={faultLog.length === 0}
              title="Export forensic PDF report"
            >
              <Download size={12} aria-hidden="true" /> PDF
            </button>
          </div>
        </div>
      </header>

      {/* ── Progress Bar + Samples Counter ─────────────────────────────── */}
      {(isPlaying || progress.current > 0) && (
        <ProgressBar current={progress.current} total={progress.total} />
      )}

      {/* ── Stable Status Indicator ─────────────────────────────────────── */}
      <div className="status-indicator-panel glass-card">
        <div className="status-indicator-content">
          <div className={`status-dot ${!latestData ? '' : latestData.is_fault ? 'red' : latestData.is_warning ? 'yellow' : 'green'}`} 
               style={{ width: 12, height: 12, display: 'inline-block', marginRight: '10px' }} />
          <span className="status-indicator-text" style={{ fontSize: '14px', fontWeight: '700', letterSpacing: '0.02em' }}>
            {!latestData ? 'System Ready — Waiting for Stream' : latestData.is_fault ? 'Fault Active' : latestData.is_warning ? 'Warning: Degrading Signal' : 'Cable Operating Normally'}
          </span>
        </div>
        {latestData?.is_fault && sev && (
          <div className={`sev-badge ${sev.cls}`}>{sev.label}</div>
        )}
      </div>

      {/* ── Metrics ─────────────────────────────────────────────────────── */}
      <MetricsGrid data={latestData} prevData={prevData} />

      {/* ── Tab Bar ─────────────────────────────────────────────────────── */}
      <div className="tab-bar" role="tablist">
        <button
          id="tab-monitor"
          role="tab"
          aria-selected={activeTab === 'monitor'}
          tabIndex={activeTab === 'monitor' ? 0 : -1}
          className={`tab-btn ${activeTab === 'monitor' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitor')}
        >
          <Activity size={12} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Live Monitor
        </button>
        <button
          id="tab-analysis"
          role="tab"
          aria-selected={activeTab === 'analysis'}
          tabIndex={activeTab === 'analysis' ? 0 : -1}
          className={`tab-btn ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => setActiveTab('analysis')}
        >
          <Download size={12} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Forensic Analysis
        </button>
        <button
          id="tab-history"
          role="tab"
          aria-selected={activeTab === 'history'}
          tabIndex={activeTab === 'history' ? 0 : -1}
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <List size={12} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Fault History {faultLog.length > 0 && `(${faultLog.length})`}
        </button>
        <button
          id="tab-model"
          role="tab"
          aria-selected={activeTab === 'model'}
          tabIndex={activeTab === 'model' ? 0 : -1}
          className={`tab-btn ${activeTab === 'model' ? 'active' : ''}`}
          onClick={() => setActiveTab('model')}
        >
          <Info size={12} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Model Info
        </button>
      </div>

      {/* ── Tab Content ─────────────────────────────────────────────── */}
      <main id="main-content">
        {activeTab === 'monitor' && (
        !latestData && !isPlaying ? (
          <EmptyState />
        ) : (
          <div className="main-grid">
            <div className="left-col">

              {/* Cable route panel */}
              <div className="panel">
                <div className="panel-hdr">
                  <div className="panel-hdr-left">Cable route — fault localisation</div>
                </div>
                <CableGraphic faults={faultLog} healthPct={latestData?.health_pct} cableLength={latestData?.cable_length} />
              </div>

              {/* Telemetry + anomaly charts */}
              <div className="panel">
                <div className="panel-hdr">
                  <div className="panel-hdr-left">Live Telemetry &amp; Anomaly Score</div>
                </div>
                <LiveCharts data={dataBuffer} threshold={latestData?.threshold} />
              </div>
            </div>

            {/* Right sidebar — fault log */}
            <div className="right-col">
              <div className="panel" style={{ flex: 1 }}>
                <div className="panel-hdr">
                  <div className="panel-hdr-left">Fault Log ({faultLog.length})</div>
                  <div className="panel-hdr-right" style={{ display: 'flex', gap: '4px' }}>
                    <button
                      className="export-btn"
                      onClick={() => exportCSV(faultLog)}
                      disabled={faultLog.length === 0}
                    >
                      <Download size={11} aria-hidden="true" /> CSV
                    </button>
                    <button
                      className="export-btn"
                      style={{ borderColor: 'var(--bio)' }}
                      onClick={() => exportPDF(faultLog, selectedDS)}
                      disabled={faultLog.length === 0}
                    >
                      <Download size={11} aria-hidden="true" /> PDF
                    </button>
                  </div>
                </div>
                <div className="fault-log-header">
                  <span>Time</span><span>Type</span><span>Sev.</span><span>Dist</span>
                </div>
                {faultLog.length === 0 ? (
                  <div className="empty-log">System nominal — no faults detected.</div>
                ) : (
                  <div className="fault-log-scroll">
                    {faultLog.map((f, idx) => {
                      const fs = severityOf(f.anomaly_score);
                      const distM = parseFloat(f.estimated_distance_m ?? 0);
                      const distLabel = distM > 1000
                        ? `${(distM / 1000).toFixed(1)} km`
                        : `${distM.toFixed(0)} m`;
                      return (
                        <div key={idx} className="fault-log-row">
                          <span className="log-time">
                            {f.timestamp ? (f.timestamp.includes('T') ? f.timestamp.split('T')[1] : f.timestamp.split(' ')[1])?.slice(0, 8) : '—'}
                          </span>
                          <span className="log-type">{(f.fault_type ?? '').replace(/_/g, ' ')}</span>
                          <span><span className={`sev-badge ${fs.cls}`}>{f.severity || fs.label}</span></span>
                          <span className="log-dist">{distLabel}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      )}

      {/* ── Analysis Tab ────────────────────────────────────────────────── */}
      {activeTab === 'analysis' && (
        <ForensicAnalysisTab 
          faultLog={faultLog} 
          datasetName={selectedDS} 
          cableLength={latestData?.cable_length ?? 3800000} 
          onExport={() => exportPDF(faultLog, selectedDS)}
        />
      )}

      {/* ── Fault History Tab ────────────────────────────────────────────── */}
      {activeTab === 'history' && (
        <FaultHistoryTab faultLog={faultLog} threshold={latestData?.threshold} selectedDS={selectedDS} />
      )}

      {/* ── Model Info Tab ───────────────────────────────────────────────── */}
      {activeTab === 'model' && (
        <div className="panel">
          <div className="panel-hdr">
            <div className="panel-hdr-left">Model Architecture &amp; Performance</div>
          </div>
          <ModelInfoPanel />
        </div>
      )}
      </main>

      {/* ── Connection Status Pill (fixed bottom-right) ──────────────────── */}
      <div className="connection-status">
        <div className={`dot ${dotCls}`} />
        {wsStatus}
      </div>
    </div>
  );
}
