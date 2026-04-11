import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Zap, Play, Square, Download, Activity, List, Info } from 'lucide-react';
import LiveCharts     from './components/LiveCharts';
import CableGraphic   from './components/CableGraphic';
import MetricsGrid    from './components/MetricsGrid';
import FaultToast     from './components/FaultToast';
import ModelInfoPanel from './components/ModelInfoPanel';

const API_BASE = 'http://localhost:8000';
const BUFFER_MAX = 200;

/* ── Severity helper ────────────────────────────────────────────────────── */
function severityOf(score, threshold) {
  const r = threshold > 0 ? score / threshold : 0;
  if (r > 5)    return { label: 'Critical', cls: 'sev-critical' };
  if (r > 3)    return { label: 'High',     cls: 'sev-high'     };
  if (r > 1.2)  return { label: 'Medium',   cls: 'sev-medium'   };
  if (r > 1.0)  return { label: 'Low',      cls: 'sev-low'      };
  if (r > 0.75) return { label: 'Degrading',cls: 'sev-warning'  };
  return               { label: 'Normal',   cls: 'sev-low'      };
}

/* ── UTC Clock ──────────────────────────────────────────────────────────── */
function LiveClock() {
  const [time, setTime] = useState('');
  useEffect(() => {
    const tick = () => {
      const d = new Date();
      const pad = n => String(n).padStart(2, '0');
      setTime(`${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <div className="hero-clock">{time}</div>;
}

/* ── Status Pill ────────────────────────────────────────────────────────── */
function StatusPill({ label, state }) {
  const dotCls = state === 'ok' || state === 'live' ? 'green'
               : state === 'connecting'             ? 'yellow'
               : 'red';
  const pillCls = dotCls === 'green' ? 'ok' : dotCls === 'yellow' ? 'warn' : 'error';
  return (
    <div className={`status-pill ${pillCls}`}>
      <div className={`status-dot ${dotCls}`} />
      {label}
    </div>
  );
}

/* ── Progress Bar ───────────────────────────────────────────────────────── */
function ProgressBar({ current, total }) {
  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
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

/* ── Empty State ────────────────────────────────────────────────────────── */
function EmptyState() {
  return (
    <div className="empty-state">
      <span className="empty-icon">⚡</span>
      <div className="empty-title">Ready to Monitor</div>
      <div className="empty-sub">
        Select your dataset and press{' '}
        <strong style={{ color: 'var(--accent-cyan)' }}>▶ Start Stream</strong>{' '}
        to begin real-time fault detection.
      </div>
    </div>
  );
}

/* ── CSV Export ─────────────────────────────────────────────────────────── */
function exportCSV(faultLog) {
  const headers = ['timestamp', 'fault_type', 'severity', 'est_distance_m', 'anomaly_score'];
  const rows    = faultLog.map(f => [
    f.Time ?? '',
    f.fault_type ?? '',
    f.Severity ?? '',
    f.est_distance ?? '',
    f.anomaly_score ?? '',
  ]);
  const csvContent = [headers, ...rows].map(r => r.join(',')).join('\n');
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `fault_log_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ── Fault History Tab ──────────────────────────────────────────────────── */
function FaultHistoryTab({ faultLog, threshold }) {
  if (faultLog.length === 0) {
    return (
      <div className="panel">
        <div className="empty-log">No faults detected yet. System running nominally.</div>
      </div>
    );
  }
  return (
    <div className="panel">
      <div className="panel-hdr">
        <div className="panel-hdr-left">Detected Fault Log ({faultLog.length})</div>
        <button
          className="export-btn"
          onClick={() => exportCSV(faultLog)}
          disabled={faultLog.length === 0}
        >
          <Download size={13} aria-hidden="true" /> Export CSV
        </button>
      </div>
      <div className="fault-log-header">
        <span>Time</span><span>Type</span><span>Severity</span><span>Dist (m)</span>
      </div>
      {faultLog.map((f, idx) => {
        const fs = severityOf(f.anomaly_score, threshold || 1);
        return (
          <div key={idx} className="fault-log-row">
            <span className="log-time">{f.Time?.split(' ')[1] ?? '—'}</span>
            <span className="log-type">{(f.fault_type ?? '').replace(/_/g, ' ')}</span>
            <span><span className={`sev-badge ${fs.cls}`}>{f.Severity || fs.label}</span></span>
            <span className="log-dist">{f.est_distance}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ── Main App ───────────────────────────────────────────────────────────── */
export default function App() {
  const [datasets,      setDatasets]     = useState([]);
  const [selectedDS,    setSelectedDS]   = useState('');
  const [speed,         setSpeed]        = useState('2×');
  const [wsStatus,      setWsStatus]     = useState('disconnected');
  const [apiStatus,     setApiStatus]    = useState('checking');
  const [dataBuffer,    setDataBuffer]   = useState([]);
  const [latestData,    setLatestData]   = useState(null);
  const [prevData,      setPrevData]     = useState(null);
  const [faultLog,      setFaultLog]     = useState([]);
  const [progress,      setProgress]     = useState({ current: 0, total: 0 });
  const [toasts,        setToasts]       = useState([]);
  const [activeTab,     setActiveTab]    = useState('monitor');

  const wsRef = useRef(null);

  /* ── API health check + datasets ─────────────────────────────────────── */
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

  /* ── Toast management ────────────────────────────────────────────────── */
  const pushToast = useCallback((fault) => {
    const id = Date.now();
    setToasts(prev => [...prev, { id, fault }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 4200);
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  /* ── WebSocket stream ────────────────────────────────────────────────── */
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

      setPrevData(prev => prev);
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
  const sev       = latestData ? severityOf(latestData.anomaly_score, latestData.threshold) : null;
  const dotCls    = wsStatus === 'live'  ? 'green'
                  : wsStatus === 'error' ? 'red'
                  : 'yellow';

  /* ── Render ──────────────────────────────────────────────────────────── */
  return (
    <div className="app-shell">

      {/* ── Toast Container ──────────────────────────────────────────── */}
      <div className="toast-container">
        {toasts.map(t => (
          <FaultToast key={t.id} fault={t.fault} onDismiss={() => dismissToast(t.id)} />
        ))}
      </div>

      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="glass-card header">
        <div className="header-left">
          <div className="header-icon">
            <Zap size={24} color="var(--accent-cyan)" aria-hidden="true" />
          </div>
          <div>
            <div className="header-title">Undersea Cable Monitor</div>
            <div className="header-sub">Conv-Transformer AE&nbsp;·&nbsp;XAI Tracking&nbsp;·&nbsp;Real-time Localisation</div>
          </div>
        </div>

        <div className="header-right">
          <LiveClock />
          <StatusPill label="API"        state={apiStatus} />
          <StatusPill label={wsStatus.toUpperCase()} state={wsStatus === 'live' ? 'ok' : wsStatus === 'error' ? 'error' : 'warn'} />

          <div className="control-panel">
            <select className="control-select" value={selectedDS} onChange={e => setSelectedDS(e.target.value)}>
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <select className="control-select" value={speed} onChange={e => setSpeed(e.target.value)}>
              {['0.25×', '0.5×', '1×', '2×', '5×', 'Max'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            {isPlaying ? (
              <button className="btn-danger" onClick={stopStream}>
                <Square size={13} aria-hidden="true" /> Stop
              </button>
            ) : (
              <button className="btn-primary" onClick={startStream}>
                <Play size={13} aria-hidden="true" /> Start Stream
              </button>
            )}
            <button
              className="export-btn"
              onClick={() => exportCSV(faultLog)}
              disabled={faultLog.length === 0}
              title="Export fault log as CSV"
            >
              <Download size={13} aria-hidden="true" /> CSV
            </button>
          </div>
        </div>
      </div>

      {/* ── Progress ──────────────────────────────────────────────────── */}
      {isPlaying && <ProgressBar current={progress.current} total={progress.total} />}

      {/* ── Alert Banner ─────────────────────────────────────────────── */}
      {latestData && (latestData.is_fault || latestData.is_warning) && (
        <div className={`alert-banner ${latestData.is_fault ? 'fault-alert' : 'warning-alert'}`}>
          <div className={`alert-icon-wrap ${latestData.is_fault ? 'alert-icon-fault' : 'alert-icon-warning'}`}>
            {latestData.is_fault ? '🚨' : '⚠️'}
          </div>
          <div className="alert-body">
            <div className="alert-headline">
              {latestData.is_fault ? 'FAULT DETECTED' : 'DEGRADING — EARLY WARNING'}
              {sev && <span className={`sev-badge ${sev.cls}`}>{sev.label}</span>}
            </div>
            <div className="alert-detail">
              Score: {latestData.anomaly_score?.toFixed(5)}&nbsp;/&nbsp;
              Threshold: {latestData.threshold?.toFixed(5)}&nbsp;&nbsp;|&nbsp;&nbsp;
              Ratio: {(latestData.anomaly_score / latestData.threshold).toFixed(2)}×
            </div>
            <div className="alert-meta">📊 Driven by: {latestData.xai_text}</div>
          </div>
        </div>
      )}

      {/* ── Metrics ──────────────────────────────────────────────────── */}
      <MetricsGrid data={latestData} prevData={prevData} />

      {/* ── Tab Bar ──────────────────────────────────────────────────── */}
      <div className="tab-bar">
        <button
          className={`tab-btn ${activeTab === 'monitor' ? 'active' : ''}`}
          onClick={() => setActiveTab('monitor')}
        >
          <Activity size={13} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Live Monitor
        </button>
        <button
          className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <List size={13} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Fault History {faultLog.length > 0 && `(${faultLog.length})`}
        </button>
        <button
          className={`tab-btn ${activeTab === 'model' ? 'active' : ''}`}
          onClick={() => setActiveTab('model')}
        >
          <Info size={13} style={{ marginRight: 6, verticalAlign: -2 }} aria-hidden="true" />
          Model Info
        </button>
      </div>

      {/* ── Tab Content ──────────────────────────────────────────────── */}

      {activeTab === 'monitor' && (
        !latestData && !isPlaying ? (
          <EmptyState />
        ) : (
          <div className="main-grid">
            <div className="left-col">
              <div className="panel">
                <div className="panel-hdr">
                  <div className="panel-hdr-left">Power Grid Link — Fault Localisation</div>
                </div>
                <CableGraphic faults={faultLog} />
              </div>
              <div className="panel">
                <div className="panel-hdr">
                  <div className="panel-hdr-left">Live Telemetry &amp; Anomaly Score</div>
                </div>
                <LiveCharts data={dataBuffer} threshold={latestData?.threshold} />
              </div>
            </div>

            <div className="right-col">
              <div className="panel" style={{ flex: 1 }}>
                <div className="panel-hdr">
                  <div className="panel-hdr-left">Fault Log ({faultLog.length})</div>
                  <button
                    className="export-btn"
                    onClick={() => exportCSV(faultLog)}
                    disabled={faultLog.length === 0}
                  >
                    <Download size={11} aria-hidden="true" /> CSV
                  </button>
                </div>
                <div className="fault-log-header">
                  <span>Time</span><span>Type</span><span>Sev.</span><span>m</span>
                </div>
                {faultLog.length === 0 ? (
                  <div className="empty-log">System nominal.</div>
                ) : (
                  faultLog.slice(0, 30).map((f, idx) => {
                    const fs = severityOf(f.anomaly_score, latestData?.threshold || 1);
                    return (
                      <div key={idx} className="fault-log-row">
                        <span className="log-time">{f.Time?.split(' ')[1] ?? '—'}</span>
                        <span className="log-type">{(f.fault_type ?? '').replace(/_/g, ' ')}</span>
                        <span><span className={`sev-badge ${fs.cls}`}>{f.Severity || fs.label}</span></span>
                        <span className="log-dist">{f.est_distance}</span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        )
      )}

      {activeTab === 'history' && (
        <FaultHistoryTab faultLog={faultLog} threshold={latestData?.threshold} />
      )}

      {activeTab === 'model' && (
        <div className="panel">
          <div className="panel-hdr">
            <div className="panel-hdr-left">Model Architecture &amp; Performance</div>
          </div>
          <ModelInfoPanel />
        </div>
      )}

      {/* ── Connection Pill ───────────────────────────────────────────── */}
      <div className="connection-status">
        <div className={`dot ${dotCls}`} />
        {wsStatus.toUpperCase()}
      </div>
    </div>
  );
}
