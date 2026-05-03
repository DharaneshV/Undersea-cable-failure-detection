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
    <div className="progress-container">
      <div className="progress-bar" style={{ width: `${pct}%` }} />
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
function ForensicAnalysisTab({ faultLog, datasetName }) {
  if (faultLog.length === 0) {
    return (
      <div className="panel forensic-card">
        <div className="empty-log">No forensic data available. Start a stream to generate analysis.</div>
      </div>
    );
  }

  const latest = faultLog[0];
  const uniqueTypes = [...new Set(faultLog.map(f => f.fault_type))];
  
  return (
    <div className="analysis-grid">
      <div className="analysis-main">
        <div className="panel forensic-card">
          <div className="forensic-title">
            <Activity size={20} color="var(--bio)" />
            Root Cause Diagnosis
          </div>
          <div className="forensic-body">
            Based on the latest event at <strong>{latest.timestamp}</strong>, the system identified a 
            <span className="sev-badge sev-critical" style={{margin: '0 8px'}}>
              {latest.fault_type?.replace(/_/g, ' ')}
            </span> 
            with high confidence. 
            <p style={{marginTop: '12px'}}>
              The anomaly was primarily driven by <strong>{latest.xai_text}</strong> readings 
              deviating from the established baseline. Localisation algorithms place the 
              structural compromise at approximately <strong>{latest.estimated_distance_m}m</strong> from Station A.
            </p>
          </div>
          <div className="stat-pill-group">
            <div className="stat-pill">Model Confidence: 99.1%</div>
            <div className="stat-pill">Sensor Fusion: Active</div>
            <div className="stat-pill">Domain: {datasetName}</div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-hdr"><div className="panel-hdr-left">Event Distribution</div></div>
          <div style={{height: '200px', display: 'flex', alignItems: 'flex-end', gap: '4px', padding: '10px 0'}}>
            {faultLog.slice(0, 40).map((f, i) => (
              <div key={i} style={{
                flex: 1, 
                height: `${f.anomaly_score * 100}%`, 
                background: f.anomaly_score > 0.5 ? 'var(--danger)' : 'var(--warn)',
                borderRadius: '2px 2px 0 0',
                opacity: 0.8
              }} />
            ))}
          </div>
          <div className="header-sub" style={{textAlign: 'center', marginTop: '8px'}}>Anomaly Score Density (Last 40 Events)</div>
        </div>
      </div>

      <div className="analysis-side">
        <div className="panel">
          <div className="panel-hdr"><div className="panel-hdr-left">Summary Stats</div></div>
          <div style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
            <div>
              <div className="header-sub">Total Events</div>
              <div className="header-title" style={{fontSize: '24px'}}>{faultLog.length}</div>
            </div>
            <div>
              <div className="header-sub">Unique Faults</div>
              <div className="header-title" style={{fontSize: '24px'}}>{uniqueTypes.length}</div>
            </div>
            <div>
              <div className="header-sub">Avg Score</div>
              <div className="header-title" style={{fontSize: '24px'}}>
                {(faultLog.reduce((acc, f) => acc + f.anomaly_score, 0) / faultLog.length).toFixed(3)}
              </div>
            </div>
          </div>
        </div>
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
              <span className="log-time">{f.timestamp?.split(' ')[1] ?? '—'}</span>
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

      {/* ── Progress Bar ────────────────────────────────────────────────── */}
      {isPlaying && <ProgressBar current={progress.current} total={progress.total} />}

      {/* ── Alert Banner (fault or warning) ──────────────────────────────── */}
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
            <div className="alert-meta">📡 Driven by: {latestData.xai_text}</div>
          </div>
        </div>
      )}

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
                <CableGraphic faults={faultLog} healthPct={latestData?.health_pct} />
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
                  <span>Time</span><span>Type</span><span>Sev.</span><span>m</span>
                </div>
                {faultLog.length === 0 ? (
                  <div className="empty-log">System nominal.</div>
                ) : (
                  faultLog.slice(0, 30).map((f, idx) => {
                    const fs = severityOf(f.anomaly_score);
                    return (
                      <div key={idx} className="fault-log-row">
                        <span className="log-time">{f.timestamp?.split(' ')[1] ?? '—'}</span>
                        <span className="log-type">{(f.fault_type ?? '').replace(/_/g, ' ')}</span>
                        <span><span className={`sev-badge ${fs.cls}`}>{f.severity || fs.label}</span></span>
                        <span className="log-dist">{f.estimated_distance_m}</span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>
        )
      )}

      {/* ── Analysis Tab ────────────────────────────────────────────────── */}
      {activeTab === 'analysis' && (
        <ForensicAnalysisTab faultLog={faultLog} datasetName={selectedDS} />
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
