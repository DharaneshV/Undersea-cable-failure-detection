import React, { useState, useEffect, useRef } from 'react';
import { Play, Square, Zap } from 'lucide-react';
import LiveCharts from './components/LiveCharts';
import CableGraphic from './components/CableGraphic';
import MetricsGrid from './components/MetricsGrid';

/* ── Severity helper (mirrors api.py) ──────────────────────────────────── */
function severityOf(score, threshold) {
  const r = threshold > 0 ? score / threshold : 0;
  if (r > 5)    return { label: 'Critical', cls: 'sev-critical' };
  if (r > 3)    return { label: 'High',     cls: 'sev-high'     };
  if (r > 1.2)  return { label: 'Medium',   cls: 'sev-medium'   };
  if (r > 1.0)  return { label: 'Low',      cls: 'sev-low'      };
  if (r > 0.75) return { label: 'Degrading',cls: 'sev-warning'  };
  return               { label: 'Normal',   cls: 'sev-low'      };
}

/* ── Live UTC clock ─────────────────────────────────────────────────────── */
function LiveClock() {
  const [time, setTime] = useState('');
  useEffect(() => {
    const tick = () => {
      const d = new Date();
      const h = String(d.getUTCHours()).padStart(2, '0');
      const m = String(d.getUTCMinutes()).padStart(2, '0');
      const s = String(d.getUTCSeconds()).padStart(2, '0');
      setTime(`${h}:${m}:${s} UTC`);
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <div className="hero-clock">{time}</div>;
}

/* ── Empty state ────────────────────────────────────────────────────────── */
function EmptyState() {
  return (
    <div className="empty-state">
      <span className="empty-icon">⚡</span>
      <div className="empty-title">Ready to Monitor</div>
      <div className="empty-sub">
        Select your dataset and press <strong style={{ color: '#00D4FF' }}>▶ Start Stream</strong> to
        begin real-time fault detection.
      </div>
    </div>
  );
}

/* ── Stream progress bar ────────────────────────────────────────────────── */
function ProgressBar({ current, total }) {
  const pct = total > 0 ? Math.min((current / total) * 100, 100) : 0;
  return (
    <div className="progress-wrap">
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="progress-label">
        Sample {current.toLocaleString()} / {total.toLocaleString()}
      </span>
    </div>
  );
}

export default function App() {
  const [datasets, setDatasets]         = useState([]);
  const [selectedDataset, setSelected]  = useState('');
  const [speed, setSpeed]               = useState('2×');
  const [status, setStatus]             = useState('disconnected');
  const [dataBuffer, setDataBuffer]     = useState([]);
  const [latestData, setLatestData]     = useState(null);
  const [faultLog, setFaultLog]         = useState([]);
  const [progress, setProgress]         = useState({ current: 0, total: 0 });
  const wsRef = useRef(null);

  /* Fetch available datasets on mount */
  useEffect(() => {
    fetch('http://localhost:8000/datasets')
      .then(r => r.json())
      .then(d => {
        if (d.datasets?.length) {
          setDatasets(d.datasets);
          setSelected(d.datasets[0]);
        }
      })
      .catch(e => console.error('Could not load datasets', e));
  }, []);

  const startStream = () => {
    if (wsRef.current) wsRef.current.close();
    setDataBuffer([]);
    setLatestData(null);
    setFaultLog([]);
    setProgress({ current: 0, total: 0 });
    setStatus('playing');

    const ws = new WebSocket(
      `ws://localhost:8000/ws/stream?dataset=${selectedDataset}&speed=${speed}`
    );

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.done || data.error) {
        setStatus('connected');
        ws.close();
        return;
      }

      setLatestData(data);
      setProgress({ current: data.index, total: data.total });

      setDataBuffer(prev => {
        const next = [...prev, data];
        if (next.length > 120) next.shift();
        return next;
      });

      if (data.new_fault) {
        setFaultLog(prev => [data.new_fault, ...prev]);
      }
    };

    ws.onopen  = () => console.log('WS connected');
    ws.onerror = () => setStatus('error');
    ws.onclose = () => { if (status !== 'connected') setStatus('disconnected'); };
    wsRef.current = ws;
  };

  const stopStream = () => {
    if (wsRef.current) wsRef.current.close();
    setStatus('connected');
  };

  /* Derived state */
  const isPlaying  = status === 'playing';
  const dotCls     = isPlaying ? 'green' : (status === 'error' ? 'red' : 'yellow');
  const sev        = latestData ? severityOf(latestData.anomaly_score, latestData.threshold) : null;

  return (
    <div>
      {/* ── Hero Bar ─────────────────────────────────────────────────── */}
      <div className="hero-bar">
        <div className="hero-left">
          <div className="hero-icon"><Zap size={24} color="#00D4FF"/></div>
          <div>
            <div className="hero-title">Smart Grid Fault Detection</div>
            <div className="hero-sub">Conv-Transformer AE &nbsp;·&nbsp; XAI Tracking &nbsp;·&nbsp; Real-time Localisation</div>
          </div>
        </div>
        <div className="hero-right">
          <LiveClock />
          <div className="sys-status-pill">
            <div className="status-dot" />
            System Online
          </div>
          <div className="control-panel">
            <select className="control-select" value={selectedDataset} onChange={e => setSelected(e.target.value)}>
              {datasets.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
            <select className="control-select" value={speed} onChange={e => setSpeed(e.target.value)}>
              {['0.25×','0.5×','1×','2×','5×','Max'].map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            {isPlaying ? (
              <button className="btn-danger" onClick={stopStream}><Square size={14}/> Stop</button>
            ) : (
              <button className="btn-primary" onClick={startStream}><Play size={14}/> Start Stream</button>
            )}
          </div>
        </div>
      </div>

      {/* ── Progress bar ─────────────────────────────────────────────── */}
      {isPlaying && <ProgressBar current={progress.current} total={progress.total} />}

      {/* ── Alert Banner ─────────────────────────────────────────────── */}
      {latestData && (latestData.is_fault || latestData.is_warning) && (
        <div className={`alert-banner ${latestData.is_fault ? 'fault-alert' : 'warning-alert'}`}>
          <div className={`alert-icon-wrap ${latestData.is_fault ? 'alert-icon-fault' : 'alert-icon-warning'}`}>
            {latestData.is_fault ? '🚨' : '⚠'}
          </div>
          <div className="alert-body">
            <div className="alert-headline">
              {latestData.is_fault ? 'FAULT DETECTED' : 'DEGRADING — EARLY WARNING'}
              {sev && <span className={`sev-badge ${sev.cls}`}>{sev.label}</span>}
            </div>
            <div className="alert-detail">
              Score: {latestData.anomaly_score?.toFixed(5)} &nbsp;/&nbsp; Threshold: {latestData.threshold?.toFixed(5)}
              &nbsp;&nbsp;|&nbsp;&nbsp;Ratio: {(latestData.anomaly_score / latestData.threshold).toFixed(2)}×
            </div>
            <div className="alert-meta">📊 Anomaly driven by: {latestData.xai_text}</div>
          </div>
        </div>
      )}

      {/* ── Metrics grid ─────────────────────────────────────────────── */}
      <MetricsGrid data={latestData} />

      {/* ── Main content ─────────────────────────────────────────────── */}
      {!latestData && !isPlaying ? (
        <EmptyState />
      ) : (
        <div className="main-grid">
          <div className="left-column">
            {/* Cable diagram */}
            <div className="cable-box">
              <div className="sec-hdr">🔌 Power Grid Link — Fault Localisation</div>
              <CableGraphic faults={faultLog} />
            </div>

            {/* Live charts */}
            <div className="chart-panel">
              <div className="sec-hdr">📉 Live Telemetry &amp; Anomaly Score</div>
              <LiveCharts data={dataBuffer} threshold={latestData?.threshold} />
            </div>
          </div>

          <div className="right-column">
            {/* Fault log */}
            <div className="log-panel">
              <div className="sec-hdr">🗂 Detected Fault Log ({faultLog.length})</div>
              <div className="fault-log-header">
                <span>Time</span>
                <span>Type</span>
                <span>Severity</span>
                <span>Dist (m)</span>
              </div>
              {faultLog.length === 0 ? (
                <div className="empty-log">System running nominally.</div>
              ) : (
                faultLog.map((f, idx) => {
                  const fs = severityOf(f.anomaly_score, latestData?.threshold || 1);
                  return (
                    <div key={idx} className="fault-log-row">
                      <span className="log-time">{f.Time?.split(' ')[1]}</span>
                      <span className="log-type">{f.fault_type}</span>
                      <span><span className={`sev-badge ${fs.cls}`}>{f.Severity || fs.label}</span></span>
                      <span className="log-dist">{f.est_distance}</span>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── Connection status pill ────────────────────────────────────── */}
      <div className="connection-status">
        <div className={`dot ${dotCls}`} />
        {status.toUpperCase()}
      </div>
    </div>
  );
}
