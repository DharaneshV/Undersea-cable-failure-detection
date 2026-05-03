import React, { useState, useEffect } from 'react';

const API = 'http://localhost:8000';

function SkeletonBar({ width = '100%', height = 18 }) {
  return (
    <div
      className="skeleton"
      style={{ width, height, borderRadius: 6, marginBottom: 8 }}
    />
  );
}

function StatBox({ label, value, color }) {
  return (
    <div className="model-stat-box">
      <div className="model-stat-value" style={{ color }}>{value}</div>
      <div className="model-stat-label">{label}</div>
    </div>
  );
}

function aucColor(auc) {
  if (auc >= 0.9) return '#2ed573';
  if (auc >= 0.8) return '#ffa502';
  return '#ff4757';
}

export default function ModelInfoPanel() {
  const [info,    setInfo]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(false);

  const fetchInfo = () => {
    setLoading(true);
    setError(false);
    fetch(`${API}/model/info`)
      .then(r => r.json())
      .then(d => { setInfo(d); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  };

  useEffect(() => { fetchInfo(); }, []);

  if (loading) {
    return (
      <div className="model-info-panel" style={{ padding: 4 }}>
        <SkeletonBar height={28} width="60%" />
        <SkeletonBar height={80} />
        <SkeletonBar height={120} />
      </div>
    );
  }

  if (error || !info) {
    return (
      <div style={{ textAlign: 'center', padding: '48px 20px', color: 'var(--text-muted)' }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>⚠️</div>
        <div style={{ marginBottom: 16, fontSize: 13 }}>Unable to load model info</div>
        <button className="retry-btn" onClick={fetchInfo}>Retry</button>
      </div>
    );
  }

  const auc       = info.roc_auc;
  const threshold = typeof info.threshold === 'number' ? info.threshold.toFixed(5) : '—';
  const features  = Array.isArray(info.features) ? info.features : [];
  const seqLen    = info.sequence_length ?? '—';
  const version   = info.version ?? '2.0';
  const modelType = (info.model_type ?? 'conv_transformer_ae').replace(/_/g, ' ');

  return (
    <div className="model-info-panel">
      {/* Title row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
            Detection model
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>
            Anomaly Detection · Predictive Maintenance
          </div>
        </div>
        <span style={{
          marginLeft: 'auto', padding: '4px 12px', borderRadius: 20,
          background: 'rgba(0,212,255,0.1)', border: '1px solid rgba(0,212,255,0.3)',
          color: 'var(--accent-cyan)', fontSize: 11, fontWeight: 700,
        }}>
          v{version}
        </span>
      </div>

      {/* Stat boxes */}
      <div className="model-stat-row">
        <StatBox
          label="ROC-AUC"
          value={auc != null ? auc.toFixed(4) : '—'}
          color={auc != null ? aucColor(auc) : 'var(--text-muted)'}
        />
        <StatBox label="Threshold" value={threshold} color="var(--accent-cyan)" />
        <StatBox label="Seq Length" value={seqLen} color="var(--accent-purple)" />
      </div>

      {/* Architecture block */}
      <div className="model-arch-block">
        <h4>Performance metrics</h4>
        <div>Real-time telemetry analysis</div>
        <div>Dynamic anomaly thresholding</div>
        <div>Multi-sensor feature fusion</div>
      </div>

      {/* Features + data source */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {features.map(f => (
          <span key={f} style={{
            padding: '4px 12px', borderRadius: 20, fontSize: 11, fontWeight: 600,
            background: 'rgba(59,130,246,0.1)', border: '1px solid rgba(59,130,246,0.25)',
            color: 'var(--accent-blue)', fontFamily: 'var(--mono)',
          }}>
            {f}
          </span>
        ))}
      </div>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 8,
        fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)',
      }}>
        <span style={{
          padding: '3px 10px', borderRadius: 12,
          background: 'rgba(46,213,115,0.1)', border: '1px solid rgba(46,213,115,0.25)',
          color: 'var(--accent-green)', fontWeight: 700, fontSize: 10,
        }}>
          Azure PDM
        </span>
        Trained on Microsoft Azure Predictive Maintenance dataset
      </div>
    </div>
  );
}
