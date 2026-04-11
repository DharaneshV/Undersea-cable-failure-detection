import React from 'react';

export default function MetricsGrid({ data }) {
  if (!data) {
    return (
      <div className="metrics-grid">
        <MetricCard label="Voltage" value="-- " unit="V" color="#3B82F6" pct={0}/>
        <MetricCard label="Current" value="--" unit="A" color="#10B981" pct={0}/>
        <MetricCard label="Temperature" value="--" unit="°C" color="#F59E0B" pct={0}/>
        <MetricCard label="Vibration" value="--" unit="g" color="#EF5941" pct={0}/>
        <MetricCard label="Anomaly Score" value="--" unit="thr --" color="#8B5CF6" pct={0}/>
        <HealthGauge hp={100} msg="WAITING" />
      </div>
    );
  }

  const v_bar = Math.max(0, Math.min(100, data.voltage / 250 * 100));
  const c_bar = Math.max(0, Math.min(100, data.current / 10  * 100));
  const t_bar = Math.max(0, Math.min(100, data.temperature / 60 * 100));
  const vib_bar = Math.max(0, Math.min(100, Math.min(Math.abs(data.vibration), 2) / 2 * 100));
  const sc_bar = Math.max(0, Math.min(100, data.anomaly_score / (data.threshold * 2) * 100));

  const isBad = data.is_fault;
  const isWarn = data.is_warning;
  const scColor = isBad ? "#EF4444" : (isWarn ? "#F59E0B" : "#10B981");

  return (
    <div className="metrics-grid">
      <MetricCard label="Voltage" value={data.voltage} unit="V" icon="⚡" color="#3B82F6" pct={v_bar}/>
      <MetricCard label="Current" value={data.current} unit="A" icon="↺" color="#10B981" pct={c_bar}/>
      <MetricCard label="Temperature" value={data.temperature} unit="°C" icon="🌡" color="#F59E0B" pct={t_bar}/>
      <MetricCard label="Vibration" value={data.vibration} unit="g" icon="〰" color="#EF5941" pct={vib_bar}/>
      <MetricCard label="Anomaly Score" value={data.anomaly_score} unit={`thr ${data.threshold}`} icon="📉" color={scColor} pct={sc_bar}/>
      <HealthGauge hp={data.health_pct} msg={isBad ? "CRITICAL" : (isWarn ? "WARNING" : "HEALTHY")} />
    </div>
  );
}

function MetricCard({ label, value, unit, icon, color, pct }) {
  return (
    <div className="metric-card">
      <span className="metric-icon">{icon}</span>
      <span className="metric-label">{label}</span>
      <div className="metric-value">{value}</div>
      <span className="metric-unit">{unit}</span>
      <div className="metric-glow" style={{ background: `linear-gradient(to top, ${color}20, transparent)` }}></div>
      <div className="metric-bar" style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}, ${color}bb)` }}></div>
    </div>
  );
}

function HealthGauge({ hp, msg }) {
  let c1 = "#10B981", c2 = "#00D4FF", ring = "rgba(16,185,129,0.15)";
  if (hp < 80) { c1 = "#F59E0B"; c2 = "#EF5941"; ring = "rgba(245,158,11,0.15)"; }
  if (hp < 50) { c1 = "#EF4444"; c2 = "#EF5941"; ring = "rgba(239,68,68,0.15)"; }

  return (
    <div className="health-ring" style={{ background: ring, border: `1px solid ${c1}30` }}>
      <div style={{ fontSize: '9px', fontWeight: 700, letterSpacing: '2px', color: `${c1}88`, marginBottom: '6px' }}>HEALTH</div>
      <div className="health-value" style={{ background: `linear-gradient(135deg, ${c1}, ${c2})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
        {hp.toFixed(0)}%
      </div>
      <div className="health-status" style={{ color: c1 }}>{msg}</div>
    </div>
  );
}
