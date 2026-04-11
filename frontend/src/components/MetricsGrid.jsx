import React, { useRef } from 'react';

/* ── SVG Icon Components ─────────────────────────────────────────────────── */
function BoltIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill={color}>
      <path d="M13 2L4.5 13.5H11L10 22L19.5 10.5H13L13 2z"/>
    </svg>
  );
}
function CurrentIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
      <polyline points="17 6 23 6 23 12"/>
    </svg>
  );
}
function ThermoIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
      <path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z"/>
    </svg>
  );
}
function WaveIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
      <path d="M2 12 C4 8 6 8 8 12 C10 16 12 16 14 12 C16 8 18 8 20 12 C21 14 22 13 22 12"/>
    </svg>
  );
}
function AlertIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  );
}

/* ── Health gauge (SVG semicircle arc) ───────────────────────────────────── */
const ARC_RADIUS  = 45;
const ARC_CX      = 60;
const ARC_CY      = 60;
const CIRCUMFERENCE = Math.PI * ARC_RADIUS; // half-circle

function healthColor(hp) {
  if (hp > 70) return '#2ed573';
  if (hp > 40) return '#ffa502';
  return '#ff4757';
}

function HealthGauge({ hp, msg }) {
  const pct    = Math.max(0, Math.min(100, hp));
  const fill   = pct / 100 * CIRCUMFERENCE;
  const gap    = CIRCUMFERENCE - fill;
  const color  = healthColor(pct);

  return (
    <div className="health-card">
      <div className="health-label-top">System Health</div>
      <svg viewBox="0 0 120 70" width="140" height="82" aria-hidden="true">
        {/* track */}
        <path
          d={`M ${ARC_CX - ARC_RADIUS},${ARC_CY} A ${ARC_RADIUS} ${ARC_RADIUS} 0 0 1 ${ARC_CX + ARC_RADIUS},${ARC_CY}`}
          fill="none" stroke="#1e2a3a" strokeWidth="10" strokeLinecap="round"
        />
        {/* fill arc */}
        <path
          d={`M ${ARC_CX - ARC_RADIUS},${ARC_CY} A ${ARC_RADIUS} ${ARC_RADIUS} 0 0 1 ${ARC_CX + ARC_RADIUS},${ARC_CY}`}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${fill} ${gap}`}
          style={{ transition: 'stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.6s ease' }}
        />
        {/* center value */}
        <text x={ARC_CX} y={ARC_CY - 4} textAnchor="middle"
          fill={color} fontSize="20" fontWeight="800"
          fontFamily="JetBrains Mono, monospace"
          style={{ transition: 'fill 0.6s ease' }}
        >
          {pct.toFixed(0)}
        </text>
        <text x={ARC_CX} y={ARC_CY + 12} textAnchor="middle"
          fill="rgba(230,237,243,0.4)" fontSize="8" fontWeight="600"
          letterSpacing="1"
        >
          HP
        </text>
      </svg>
      <div className="health-label-status" style={{ color }}>{msg}</div>
    </div>
  );
}

/* ── Metric card ─────────────────────────────────────────────────────────── */
const ICONS = {
  voltage:     (c) => <BoltIcon    color={c} />,
  current:     (c) => <CurrentIcon color={c} />,
  temperature: (c) => <ThermoIcon  color={c} />,
  vibration:   (c) => <WaveIcon    color={c} />,
  anomaly:     (c) => <AlertIcon   color={c} />,
};

function formatValue(key, val) {
  if (val == null || isNaN(val)) return '--';
  const n = Number(val);
  if (key === 'voltage')     return n.toFixed(1);
  if (key === 'current')     return n.toFixed(2);
  if (key === 'temperature') return n.toFixed(1);
  if (key === 'vibration')   return n.toFixed(3);
  return n.toFixed(4);
}

function deltaArrow(delta) {
  if (delta == null || Math.abs(delta) < 0.0001) return { symbol: '–', cls: 'delta-flat' };
  if (delta > 0) return { symbol: `▲ ${Math.abs(delta).toFixed(3)}`, cls: 'delta-up' };
  return { symbol: `▼ ${Math.abs(delta).toFixed(3)}`, cls: 'delta-down' };
}

function MetricCard({ field, label, value, prevValue, unit, color, pct, animDelay, anomalyActive }) {
  const delta = (value != null && prevValue != null) ? value - prevValue : null;
  const arrow = deltaArrow(delta);
  return (
    <div
      className={`metric-card${anomalyActive ? ' anomaly-active' : ''}`}
      style={{ animationDelay: `${animDelay}ms` }}
    >
      <div className="metric-header">
        <div className="metric-icon-wrap" style={{ background: `${color}18` }}>
          {ICONS[field]?.(color)}
        </div>
        <span className="metric-label">{label}</span>
      </div>
      <div className="metric-value">{formatValue(field, value)}</div>
      <div className="metric-unit">{unit}</div>
      <div className={`metric-delta ${arrow.cls}`}>{arrow.symbol}</div>
      <div className="metric-glow" style={{ background: `linear-gradient(to top, ${color}20, transparent)` }}/>
      <div className="metric-bar"   style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}, ${color}99)` }}/>
    </div>
  );
}

/* ── Skeleton ────────────────────────────────────────────────────────────── */
function SkeletonCard({ delay }) {
  return (
    <div className="metric-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="skeleton" style={{ height: 12, width: '60%', marginBottom: 14 }}/>
      <div className="skeleton" style={{ height: 30, width: '80%', marginBottom: 8 }}/>
      <div className="skeleton" style={{ height: 10, width: '40%' }}/>
    </div>
  );
}

/* ── Main export ─────────────────────────────────────────────────────────── */
export default function MetricsGrid({ data, prevData }) {
  if (!data) {
    return (
      <div className="metrics-grid">
        <SkeletonCard delay={0}  />
        <SkeletonCard delay={60} />
        <SkeletonCard delay={120}/>
        <SkeletonCard delay={180}/>
        <SkeletonCard delay={240}/>
        <div className="health-card">
          <div className="health-label-top">System Health</div>
          <svg viewBox="0 0 120 70" width="140" height="82" aria-hidden="true">
            <path d={`M ${ARC_CX - ARC_RADIUS},${ARC_CY} A ${ARC_RADIUS} ${ARC_RADIUS} 0 0 1 ${ARC_CX + ARC_RADIUS},${ARC_CY}`}
              fill="none" stroke="#1e2a3a" strokeWidth="10" strokeLinecap="round"/>
          </svg>
          <div className="health-label-status" style={{ color: 'var(--text-muted)' }}>WAITING</div>
        </div>
      </div>
    );
  }

  const isBad  = data.is_fault;
  const isWarn = data.is_warning;
  const scColor = isBad ? '#ff4757' : (isWarn ? '#ffa502' : '#2ed573');

  const pct = (v, max) => Math.max(0, Math.min(100, (v / max) * 100));

  return (
    <div className="metrics-grid">
      <HealthGauge
        hp={data.health_pct ?? 100}
        msg={isBad ? 'CRITICAL' : (isWarn ? 'WARNING' : 'HEALTHY')}
      />
      <MetricCard
        field="voltage" label="Voltage" value={data.voltage} prevValue={prevData?.voltage}
        unit="V" color="#3B82F6" pct={pct(data.voltage, 260)} animDelay={60}
      />
      <MetricCard
        field="current" label="Current" value={data.current} prevValue={prevData?.current}
        unit="A" color="#2ed573" pct={pct(data.current, 10)} animDelay={120}
      />
      <MetricCard
        field="temperature" label="Temperature" value={data.temperature} prevValue={prevData?.temperature}
        unit="°C" color="#ffa502" pct={pct(data.temperature, 60)} animDelay={180}
      />
      <MetricCard
        field="anomaly" label="Anomaly Score" value={data.anomaly_score} prevValue={prevData?.anomaly_score}
        unit={`thr ${data.threshold?.toFixed(5) ?? '--'}`}
        color={scColor} pct={pct(data.anomaly_score, data.threshold * 2)} animDelay={240}
        anomalyActive={isBad}
      />
    </div>
  );
}
