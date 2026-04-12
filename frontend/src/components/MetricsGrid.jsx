import React from 'react';

/* ── Sea-themed sensor colour palette ────────────────────────────────────── */
// Each sensor maps to a distinct oceanic hue
const SENSOR_OCEAN = {
  voltage:     { color: '#0084ff', bar: 'var(--current)' },   // deep current blue
  current:     { color: '#00ffc8', bar: 'var(--bio)'     },   // bioluminescent
  temperature: { color: '#ffab40', bar: 'var(--warn)'    },   // hydrothermal amber
  vibration:   { color: '#ff4d6d', bar: 'var(--danger)'  },   // danger red
  anomaly:     { color: null,      bar: null              },   // dynamic — set by health
};

/* ── SVG icon components ─────────────────────────────────────────────────── */
function BoltIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24" fill={color}>
      <path d="M13 2L4.5 13.5H11L10 22L19.5 10.5H13L13 2z"/>
    </svg>
  );
}
function CurrentIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round">
      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/>
      <polyline points="17 6 23 6 23 12"/>
    </svg>
  );
}
function ThermoIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
      <path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z"/>
    </svg>
  );
}
function WaveIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
      <path d="M2 12 C4 8 6 8 8 12 C10 16 12 16 14 12 C16 8 18 8 20 12 C21 14 22 13 22 12"/>
    </svg>
  );
}
function AlertIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2" strokeLinecap="round">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
      <line x1="12" y1="9"  x2="12"  y2="13"/>
      <line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
  );
}

/* ── Health gauge — SVG semicircle arc ───────────────────────────────────── */
const ARC_RADIUS    = 45;
const ARC_CX        = 60;
const ARC_CY        = 62;
const CIRCUMFERENCE = Math.PI * ARC_RADIUS;  // half-circle arc length

function healthColor(hp) {
  if (hp > 70) return '#00ffc8';  // bio-green — healthy
  if (hp > 40) return '#ffab40';  // amber    — warning
  return '#ff4d6d';               // danger   — critical
}
function healthMsg(hp) {
  if (hp > 70) return 'HEALTHY';
  if (hp > 40) return 'DEGRADED';
  return 'CRITICAL';
}

/* Thin secondary track arc path builder */
const arcPath = `M ${ARC_CX - ARC_RADIUS},${ARC_CY} A ${ARC_RADIUS} ${ARC_RADIUS} 0 0 1 ${ARC_CX + ARC_RADIUS},${ARC_CY}`;

function HealthGauge({ hp, msg }) {
  const pct   = Math.max(0, Math.min(100, hp ?? 100));
  const fill  = (pct / 100) * CIRCUMFERENCE;
  const gap   = CIRCUMFERENCE - fill;
  const color = healthColor(pct);

  return (
    <div className="health-card">
      <div className="health-label-top">System Health</div>
      <svg viewBox="0 0 120 76" width="150" height="90" aria-label={`Health ${pct.toFixed(0)}%`}>
        {/* Track ring */}
        <path d={arcPath} fill="none" stroke="rgba(0,255,200,0.08)" strokeWidth="10" strokeLinecap="round"/>
        {/* Tick marks every 20% */}
        {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((t, i) => {
          const angle = Math.PI * t;           // 0 → π (left → right)
          const ix = ARC_CX - ARC_RADIUS * Math.cos(angle);
          const iy = ARC_CY - ARC_RADIUS * Math.sin(angle);
          const ox = ARC_CX - (ARC_RADIUS + 6) * Math.cos(angle);
          const oy = ARC_CY - (ARC_RADIUS + 6) * Math.sin(angle);
          return (
            <line key={i}
              x1={ix} y1={iy} x2={ox} y2={oy}
              stroke="rgba(0,184,212,0.3)" strokeWidth="1.5"
            />
          );
        })}
        {/* Filled arc — transitions smoothly */}
        <path
          d={arcPath}
          fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={`${fill} ${gap}`}
          style={{ transition: 'stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.6s ease' }}
        />
        {/* Glow duplicate */}
        <path
          d={arcPath}
          fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${fill} ${gap}`}
          opacity="0.12"
          style={{ transition: 'stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1), stroke 0.6s ease' }}
        />
        {/* Centre percentage */}
        <text x={ARC_CX} y={ARC_CY - 6} textAnchor="middle"
          fill={color} fontSize="22" fontWeight="700"
          fontFamily="Space Mono, monospace"
          style={{ transition: 'fill 0.6s ease' }}>
          {pct.toFixed(0)}
        </text>
        <text x={ARC_CX} y={ARC_CY + 12} textAnchor="middle"
          fill="rgba(200,240,245,0.3)" fontSize="7" fontWeight="700"
          letterSpacing="2" fontFamily="Oxanium, sans-serif">
          HEALTH %
        </text>
      </svg>
      <div className="health-label-status" style={{ color }}>{msg}</div>
    </div>
  );
}

/* ── Metric card ─────────────────────────────────────────────────────────── */
const ICONS = {
  voltage:     c => <BoltIcon    color={c} />,
  current:     c => <CurrentIcon color={c} />,
  temperature: c => <ThermoIcon  color={c} />,
  vibration:   c => <WaveIcon    color={c} />,
  anomaly:     c => <AlertIcon   color={c} />,
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
  if (delta > 0) return { symbol: `▲ ${Math.abs(delta).toFixed(3)}`, cls: 'delta-up'   };
  return               { symbol: `▼ ${Math.abs(delta).toFixed(3)}`, cls: 'delta-down' };
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
      <div className="metric-value" style={{ color }}>{formatValue(field, value)}</div>
      <div className="metric-unit">{unit}</div>
      <div className={`metric-delta ${arrow.cls}`}>{arrow.symbol}</div>
      {/* Ambient glow at card base */}
      <div className="metric-glow" style={{ background: `linear-gradient(to top, ${color}28, transparent)` }}/>
      {/* Coloured sensor bar at card bottom */}
      <div className="metric-bar"
        style={{
          width: `${pct}%`,
          background: `linear-gradient(90deg, ${color}, ${color}66)`,
          boxShadow: `0 0 8px ${color}44`,
        }}
      />
    </div>
  );
}

/* ── Skeleton ────────────────────────────────────────────────────────────── */
function SkeletonCard({ delay }) {
  return (
    <div className="metric-card" style={{ animationDelay: `${delay}ms` }}>
      <div className="skeleton" style={{ height: 10, width: '55%', marginBottom: 14 }}/>
      <div className="skeleton" style={{ height: 28, width: '75%', marginBottom: 8  }}/>
      <div className="skeleton" style={{ height: 8,  width: '35%' }}/>
    </div>
  );
}

/* ── Main export ─────────────────────────────────────────────────────────── */
export default function MetricsGrid({ data, prevData }) {
  if (!data) {
    return (
      <div className="metrics-grid">
        <SkeletonCard delay={0}   />
        <SkeletonCard delay={60}  />
        <SkeletonCard delay={120} />
        <SkeletonCard delay={180} />
        <SkeletonCard delay={240} />
        {/* Empty health gauge */}
        <div className="health-card">
          <div className="health-label-top">System Health</div>
          <svg viewBox="0 0 120 76" width="150" height="90" aria-hidden="true">
            <path d={arcPath} fill="none" stroke="rgba(0,255,200,0.07)" strokeWidth="10" strokeLinecap="round"/>
          </svg>
          <div className="health-label-status" style={{ color: 'var(--txt-muted)' }}>WAITING</div>
        </div>
      </div>
    );
  }

  const isBad  = data.is_fault;
  const isWarn = data.is_warning;
  // Anomaly card colour responds to state
  const scColor = isBad ? '#ff4d6d' : isWarn ? '#ffab40' : '#00ffc8';

  const pct = (v, max) => Math.max(0, Math.min(100, (v / max) * 100));

  return (
    <div className="metrics-grid">
      <HealthGauge
        hp={data.health_pct ?? 100}
        msg={healthMsg(data.health_pct ?? 100)}
      />
      <MetricCard
        field="voltage" label="Voltage" value={data.voltage} prevValue={prevData?.voltage}
        unit="V"
        color={SENSOR_OCEAN.voltage.color}
        pct={pct(data.voltage, 260)} animDelay={60}
      />
      <MetricCard
        field="current" label="Current" value={data.current} prevValue={prevData?.current}
        unit="A"
        color={SENSOR_OCEAN.current.color}
        pct={pct(data.current, 10)} animDelay={120}
      />
      <MetricCard
        field="temperature" label="Temperature" value={data.temperature} prevValue={prevData?.temperature}
        unit="°C"
        color={SENSOR_OCEAN.temperature.color}
        pct={pct(data.temperature, 60)} animDelay={180}
      />
      <MetricCard
        field="anomaly" label="Anomaly Score" value={data.anomaly_score} prevValue={prevData?.anomaly_score}
        unit={`thr ${data.threshold?.toFixed(5) ?? '--'}`}
        color={scColor}
        pct={pct(data.anomaly_score, data.threshold * 2)} animDelay={240}
        anomalyActive={isBad}
      />
    </div>
  );
}
