import React from 'react';

/* ── Sea-themed sensor colour palette ────────────────────────────────────── */
const SENSOR_OCEAN = {
  voltage:     { color: '#0084ff', bar: 'var(--current)' },
  current:     { color: '#0ea5e9', bar: 'var(--bio)'     },
  temperature: { color: '#f59e0b', bar: 'var(--warn)'    },
  vibration:   { color: '#f43f5e', bar: 'var(--danger)'  },
  anomaly:     { color: null,      bar: null              },
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
function OsnrIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 10h3l3 9 5-15 3 9h5" />
    </svg>
  );
}
function BerIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="6" height="6" rx="1" />
      <rect x="15" y="3" width="6" height="6" rx="1" />
      <rect x="3" y="15" width="6" height="6" rx="1" />
      <path d="M16 16l4 4m0-4l-4 4" strokeWidth="2.5" />
      <path d="M9 6h6M6 9v6" strokeWidth="1.5" strokeDasharray="2 2" />
    </svg>
  );
}
function PowerIcon({ color }) {
  return (
    <svg aria-hidden="true" width="14" height="14" viewBox="0 0 24 24"
      fill="none" stroke={color} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="6" cy="12" r="3" />
      <path d="M9 12h13M17 8l5 4-5 4" />
      <path d="M3 8l1-1M3 16l1 1M6 5V3M6 19v2" strokeWidth="1.5" />
    </svg>
  );
}

/* ── Health Bar ───────────────────────────────────────────────────────────── */
function healthColor(hp) {
  if (hp > 70) return '#10b981';
  if (hp > 40) return '#f59e0b';
  return '#f43f5e';
}
function healthMsg(hp) {
  if (hp > 70) return 'Healthy';
  if (hp > 40) return 'Degraded';
  return 'Critical';
}

function HealthBar({ hp }) {
  const pct   = Math.max(0, Math.min(100, hp ?? 100));
  const color = healthColor(pct);
  const msg   = healthMsg(pct);

  return (
    <div
      className="health-card"
      role="meter"
      aria-valuenow={Math.round(pct)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label="Cable health"
    >
      <div className="health-label-top">System Health</div>

      {/* Big percentage number */}
      <div
        className="health-pct-value"
        style={{
          color,
          transition: 'color 0.6s ease',
        }}
      >
        {pct.toFixed(0)}
        <span className="health-pct-sign">%</span>
      </div>

      {/* Progress bar track */}
      <div className="health-bar-track">
        <div
          className="health-bar-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}cc, ${color})`,
            boxShadow: `0 0 8px ${color}66`,
            transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1), background 0.6s ease, box-shadow 0.6s ease',
          }}
        />
      </div>

      {/* Tick labels */}
      <div className="health-bar-ticks">
        <span>0</span>
        <span>25</span>
        <span>50</span>
        <span>75</span>
        <span>100</span>
      </div>

      <div
        className="health-label-status"
        style={{ color, transition: 'color 0.6s ease' }}
      >
        {msg}
      </div>
    </div>
  );
}

/* ── Metric card ─────────────────────────────────────────────────────────── */
const ICONS = {
  voltage:       c => <BoltIcon    color={c} />,
  current:       c => <CurrentIcon color={c} />,
  temperature:   c => <ThermoIcon  color={c} />,
  vibration:     c => <WaveIcon    color={c} />,
  anomaly:       c => <AlertIcon   color={c} />,
  optical_osnr:  c => <OsnrIcon    color={c} />,
  optical_ber:   c => <BerIcon     color={c} />,
  optical_power: c => <PowerIcon   color={c} />,
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
      <div className="metric-glow" style={{ background: `linear-gradient(to top, ${color}28, transparent)` }}/>
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
        <div className="health-card">
          <div className="health-label-top">System Health</div>
          <div className="health-pct-value" style={{ color: 'var(--txt-muted)' }}>
            --<span className="health-pct-sign">%</span>
          </div>
          <div className="health-bar-track">
            <div className="health-bar-fill" style={{ width: '0%', background: 'var(--depth-3)' }} />
          </div>
          <div className="health-bar-ticks">
            <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
          </div>
          <div className="health-label-status" style={{ color: 'var(--txt-muted)' }}>Waiting</div>
        </div>
        <SkeletonCard delay={0}   />
        <SkeletonCard delay={60}  />
        <SkeletonCard delay={120} />
        <SkeletonCard delay={180} />
        <SkeletonCard delay={240} />
      </div>
    );
  }

  const isBad  = data.is_fault;
  const isWarn = data.is_warning;
  const scColor = isBad ? '#f43f5e' : isWarn ? '#f59e0b' : '#10b981';

  const pct = (v, max) => Math.max(0, Math.min(100, (v / max) * 100));

  return (
    <div className="metrics-grid">
      <HealthBar hp={data.health_pct ?? 100} />

      {data.cable_domain_id === 1 ? (
        <>
          <MetricCard
            field="optical_osnr" label="OSNR" value={data.optical_osnr} prevValue={prevData?.optical_osnr}
            unit="dB" color={SENSOR_OCEAN.voltage.color} pct={pct(data.optical_osnr, 40)} animDelay={60}
          />
          <MetricCard
            field="optical_ber" label="Log BER" value={data.optical_ber} prevValue={prevData?.optical_ber}
            unit="log" color={SENSOR_OCEAN.current.color} pct={pct(data.optical_ber + 15, 15)} animDelay={120}
          />
          <MetricCard
            field="optical_power" label="Out Power" value={data.optical_power} prevValue={prevData?.optical_power}
            unit="dBm" color={SENSOR_OCEAN.temperature.color} pct={pct(data.optical_power, 20)} animDelay={180}
          />
        </>
      ) : (
        <>
          <MetricCard
            field="voltage" label="Voltage" value={data.voltage} prevValue={prevData?.voltage}
            unit="V" color={SENSOR_OCEAN.voltage.color} pct={pct(data.voltage, 260)} animDelay={60}
          />
          <MetricCard
            field="current" label="Current" value={data.current} prevValue={prevData?.current}
            unit="A" color={SENSOR_OCEAN.current.color} pct={pct(data.current, 10)} animDelay={120}
          />
          <MetricCard
            field="temperature" label="Temp" value={data.temperature} prevValue={prevData?.temperature}
            unit="°C" color={SENSOR_OCEAN.temperature.color} pct={pct(data.temperature, 60)} animDelay={180}
          />
        </>
      )}
      <MetricCard
        field="vibration" label="Vibration" value={data.vibration} prevValue={prevData?.vibration}
        unit="g" color={SENSOR_OCEAN.vibration.color} pct={pct(Math.abs(data.vibration ?? 0), 2)} animDelay={240}
      />
      <MetricCard
        field="anomaly" label="Anomaly Score" value={data.anomaly_score} prevValue={prevData?.anomaly_score}
        unit={`thr ${data.threshold?.toFixed(4) ?? '--'}`}
        color={scColor}
        pct={pct(data.anomaly_score, (data.threshold ?? 0.1) * 2)} animDelay={300}
        anomalyActive={isBad}
      />
    </div>
  );
}
