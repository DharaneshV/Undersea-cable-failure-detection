import React, { useMemo } from 'react';
import {
  AreaChart, Area,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea,
} from 'recharts';

/* ── Design constants ───────────────────────────────────────────────────── */
const SENSORS = [
  { key: 'voltage',     label: 'Voltage',     color: '#3B82F6', unit: 'V',  gradId: 'gVoltage'  },
  { key: 'current',     label: 'Current',     color: '#2ed573', unit: 'A',  gradId: 'gCurrent'  },
  { key: 'temperature', label: 'Temperature', color: '#ffa502', unit: '°C', gradId: 'gTemp'     },
  { key: 'vibration',   label: 'Vibration',   color: '#ff6b35', unit: 'g',  gradId: 'gVibration'},
];

const AXIS_STYLE = {
  stroke: 'rgba(255,255,255,0.04)',
  tick:   { fill: 'rgba(230,237,243,0.3)', fontFamily: 'JetBrains Mono, monospace', fontSize: 9 },
};

const GRID_STYLE = { strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.05)' };

/* ── Custom tooltip ─────────────────────────────────────────────────────── */
function GlassTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#1e2a3a', border: '1px solid rgba(255,255,255,0.10)',
      borderRadius: 8, padding: '10px 14px',
      fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
      boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
    }}>
      {payload.map((entry, i) => (
        <div key={i} style={{
          display: 'flex', justifyContent: 'space-between',
          gap: 16, marginBottom: i < payload.length - 1 ? 3 : 0,
        }}>
          <span style={{ color: 'rgba(230,237,243,0.5)' }}>{entry.name}</span>
          <span style={{ color: entry.color || entry.stroke, fontWeight: 600 }}>
            {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ── Gradient defs ──────────────────────────────────────────────────────── */
function GradientDefs({ aboveThreshold }) {
  return (
    <defs>
      {SENSORS.map(s => (
        <linearGradient key={s.gradId} id={s.gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor={s.color} stopOpacity={0.55} />
          <stop offset="100%" stopColor={s.color} stopOpacity={0.02} />
        </linearGradient>
      ))}
      <linearGradient id="gAnomaly" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%"   stopColor={aboveThreshold ? '#ff4757' : '#8B5CF6'} stopOpacity={0.5} />
        <stop offset="100%" stopColor={aboveThreshold ? '#ff4757' : '#8B5CF6'} stopOpacity={0.02} />
      </linearGradient>
    </defs>
  );
}

/* ── Sparkline strip ────────────────────────────────────────────────────── */
function SparklineStrip({ data }) {
  return (
    <div>
      <div className="chart-label-strip">
        <span className="chart-label">Sensor Overview</span>
        <div className="legend-pills">
          {SENSORS.map(s => (
            <div key={s.key} className="legend-pill">
              <div className="legend-dot" style={{ background: s.color }} />
              {s.label}
            </div>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={52}>
        <LineChart data={data} margin={{ top: 2, right: 4, left: 4, bottom: 2 }}>
          {SENSORS.map(s => (
            <Line
              key={s.key}
              type="monotone" dataKey={s.key} name={s.label}
              stroke={s.color} strokeWidth={1.5}
              dot={false} isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Individual sensor area chart ───────────────────────────────────────── */
function SensorChart({ sensor, data }) {
  const ticks = data.length > 0 ? [data[data.length - 1]?.timestamp] : [];
  return (
    <div>
      <div className="chart-label-strip">
        <div className="legend-dot" style={{ background: sensor.color, width: 8, height: 8, borderRadius: '50%' }} />
        <span className="chart-label">{sensor.label} ({sensor.unit})</span>
      </div>
      <ResponsiveContainer width="100%" height={110}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id={`${sensor.gradId}-inner`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={sensor.color} stopOpacity={0.55} />
              <stop offset="100%" stopColor={sensor.color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis
            dataKey="timestamp" hide={false}
            ticks={ticks}
            tickFormatter={v => v ? String(v).split(' ')[1]?.slice(0, 5) ?? '' : ''}
            {...AXIS_STYLE}
            axisLine={false} tickLine={false}
          />
          <YAxis {...AXIS_STYLE} axisLine={false} tickLine={false} width={36} />
          <Tooltip content={<GlassTooltip />} />
          <Area
            type="monotone" dataKey={sensor.key} name={sensor.label}
            stroke={sensor.color} strokeWidth={2}
            fill={`url(#${sensor.gradId}-inner)`}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Anomaly score chart ─────────────────────────────────────────────────── */
function AnomalyChart({ data, threshold }) {
  const latestScore   = data[data.length - 1]?.anomaly_score ?? 0;
  const aboveThreshold = threshold && latestScore > threshold;
  const fillColor     = aboveThreshold ? '#ff4757' : '#8B5CF6';

  return (
    <div>
      <div className="chart-label-strip">
        <div className="legend-dot" style={{ background: fillColor, width: 8, height: 8, borderRadius: '50%' }} />
        <span className="chart-label">Anomaly Score</span>
        {threshold && (
          <span style={{ fontSize: 10, color: 'rgba(255,71,87,0.7)', fontFamily: 'var(--mono)' }}>
            thr: {threshold.toFixed(5)}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={120}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id="gAnomalyDyn" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={fillColor} stopOpacity={0.5}  />
              <stop offset="100%" stopColor={fillColor} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis dataKey="timestamp" hide {...AXIS_STYLE} />
          <YAxis {...AXIS_STYLE} axisLine={false} tickLine={false} width={36} domain={[0, 'auto']} />
          <Tooltip content={<GlassTooltip />} />
          {threshold && (
            <ReferenceArea
              y1={threshold} y2={1.0}
              fill="rgba(255,71,87,0.08)"
              strokeOpacity={0}
            />
          )}
          {threshold && (
            <ReferenceLine
              y={threshold} stroke="#ff4757" strokeDasharray="4 4" strokeWidth={1.5}
              label={{ position: 'right', value: 'threshold', fill: '#ff4757', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
            />
          )}
          <Area
            type="monotone" dataKey="anomaly_score" name="Score"
            stroke={fillColor} strokeWidth={2}
            fill="url(#gAnomalyDyn)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Main export ─────────────────────────────────────────────────────────── */
export default function LiveCharts({ data, threshold }) {
  if (!data || data.length === 0) {
    return (
      <div style={{
        height: 480, display: 'flex', alignItems: 'center',
        justifyContent: 'center', color: 'rgba(230,237,243,0.25)',
        fontStyle: 'italic', fontSize: 13,
      }}>
        Awaiting telemetry…
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Sparkline overview */}
      <SparklineStrip data={data} />

      {/* 2×2 sensor grid */}
      <div className="chart-grid-2x2">
        {SENSORS.map(s => (
          <SensorChart key={s.key} sensor={s} data={data} />
        ))}
      </div>

      {/* Anomaly score — full width */}
      <AnomalyChart data={data} threshold={threshold} />
    </div>
  );
}
