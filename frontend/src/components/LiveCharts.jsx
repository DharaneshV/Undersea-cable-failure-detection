import React, { useMemo } from 'react';
import {
  AreaChart, Area,
  LineChart, Line,
  XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea,
} from 'recharts';

/* ── Ocean sensor palette — matches MetricsGrid ──────────────────────────── */
const SENSORS = [
  { key: 'voltage',     label: 'Voltage',     color: '#0084ff', unit: 'V',  gradId: 'gVoltage'  },
  { key: 'current',     label: 'Current',     color: '#00ffc8', unit: 'A',  gradId: 'gCurrent'  },
  { key: 'temperature', label: 'Temperature', color: '#ffab40', unit: '°C', gradId: 'gTemp'     },
  { key: 'vibration',   label: 'Vibration',   color: '#ff4d6d', unit: 'g',  gradId: 'gVibration'},
];

const AXIS_STYLE = {
  stroke: 'rgba(0,184,212,0.06)',
  tick:   { fill: 'rgba(107,163,176,0.6)', fontFamily: 'Space Mono, monospace', fontSize: 9 },
};

const GRID_STYLE = {
  strokeDasharray: '3 6',
  stroke: 'rgba(0,184,212,0.07)',
};

/* ── Deep-sea glass tooltip ──────────────────────────────────────────────── */
function OceanTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(4,19,34,0.97)',
      border: '1px solid rgba(0,184,212,0.18)',
      borderRadius: 8, padding: '10px 14px',
      fontSize: 11, fontFamily: 'Space Mono, monospace',
      boxShadow: '0 8px 32px rgba(0,0,0,0.7)',
    }}>
      {payload.map((entry, i) => (
        <div key={i} style={{
          display: 'flex', justifyContent: 'space-between',
          gap: 16, marginBottom: i < payload.length - 1 ? 3 : 0,
        }}>
          <span style={{ color: 'rgba(107,163,176,0.7)' }}>{entry.name}</span>
          <span style={{ color: entry.color || entry.stroke, fontWeight: 700 }}>
            {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ── Gradient defs ───────────────────────────────────────────────────────── */
function GradientDefs({ aboveThreshold }) {
  return (
    <defs>
      {SENSORS.map(s => (
        <linearGradient key={s.gradId} id={s.gradId} x1="0" y1="0" x2="0" y2="1">
          {/* Two-stop ocean gradient — strong at top, near-transparent at base */}
          <stop offset="0%"   stopColor={s.color} stopOpacity={0.50} />
          <stop offset="75%"  stopColor={s.color} stopOpacity={0.04} />
          <stop offset="100%" stopColor={s.color} stopOpacity={0.00} />
        </linearGradient>
      ))}
      {/* Anomaly gradient shifts from bioluminescent to danger-red on breach */}
      <linearGradient id="gAnomalyDyn" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%"   stopColor={aboveThreshold ? '#ff4d6d' : '#a855f7'} stopOpacity={0.55} />
        <stop offset="80%"  stopColor={aboveThreshold ? '#ff4d6d' : '#a855f7'} stopOpacity={0.03} />
        <stop offset="100%" stopColor={aboveThreshold ? '#ff4d6d' : '#a855f7'} stopOpacity={0.00} />
      </linearGradient>
    </defs>
  );
}

/* ── Sparkline overview — all 4 sensors on one overlay strip ─────────────── */
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
      <ResponsiveContainer width="100%" height={48}>
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

/* ── Individual sensor area chart ────────────────────────────────────────── */
function SensorChart({ sensor, data }) {
  // Only show the last timestamp as a tick (avoids clutter)
  const ticks = data.length > 0 ? [data[data.length - 1]?.timestamp] : [];
  return (
    <div>
      <div className="chart-label-strip">
        <div className="legend-dot" style={{ background: sensor.color, width: 7, height: 7, borderRadius: '50%' }} />
        <span className="chart-label">{sensor.label} ({sensor.unit})</span>
      </div>
      <ResponsiveContainer width="100%" height={108}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id={`${sensor.gradId}-inner`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={sensor.color} stopOpacity={0.50} />
              <stop offset="80%"  stopColor={sensor.color} stopOpacity={0.03} />
              <stop offset="100%" stopColor={sensor.color} stopOpacity={0.00} />
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
          <Tooltip content={<OceanTooltip />} />
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

/* ── Anomaly score chart — full width ────────────────────────────────────── */
function AnomalyChart({ data, threshold }) {
  const latestScore    = data[data.length - 1]?.anomaly_score ?? 0;
  const aboveThreshold = threshold && latestScore > threshold;
  const lineColor      = aboveThreshold ? '#ff4d6d' : '#a855f7';

  return (
    <div>
      <div className="chart-label-strip">
        <div className="legend-dot"
          style={{ background: lineColor, width: 7, height: 7, borderRadius: '50%',
                   boxShadow: `0 0 6px ${lineColor}88` }} />
        <span className="chart-label">Anomaly Score</span>
        {threshold && (
          <span style={{ fontSize: 10, color: 'rgba(255,77,109,0.7)', fontFamily: 'var(--mono)' }}>
            thr: {threshold.toFixed(5)}
          </span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={118}>
        <AreaChart data={data} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id="gAnomalyInner" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={lineColor} stopOpacity={0.55} />
              <stop offset="80%"  stopColor={lineColor} stopOpacity={0.03} />
              <stop offset="100%" stopColor={lineColor} stopOpacity={0.00} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID_STYLE} />
          <XAxis dataKey="timestamp" hide {...AXIS_STYLE} />
          <YAxis {...AXIS_STYLE} axisLine={false} tickLine={false} width={36} domain={[0, 'auto']} />
          <Tooltip content={<OceanTooltip />} />

          {/* Red zone above threshold */}
          {threshold && (
            <ReferenceArea
              y1={threshold} y2="auto"
              fill="rgba(255,77,109,0.06)"
              strokeOpacity={0}
            />
          )}
          {/* Threshold line */}
          {threshold && (
            <ReferenceLine
              y={threshold}
              stroke="#ff4d6d"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                position: 'right',
                value: 'threshold',
                fill: '#ff4d6d',
                fontSize: 9,
                fontFamily: 'Space Mono, monospace',
              }}
            />
          )}
          <Area
            type="monotone" dataKey="anomaly_score" name="Score"
            stroke={lineColor} strokeWidth={2}
            fill="url(#gAnomalyInner)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Main export ─────────────────────────────────────────────────────────── */
export default function LiveCharts({ data, threshold }) {
  const aboveThreshold = useMemo(
    () => threshold && data?.length > 0 && (data[data.length - 1]?.anomaly_score ?? 0) > threshold,
    [data, threshold]
  );

  if (!data || data.length === 0) {
    return (
      <div style={{
        height: 480, display: 'flex', alignItems: 'center',
        justifyContent: 'center', color: 'rgba(107,163,176,0.35)',
        fontStyle: 'italic', fontSize: 13, fontFamily: 'var(--mono)',
      }}>
        Awaiting telemetry signal…
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      {/* Top sparkline strip — all sensors overlaid */}
      <SparklineStrip data={data} />

      {/* 2×2 sensor grid — individual area charts */}
      <div className="chart-grid-2x2">
        {SENSORS.map(s => (
          <SensorChart key={s.key} sensor={s} data={data} />
        ))}
      </div>

      {/* Anomaly score — full width, threshold band */}
      <AnomalyChart data={data} threshold={threshold} aboveThreshold={aboveThreshold} />
    </div>
  );
}
