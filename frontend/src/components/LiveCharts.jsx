import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, AreaChart, Area
} from 'recharts';

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: '#131D2E', border: '1px solid rgba(255,255,255,0.12)',
      padding: '10px 14px', borderRadius: '8px',
      fontSize: '11px', fontFamily: 'JetBrains Mono, monospace'
    }}>
      {payload.map((entry, i) => (
        <div key={i} style={{
          color: entry.color, display: 'flex',
          justifyContent: 'space-between', gap: '16px', marginBottom: '2px'
        }}>
          <span style={{ color: 'rgba(240,246,255,0.5)' }}>{entry.name}</span>
          <span style={{ fontWeight: 600 }}>{entry.value}</span>
        </div>
      ))}
    </div>
  );
};

const chartLabel = (text) => (
  <div style={{
    fontSize: '9px', fontWeight: 700, letterSpacing: '1.5px',
    textTransform: 'uppercase', color: 'rgba(240,246,255,0.3)',
    marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px'
  }}>
    <span style={{
      display: 'inline-block', width: '3px', height: '10px',
      background: 'linear-gradient(180deg, #00D4FF, #3B82F6)',
      borderRadius: '2px'
    }}/>
    {text}
  </div>
);

const axisProps = {
  stroke: 'rgba(255,255,255,0.06)',
  fontSize: 10,
  tick: { fill: 'rgba(240,246,255,0.25)', fontFamily: 'JetBrains Mono, monospace' },
};

export default function LiveCharts({ data, threshold }) {
  if (!data || data.length === 0) {
    return (
      <div style={{
        height: 480, display: 'flex', alignItems: 'center',
        justifyContent: 'center', color: 'rgba(240,246,255,0.25)',
        fontStyle: 'italic', fontSize: 13
      }}>
        Awaiting telemetry…
      </div>
    );
  }

  const common = { data, margin: { top: 8, right: 8, left: -20, bottom: 0 } };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', height: '480px' }}>

      {/* Voltage & Current */}
      <div style={{ flex: 1, minHeight: '130px' }}>
        {chartLabel('Voltage & Current')}
        <ResponsiveContainer width="100%" height="100%">
          <LineChart {...common}>
            <XAxis dataKey="timestamp" hide />
            <YAxis {...axisProps} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="voltage"  stroke="#378ADD" strokeWidth={1.4} dot={false} isAnimationActive={false} name="Voltage"/>
            <Line type="monotone" dataKey="current"  stroke="#1D9E75" strokeWidth={1.4} dot={false} isAnimationActive={false} name="Current"/>
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Temperature & Vibration */}
      <div style={{ flex: 1, minHeight: '130px' }}>
        {chartLabel('Temperature & Vibration')}
        <ResponsiveContainer width="100%" height="100%">
          <LineChart {...common}>
            <XAxis dataKey="timestamp" hide />
            <YAxis {...axisProps} />
            <Tooltip content={<CustomTooltip />} />
            <Line type="monotone" dataKey="temperature" stroke="#EF9F27" strokeWidth={1.4} dot={false} isAnimationActive={false} name="Temp"/>
            <Line type="monotone" dataKey="vibration"   stroke="#D85A30" strokeWidth={1.1} dot={false} isAnimationActive={false} name="Vibration"/>
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Anomaly Score */}
      <div style={{ flex: 1, minHeight: '130px' }}>
        {chartLabel('Anomaly Score')}
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart {...common}>
            <XAxis dataKey="timestamp" hide />
            <YAxis {...axisProps} domain={[0, 'auto']} />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone" dataKey="anomaly_score" name="Score"
              stroke="#7F77DD" fill="rgba(127,119,221,0.10)"
              strokeWidth={1.5} isAnimationActive={false}
            />
            {threshold && (
              <ReferenceLine
                y={threshold} stroke="#E24B4A" strokeDasharray="4 3"
                label={{ position: 'right', value: 'threshold', fill: '#E24B4A', fontSize: 10 }}
              />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </div>

    </div>
  );
}
