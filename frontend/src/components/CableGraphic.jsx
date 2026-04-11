import React, { useRef, useEffect } from 'react';

const CABLE_LENGTH = 500;
const VB_W = 900;
const VB_H = 300;
const MX   = 80;   // margin x
const CY   = 140;  // cable y position
const CW   = VB_W - 2 * MX;

const FAULT_COLORS = {
  cable_cut:          '#ff4757',
  anchor_drag:        '#ffa502',
  overheating:        '#ff6b35',
  insulation_failure: '#7F77DD',
};

function getFaultColor(ftypeRaw) {
  return FAULT_COLORS[ftypeRaw] ?? '#ff4757';
}

function SeabedTerrain() {
  // organic wave below the cable
  const points = [
    `${MX},${CY + 60}`,
    `${MX + CW * 0.1},${CY + 80}`,
    `${MX + CW * 0.2},${CY + 65}`,
    `${MX + CW * 0.35},${CY + 95}`,
    `${MX + CW * 0.5},${CY + 72}`,
    `${MX + CW * 0.62},${CY + 105}`,
    `${MX + CW * 0.75},${CY + 78}`,
    `${MX + CW * 0.88},${CY + 88}`,
    `${VB_W - MX},${CY + 68}`,
  ].join(' ');

  return (
    <>
      <defs>
        <linearGradient id="seabed-grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%"   stopColor="#0a1628" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#040a12" stopOpacity="1" />
        </linearGradient>
        <filter id="turbulence">
          <feTurbulence type="fractalNoise" baseFrequency="0.02 0.04" numOctaves="2" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="3" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </defs>
      {/* depth dotted lines */}
      {[0, -25, -50, -75].map((d, i) => {
        const y = CY + 60 + i * 35;
        return (
          <g key={d}>
            <line
              x1={MX - 30} y1={y} x2={VB_W - MX + 10} y2={y}
              stroke="rgba(255,255,255,0.06)"
              strokeWidth="1"
              strokeDasharray="4 8"
            />
            <text
              x={MX - 35} y={y + 4}
              textAnchor="end" fill="rgba(255,255,255,0.2)"
              fontSize="9" fontFamily="JetBrains Mono, monospace"
            >
              {d}m
            </text>
          </g>
        );
      })}
      {/* seabed fill */}
      <polygon
        points={`${MX},${VB_H} ${points} ${VB_W - MX},${VB_H}`}
        fill="url(#seabed-grad)"
        filter="url(#turbulence)"
        opacity="0.7"
      />
      {/* seafloor highlight line */}
      <polyline
        points={points}
        fill="none"
        stroke="rgba(0,212,255,0.12)"
        strokeWidth="1.5"
      />
    </>
  );
}

function DataPackets({ hasFault }) {
  const color = hasFault ? '#ff4757' : '#00d4ff';
  const dur   = hasFault ? '1.5s' : '3s';
  const cablePath = `M${MX},${CY} L${VB_W - MX},${CY}`;

  return (
    <>
      <defs>
        <filter id="packet-glow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {[0, 1, 2].map(i => (
        <g key={i}>
          <circle r="4" fill={color} opacity="0.9" filter="url(#packet-glow)">
            <animateMotion
              dur={dur}
              begin={`${i}s`}
              repeatCount="indefinite"
              path={cablePath}
            />
          </circle>
          <circle r="8" fill="none" stroke={color} strokeWidth="1" opacity="0">
            <animateMotion
              dur={dur}
              begin={`${i}s`}
              repeatCount="indefinite"
              path={cablePath}
            />
            <animate
              attributeName="opacity"
              values="0.5;0"
              dur="0.8s"
              begin={`${i}s`}
              repeatCount="indefinite"
            />
            <animate
              attributeName="r"
              values="4;14"
              dur="0.8s"
              begin={`${i}s`}
              repeatCount="indefinite"
            />
          </circle>
        </g>
      ))}
    </>
  );
}

function Station({ x, label, distLabel }) {
  return (
    <g>
      <rect
        x={x - 20} y={CY - 18} width="40" height="36" rx="7"
        fill="rgba(0,212,255,0.08)" stroke="#00d4ff" strokeWidth="1.5"
      >
        <animate attributeName="stroke-opacity" values="0.6;1;0.6" dur="3s" repeatCount="indefinite"/>
      </rect>
      <text x={x} y={CY + 5} textAnchor="middle" fill="#00d4ff" fontSize="11" fontWeight="bold">⬡</text>
      <text x={x} y={CY + 40} textAnchor="middle" fill="rgba(240,246,255,0.45)" fontSize="9" fontWeight="600">{label}</text>
      <text x={x} y={CY - 26} textAnchor="middle" fill="rgba(0,212,255,0.4)" fontSize="9" fontFamily="monospace">{distLabel}</text>
    </g>
  );
}

function Repeater({ x, index }) {
  const dist = Math.round(CABLE_LENGTH * index / 5);
  return (
    <g>
      <rect
        x={x - 11} y={CY - 9} width="22" height="18" rx="5"
        fill="rgba(59,130,246,0.1)" stroke="rgba(59,130,246,0.4)" strokeWidth="1"
      />
      <text x={x} y={CY + 4} textAnchor="middle" fill="rgba(59,130,246,0.7)" fontSize="7" fontWeight="bold">R{index}</text>
      <text x={x} y={CY + 22} textAnchor="middle" fill="rgba(240,246,255,0.22)" fontSize="7" fontFamily="monospace">{dist}m</text>
    </g>
  );
}

export default function CableGraphic({ faults }) {
  const hasFault = faults && faults.length > 0;

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        style={{ width: '100%', height: 'auto', minWidth: '600px' }}
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <linearGradient id="cable-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#00d4ff" stopOpacity="0.9" />
            <stop offset="30%"  stopColor="#3B82F6" stopOpacity="0.7" />
            <stop offset="70%"  stopColor="#3B82F6" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#00d4ff" stopOpacity="0.9" />
          </linearGradient>
          <linearGradient id="cable-shadow-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#00d4ff" stopOpacity="0.15" />
            <stop offset="50%"  stopColor="#8B5CF6" stopOpacity="0.1"  />
            <stop offset="100%" stopColor="#00d4ff" stopOpacity="0.15" />
          </linearGradient>
          <filter id="cable-glow">
            <feGaussianBlur stdDeviation="3" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="fault-glow">
            <feGaussianBlur stdDeviation="5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* Seabed terrain */}
        <SeabedTerrain />

        {/* Cable shadow */}
        <line
          x1={MX} y1={CY + 4} x2={VB_W - MX} y2={CY + 4}
          stroke="url(#cable-shadow-grad)" strokeWidth="10" strokeLinecap="round"
        />

        {/* Main cable */}
        <path
          d={`M${MX},${CY} L${VB_W - MX},${CY}`}
          fill="none" stroke="url(#cable-grad)"
          strokeWidth="5" strokeLinecap="round"
          filter="url(#cable-glow)"
        />

        {/* Data packets */}
        <DataPackets hasFault={hasFault} />

        {/* Stations */}
        <Station x={MX}      label="Station A" distLabel="0 m"            />
        <Station x={VB_W-MX} label="Station B" distLabel={`${CABLE_LENGTH} m`} />

        {/* Repeaters */}
        {[1, 2, 3, 4].map(k => (
          <Repeater key={k} x={MX + CW * k / 5} index={k} />
        ))}

        {/* Fault markers */}
        {(faults ?? []).slice(0, 4).map((f, i) => {
          const dist = parseFloat(f.est_distance ?? 0);
          const x    = MX + (dist / CABLE_LENGTH) * CW;
          const fc   = getFaultColor(f.ftype_raw);
          const labelY = i % 2 === 0 ? CY - 22 : CY - 38;
          return (
            <g key={i}>
              {/* pulse ring */}
              <circle cx={x} cy={CY} r="14" fill="none" stroke={fc} strokeWidth="1.5" opacity="0.3">
                <animate attributeName="r"         values="10;22;10" dur="1.8s" repeatCount="indefinite"/>
                <animate attributeName="opacity"   values="0.4;0;0.4" dur="1.8s" repeatCount="indefinite"/>
              </circle>
              {/* core dot */}
              <circle cx={x} cy={CY} r="7" fill={fc} opacity="0.95" filter="url(#fault-glow)">
                <animate attributeName="r"       values="6;9;6"     dur="1.4s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="1;0.65;1"  dur="1.4s" repeatCount="indefinite"/>
              </circle>
              {/* label line */}
              <line x1={x} y1={CY - 9} x2={x} y2={labelY + 4} stroke={fc} strokeWidth="1" opacity="0.5" strokeDasharray="2,2"/>
              {/* label */}
              <text x={x} y={labelY} textAnchor="middle" fill={fc} fontSize="8.5" fontWeight="700">
                {(f.fault_type ?? 'Fault').replace(/_/g, ' ')}
              </text>
              {/* distance badge */}
              <text x={x} y={CY + 20} textAnchor="middle" fill={fc} fontSize="7.5" fontFamily="monospace" opacity="0.7">
                {dist}m
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
