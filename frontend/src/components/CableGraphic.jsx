import React from 'react';

const CABLE_LENGTH = 500;
const VB_W = 900;
const VB_H = 320;
const MX   = 80;    // margin x
const CY   = 130;   // cable y position — higher to leave room for seabed
const CW   = VB_W - 2 * MX;

const FAULT_COLORS = {
  cable_cut:          '#ff4d6d',
  anchor_drag:        '#ffab40',
  overheating:        '#ff7d45',
  insulation_failure: '#a855f7',
};

function getFaultColor(ftypeRaw) {
  return FAULT_COLORS[ftypeRaw] ?? '#ff4d6d';
}

/* ── Seabed terrain ──────────────────────────────────────────────────────── */
function SeabedTerrain() {
  // Organic undulating seafloor 70px below the cable
  const floorY  = CY + 70;
  const pts = [
    [MX        , floorY + 12],
    [MX + CW*.10, floorY + 34],
    [MX + CW*.20, floorY + 16],
    [MX + CW*.30, floorY + 48],
    [MX + CW*.42, floorY + 22],
    [MX + CW*.54, floorY + 58],
    [MX + CW*.64, floorY + 30],
    [MX + CW*.76, floorY + 52],
    [MX + CW*.88, floorY + 38],
    [VB_W - MX  , floorY + 24],
  ];
  const polyPts = pts.map(p => p.join(',')).join(' ');

  return (
    <>
      <defs>
        {/* Seabed depth gradient — dark teal to near-black */}
        <linearGradient id="seabed-grad" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%"   stopColor="#0a2035" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#020912" stopOpacity="1"    />
        </linearGradient>
        {/* Caustic light blobs — bioluminescent patches on seafloor */}
        <radialGradient id="caustic-1" cx="30%" cy="80%" r="15%">
          <stop offset="0%"   stopColor="#00ffc8" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#00ffc8" stopOpacity="0"    />
        </radialGradient>
        <radialGradient id="caustic-2" cx="68%" cy="90%" r="12%">
          <stop offset="0%"   stopColor="#00b8d4" stopOpacity="0.10" />
          <stop offset="100%" stopColor="#00b8d4" stopOpacity="0"    />
        </radialGradient>
        {/* slight wave turbulence on seafloor edge */}
        <filter id="seafloor-blur">
          <feTurbulence type="fractalNoise" baseFrequency="0.015 0.03" numOctaves="2" result="noise" />
          <feDisplacementMap in="SourceGraphic" in2="noise" scale="4" />
        </filter>
      </defs>

      {/* Depth ruler lines */}
      {[0, 20, 40, 60].map((d, i) => {
        const y = CY + (i + 1) * 18;
        return (
          <g key={d}>
            <line
              x1={MX - 28} y1={y} x2={VB_W - MX + 8} y2={y}
              stroke="rgba(0,184,212,0.07)" strokeWidth="1" strokeDasharray="3 7"
            />
            <text
              x={MX - 32} y={y + 4}
              textAnchor="end" fill="rgba(0,184,212,0.2)"
              fontSize="8" fontFamily="Space Mono, monospace"
            >
              -{d}m
            </text>
          </g>
        );
      })}

      {/* Caustic light patches */}
      <rect x={MX} y={floorY} width={CW} height={VB_H - floorY}
        fill="url(#caustic-1)" />
      <rect x={MX} y={floorY} width={CW} height={VB_H - floorY}
        fill="url(#caustic-2)" />

      {/* Seabed fill */}
      <polygon
        points={`${MX},${VB_H} ${polyPts} ${VB_W - MX},${VB_H}`}
        fill="url(#seabed-grad)"
        filter="url(#seafloor-blur)"
        opacity="0.85"
      />
      {/* Seafloor surface highlight — thin bioluminescent line */}
      <polyline
        points={polyPts}
        fill="none"
        stroke="rgba(0,255,200,0.15)"
        strokeWidth="1.5"
      />
    </>
  );
}

/* ── Animated data packets ───────────────────────────────────────────────── */
function DataPackets({ hasFault, healthPct }) {
  // Color shifts from bio-green → danger-red as health drops
  const color = hasFault ? '#ff4d6d'
              : healthPct != null && healthPct < 50 ? '#ffab40'
              : '#00ffc8';
  const dur   = hasFault ? '1.6s' : '3.2s';
  const cablePath = `M${MX},${CY} L${VB_W - MX},${CY}`;

  return (
    <>
      <defs>
        <filter id="packet-glow">
          <feGaussianBlur stdDeviation="2.5" result="blur" />
          <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
        </filter>
      </defs>
      {[0, 1, 2].map(i => (
        <g key={i}>
          {/* Packet core */}
          <circle r="4" fill={color} opacity="0.95" filter="url(#packet-glow)">
            <animateMotion dur={dur} begin={`${i * 1.07}s`} repeatCount="indefinite" path={cablePath} />
          </circle>
          {/* Expanding wake ring */}
          <circle r="6" fill="none" stroke={color} strokeWidth="1" opacity="0">
            <animateMotion dur={dur} begin={`${i * 1.07}s`} repeatCount="indefinite" path={cablePath} />
            <animate attributeName="opacity" values="0.55;0"  dur="0.9s" begin={`${i * 1.07}s`} repeatCount="indefinite" />
            <animate attributeName="r"       values="4;16"    dur="0.9s" begin={`${i * 1.07}s`} repeatCount="indefinite" />
          </circle>
        </g>
      ))}
    </>
  );
}

/* ── Terminal station ────────────────────────────────────────────────────── */
function Station({ x, label, distLabel }) {
  return (
    <g>
      <rect x={x - 22} y={CY - 20} width="44" height="40" rx="8"
        fill="rgba(0,255,200,0.07)" stroke="#00ffc8" strokeWidth="1.5">
        <animate attributeName="stroke-opacity" values="0.5;0.9;0.5" dur="3.5s" repeatCount="indefinite"/>
      </rect>
      {/* Sonar icon */}
      <circle cx={x} cy={CY} r="5" fill="none" stroke="#00ffc8" strokeWidth="1.5" opacity="0.8" />
      <circle cx={x} cy={CY} r="2" fill="#00ffc8" opacity="0.9" />
      <text x={x} y={CY + 32} textAnchor="middle" fill="rgba(200,240,245,0.5)"
        fontSize="9" fontFamily="Oxanium, sans-serif" fontWeight="600">
        {label}
      </text>
      <text x={x} y={CY - 28} textAnchor="middle" fill="rgba(0,255,200,0.4)"
        fontSize="8" fontFamily="Space Mono, monospace">
        {distLabel}
      </text>
    </g>
  );
}

/* ── Mid-route repeater node ─────────────────────────────────────────────── */
function Repeater({ x, index }) {
  const dist = Math.round(CABLE_LENGTH * index / 5);
  return (
    <g>
      <rect x={x - 10} y={CY - 8} width="20" height="16" rx="4"
        fill="rgba(0,132,255,0.10)" stroke="rgba(0,132,255,0.40)" strokeWidth="1"
      />
      <text x={x} y={CY + 4} textAnchor="middle"
        fill="rgba(0,132,255,0.75)" fontSize="7" fontWeight="700"
        fontFamily="Space Mono, monospace">
        R{index}
      </text>
      <text x={x} y={CY + 22} textAnchor="middle"
        fill="rgba(200,240,245,0.18)" fontSize="7" fontFamily="Space Mono, monospace">
        {dist}m
      </text>
    </g>
  );
}

/* ── Main export ─────────────────────────────────────────────────────────── */
export default function CableGraphic({ faults, healthPct }) {
  const hasFault = faults && faults.length > 0;

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        style={{ width: '100%', height: 'auto', minWidth: '600px' }}
        xmlns="http://www.w3.org/2000/svg"
        aria-label="Undersea cable route with fault markers"
        role="img"
      >
        <defs>
          {/* Cable gradient — bio-teal → deep-blue → bio-teal */}
          <linearGradient id="cable-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#00ffc8" stopOpacity="0.9" />
            <stop offset="30%"  stopColor="#00b8d4" stopOpacity="0.75" />
            <stop offset="70%"  stopColor="#00b8d4" stopOpacity="0.75" />
            <stop offset="100%" stopColor="#00ffc8" stopOpacity="0.9" />
          </linearGradient>
          {/* Shadow glow under cable */}
          <linearGradient id="cable-shadow-grad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#00ffc8" stopOpacity="0.12" />
            <stop offset="50%"  stopColor="#0084ff" stopOpacity="0.08" />
            <stop offset="100%" stopColor="#00ffc8" stopOpacity="0.12" />
          </linearGradient>
          <filter id="cable-glow">
            <feGaussianBlur stdDeviation="3.5" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
          <filter id="fault-glow">
            <feGaussianBlur stdDeviation="6" result="b"/>
            <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        {/* Deep ocean background wash */}
        <rect x="0" y="0" width={VB_W} height={VB_H}
          fill="rgba(2,9,18,0)" />

        {/* Seabed terrain (with caustics, ruler, sediment) */}
        <SeabedTerrain />

        {/* Cable shadow glow */}
        <line
          x1={MX} y1={CY + 5} x2={VB_W - MX} y2={CY + 5}
          stroke="url(#cable-shadow-grad)" strokeWidth="12" strokeLinecap="round"
        />

        {/* Main cable line */}
        <path
          d={`M${MX},${CY} L${VB_W - MX},${CY}`}
          fill="none" stroke="url(#cable-grad)"
          strokeWidth="4.5" strokeLinecap="round"
          filter="url(#cable-glow)"
        />

        {/* Animated data packets */}
        <DataPackets hasFault={hasFault} healthPct={healthPct} />

        {/* Terminal stations */}
        <Station x={MX}       label="Station A" distLabel="0 m" />
        <Station x={VB_W - MX} label="Station B" distLabel={`${CABLE_LENGTH} m`} />

        {/* Repeater nodes */}
        {[1, 2, 3, 4].map(k => (
          <Repeater key={k} x={MX + CW * k / 5} index={k} />
        ))}

        {/* Fault markers */}
        {(faults ?? []).slice(0, 5).map((f, i) => {
          const dist = parseFloat(f.est_distance ?? 0);
          const x    = MX + (dist / CABLE_LENGTH) * CW;
          const fc   = getFaultColor(f.ftype_raw);
          // Alternate labels above/below to prevent overlap
          const labelY = i % 2 === 0 ? CY - 24 : CY - 42;
          return (
            <g key={i} role="graphics-symbol" aria-label={`${f.fault_type} at ${dist}m`}>
              {/* Pressure-wave outer ring */}
              <circle cx={x} cy={CY} r="10" fill="none" stroke={fc} strokeWidth="1.5" opacity="0.25">
                <animate attributeName="r"       values="8;26;8"   dur="2s"   repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.5;0;0.5" dur="2s"  repeatCount="indefinite"/>
              </circle>
              {/* Mid ring */}
              <circle cx={x} cy={CY} r="6" fill="none" stroke={fc} strokeWidth="1" opacity="0.4">
                <animate attributeName="r"       values="6;16;6"    dur="2s" begin="0.5s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="0.4;0;0.4" dur="2s" begin="0.5s" repeatCount="indefinite"/>
              </circle>
              {/* Core fault dot */}
              <circle cx={x} cy={CY} r="6" fill={fc} opacity="0.95" filter="url(#fault-glow)">
                <animate attributeName="r"       values="5;8;5"    dur="1.5s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="1;0.6;1"  dur="1.5s" repeatCount="indefinite"/>
              </circle>
              {/* Stem line to label */}
              <line
                x1={x} y1={CY - 8} x2={x} y2={labelY + 6}
                stroke={fc} strokeWidth="1" opacity="0.45" strokeDasharray="2,2"
              />
              {/* Fault type label */}
              <text x={x} y={labelY} textAnchor="middle" fill={fc}
                fontSize="8" fontWeight="700" fontFamily="Oxanium, sans-serif">
                {(f.fault_type ?? 'Fault').replace(/_/g, ' ')}
              </text>
              {/* Distance badge */}
              <text x={x} y={CY + 22} textAnchor="middle" fill={fc}
                fontSize="7.5" fontFamily="Space Mono, monospace" opacity="0.7">
                {dist}m
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
