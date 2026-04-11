import React from 'react';

export default function CableGraphic({ faults }) {
  const W = 780, H = 130, MX = 70, CY = 62;
  const CW = W - 2 * MX;
  const CABLE_LENGTH = 500;

  const getFaultColor = (ftypeRaw) => {
    const FAULT_COLORS = {
      "cable_cut": "#E24B4A",
      "anchor_drag": "#EF9F27",
      "overheating": "#D85A30",
      "insulation_failure": "#7F77DD"
    };
    return FAULT_COLORS[ftypeRaw] || "#EF4444";
  };

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', minWidth: '700px' }} xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="cg" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00D4FF" stopOpacity="0.9"/>
            <stop offset="30%" stopColor="#3B82F6" stopOpacity="0.7"/>
            <stop offset="70%" stopColor="#3B82F6" stopOpacity="0.7"/>
            <stop offset="100%" stopColor="#00D4FF" stopOpacity="0.9"/>
          </linearGradient>
          <linearGradient id="cg-shadow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00D4FF" stopOpacity="0.2"/>
            <stop offset="50%" stopColor="#8B5CF6" stopOpacity="0.15"/>
            <stop offset="100%" stopColor="#00D4FF" stopOpacity="0.2"/>
          </linearGradient>
          <radialGradient id="depth" cx="50%" cy="100%" r="60%">
            <stop offset="0%" stopColor="#00D4FF" stopOpacity="0.04"/>
            <stop offset="100%" stopColor="transparent"/>
          </radialGradient>
          <filter id="gl"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <filter id="glow-strong"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>

        <rect x="0" y="0" width={W} height={H} fill="url(#depth)"/>
        <line x1={MX} y1={CY+20} x2={W-MX} y2={CY+20} stroke="rgba(0,212,255,0.07)" strokeWidth="1" strokeDasharray="6,6"/>
        <line x1={MX} y1={CY+3} x2={W-MX} y2={CY+3} stroke="url(#cg-shadow)" strokeWidth="10" strokeLinecap="round"/>
        
        <path id="cable-path" d={`M${MX},${CY} L${W-MX},${CY}`} fill="none" stroke="url(#cg)" strokeWidth="5" strokeLinecap="round" filter="url(#gl)"/>
        
        <circle r="4" fill="#00D4FF" opacity="0.85" filter="url(#gl)">
          <animateMotion dur="3s" repeatCount="indefinite" path={`M${MX},${CY} L${W-MX},${CY}`}/>
        </circle>

        {/* Stations */}
        <rect x={MX-18} y={CY-16} width="36" height="32" rx="6" fill="rgba(0,212,255,0.1)" stroke="#00D4FF" strokeWidth="1.5" filter="url(#gl)"/>
        <text x={MX} y={CY+4} textAnchor="middle" fill="#00D4FF" fontSize="10" fontWeight="bold">⬡</text>
        <text x={MX} y={CY+36} textAnchor="middle" fill="rgba(240,246,255,0.45)" fontSize="9" fontWeight="600">Station A</text>
        <text x={MX} y={CY-22} textAnchor="middle" fill="rgba(0,212,255,0.4)" fontSize="8" fontFamily="monospace">0m</text>

        <rect x={W-MX-18} y={CY-16} width="36" height="32" rx="6" fill="rgba(0,212,255,0.1)" stroke="#00D4FF" strokeWidth="1.5" filter="url(#gl)"/>
        <text x={W-MX} y={CY+4} textAnchor="middle" fill="#00D4FF" fontSize="10" fontWeight="bold">⬡</text>
        <text x={W-MX} y={CY+36} textAnchor="middle" fill="rgba(240,246,255,0.45)" fontSize="9" fontWeight="600">Station B</text>
        <text x={W-MX} y={CY-22} textAnchor="middle" fill="rgba(0,212,255,0.4)" fontSize="8" fontFamily="monospace">{CABLE_LENGTH}m</text>

        {/* Repeaters */}
        {[1,2,3,4].map(k => {
          const rx = MX + CW * k / 5;
          return (
            <g key={k}>
              <rect x={rx-10} y={CY-8} width="20" height="16" rx="4" fill="rgba(59,130,246,0.12)" stroke="rgba(59,130,246,0.5)" strokeWidth="1"/>
              <text x={rx} y={CY+3} textAnchor="middle" fill="rgba(59,130,246,0.7)" fontSize="7" fontWeight="bold">R{k}</text>
              <text x={rx} y={CY+20} textAnchor="middle" fill="rgba(240,246,255,0.22)" fontSize="7" fontFamily="monospace">{Math.round(CABLE_LENGTH*k/5)}m</text>
              <text x={rx} y={CY-14} textAnchor="middle" fill="rgba(240,246,255,0.15)" fontSize="7">Node {k}</text>
            </g>
          )
        })}

        {/* Faults */}
        {faults.slice(0, 3).map((f, i) => {
          const dist = parseFloat(f.est_distance || 0);
          const x = MX + (dist / CABLE_LENGTH) * CW;
          const fc = getFaultColor(f.ftype_raw);
          return (
            <g key={i}>
              <circle cx={x} cy={CY} r="7" fill={fc} opacity="0.95" filter="url(#glow-strong)">
                <animate attributeName="r" values="6;11;6" dur="1.6s" repeatCount="indefinite"/>
                <animate attributeName="opacity" values="1;0.55;1" dur="1.6s" repeatCount="indefinite"/>
              </circle>
              <text x={x} y={CY-18} textAnchor="middle" fill={fc} fontSize="8" fontWeight="bold">{f.fault_type}</text>
              <line x1={x} y1={CY-10} x2={x} y2={CY-3} stroke={fc} strokeWidth="1" opacity="0.5" strokeDasharray="2,2"/>
            </g>
          )
        })}

      </svg>
    </div>
  );
}
