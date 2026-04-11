import React from 'react';

const SEVERITY_CLS = {
  Critical: 'severity-critical',
  High:     'severity-critical',
  Medium:   'severity-medium',
  Low:      'severity-low',
};

function WarningIcon({ color }) {
  return (
    <svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none">
      <path
        d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
        stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      />
      <line x1="12" y1="9" x2="12" y2="13" stroke={color} strokeWidth="2" strokeLinecap="round"/>
      <line x1="12" y1="17" x2="12.01" y2="17" stroke={color} strokeWidth="2.5" strokeLinecap="round"/>
    </svg>
  );
}

function severityColor(sev) {
  if (sev === 'Critical' || sev === 'High') return '#ff4757';
  if (sev === 'Medium')                     return '#ffa502';
  return '#ffa502';
}

export default function FaultToast({ fault, onDismiss }) {
  const sev      = fault?.Severity || 'High';
  const sevCls   = SEVERITY_CLS[sev] || 'severity-critical';
  const iconColor = severityColor(sev);
  const faultType = fault?.fault_type || 'Unknown';
  const location  = fault?.est_distance != null ? `${fault.est_distance} m` : '—';
  const time      = fault?.Time ? fault.Time.split(' ')[1] || fault.Time : '—';

  return (
    <div className={`fault-toast ${sevCls}`} role="alert">
      <div className="toast-icon">
        <WarningIcon color={iconColor} />
      </div>
      <div className="toast-body">
        <span className="toast-title">Fault Detected — {faultType.replace(/_/g, ' ')}</span>
        <span className="toast-meta">Location: {location} · {time}</span>
      </div>
      <button className="toast-close" onClick={onDismiss} aria-label="Dismiss">×</button>
      <div className="toast-progress" />
    </div>
  );
}
