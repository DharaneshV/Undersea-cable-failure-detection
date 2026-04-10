"""
dashboard.py
Unified Streamlit live monitoring & analysis dashboard.
"""

import time
import logging
import os
import io

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CABLE_LENGTH, FAULT_COLORS, SENSOR_COLORS, FEATURES, SEQ_LEN
from simulator import generate_dataset
from model import CableFaultDetector
from utils import ema
from evaluate import run_evaluation

log = logging.getLogger(__name__)

st.set_page_config(page_title="Undersea Cable Monitor", page_icon="🌊", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Design Tokens ─────────────────────────────────────────────────── */
:root {
  --bg-deep:       #080C14;
  --bg-surface:    #0D1420;
  --bg-elevated:   #131D2E;
  --bg-card:       rgba(19,29,46,0.85);
  --border-subtle: rgba(255,255,255,0.07);
  --border-medium: rgba(255,255,255,0.12);
  --accent-cyan:   #00D4FF;
  --accent-blue:   #3B82F6;
  --accent-purple: #8B5CF6;
  --success:       #10B981;
  --warning:       #F59E0B;
  --danger:        #EF4444;
  --text-primary:  #F0F6FF;
  --text-secondary:rgba(240,246,255,0.55);
  --text-muted:    rgba(240,246,255,0.30);
  --mono:          'JetBrains Mono', monospace;
  --radius-lg:     18px;
  --radius-md:     12px;
  --radius-sm:     8px;
  --shadow-glow-cyan:  0 0 24px rgba(0,212,255,0.15);
  --shadow-glow-blue:  0 0 24px rgba(59,130,246,0.15);
  --shadow-card:       0 4px 24px rgba(0,0,0,0.4);
}

/* ── Base ──────────────────────────────────────────────────────────── */
.stApp { font-family: 'Inter', sans-serif !important; background: var(--bg-deep) !important; }
.stApp * { box-sizing: border-box; }

/* ── Hero Header ───────────────────────────────────────────────────── */
.hero-bar {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(135deg, rgba(0,212,255,0.06) 0%, rgba(139,92,246,0.06) 100%);
  border: 1px solid var(--border-subtle);
  border-radius: var(--radius-lg); padding: 18px 28px; margin-bottom: 20px;
  backdrop-filter: blur(16px); position: relative; overflow: hidden;
}
.hero-bar::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
  opacity: 0.6;
}
.hero-left { display: flex; align-items: center; gap: 16px; }
.hero-icon {
  width: 48px; height: 48px; border-radius: 14px;
  background: linear-gradient(135deg, rgba(0,212,255,0.2), rgba(59,130,246,0.2));
  border: 1px solid rgba(0,212,255,0.3); display: flex; align-items: center;
  justify-content: center; font-size: 24px; flex-shrink: 0;
}
.hero-title {
  font-size: 22px; font-weight: 800; letter-spacing: -0.3px;
  background: linear-gradient(135deg, #00D4FF 0%, #3B82F6 45%, #8B5CF6 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1.2;
}
.hero-sub {
  font-size: 11px; color: var(--text-muted); letter-spacing: 0.8px;
  font-weight: 500; margin-top: 3px; font-family: var(--mono);
}
.hero-right { display: flex; align-items: center; gap: 12px; }
.sys-status-pill {
  display: flex; align-items: center; gap: 7px; padding: 7px 14px;
  background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25);
  border-radius: 50px; font-size: 11px; font-weight: 700; color: #10B981;
  letter-spacing: 1px; text-transform: uppercase;
}
.status-dot {
  width: 7px; height: 7px; border-radius: 50%; background: #10B981;
  animation: pulse-dot 2s ease infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
  50% { opacity:0.7; box-shadow: 0 0 0 5px rgba(16,185,129,0); }
}
.hero-clock {
  font-family: var(--mono); font-size: 12px; font-weight: 500;
  color: var(--text-secondary); padding: 7px 14px;
  background: rgba(255,255,255,0.04); border: 1px solid var(--border-subtle);
  border-radius: 50px; letter-spacing: 0.5px;
}

/* ── Section Header ────────────────────────────────────────────────── */
.sec-hdr {
  font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
  color: var(--text-muted); margin: 20px 0 12px;
  padding-bottom: 8px; border-bottom: 1px solid var(--border-subtle);
  display: flex; align-items: center; gap: 8px;
}
.sec-hdr::before {
  content: ''; display: inline-block; width: 3px; height: 14px;
  background: linear-gradient(180deg, var(--accent-cyan), var(--accent-blue));
  border-radius: 2px; flex-shrink: 0;
}

/* ── Glass Card ────────────────────────────────────────────────────── */
.glass-card {
  background: var(--bg-card); backdrop-filter: blur(20px);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-lg);
  padding: 28px; margin: 8px 0; box-shadow: var(--shadow-card);
}

/* ── Metric Cards ──────────────────────────────────────────────────── */
.metric-card {
  background: linear-gradient(160deg, rgba(19,29,46,0.9) 0%, rgba(13,20,32,0.95) 100%);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-md);
  padding: 18px 16px 22px; text-align: center;
  position: relative; overflow: hidden;
  transition: transform 0.22s cubic-bezier(0.34,1.56,0.64,1), box-shadow 0.22s ease;
  box-shadow: var(--shadow-card);
}
.metric-card:hover {
  transform: translateY(-3px) scale(1.01);
  box-shadow: 0 12px 32px rgba(0,0,0,0.5);
}
.metric-card::after {
  content: ''; position: absolute; inset: 0; border-radius: var(--radius-md);
  opacity: 0; transition: opacity 0.22s; pointer-events: none;
  background: radial-gradient(ellipse at 50% 0%, rgba(255,255,255,0.05), transparent 70%);
}
.metric-card:hover::after { opacity: 1; }
.metric-icon {
  font-size: 18px; margin-bottom: 8px; opacity: 0.75;
  display: block; line-height: 1;
}
.metric-label {
  font-size: 9px; font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase;
  color: var(--text-muted); margin-bottom: 8px; display: block;
}
.metric-value {
  font-size: 28px; font-weight: 800; color: var(--text-primary);
  line-height: 1.05; font-family: var(--mono); letter-spacing: -1px;
}
.metric-unit {
  font-size: 11px; font-weight: 500; color: var(--text-muted);
  letter-spacing: 0.3px; display: block; margin-top: 3px;
}
.metric-bar {
  position: absolute; bottom: 0; left: 0; height: 3px;
  border-radius: 0 2px 0 var(--radius-md); transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
}
.metric-glow {
  position: absolute; bottom: 0; left: 0; right: 0; height: 60px;
  border-radius: 0 0 var(--radius-md) var(--radius-md);
  opacity: 0.06; pointer-events: none;
}

/* ── Health Gauge ──────────────────────────────────────────────────── */
.health-ring { text-align: center; padding: 6px 0; }
.health-value { font-size: 42px; font-weight: 800; line-height: 1; font-family: var(--mono); }
.health-label { font-size: 9px; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase; margin-top: 4px; }
.health-status { font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }

/* ── Status Pills ──────────────────────────────────────────────────── */
.status-pill {
  border-radius: 50px; padding: 10px 18px; text-align: center;
  font-weight: 700; font-size: 12px; letter-spacing: 0.8px;
  text-transform: uppercase; transition: box-shadow 0.2s;
}
.status-ok   { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); color: #10B981; }
.status-warn { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); color: #F59E0B; }
.status-bad  { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  color: #EF4444; animation: pulse-ring 2s infinite; }

/* ── Severity Badges ───────────────────────────────────────────────── */
.sev-badge {
  display: inline-block; padding: 3px 10px; border-radius: 6px;
  font-size: 10px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;
}
.sev-low      { background: rgba(16,185,129,0.12); color: #10B981; border: 1px solid rgba(16,185,129,0.2); }
.sev-degrading{ background: rgba(245,158,11,0.1);  color: #F59E0B; border: 1px dotted rgba(245,158,11,0.4); }
.sev-warning  { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.3); }
.sev-medium   { background: rgba(245,158,11,0.15); color: #F59E0B; border: 1px solid rgba(245,158,11,0.35); }
.sev-high     { background: rgba(239,89,65,0.12);  color: #EF5941; border: 1px solid rgba(239,89,65,0.3); }
.sev-critical { background: rgba(239,68,68,0.15);  color: #EF4444; border: 1px solid rgba(239,68,68,0.35); }
.sev-normal   { background: rgba(16,185,129,0.1);  color: #10B981; border: 1px solid rgba(16,185,129,0.2); }

/* ── Alert Banners ─────────────────────────────────────────────────── */
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0 rgba(239,68,68,0.35); }
  70%  { box-shadow: 0 0 0 12px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
@keyframes pulse-warn {
  0%   { box-shadow: 0 0 0 0 rgba(245,158,11,0.35); }
  70%  { box-shadow: 0 0 0 10px rgba(245,158,11,0); }
  100% { box-shadow: 0 0 0 0 rgba(245,158,11,0); }
}
.alert-banner {
  display: flex; align-items: flex-start; gap: 16px;
  border-radius: var(--radius-md); padding: 16px 20px; margin: 10px 0;
  backdrop-filter: blur(12px);
}
.fault-alert {
  background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.03) 100%);
  border: 1px solid rgba(239,68,68,0.3); border-left: 4px solid #EF4444;
  animation: pulse-ring 2s infinite;
}
.warning-alert {
  background: linear-gradient(135deg, rgba(245,158,11,0.1) 0%, rgba(245,158,11,0.03) 100%);
  border: 1px solid rgba(245,158,11,0.25); border-left: 4px solid #F59E0B;
  animation: pulse-warn 2.5s infinite;
}
.alert-icon-wrap {
  width: 38px; height: 38px; border-radius: 50%; display: flex;
  align-items: center; justify-content: center; flex-shrink: 0; font-size: 18px;
}
.alert-icon-fault   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.3); }
.alert-icon-warning { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); }
.alert-body { flex: 1; min-width: 0; }
.alert-headline { font-size: 13px; font-weight: 700; color: var(--text-primary); letter-spacing: 0.3px; margin-bottom: 4px; }
.alert-detail   { font-size: 11px; color: var(--text-secondary); font-family: var(--mono); }
.alert-meta     { font-size: 11px; color: var(--text-muted); margin-top: 6px; }

/* ── Cable Box ─────────────────────────────────────────────────────── */
.cable-box {
  background: linear-gradient(180deg, rgba(13,20,32,0.9) 0%, rgba(8,12,20,0.9) 100%);
  border: 1px solid var(--border-subtle); border-radius: var(--radius-lg);
  padding: 20px 20px 14px; margin: 10px 0; overflow-x: auto;
  box-shadow: inset 0 1px 0 rgba(0,212,255,0.05), var(--shadow-card);
}

/* ── Fault Log ─────────────────────────────────────────────────────── */
.fault-log-header {
  display: grid; grid-template-columns: 2fr 1.8fr 1fr 1.3fr 1.3fr;
  padding: 8px 16px; font-size: 9px; font-weight: 700; letter-spacing: 1.5px;
  text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;
}
.fault-log-row {
  display: grid; grid-template-columns: 2fr 1.8fr 1fr 1.3fr 1.3fr;
  padding: 10px 16px; border-radius: var(--radius-sm); margin-bottom: 3px;
  background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
  transition: background 0.15s, border-color 0.15s; align-items: center;
}
.fault-log-row:hover { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.09); }
.fault-log-time  { font-family: var(--mono); font-size: 11px; color: var(--text-secondary); }
.fault-log-type  { font-size: 12px; font-weight: 600; color: var(--text-primary); }
.fault-log-score { font-family: var(--mono); font-size: 12px; color: var(--text-secondary); }
.fault-log-dist  { font-family: var(--mono); font-size: 12px; color: var(--accent-cyan); }

/* ── Empty State ───────────────────────────────────────────────────── */
.empty-state {
  text-align: center; padding: 64px 28px;
  background: linear-gradient(160deg, rgba(13,20,32,0.8), rgba(8,12,20,0.9));
  border: 1px solid var(--border-subtle); border-radius: var(--radius-lg);
  position: relative; overflow: hidden;
}
.empty-state::before {
  content: ''; position: absolute; bottom: -40px; left: 50%; transform: translateX(-50%);
  width: 300px; height: 80px;
  background: radial-gradient(ellipse, rgba(0,212,255,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.empty-icon { font-size: 56px; display: block; margin-bottom: 18px; animation: float 3s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
.empty-title { font-size: 18px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
.empty-sub   { font-size: 13px; color: var(--text-muted); max-width: 340px; margin: 0 auto; line-height: 1.6; }

/* ── Architecture Flow diagram ─────────────────────────────────────── */
.arch-flow { display: flex; flex-direction: column; align-items: center; gap: 0; padding: 20px 0; }
.arch-block {
  background: linear-gradient(135deg, rgba(19,29,46,0.95), rgba(13,20,32,0.98));
  border: 1px solid var(--border-subtle); border-radius: var(--radius-md);
  padding: 14px 28px; text-align: center; min-width: 280px;
  position: relative; box-shadow: var(--shadow-card);
}
.arch-block.accent-cyan  { border-color: rgba(0,212,255,0.35); box-shadow: 0 0 20px rgba(0,212,255,0.08); }
.arch-block.accent-blue  { border-color: rgba(59,130,246,0.35); box-shadow: 0 0 20px rgba(59,130,246,0.08); }
.arch-block.accent-purple{ border-color: rgba(139,92,246,0.35); box-shadow: 0 0 20px rgba(139,92,246,0.08); }
.arch-block.accent-green { border-color: rgba(16,185,129,0.35); box-shadow: 0 0 20px rgba(16,185,129,0.08); }
.arch-title { font-size: 13px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px; }
.arch-sub   { font-size: 11px; color: var(--text-muted); font-family: var(--mono); }
.arch-arrow { font-size: 20px; color: var(--text-muted); line-height: 1.4; }

/* ── Tab styling ───────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 4px; background: rgba(13,20,32,0.6);
  border-radius: var(--radius-md); padding: 4px; border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 8px; padding: 8px 18px; font-size: 13px; font-weight: 600;
  color: var(--text-muted); transition: all 0.2s; background: transparent; border: none;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,212,255,0.12), rgba(59,130,246,0.12)) !important;
  color: var(--accent-cyan) !important;
  border: 1px solid rgba(0,212,255,0.2) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── Sidebar ───────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0D1420 0%, #0A1018 100%) !important;
  border-right: 1px solid var(--border-subtle) !important;
}
.sidebar-brand {
  text-align: center; padding: 20px 0 16px;
  border-bottom: 1px solid var(--border-subtle); margin-bottom: 16px;
}
.sidebar-brand-icon {
  font-size: 36px; display: block; margin-bottom: 8px;
}
.sidebar-brand-name {
  font-size: 15px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.2px;
}
.sidebar-brand-tagline {
  font-size: 10px; color: var(--text-muted); letter-spacing: 0.8px; margin-top: 2px;
  text-transform: uppercase; font-family: var(--mono);
}
.sidebar-section {
  font-size: 9px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
  color: var(--accent-cyan); margin: 18px 0 10px; opacity: 0.8;
  display: flex; align-items: center; gap: 6px;
}

/* ── Progress bar ──────────────────────────────────────────────────── */
.stProgress > div > div > div { border-radius: 4px; }

/* ── Scrollbar ─────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Info / warning / error boxes ─────────────────────────────────── */
.stAlert { border-radius: var(--radius-md) !important; }

/* legacy compat */
.dash-title { font-size: 26px; font-weight: 800;
  background: linear-gradient(135deg,#00D4FF 0%,#3B82F6 50%,#8B5CF6 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.dash-sub { font-size: 12px; color: var(--text-muted); letter-spacing: .5px; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────────────────────────
if "active_dataset" not in st.session_state:
    st.session_state.active_dataset = {
        "data": None,
        "predictions": None,
        "labels": None,
        "source": None,
        "fault_log": None
    }

if "manual_rows" not in st.session_state:
    st.session_state.manual_rows = []


# ── check pretrained model ────────────────────────────────────────────────────
if not os.path.exists("saved_model/conv_transformer_ae.keras"):
    st.error("⚠️ No saved model found.")
    st.markdown("""
    Run the following command to train the model before launching the dashboard:
    
    ```bash
    python model.py
    ```
    Then restart the app.
    """)
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model():
    detector = CableFaultDetector()
    detector.load()
    return detector

with st.spinner("Loading pretrained Transformer model…"):
    detector = load_model()


# ── helper functions ──────────────────────────────────────────────────────────
def severity_of(score: float, threshold: float) -> tuple[str, str]:
    r = score / threshold if threshold > 0 else 0
    if r > 5:     return "Critical", "sev-critical"
    if r > 3:     return "High",     "sev-high"
    if r > 1.2:   return "Medium",   "sev-medium"
    if r > 1.0:   return "Low",      "sev-low"
    if r > 0.75:  return "Degrading","sev-warning"
    return               "Normal",   "sev-low"

METRIC_ICONS = {"Voltage": "⚡", "Current": "↺", "Temperature": "🌡", "Vibration": "〰", "Anomaly Score": "📉"}

def metric_card(label, value, unit, bar_color, delta=None, bar_pct=50):
    icon = METRIC_ICONS.get(label, "·")
    glow_color = bar_color.replace("#", "") if bar_color.startswith("#") else "ffffff"
    return (
        f'<div class="metric-card" style="box-shadow:0 4px 24px rgba(0,0,0,0.4),0 0 12px rgba(0,0,0,0.2)">'
        f'<span class="metric-icon">{icon}</span>'
        f'<span class="metric-label">{label}</span>'
        f'<div class="metric-value">{value}</div>'
        f'<span class="metric-unit">{unit}</span>'
        f'<div class="metric-glow" style="background:linear-gradient(to top,{bar_color}20,transparent)"></div>'
        f'<div class="metric-bar" style="width:{bar_pct:.0f}%;background:linear-gradient(90deg,{bar_color},{bar_color}bb)"></div>'
        f'</div>'
    )

def health_gauge_html(hp: float) -> str:
    if hp >= 80:   c1, c2, tag, ring = "#10B981", "#00D4FF", "HEALTHY",  "rgba(16,185,129,0.15)"
    elif hp >= 50: c1, c2, tag, ring = "#F59E0B", "#EF5941", "WARNING",  "rgba(245,158,11,0.15)"
    else:          c1, c2, tag, ring = "#EF4444", "#EF5941", "CRITICAL", "rgba(239,68,68,0.15)"
    return (
        f'<div class="health-ring" style="background:{ring};border:1px solid {c1}30;border-radius:14px;padding:12px 8px">'
        f'<div style="font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{c1}88;margin-bottom:6px">HEALTH</div>'
        f'<div class="health-value" style="background:linear-gradient(135deg,{c1},{c2});-webkit-background-clip:text;-webkit-text-fill-color:transparent">{hp:.0f}%</div>'
        f'<div class="health-status" style="color:{c1};margin-top:6px">{tag}</div>'
        f'</div>'
    )

def health_pct_ema(scores: np.ndarray, threshold: float) -> float:
    v = scores[~np.isnan(scores)]
    if len(v) == 0: return 100.0
    smoothed = ema(v / threshold, alpha=0.1)
    return float(np.clip(100 * (1 - smoothed[-1]), 0, 100))

def match_fault_distance(sample_idx: int, fault_log: list[dict]) -> float:
    for fl in fault_log:
        if fl["start_sample"] <= sample_idx < fl["start_sample"] + fl["duration_samples"]:
            return fl["fault_distance_m"]
    return 0.0

def build_status_cards(row, sc, thr, i, is_fault, is_warning):
    v = row.get("voltage", 0)
    c = row.get("current", 0)
    tmp = row.get("temperature", 0)
    vib = row.get("vibration", 0)
    
    v_bar = max(0, min(100, v / 250 * 100))
    c_bar = max(0, min(100, c / 10  * 100))
    t_bar = max(0, min(100, tmp / 60 * 100))
    vib_bar = max(0, min(100, min(abs(vib), 2) / 2 * 100))
    sc_bar = max(0, min(100, sc / (thr * 2) * 100))
    
    if is_fault: status_html = '<div class="status-pill status-bad">🚨 FAULT</div>'
    elif is_warning: status_html = '<div class="status-pill status-warn">⚠ WARNING</div>'
    else: status_html = '<div class="status-pill status-ok">✓ Normal</div>'

    score_color = "#EF4444" if is_fault else ("#F59E0B" if is_warning else "#10B981")
    return (
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:4px 0">'
        + metric_card("Voltage", f"{v:.1f}", "V", SENSOR_COLORS.get("voltage", "#3B82F6"), None, v_bar)
        + metric_card("Current", f"{c:.2f}", "A", SENSOR_COLORS.get("current", "#10B981"), None, c_bar)
        + metric_card("Temperature", f"{tmp:.1f}", "°C", SENSOR_COLORS.get("temperature", "#F59E0B"), None, t_bar)
        + metric_card("Vibration", f"{vib:.3f}", "g", SENSOR_COLORS.get("vibration", "#EF5941"), None, vib_bar)
        + metric_card("Anomaly Score", f"{sc:.4f}", f"thr {thr:.4f}", score_color, None, sc_bar)
        + f'<div style="display:flex;align-items:center;justify-content:center;padding:8px">{status_html}</div>'
        + '</div>'
    )


def cable_svg(cable_len: int, faults: list[dict]) -> str:
    W, H, MX, CY = 780, 130, 70, 62
    CW = W - 2 * MX
    # depth glow radius
    parts = [
        f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
        '<defs>',
        '  <linearGradient id="cg" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%"   stop-color="#00D4FF" stop-opacity="0.9"/>',
        '    <stop offset="30%"  stop-color="#3B82F6" stop-opacity="0.7"/>',
        '    <stop offset="70%"  stop-color="#3B82F6" stop-opacity="0.7"/>',
        '    <stop offset="100%" stop-color="#00D4FF" stop-opacity="0.9"/>',
        '  </linearGradient>',
        '  <linearGradient id="cg-shadow" x1="0%" y1="0%" x2="100%" y2="0%">',
        '    <stop offset="0%"   stop-color="#00D4FF" stop-opacity="0.2"/>',
        '    <stop offset="50%"  stop-color="#8B5CF6" stop-opacity="0.15"/>',
        '    <stop offset="100%" stop-color="#00D4FF" stop-opacity="0.2"/>',
        '  </linearGradient>',
        '  <radialGradient id="depth" cx="50%" cy="100%" r="60%">',
        '    <stop offset="0%" stop-color="#00D4FF" stop-opacity="0.04"/>',
        '    <stop offset="100%" stop-color="transparent"/>',
        '  </radialGradient>',
        '  <filter id="gl"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '  <filter id="glow-strong"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '</defs>',
        # depth bg
        f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#depth)"/>',
        # seabed dashed guide
        f'<line x1="{MX}" y1="{CY+20}" x2="{W-MX}" y2="{CY+20}" stroke="rgba(0,212,255,0.07)" stroke-width="1" stroke-dasharray="6,6"/>',
        # cable shadow
        f'<line x1="{MX}" y1="{CY+3}" x2="{W-MX}" y2="{CY+3}" stroke="url(#cg-shadow)" stroke-width="10" stroke-linecap="round"/>',
        # main cable
        f'<line x1="{MX}" y1="{CY}" x2="{W-MX}" y2="{CY}" stroke="url(#cg)" stroke-width="5" stroke-linecap="round" filter="url(#gl)"/>',
        # animated signal pulse
        f'<circle r="4" fill="#00D4FF" opacity="0.85" filter="url(#gl)">'
        f'  <animateMotion dur="3s" repeatCount="indefinite">'
        f'    <mpath href="#cable-path"/>'
        f'  </animateMotion>'
        f'</circle>',
        f'<path id="cable-path" d="M{MX},{CY} L{W-MX},{CY}" fill="none"/>',
    ]
    # Station boxes
    for sx, label, dist_label in [(MX, "Substation A", "0 m"), (W-MX, "Substation B", f"{cable_len} m")]:
        anchor = "middle"
        parts += [
            f'<rect x="{sx-18}" y="{CY-16}" width="36" height="32" rx="6" fill="rgba(0,212,255,0.1)" stroke="#00D4FF" stroke-width="1.5" filter="url(#gl)"/>',
            f'<text x="{sx}" y="{CY+4}" text-anchor="{anchor}" fill="#00D4FF" font-size="9" font-family="Inter" font-weight="700">⬡</text>',
            f'<text x="{sx}" y="{CY+36}" text-anchor="{anchor}" fill="rgba(240,246,255,0.45)" font-size="9" font-family="Inter" font-weight="600">{label}</text>',
            f'<text x="{sx}" y="{CY-22}" text-anchor="{anchor}" fill="rgba(0,212,255,0.4)" font-size="8" font-family="JetBrains Mono,monospace">{dist_label}</text>',
        ]
    # Repeater nodes
    for k in range(1, 5):
        x = MX + CW * k / 5
        parts += [
            f'<rect x="{x-10}" y="{CY-8}" width="20" height="16" rx="4" fill="rgba(59,130,246,0.12)" stroke="rgba(59,130,246,0.5)" stroke-width="1"/>',
            f'<text x="{x}" y="{CY+3}" text-anchor="middle" fill="rgba(59,130,246,0.7)" font-size="7" font-family="Inter" font-weight="700">R{k}</text>',
            f'<text x="{x}" y="{CY+20}" text-anchor="middle" fill="rgba(240,246,255,0.22)" font-size="7" font-family="JetBrains Mono,monospace">{int(cable_len*k/5)}m</text>',
            f'<text x="{x}" y="{CY-14}" text-anchor="middle" fill="rgba(240,246,255,0.15)" font-size="7" font-family="Inter">Node {k}</text>',
        ]
    # Fault markers
    for f in faults:
        dist = float(str(f.get("Est. distance", "0")).replace(" m", ""))
        x = MX + (dist / cable_len) * CW
        fc = FAULT_COLORS.get(f.get("_ftype", "cable_cut"), "#EF4444")
        parts.append(
            f'<circle cx="{x}" cy="{CY}" r="7" fill="{fc}" opacity="0.95" filter="url(#glow-strong)">'
            f'<animate attributeName="r" values="6;11;6" dur="1.6s" repeatCount="indefinite"/>'
            f'<animate attributeName="opacity" values="1;0.55;1" dur="1.6s" repeatCount="indefinite"/></circle>'
        )
        parts.append(f'<text x="{x}" y="{CY-18}" text-anchor="middle" fill="{fc}" font-size="8" font-family="Inter" font-weight="700">{f["Fault type"]}</text>')
        parts.append(f'<line x1="{x}" y1="{CY-10}" x2="{x}" y2="{CY-3}" stroke="{fc}" stroke-width="1" opacity="0.5" stroke-dasharray="2,2"/>')
    parts.append('</svg>')
    return '\n'.join(parts)


# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-bar">
  <div class="hero-left">
    <div class="hero-icon">⚡</div>
    <div>
      <div class="hero-title">Smart Grid Fault Detection</div>
      <div class="hero-sub">LSTM-AE &nbsp;·&nbsp; Power Network Localisation &nbsp;·&nbsp; Real-time XAI Scoring</div>
    </div>
  </div>
  <div class="hero-right">
    <div class="hero-clock" id="hero-clock">-- : -- : -- UTC</div>
    <div class="sys-status-pill"><div class="status-dot"></div>System Online</div>
  </div>
</div>
<script>
(function() {
  function tick() {
    var d = new Date();
    var h = String(d.getUTCHours()).padStart(2,'0');
    var m = String(d.getUTCMinutes()).padStart(2,'0');
    var s = String(d.getUTCSeconds()).padStart(2,'0');
    var el = document.getElementById('hero-clock');
    if (el) el.textContent = h + ':' + m + ':' + s + ' UTC';
  }
  tick(); setInterval(tick, 1000);
})();
</script>
""", unsafe_allow_html=True)

tab_live, tab_upload, tab_analytics, tab_model = st.tabs([
    "📡  Live Monitor", "📤  Data Upload & Analyze", "📊  Evaluation & Analytics", "🧠  Model Info"
])

# ── tab 1: Live Monitor ───────────────────────────────────────────────────────
with tab_live:
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
          <span class="sidebar-brand-icon">⚡</span>
          <div class="sidebar-brand-name">Smart Grid Monitor</div>
          <div class="sidebar-brand-tagline">LSTM · XAI Tracker</div>
        </div>
        <div class="sidebar-section">⚙ Data Source</div>
        """, unsafe_allow_html=True)
        
        # Look for CSVs in datasets directory
        dataset_dir = "datasets"
        if os.path.exists(dataset_dir):
            csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv") and not f.endswith("_fault_log.csv")]
        else:
            csv_files = []
            
        selected_csv = st.selectbox("Select dataset to stream:", csv_files)
        
        st.markdown('<div class="sidebar-section">▶ Playback</div>', unsafe_allow_html=True)
        playback_spd = st.select_slider("Speed", options=["0.25×", "0.5×", "1×", "2×", "5×", "Max"], value="2×")
        speed_map = {"0.25×": 0.10, "0.5×": 0.05, "1×": 0.02, "2×": 0.01, "5×": 0.004, "Max": 0.0}
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        start_btn = st.button("▶  Start Live Stream", type="primary", use_container_width=True)

    if not start_btn and st.session_state.active_dataset["source"] != "simulation":
        st.markdown("""
        <div class="empty-state">
          <span class="empty-icon">⚡</span>
          <div class="empty-title">Ready to Monitor</div>
          <div class="empty-sub">Select your real-world dataset in the sidebar, then press <strong style="color:#00D4FF">▶ Start Live Stream</strong> to begin point-based fault detection.</div>
        </div>
        """, unsafe_allow_html=True)
    elif start_btn:
        with st.spinner("Loading smart grid data & running inference…"):
            if not selected_csv:
                st.error("No dataset selected. Generate or fetch one first.")
                st.stop()
            
            df_full = pd.read_csv(os.path.join(dataset_dir, selected_csv))
            
            log_name = selected_csv.replace(".csv", "_fault_log.csv")
            log_path = os.path.join(dataset_dir, log_name)
            fault_log = []
            if os.path.exists(log_path):
                fault_log = pd.read_csv(log_path).to_dict('records')
                
            result_full = detector.predict(df_full)

        total_frames = max(len(result_full) - SEQ_LEN, 1)
        thr = detector.threshold
        detected_faults = []
        progress_bar = st.progress(0, text="Simulation progress")

        row_top = st.columns([1, 5])
        health_ph = row_top[0].empty()
        metrics_ph = row_top[1].empty()
        alert_ph = st.empty()
        cable_ph = st.empty()
        chart_ph = st.empty()
        ftbl_ph = st.empty()
        
        delay = speed_map[playback_spd]
        window = SEQ_LEN
        
        skip_map = {"0.25×": 1, "0.5×": 2, "1×": 4, "2×": 10, "5×": 25, "Max": 100}
        frame_skip = skip_map.get(playback_spd, 1)

        for i in range(window, len(result_full)):
            row = result_full.iloc[i]
            
            sc = float(row["anomaly_score"]) if not pd.isna(row["anomaly_score"]) else 0.0
            is_fault = bool(sc > thr)
            is_warning = bool(sc > 0.75 * thr and not is_fault and not pd.isna(row["anomaly_score"]))
            ftype = row.get("fault_type", "none")
            
            # process faults under the hood unconditionally so we don't miss any during skips
            if is_fault and (not detected_faults or detected_faults[-1]["_idx"] < i - 50):
                dist = match_fault_distance(i, fault_log)
                sev_label, _ = severity_of(sc, thr)
                detected_faults.append({
                    "_idx": i, "_ftype": ftype, "Time": str(row["timestamp"])[:19],
                    "Fault type": ftype.replace("_", " ").title(), "Severity": sev_label,
                    "Anomaly score": f"{sc:.4f}", "Est. distance": f"{dist} m"
                })

            # skip rendering for UI lag prevention
            if i % frame_skip == 0 or i == len(result_full) - 1:
                chunk = result_full.iloc[max(0, i - 150): i]
                progress_bar.progress(min((i - window) / total_frames, 1.0), text=f"Sample {i:,} / {len(result_full):,}")
                
                hp = health_pct_ema(chunk["anomaly_score"].values, thr)
                health_ph.markdown(health_gauge_html(hp), unsafe_allow_html=True)
                metrics_ph.markdown(build_status_cards(row, sc, thr, i, is_fault, is_warning), unsafe_allow_html=True)
                
                if is_fault or is_warning:
                    sev_label, sev_cls = severity_of(sc, thr)
                    feat_errs = {feat: float(row.get(f"err_{feat}", 0)) for feat in FEATURES}
                    tot_err = sum(feat_errs.values())
                    xai_text = " | ".join([f"{f.title()}: {v/tot_err*100:.0f}%" for f, v in sorted(feat_errs.items(), key=lambda x: x[1], reverse=True)[:2]]) if tot_err > 0 else "Unknown"
                    
                    if is_fault:
                        alert_cls, icon_cls, icon, headline = "fault-alert", "alert-icon-fault", "🚨", "FAULT DETECTED"
                    else:
                        alert_cls, icon_cls, icon, headline = "warning-alert", "alert-icon-warning", "⚠", "DEGRADING — EARLY WARNING"
                    alert_ph.markdown(
                        f'<div class="alert-banner {alert_cls}">'
                        f'<div class="alert-icon-wrap {icon_cls}">{icon}</div>'
                        f'<div class="alert-body">'
                        f'<div class="alert-headline">{headline} &mdash; <span style="font-weight:500">{ftype.replace("_"," ").upper()}</span> &nbsp;<span class="sev-badge {sev_cls}">{sev_label}</span></div>'
                        f'<div class="alert-detail">Score: {sc:.5f} &nbsp;/&nbsp; Threshold: {thr:.5f}&nbsp;&nbsp;|&nbsp;&nbsp;Ratio: {sc/thr:.2f}×</div>'
                        f'<div class="alert-meta">📊 Anomaly driven by: {xai_text}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
                else:
                    alert_ph.empty()
                
                cable_ph.markdown(f'<div class="cable-box"><div class="sec-hdr">🔌 Power Grid Link — Fault Localisation</div>{cable_svg(CABLE_LENGTH, detected_faults)}</div>', unsafe_allow_html=True)

                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Voltage & Current", "Temperature & Vibration", "Anomaly Score"), vertical_spacing=0.09, row_heights=[0.35, 0.35, 0.30])
                ts = chunk["timestamp"]
                fig.add_trace(go.Scatter(x=ts, y=chunk["voltage"], name="Voltage", line=dict(color="#378ADD", width=1.4)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ts, y=chunk["current"], name="Current", line=dict(color="#1D9E75", width=1.4)), row=1, col=1)
                fig.add_trace(go.Scatter(x=ts, y=chunk["temperature"], name="Temp", line=dict(color="#EF9F27", width=1.4)), row=2, col=1)
                fig.add_trace(go.Scatter(x=ts, y=chunk["vibration"], name="Vibration", line=dict(color="#D85A30", width=1.1)), row=2, col=1)
                fig.add_trace(go.Scatter(x=ts, y=chunk["anomaly_score"], name="Score", line=dict(color="#7F77DD", width=1.5), fill="tozeroy", fillcolor="rgba(127,119,221,0.08)"), row=3, col=1)
                fig.add_hline(y=thr, line_dash="dash", line_color="#E24B4A", annotation_text="threshold", annotation_font_color="#E24B4A", row=3, col=1)
                for fl in fault_log:
                    fs = df_full.iloc[fl["start_sample"]]["timestamp"]
                    fe = df_full.iloc[min(fl["start_sample"] + fl["duration_samples"], len(df_full) - 1)]["timestamp"]
                    fc = FAULT_COLORS.get(fl["fault_type"], "#888")
                    fig.add_vrect(x0=fs, x1=fe, fillcolor=fc, opacity=0.07, line_width=0, row="all", col=1)
                fig.update_layout(
                    height=460, margin=dict(l=0, r=0, t=32, b=0), showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                                bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="rgba(240,246,255,0.5)")),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(8,12,20,0.6)",
                    font=dict(family="Inter", color="rgba(240,246,255,0.5)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    xaxis2=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    xaxis3=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    yaxis2=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                    yaxis3=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
                )
                chart_ph.plotly_chart(fig, use_container_width=True, key=f"live_monitor_chart_{i}")
                
                if detected_faults:
                    rows_html = "".join([
                        f'<div class="fault-log-row">'
                        f'<span class="fault-log-time">{f["Time"]}</span>'
                        f'<span class="fault-log-type">{f["Fault type"]}</span>'
                        f'<span><span class="sev-badge sev-{f["Severity"].lower()}">{f["Severity"]}</span></span>'
                        f'<span class="fault-log-score">{f["Anomaly score"]}</span>'
                        f'<span class="fault-log-dist">{f["Est. distance"]}</span>'
                        f'</div>'
                        for f in detected_faults
                    ])
                    ftbl_ph.markdown(
                        f'<div class="sec-hdr">🗂 Detected Fault Log ({len(detected_faults)} events)</div>'
                        f'<div class="fault-log-header"><span>Time</span><span>Type</span><span>Severity</span><span>Score</span><span>Distance</span></div>'
                        f'{rows_html}',
                        unsafe_allow_html=True
                    )
                
                time.sleep(delay * frame_skip)
        
        progress_bar.progress(1.0, text="✅ Complete")
        st.session_state.active_dataset = {
            "data": df_full,
            "predictions": result_full,
            "labels": df_full["label"].values,
            "source": "simulation",
            "fault_log": fault_log
        }
        
    csv_data = None
    if st.session_state.active_dataset["source"] == "simulation":
        res_full = st.session_state.active_dataset["predictions"]
        if res_full is not None:
            csv_data = res_full.to_csv(index=False)
            st.download_button("📥 Export Simulation as CSV", data=csv_data, file_name="cable_simulation.csv", mime="text/csv")
            st.info("Simulation data also available in Evaluation & Analytics tab ▶")


# ── tab 2: Data Upload & Analyze ──────────────────────────────────────────────
with tab_upload:
    mod = st.radio("Input Mode", ["Upload CSV", "Manual Entry"], horizontal=True)
    
    if mod == "Upload CSV":
        up_file = st.file_uploader("Upload sensor data CSV", type=["csv"])
        if up_file:
            df_up = pd.read_csv(up_file)
            st.write(f"Detected {len(df_up)} rows and {len(df_up.columns)} columns.")
            
            cols = list(df_up.columns)
            st.markdown("### Match Columns")
            c1, c2, c3, c4 = st.columns(4)
            m_v   = c1.selectbox("Voltage", ["<skip>"] + cols, index=cols.index("voltage")+1 if "voltage" in cols else 0)
            m_c   = c2.selectbox("Current", ["<skip>"] + cols, index=cols.index("current")+1 if "current" in cols else 0)
            m_tmp = c3.selectbox("Temperature", ["<skip>"] + cols, index=cols.index("temperature")+1 if "temperature" in cols else (cols.index("temp")+1 if "temp" in cols else 0))
            m_vib = c4.selectbox("Vibration", ["<skip>"] + cols, index=cols.index("vibration")+1 if "vibration" in cols else 0)
            
            if st.button("Confirm Mapping & Predict", type="primary"):
                missing = [n for n, m in zip(FEATURES, [m_v, m_c, m_tmp, m_vib]) if m == "<skip>"]
                if missing:
                    st.error(f"Missing mapping for: {', '.join(missing)}")
                else:
                    df_run = pd.DataFrame({
                        "voltage": df_up[m_v], "current": df_up[m_c],
                        "temperature": df_up[m_tmp], "vibration": df_up[m_vib]
                    })
                    if "timestamp" in cols: df_run["timestamp"] = pd.to_datetime(df_up["timestamp"])
                    else: df_run["timestamp"] = pd.to_datetime(np.arange(len(df_run))/10.0, unit="s", origin="2026-01-01")
                    
                    if "label" in cols: df_run["label"] = df_up["label"].values.astype(int)
                    if "fault_type" in cols: df_run["fault_type"] = df_up["fault_type"].values
                    else: df_run["fault_type"] = "none"
                    
                    with st.spinner("Running model..."):
                        preds = detector.predict(df_run)
                    
                    st.session_state.active_dataset = {
                        "data": df_run, "predictions": preds,
                        "labels": df_run["label"].values if "label" in cols else None,
                        "source": "upload", "fault_log": None
                    }
                    
                    st.success("✅ Prediction complete!")
                    if "label" in cols:
                        st.info("Labels detected — full evaluation metrics available in Evaluation & Analytics tab ▶")
                    else:
                        st.info("ℹ️ No label column found. Evaluation metrics skipped. Only anomaly scores are shown.")
                    
                    st.plotly_chart(run_evaluation(preds, detector.threshold)["score_timeline_fig"], use_container_width=True, key="upload_score_chart")
                    
                    csv_export = preds.to_csv(index=False)
                    st.download_button("📥 Download Results CSV", csv_export, "predictions.csv", "text/csv")

    else:
        st.markdown('<div class="sec-hdr">✏ Add Sensor Reading</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        v_in = c1.number_input("Voltage", value=220.0)
        c_in = c2.number_input("Current", value=5.0)
        t_in = c3.number_input("Temp", value=18.0)
        vi_in = c4.number_input("Vib", value=0.0)
        
        col_btn1, col_btn2 = st.columns([1, 4])
        if col_btn1.button("Add Row"):
            st.session_state.manual_rows.append({"voltage": v_in, "current": c_in, "temperature": t_in, "vibration": vi_in})
        if col_btn2.button("Clear Data"):
            st.session_state.manual_rows = []
        
        nr = len(st.session_state.manual_rows)
        st.write(f"Rows collected: {nr} / {SEQ_LEN}")
        st.progress(min(nr/SEQ_LEN, 1.0))
        
        if nr > 0:
            df_man = pd.DataFrame(st.session_state.manual_rows)
            df_man["timestamp"] = pd.to_datetime(np.arange(nr)/10.0, unit="s", origin="2026-01-01")
            df_man["label"] = 0
            df_man["fault_type"] = "none"
            st.dataframe(df_man)
            
            with st.spinner("Running prediction..."):
                if nr < SEQ_LEN:
                    # Pad to run a partial check
                    pad_df = pd.concat([pd.DataFrame(np.zeros((SEQ_LEN-nr, len(df_man.columns))), columns=df_man.columns), df_man])
                    for f in FEATURES: pad_df[f] = pad_df[f].fillna(0)
                    preds = detector.predict(pad_df).tail(nr).reset_index(drop=True)
                    st.warning(f"⚠ Partial window ({nr}/{SEQ_LEN}) — score is indicative only")
                else:
                    preds = detector.predict(df_man)
                    st.session_state.active_dataset = {
                        "data": df_man, "predictions": preds, "labels": None, "source": "manual", "fault_log": None
                    }
                    st.success("✅ Full window processed.")
                
                sc = preds["anomaly_score"].iloc[-1]
                thr = detector.threshold
                st.metric("Latest Anomaly Score", f"{sc:.4f}", delta=f"{sc - thr:.4f}", delta_color="inverse")


# ── tab 3: Evaluation & Analytics ─────────────────────────────────────────────
with tab_analytics:
    act = st.session_state.active_dataset
    if act["data"] is None:
        st.info("📊 No data to evaluate yet.\nRun a simulation in Live Monitor, or upload a CSV in Data Upload & Analyze, then return here.")
    else:
        results = run_evaluation(act["predictions"], detector.threshold, act.get("fault_log"))
        
        st.markdown(f'<div style="text-align:right; font-size:12px; color:#888;">Source: {str(act["source"]).title()}</div>', unsafe_allow_html=True)
        
        if results.get("has_labels", False):
            with st.expander("Model Performance", expanded=True):
                mcols = st.columns(5)
                met = results["metrics"]
                mcols[0].metric("Precision", f'{met["precision"]:.3f}')
                mcols[1].metric("Recall", f'{met["recall"]:.3f}')
                mcols[2].metric("F1-Score", f'{met["f1"]:.3f}')
                mcols[3].metric("ROC-AUC", f'{met["roc_auc"]:.3f}')
                mcols[4].metric("PR-AUC", f'{met["pr_auc"]:.3f}')
                
                c1, c2 = st.columns(2)
                c1.plotly_chart(results["roc_fig"], use_container_width=True, key="roc_chart")
                c2.plotly_chart(results["pr_fig"], use_container_width=True, key="pr_chart")
                st.plotly_chart(results["f1_fig"], use_container_width=True, key="f1_chart")
                
            with st.expander("Error Analysis", expanded=False):
                c1, c2 = st.columns(2)
                c1.plotly_chart(results["cm_fig"], use_container_width=True, key="cm_chart")
                c2.plotly_chart(results["error_dist_fig"], use_container_width=True, key="err_dist_chart_labeled")
                if "per_sensor_fig" in results:
                    st.plotly_chart(results["per_sensor_fig"], use_container_width=True, key="per_sensor_chart_labeled")
        else:
            if act["source"] != "manual":
                st.info("ℹ️ Evaluation metrics require ground-truth labels. Upload a CSV with a 'label' column to unlock.")
            if "error_dist_fig" in results:
                st.plotly_chart(results["error_dist_fig"], use_container_width=True, key="err_dist_chart_unlabeled")
            if "per_sensor_fig" in results:
                st.plotly_chart(results["per_sensor_fig"], use_container_width=True, key="per_sensor_chart_unlabeled")

        with st.expander("Signal View", expanded=True):
            st.plotly_chart(results["score_timeline_fig"], use_container_width=True, key="timeline_chart")
            st.plotly_chart(results["signal_fig"], use_container_width=True, key="signal_chart")


# ── tab 4: Model Info ─────────────────────────────────────────────────────────
with tab_model:
    c_info, c_stats = st.columns([3, 2])
    with c_info:
        st.markdown('<div class="sec-hdr">🧠 LSTM Autoencoder Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-flow">
          <div class="arch-block accent-cyan">
            <div class="arch-title">📥 Input Window</div>
            <div class="arch-sub">Shape: (batch, 50 timesteps, 4 features)</div>
          </div>
          <div class="arch-arrow">↓</div>
          <div class="arch-block accent-blue">
            <div class="arch-title">🔒 LSTM Encoder — Layer 1</div>
            <div class="arch-sub">LSTM(64, return_sequences=True) → Dropout(0.20)</div>
          </div>
          <div class="arch-arrow">↓</div>
          <div class="arch-block accent-blue">
            <div class="arch-title">🔒 LSTM Encoder — Layer 2 (Bottleneck)</div>
            <div class="arch-sub">LSTM(32, return_sequences=False) → latent vector</div>
          </div>
          <div class="arch-arrow">↓</div>
          <div class="arch-block accent-purple">
            <div class="arch-title">🔓 LSTM Decoder — Layer 1</div>
            <div class="arch-sub">RepeatVector(50) → LSTM(32, return_sequences=True) → Dropout(0.20)</div>
          </div>
          <div class="arch-arrow">↓</div>
          <div class="arch-block accent-purple">
            <div class="arch-title">🔓 LSTM Decoder — Layer 2</div>
            <div class="arch-sub">LSTM(64, return_sequences=True) → TimeDistributed Dense(4)</div>
          </div>
          <div class="arch-arrow">↓</div>
          <div class="arch-block accent-green">
            <div class="arch-title">📊 Anomaly Scoring</div>
            <div class="arch-sub">MAE(input, reconstruction) → compare vs threshold (p95 of val errors)</div>
          </div>
        </div>
        <div style="margin-top:20px;padding:14px 18px;background:rgba(0,212,255,0.05);border:1px solid rgba(0,212,255,0.15);border-radius:12px;font-size:12px;color:rgba(240,246,255,0.6);line-height:1.7">
          <strong style="color:#00D4FF">Training regime:</strong> Trained exclusively on <em>normal</em> cable readings (label=0). At inference, any window that the model cannot reconstruct well (high MAE) is flagged as anomalous — the model has never seen faults and therefore cannot reconstruct them accurately.
        </div>
        """, unsafe_allow_html=True)

    with c_stats:
        st.markdown('<div class="sec-hdr">📐 Input Feature Statistics</div>', unsafe_allow_html=True)
        scaler = detector.scaler
        if scaler:
            df_stats = pd.DataFrame({
                "Sensor": FEATURES,
                "Train Min": [f"{v:.3f}" for v in scaler.data_min_],
                "Train Max": [f"{v:.3f}" for v in scaler.data_max_]
            })
            st.dataframe(df_stats, hide_index=True, use_container_width=True)
        st.markdown('<div class="sec-hdr" style="margin-top:24px">⚠ Fault Type Legend</div>', unsafe_allow_html=True)
        for ftype, color in FAULT_COLORS.items():
            if ftype == "none": continue
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:7px">'
                f'<div style="width:12px;height:12px;border-radius:50%;background:{color};flex-shrink:0;box-shadow:0 0 6px {color}88"></div>'
                f'<span style="font-size:12px;color:rgba(240,246,255,0.65);font-weight:500">{ftype.replace("_"," ").title()}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
