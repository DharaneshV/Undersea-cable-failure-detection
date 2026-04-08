# Undersea Cable Fault Detection System
### LSTM Autoencoder · Real-time anomaly detection · No hardware required

---

## Project overview

This system simulates an undersea cable monitoring network. It generates
realistic sensor data (voltage, current, temperature, vibration), injects
fault events (cable cuts, anchor drag, overheating, insulation failure),
and uses an **LSTM Autoencoder** to detect anomalies in real time.

The live dashboard streams the simulation with fault alerts, estimated
fault distance (Time Domain Reflectometry), and rich visual analytics.

---

## Files

| File | Purpose |
|------|---------|
| `simulator.py` | Generates synthetic cable sensor data + injects faults |
| `model.py`     | LSTM Autoencoder training, inference, save/load |
| `dashboard.py` | Streamlit live monitoring dashboard (enhanced) |
| `evaluate.py`  | Generates evaluation plots (ROC, confusion matrix, etc.) |
| `requirements.txt` | Python dependencies |

---

## Setup (first time)

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Run the dashboard

```bash
python -m streamlit run dashboard.py
```

Open your browser at **http://localhost:8501**

- Use the sidebar to set simulation duration, fault count, and playback speed
- Click **▶ Start Simulation**
- Watch the LSTM model detect faults in real time

---

## Dashboard features

| Feature | Description |
|---------|-------------|
| 🎨 Premium dark theme | Glassmorphism styling with Inter font |
| 📊 Custom metric cards | Live voltage, current, temperature, vibration gauges |
| 💚 Health gauge | Real-time cable health score (0–100%) |
| 🔌 Cable route diagram | Animated SVG showing fault locations on the cable |
| ⚠️ Fault severity | Faults classified as Low / Medium / High / Critical |
| 📈 Enhanced charts | Dark-themed Plotly charts with fault-region shading |
| 🗂 Fault log table | Styled, color-coded log of detected faults |
| 📥 CSV export | Download detected faults as a report |
| 🧠 Model metrics | Precision, recall, F1-score after simulation |
| 📋 Summary panel | End-of-simulation statistics and health assessment |

---

## Run model training only (no UI)

```bash
python model.py
```

This trains the LSTM autoencoder on normal data, evaluates it on fault
data, and prints classification metrics (precision, recall, F1, ROC-AUC).

---

## How the LSTM Autoencoder works

```
Normal training data
       ↓
  [LSTM Encoder] → compressed representation (bottleneck)
       ↓
  [LSTM Decoder] → reconstructs original signal
       ↓
  Reconstruction error (MAE) on normal data → set threshold (95th percentile)

At inference:
  New sensor window → model reconstructs it
  If error > threshold → ANOMALY (fault detected)
```

The model is trained **only on normal data** — no fault labels needed.
Faults cause large reconstruction errors because the model has never seen them.

---

## Fault types simulated

| Fault | Sensor signature |
|-------|-----------------| 
| Cable cut | Sudden voltage/current collapse |
| Anchor drag | Vibration spike + voltage dip |
| Overheating | Rising temperature + current |
| Insulation failure | Slow voltage leak + temperature rise |

---

## Fault localisation (TDR)

Distance = (Signal speed × Time delay) / 2

Signal speed in fibre ≈ 2×10⁸ m/s.
Time delay is estimated from the sample index of the detected anomaly.

---
l.
7. Health monitoring — Rolling window health metric for operational awareness.
