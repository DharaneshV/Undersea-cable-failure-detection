# Undersea Cable Fault Detection System
### Conv-Transformer Autoencoder · Real-time anomaly detection · Multi-modal sensor fusion

---

## Project overview

This system monitors undersea fibre-optic and electrical cables in real time.
It ingests multi-modal telemetry (voltage, current, temperature, vibration,
acoustic strain, optical OSNR/BER/power) and uses a **Conv-Transformer
Autoencoder** with a classification head to detect and localise faults.

The live dashboard (React + Vite) streams data via WebSocket from a FastAPI
backend, with forensic PDF/CSV reporting and XAI-driven root-cause analysis.

---

## Architecture

```
┌────────────┐      WebSocket       ┌──────────────────┐
│  React UI  │ ◄──────────────────► │  FastAPI Backend  │
│  (Vite)    │      REST API        │  (api.py)         │
└────────────┘                      └────────┬─────────┘
                                             │
                                    ┌────────▼─────────┐
                                    │  Conv-Transformer │
                                    │  Autoencoder      │
                                    │  (model.py)       │
                                    └──────────────────┘
```

---

## Files

| File | Purpose |
|------|---------|
| `api.py`       | FastAPI backend with WebSocket streaming & model inference |
| `frontend/`    | React (Vite) dashboard with real-time charts & forensic tools |
| `model.py`     | Conv-Transformer Autoencoder architecture (9 features + 10 domain channels) |
| `config.py`    | Single source of truth for all hyper-parameters and constants |
| `simulator.py` | Generates multi-modal sensor data with fault injection |
| `evaluate.py`  | Performance analytics (ROC, PR, Confusion Matrix) |
| `reports/`     | PDF/CSV forensic report generator |
| `tests/`       | API and model integration tests |

---

## Setup & Running

### 1. Backend (FastAPI)
```bash
# Create venv and install
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the API
make run-api
# or: uvicorn api:app --reload --port 8000
```

### 2. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** for the dashboard.

### 3. Docker (full stack)
```bash
docker compose up -d        # API + Frontend
docker compose ps           # verify services
docker compose down         # teardown
```

---

## Input Features (9 sensors + domain encoding)

| Feature | Domain | Unit |
|---------|--------|------|
| `voltage` | Electrical | V |
| `current` | Electrical | A |
| `temperature` | Electrical | °C |
| `vibration` | Mechanical | g |
| `acoustic_strain` | Acoustic | µε |
| `optical_osnr` | Optical | dB |
| `optical_ber` | Optical | log₁₀ |
| `optical_power` | Optical | dBm |
| `cable_distance_norm` | Spatial | [0–1] |

The model also receives a **10-channel one-hot domain embedding** (`cable_domain_id`)
for a total of 19 input features per timestep.

---

## Dashboard features

| Feature | Description |
|---------|-------------|
| 🌊 Real-time Streaming | High-frequency WebSocket updates from the backend |
| 📊 Metrics Grid | Live voltage, current, temperature, vibration, and optical OSNR |
| 🔌 Cable Route SVG | Animated path showing fault localisation (TDR estimate) |
| 🧠 Transformer XAI | Explanation of which sensors triggered the anomaly |
| 📄 Forensic Reports | Generate PDF/CSV reports of detected events |
| 🌓 Dark Mode | Premium glassmorphism UI with bioluminescent accents |

---

## How the model works

```
Input window (60 timesteps × 19 features)
       ↓
  [Conv1D + SinePositionalEncoding + TransformerEncoder × 3]
       ↓
  Dual-head output:
    1. Reconstruction → MAE loss (anomaly score = 1 − P(Normal))
    2. Classification → 4-class (Normal, Short, Open, High-Z)
       ↓
  Threshold calibrated via F1-sweep on validation set
```

Trained on **normal data** — faults produce high reconstruction error.

---

## Fault types

| Fault | Class | Sensor signature |
|-------|-------|-----------------|
| Cable cut | Open Circuit | Sudden voltage/current collapse |
| Anchor drag | High-Impedance | Vibration spike + voltage dip |
| Overheating | High-Impedance | Rising temperature + current |
| Insulation failure | Short Circuit | Slow voltage leak + temperature rise |

---

## Fault localisation (TDR)

Distance = (Signal speed × Time delay) / 2

Signal speed in fibre ≈ 2×10⁸ m/s.
Time delay is estimated from the sample index of the detected anomaly.

---

## Make targets

```bash
make install          # Install backend dependencies
make frontend-install # Install frontend dependencies
make train            # Train the model
make test             # Run pytest suite
make run-api          # Start FastAPI server
make run-frontend     # Start Vite dev server
make lint             # Syntax-check all Python modules
make clean            # Remove generated files
```
