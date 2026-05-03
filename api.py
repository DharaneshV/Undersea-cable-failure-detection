import asyncio
import json
import logging
import os
import time

import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from functools import wraps
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from pydantic import BaseModel, Field, validator
from model import CableFaultDetector
from config import SEQ_LEN, FEATURES, NORMAL_PROFILES
from utils import ema
from reports import ReportGenerator

# In-memory storage for report links (for session duration)
REPORTS_DB = {}

def _safe_remote_address(request: Request) -> str:
    """Rate-limit key function that handles TestClient (no real client socket)."""
    return (request.client.host if request.client else None) or "127.0.0.1"

limiter = Limiter(key_func=_safe_remote_address, config_filename=None)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting Smart Grid API...")
    yield
    log.info("Shutting down Smart Grid API...")


app = FastAPI(
    title="Smart Grid Flow API",
    description="Real-time undersea cable fault detection API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.limiter = limiter


class SensorReading(BaseModel):
    # Core electrical sensors (always required)
    voltage:     float = Field(..., ge=0,   le=500,  description="Voltage in volts")
    current:     float = Field(..., ge=0,   le=20,   description="Current in amperes")
    temperature: float = Field(..., ge=-10, le=100,  description="Temperature in Celsius")
    vibration:   float = Field(..., ge=-10, le=10,   description="Vibration in g")
    # Extended multi-modal sensors (optional — default to neutral/safe values)
    acoustic_strain:     float = Field(default=0.0,  ge=-100, le=100,  description="Acoustic strain (µε)")
    optical_osnr:        float = Field(default=20.0, ge=-30,  le=50,   description="Optical OSNR (dB)")
    optical_ber:         float = Field(default=0.0,  ge=-20,  le=5,    description="Optical BER (log10)")
    optical_power:       float = Field(default=0.0,  ge=-10,  le=5,    description="Optical power (dBm)")
    cable_distance_norm: float = Field(default=0.0,  ge=0,    le=1,    description="Normalised fault position [0–1]")
    # Domain identifier — 0=Electrical, 1=Optical, 2=Hybrid, 3=Acoustic
    cable_domain_id:     int   = Field(default=0,    ge=0,    le=9,    description="Cable domain ID")

    @validator("voltage")
    def validate_voltage_range(cls, v):
        profile = NORMAL_PROFILES["voltage"]
        if abs(v - profile[0]) > profile[1] * 10:
            log.warning(f"Unusual voltage reading: {v}")
        return v

    @validator("current")
    def validate_current_range(cls, c):
        profile = NORMAL_PROFILES["current"]
        if abs(c - profile[0]) > profile[1] * 10:
            log.warning(f"Unusual current reading: {c}")
        return c


class BatchPredictionRequest(BaseModel):
    readings: list[SensorReading]
    batch_id: str = Field(default="batch", description="Batch identifier for tracking")

# Lazy-load model helper
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        try:
            log.info("Lazy-loading model for inference...")
            _detector = CableFaultDetector()
            _detector.load()
            log.info("Model loaded successfully.")
        except Exception as e:
            log.error(f"Failed to load model: {e}")
            # We don't raise here to allow the API to stay up, 
            # but prediction endpoints will fail gracefully.
    return _detector

def severity_of(score: float, threshold: float) -> tuple[str, str]:
    """Map 1-P(Normal) fault probability to a severity label.
    score is now [0, 1] — no longer a ratio against threshold.
    threshold is kept as parameter for API compatibility but not used for ratio.
    """
    if score > 0.70:   return "Critical",  "sev-critical"
    if score > 0.50:   return "High",       "sev-high"
    if score > 0.30:   return "Medium",     "sev-medium"
    if score > 0.15:   return "Low",        "sev-low"
    if score > 0.05:   return "Degrading",  "sev-warning"
    return                    "Normal",     "sev-normal"

def match_fault_distance(sample_idx: int, fault_log: list) -> float:
    for fl in fault_log:
        if fl["start_sample"] <= sample_idx < fl["start_sample"] + fl["duration_samples"]:
            return fl["fault_distance_m"]
    return 0.0

@app.get("/datasets")
def get_datasets():
    if not os.path.exists("datasets"):
        return {"datasets": []}
    files = [f for f in os.listdir("datasets") if f.endswith(".csv") and not f.endswith("_fault_log.csv")]
    return {"datasets": files}


@app.get("/status")
def status():
    det = get_detector()
    return {
        "status": "online" if det else "degraded (model missing)",
        "threshold": det.threshold if det else 0.0,
        "model_type": "conv_transformer_ae",
        "seq_len": SEQ_LEN,
    }


@app.get("/model/info")
def model_info():
    """Get detailed model information."""
    det = get_detector()
    if not det:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    import os, pickle
    # Try to load cached roc_auc from a sidecar file if it exists
    roc_auc = None
    roc_path = os.path.join("saved_model", "roc_auc.pkl")
    if os.path.exists(roc_path):
        with open(roc_path, "rb") as f:
            roc_auc = pickle.load(f)
    return {
        "model_type": "conv_transformer_ae",
        "version": "2.0",
        "features": FEATURES,
        "sequence_length": SEQ_LEN,
        "threshold": det.threshold,
        "roc_auc": roc_auc,
        "dataset": "Azure Predictive Maintenance",
        "architecture": "Conv1D + SinePositionalEncoding + TransformerEncoder × 2 + GlobalAvgPool → Dense(32) bottleneck",
    }


@app.post("/predict/single")
@limiter.limit("30/minute")
async def predict_single(request: Request, reading: SensorReading):
    """Predict anomaly for a single sensor reading."""
    det = get_detector()
    if not det:
        raise HTTPException(status_code=503, detail="Model not loaded")

    log.info(f"Prediction request: {reading.dict()}")
    
    df = pd.DataFrame([{
        "voltage":            reading.voltage,
        "current":            reading.current,
        "temperature":        reading.temperature,
        "vibration":          reading.vibration,
        "acoustic_strain":    reading.acoustic_strain,
        "optical_osnr":       reading.optical_osnr,
        "optical_ber":        reading.optical_ber,
        "optical_power":      reading.optical_power,
        "cable_distance_norm": reading.cable_distance_norm,
        "cable_domain_id":    reading.cable_domain_id,
        "timestamp":          pd.Timestamp.now(),
        "label":              0,
        "fault_type":         "none",
    }])
    
    result = det.predict(df)
    score = float(result["anomaly_score"].iloc[-1])
    
    return {
        "anomaly_score": score,
        "threshold": det.threshold,
        "is_anomaly": score > det.threshold,
        "severity": severity_of(score, det.threshold)[0],
        "diagnosis": result["fault_diagnosis"].iloc[-1],
    }


@app.post("/predict/batch")
@limiter.limit("10/minute")
async def predict_batch(request: Request, batch: BatchPredictionRequest):
    """Predict anomalies for a batch of sensor readings."""
    det = get_detector()
    if not det:
        raise HTTPException(status_code=503, detail="Model not loaded")

    log.info(f"Batch prediction: {len(batch.readings)} readings, batch_id={batch.batch_id}")
    start_time = time.time()
    
    readings_data = [r.dict() for r in batch.readings]
    df = pd.DataFrame(readings_data)
    df["timestamp"] = pd.date_range("now", periods=len(df), freq="100ms")
    df["label"] = 0
    df["fault_type"] = "none"
    
    if len(df) < SEQ_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {SEQ_LEN} readings for prediction"
        )
    
    result = det.predict(df)
    thr = det.threshold
    
    predictions = []
    for i, row in result.iterrows():
        if pd.isna(row["anomaly_score"]):
            continue
        sc = float(row["anomaly_score"])
        predictions.append({
            "index": i,
            "anomaly_score": sc,
            "is_anomaly": sc > thr,
            "severity": severity_of(sc, thr)[0],
            "diagnosis": row["fault_diagnosis"],
        })
    
    anomaly_count = sum(1 for p in predictions if p["is_anomaly"])
    elapsed = time.time() - start_time
    
    log.info(f"Batch complete: {anomaly_count} anomalies in {elapsed:.2f}s")
    
    return {
        "batch_id": batch.batch_id,
        "total_readings": len(batch.readings),
        "predictions": predictions,
        "anomaly_count": anomaly_count,
        "processing_time": elapsed,
    }


class ReportRequest(BaseModel):
    fault_log: list
    metadata: dict = {}
    format: str = "pdf"  # "pdf" or "csv"

@app.post("/report/generate")
async def generate_report(request: ReportRequest):
    """Generate a forensic report on the server and return a download ID."""
    import uuid
    report_id = str(uuid.uuid4())
    os.makedirs("generated_reports", exist_ok=True)
    
    ext = "pdf" if request.format == "pdf" else "csv"
    file_path = f"generated_reports/report_{report_id}.{ext}"
    
    # Add system context to metadata
    det = get_detector()
    full_meta = {
        "deployment_id": "CABLE-ALPHA-9",
        "threshold": float(det.threshold or 0.0) if det else 0.0,
        "model_version": "2.1.0-transformer",
        "source": request.metadata.get("selected_dataset", "Live Stream"),
        **request.metadata
    }

    try:
        if request.format == "pdf":
            ReportGenerator.generate_pdf(request.fault_log, full_meta, file_path)
        else:
            ReportGenerator.generate_csv(request.fault_log, file_path)
        
        REPORTS_DB[report_id] = file_path
        return {"report_id": report_id, "format": request.format}
    except Exception as e:
        log.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@app.get("/report/download/{report_id}")
async def download_report(report_id: str):
    """Download a previously generated report."""
    from fastapi.responses import FileResponse
    file_path = REPORTS_DB.get(report_id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found or expired")
    
    filename = os.path.basename(file_path)
    return FileResponse(path=file_path, filename=filename, media_type='application/octet-stream')

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, dataset: str, speed: str = "2×"):
    await websocket.accept()
    try:
        csv_path = os.path.join("datasets", dataset)
        if not os.path.exists(csv_path):
            await websocket.send_text(json.dumps({"error": "Dataset not found"}))
            await websocket.close()
            return

        print(f"Streaming {csv_path} at {speed} speed.")
        df_full = pd.read_csv(csv_path)

        log_name = dataset.replace(".csv", "_fault_log.csv")
        log_path = os.path.join("datasets", log_name)
        fault_log = []
        if os.path.exists(log_path):
            fault_log = pd.read_csv(log_path).to_dict('records')

        det = get_detector()
        if not det:
            await websocket.send_text(json.dumps({"error": "Model not loaded on server"}))
            await websocket.close()
            return

        result_full = det.predict(df_full)
        thr = det.threshold
        window = SEQ_LEN

        speed_map = {"0.25×": 0.10, "0.5×": 0.05, "1×": 0.02, "2×": 0.01, "5×": 0.004, "Max": 0}
        delay = speed_map.get(speed, 0.01)
        
        # Emit frames at the configured skip rate for smooth WebSocket streaming
        skip_map = {"0.25×": 1, "0.5×": 1, "1×": 2, "2×": 3, "5×": 8, "Max": 20}
        frame_skip = skip_map.get(speed, 1)

        detected_faults = []

        for i in range(window, len(result_full)):
            row = result_full.iloc[i]

            sc = float(row["anomaly_score"]) if not pd.isna(row["anomaly_score"]) else 0.0
            # anomaly_score is now 1-P(Normal) in [0,1] — use fixed probability thresholds
            is_fault   = bool(sc > 0.30)   # Medium severity and above = detected fault
            is_warning = bool(0.05 < sc <= 0.30 and not pd.isna(row["anomaly_score"]))
            ftype = row.get("fault_type", "none")

            new_fault = None
            if is_fault and (not detected_faults or detected_faults[-1]["_idx"] < i - 50):
                dist = match_fault_distance(i, fault_log)
                sev_label, sev_cls = severity_of(sc, thr)
                
                # Calculate XAI for this specific fault
                feat_errs = {feat: float(row.get(f"err_{feat}", 0)) for feat in FEATURES}
                tot_err = sum(feat_errs.values())
                xai_text = " | ".join([f"{f.title()}: {v/tot_err*100:.0f}%" for f, v in sorted(feat_errs.items(), key=lambda x: x[1], reverse=True)[:2]]) if tot_err > 0 else "Unknown"

                fault_obj = {
                    "_idx": i, 
                    "timestamp": str(row["timestamp"])[:19],
                    "fault_type": row["fault_diagnosis"], 
                    "severity": sev_label,
                    "anomaly_score": round(sc, 4), 
                    "estimated_distance_m": float(dist),
                    "xai_text": xai_text
                }
                detected_faults.append(fault_obj)
                new_fault = fault_obj

            if i % frame_skip == 0 or i == len(result_full) - 1:
                # Calculate health
                chunk_scores = result_full.iloc[max(0, i - 150): i]["anomaly_score"].values
                v = chunk_scores[~np.isnan(chunk_scores)]
                if len(v) == 0:
                    hp = 100.0
                else:
                    smoothed = ema(v / thr, alpha=0.1)
                    hp = float(np.clip(100 * (1 - smoothed[-1]), 0, 100))

                feat_errs = {feat: float(row.get(f"err_{feat}", 0)) for feat in FEATURES}
                tot_err = sum(feat_errs.values())
                xai_text = " | ".join([f"{f.title()}: {v/tot_err*100:.0f}%" for f, v in sorted(feat_errs.items(), key=lambda x: x[1], reverse=True)[:2]]) if tot_err > 0 else "Unknown"
                
                payload = {
                    "index": int(i),
                    "total": int(len(result_full)),
                    "timestamp": str(row["timestamp"])[:19],
                    "voltage":             round(float(row.get("voltage", 0.0)), 2),
                    "current":             round(float(row.get("current", 0.0)), 3),
                    "temperature":         round(float(row.get("temperature", 0.0)), 2),
                    "vibration":           round(float(row.get("vibration", 0.0)), 4),
                    "acoustic_strain":     round(float(row.get("acoustic_strain", 0.0)), 4),
                    "optical_osnr":        round(float(row.get("optical_osnr", 20.0)), 2),
                    "optical_ber":         round(float(row.get("optical_ber", 0.0)), 6),
                    "optical_power":       round(float(row.get("optical_power", 0.0)), 3),
                    "cable_distance_norm": round(float(row.get("cable_distance_norm", 0.0)), 4),
                    "anomaly_score": round(sc, 5),
                    "threshold":     round(thr, 5),
                    "recon_error":   round(float(row.get("recon_error", 0.0)) if pd.notna(row.get("recon_error", 0.0)) else 0.0, 5),
                    "is_fault":     is_fault,
                    "is_warning":   is_warning,
                    "health_pct":   round(hp, 1),
                    "xai_text":     xai_text,
                    "new_fault":    new_fault,
                }
                await websocket.send_text(json.dumps(payload))
                
                if delay > 0:
                    await asyncio.sleep(delay * frame_skip)

        await websocket.send_text(json.dumps({"done": True}))
        await websocket.close()
        print("Done streaming.")
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        log.error(f"Error in websocket loop: {e}")
        try:
            await websocket.close()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
