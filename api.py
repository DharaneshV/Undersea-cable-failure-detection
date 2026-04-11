import asyncio
import json
import os
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from model import CableFaultDetector
from config import SEQ_LEN, FEATURES
from utils import ema

app = FastAPI(title="Smart Grid Flow API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-load model on startup
print("Loading model for fast API inference...")
detector = CableFaultDetector()
detector.load()
print("Model loaded.")

def severity_of(score: float, threshold: float) -> tuple[str, str]:
    r = score / threshold if threshold > 0 else 0
    if r > 5:     return "Critical", "sev-critical"
    if r > 3:     return "High",     "sev-high"
    if r > 1.2:   return "Medium",   "sev-medium"
    if r > 1.0:   return "Low",      "sev-low"
    if r > 0.75:  return "Degrading","sev-warning"
    return               "Normal",   "sev-low"

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
    return {"status": "online", "threshold": detector.threshold}

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

        result_full = detector.predict(df_full)
        thr = detector.threshold
        window = SEQ_LEN

        speed_map = {"0.25×": 0.10, "0.5×": 0.05, "1×": 0.02, "2×": 0.01, "5×": 0.004, "Max": 0}
        delay = speed_map.get(speed, 0.01)
        
        # We can emit more frames over websocket than streamlit could
        skip_map = {"0.25×": 1, "0.5×": 1, "1×": 2, "2×": 3, "5×": 8, "Max": 20}
        frame_skip = skip_map.get(speed, 1)

        detected_faults = []

        for i in range(window, len(result_full)):
            row = result_full.iloc[i]

            sc = float(row["anomaly_score"]) if not pd.isna(row["anomaly_score"]) else 0.0
            is_fault = bool(sc > thr)
            is_warning = bool(sc > 0.75 * thr and not is_fault and not pd.isna(row["anomaly_score"]))
            ftype = row.get("fault_type", "none")

            new_fault = None
            if is_fault and (not detected_faults or detected_faults[-1]["_idx"] < i - 50):
                dist = match_fault_distance(i, fault_log)
                sev_label, sev_cls = severity_of(sc, thr)
                fault_obj = {
                    "_idx": i, "ftype_raw": str(ftype), "Time": str(row["timestamp"])[:19],
                    "fault_type": str(ftype).replace("_", " ").title(), "Severity": sev_label,
                    "anomaly_score": round(sc, 4), "est_distance": float(dist)
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
                    "voltage": round(float(row["voltage"]), 2),
                    "current": round(float(row["current"]), 3),
                    "temperature": round(float(row["temperature"]), 2),
                    "vibration": round(float(row["vibration"]), 4),
                    "anomaly_score": round(sc, 5),
                    "threshold": round(thr, 5),
                    "is_fault": is_fault,
                    "is_warning": is_warning,
                    "health_pct": round(hp, 1),
                    "xai_text": xai_text,
                    "new_fault": new_fault
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
        print(f"Error in websocket loop: {e}")
        try:
            await websocket.close()
        except:
            pass
