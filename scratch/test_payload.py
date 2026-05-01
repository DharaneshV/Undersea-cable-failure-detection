import sys
import pandas as pd
import numpy as np
import json

sys.path.append("d:\\undersea cable failure detection")
from model import CableFaultDetector
from config import FEATURES, SEQ_LEN

print("Loading model...")
detector = CableFaultDetector()
detector.load()

print("Loading dataset...")
df_full = pd.read_csv("datasets/ai4i2020.csv")

print("Predicting...")
result_full = detector.predict(df_full)

print("Extracting payload...")
window = SEQ_LEN
i = window

row = result_full.iloc[i]
sc = float(row["anomaly_score"]) if not pd.isna(row["anomaly_score"]) else 0.0
thr = float(detector.threshold) if detector.threshold else 1.0
is_fault = bool(row["predicted_label"] == 1)
is_warning = bool(sc > thr * 0.75 and not is_fault)
new_fault = None
hp = 100.0

def ema(data, alpha):
    if len(data) == 0: return np.array([])
    out = np.zeros_like(data)
    out[0] = data[0]
    for j in range(1, len(data)): out[j] = alpha * data[j] + (1 - alpha) * out[j - 1]
    return out

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

try:
    payload = {
        "index": int(i),
        "total": int(len(result_full)),
        "timestamp": str(row["timestamp"])[:19],
        "voltage": round(float(row.get("voltage", 0.0)) if pd.notna(row.get("voltage", 0.0)) else 0.0, 2),
        "current": round(float(row.get("current", 0.0)) if pd.notna(row.get("current", 0.0)) else 0.0, 3),
        "temperature": round(float(row.get("temperature", 0.0)) if pd.notna(row.get("temperature", 0.0)) else 0.0, 2),
        "vibration": round(float(row.get("vibration", 0.0)) if pd.notna(row.get("vibration", 0.0)) else 0.0, 4),
        "anomaly_score": round(sc, 5),
        "threshold": round(thr, 5),
        "is_fault": is_fault,
        "is_warning": is_warning,
        "health_pct": round(hp, 1),
        "xai_text": xai_text,
        "new_fault": new_fault
    }
    s = json.dumps(payload)
    print("Payload JSON:", s)
except Exception as e:
    import traceback
    traceback.print_exc()

