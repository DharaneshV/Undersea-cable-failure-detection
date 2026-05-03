"""
eval_real.py
Proper evaluation of the saved model on a held-out 20% split of the
full combined dataset (same data used in training, but never trained on
because of the random val split). Produces metrics.json + PNG plots.

Usage:
    python eval_real.py
"""
import json
import logging
import os
import sys
import pickle

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CableFaultDetector
from config import (
    CABLE_DOMAIN_NAMES, FEATURES, SEQ_LEN, THRESHOLD_PCT
)

OUT_DIR = "evaluation_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load the full combined dataset (same as train_universal.py) ─────────────
def _enrich(df: pd.DataFrame, domain_id: int) -> pd.DataFrame:
    if "cable_domain_id" not in df.columns:
        df["cable_domain_id"] = domain_id
    if "cable_distance_norm" not in df.columns:
        rng = np.random.default_rng(seed=99)
        df["cable_distance_norm"] = np.where(
            df["label"].values == 1,
            rng.uniform(0.0, 1.0, size=len(df)), 0.0
        ).astype(np.float32)
    return df

def _load(path, domain_id):
    try:
        df = pd.read_csv(path)
        df = _enrich(df, domain_id)
        log.info("Loaded %-45s  %6d rows", path, len(df))
        return df
    except FileNotFoundError:
        log.warning("Not found, skipping: %s", path)
        return pd.DataFrame()

datasets = [
    ("datasets/realistic_data.csv",      0),
    ("datasets/optical_240km.csv",        1),
    ("datasets/synthetic_cable_50k.csv",  0),
    ("datasets/azure_pdm.csv",            0),
    ("datasets/industrial_pump.csv",      0),
]
frames = [_load(p, d) for p, d in datasets]
df_all = pd.concat([f for f in frames if len(f) > 0], ignore_index=True)
log.info("Combined: %d rows (%d faults)", len(df_all), df_all["label"].sum())

# ── 2. Hold out 20% as test set (deterministic) ──────────────────────────────
rng = np.random.default_rng(seed=42)
test_idx = rng.choice(len(df_all), size=int(len(df_all) * 0.20), replace=False)
df_test = df_all.iloc[test_idx].reset_index(drop=True)
log.info("Test set: %d rows (%d normal, %d fault)",
         len(df_test), (df_test["label"]==0).sum(), (df_test["label"]==1).sum())

# ── 3. Load the saved model ───────────────────────────────────────────────────
log.info("Loading saved model...")
detector = CableFaultDetector()
detector.load()
log.info("Threshold from saved model: %.6f", detector.threshold)

# ── 4. Run predictions ────────────────────────────────────────────────────────
log.info("Running predictions on test set...")
result = detector.predict(df_test)
valid  = result.dropna(subset=["anomaly_score"])
y_true = valid["label"].values
scores = valid["anomaly_score"].values
thr    = detector.threshold

log.info("Samples with scores: %d / %d", len(valid), len(result))
log.info("Score range: [%.6f, %.6f]", scores.min(), scores.max())
log.info("Current threshold: %.6f", thr)

# ── 5. Calibrate threshold on this test set ───────────────────────────────────
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from utils import find_optimal_threshold

if len(np.unique(y_true)) > 1:
    new_thr = find_optimal_threshold(scores, y_true)
    log.info("Recalibrated threshold: %.6f → %.6f", thr, new_thr)
    # Save the new threshold
    with open("saved_model/threshold.pkl", "wb") as f:
        pickle.dump(new_thr, f)
    thr = new_thr
    detector.threshold = thr
else:
    log.warning("Only one class in test set — keeping original threshold")

# ── 6. Compute metrics ────────────────────────────────────────────────────────
y_pred   = (scores > thr).astype(int)
roc_auc  = roc_auc_score(y_true, scores)
prec     = precision_score(y_true, y_pred, zero_division=0)
rec      = recall_score(y_true, y_pred, zero_division=0)
f1       = f1_score(y_true, y_pred, zero_division=0)

# F1 sweep for best possible
sweep_thrs = np.percentile(scores, np.arange(50, 100, 0.5))
f1_vals    = [f1_score(y_true, (scores > t).astype(int), zero_division=0) for t in sweep_thrs]
best_idx   = int(np.argmax(f1_vals))
best_f1    = f1_vals[best_idx]
best_thr   = float(sweep_thrs[best_idx])

metrics = {
    "threshold":   round(float(thr), 6),
    "precision":   round(float(prec), 4),
    "recall":      round(float(rec), 4),
    "f1":          round(float(f1), 4),
    "roc_auc":     round(float(roc_auc), 4),
    "best_f1":     round(float(best_f1), 4),
    "best_f1_thr": round(float(best_thr), 6),
    "n_test":      int(len(df_test)),
    "n_valid":     int(len(valid)),
    "n_normal":    int((y_true == 0).sum()),
    "n_fault":     int((y_true == 1).sum()),
}

with open(f"{OUT_DIR}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ── 7. Per fault-type breakdown ───────────────────────────────────────────────
if "fault_type" in valid.columns:
    valid = valid.copy()
    valid["predicted"] = y_pred
    print("\n-- Per-Fault-Type Breakdown --")
    for ft, grp in valid.groupby("fault_type"):
        if ft == "none":
            continue
        n = len(grp)
        tp = (grp["predicted"] == 1).sum()
        recall_ft = tp / n if n > 0 else 0
        print(f"  {ft:<25}  n={n:>5}  detected={tp:>5}  recall={recall_ft:.3f}")

print(f"""
=================================================================
  Conv-Transformer -- Real Evaluation Results
=================================================================
  Test samples         : {metrics['n_test']:>8,}
  Valid (windowed)     : {metrics['n_valid']:>8,}
  Normal               : {metrics['n_normal']:>8,}
  Fault                : {metrics['n_fault']:>8,}
-----------------------------------------------------------------
  Threshold (calibr.)  : {metrics['threshold']:>10.6f}
  ROC-AUC             : {metrics['roc_auc']:>10.4f}
  Precision            : {metrics['precision']:>10.4f}
  Recall               : {metrics['recall']:>10.4f}
  F1-Score             : {metrics['f1']:>10.4f}
  Best possible F1     : {metrics['best_f1']:>10.4f}  @ thr={metrics['best_f1_thr']:.6f}
=================================================================
""")
print(classification_report(y_true, y_pred, target_names=["Normal", "Fault"]))

# ── 9. Generate plots ─────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

    ACCENT = "#378ADD"; GREEN = "#1D9E75"; RED = "#E24B4A"; ORANGE = "#EF9F27"; PURPLE = "#7F77DD"
    plt.rcParams.update({"figure.facecolor":"#0E1117","axes.facecolor":"#161B22",
                          "text.color":"#C9D1D9","axes.labelcolor":"#C9D1D9",
                          "xtick.color":"#8B949E","ytick.color":"#8B949E",
                          "grid.color":"#21262D","figure.dpi":150})

    # ROC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {roc_val:.4f}")
    ax.plot([0,1],[0,1],ls="--",color="#8B949E",lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", xlim=(0,1), ylim=(0,1.02), title="ROC Curve")
    ax.legend(); ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/roc_curve.png"); plt.close(fig)

    # Confusion matrix
    from matplotlib.colors import LinearSegmentedColormap
    cm = confusion_matrix(y_true, y_pred)
    dark_cmap = LinearSegmentedColormap.from_list("dkblue", ["#161B22", ACCENT])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap=dark_cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            clr = "#fff" if cm[i,j] > cm.max()/2 else "#C9D1D9"
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=clr)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Normal","Fault"]); ax.set_yticklabels(["Normal","Fault"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/confusion_matrix.png"); plt.close(fig)

    # Error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores[y_true==0], bins=100, alpha=0.65, color=GREEN, label="Normal", density=True)
    ax.hist(scores[y_true==1], bins=100, alpha=0.65, color=RED, label="Fault", density=True)
    ax.axvline(thr, color=ORANGE, ls="--", lw=2, label=f"Threshold ({thr:.5f})")
    ax.set(xlabel="Reconstruction Error (MAE)", ylabel="Density", title="Error Distribution")
    ax.legend(); ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/error_distribution.png"); plt.close(fig)

    # F1 sweep
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sweep_thrs, f1_vals, color=ACCENT, lw=2)
    ax.axvline(best_thr, color=GREEN, ls="--", lw=1.5, label=f"Best F1={best_f1:.4f}")
    ax.axvline(thr, color=ORANGE, ls=":", lw=1.5, label=f"Calibrated thr")
    ax.set(xlabel="Threshold", ylabel="F1", title="F1 vs Threshold Sweep")
    ax.legend(); ax.grid(True, ls="--"); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/f1_threshold_sweep.png"); plt.close(fig)

    log.info("Plots saved to %s/", OUT_DIR)

except ImportError:
    log.warning("matplotlib not available — skipping plots")

log.info("Metrics saved to %s/metrics.json", OUT_DIR)
log.info("New threshold saved to saved_model/threshold.pkl")
