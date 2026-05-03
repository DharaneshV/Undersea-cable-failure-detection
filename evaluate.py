"""
evaluate.py
Evaluation analytics for the LSTM Autoencoder.

Exposes two interfaces:
  1. run_evaluation(result_df, threshold, fault_log=None)
       → returns a dict of Plotly figures + metric values. Called by api.py.
  2. CLI entrypoint (python evaluate.py)
       → generates matplotlib PNGs, metrics JSON, and summary report.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score,
    recall_score, roc_curve,
)

from config import FEATURES, SEQ_LEN, SENSOR_COLORS

log = logging.getLogger(__name__)

# ── colours ───────────────────────────────────────────────────────────────────
ACCENT = "#378ADD"
GREEN  = "#1D9E75"
RED    = "#E24B4A"
ORANGE = "#EF9F27"
PURPLE = "#7F77DD"


# ── dark Plotly layout ────────────────────────────────────────────────────────
def _dark_layout(title: str, height: int = 400) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color="rgba(255,255,255,.75)")),
        height=height,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.4)",
        font=dict(family="Inter", color="rgba(255,255,255,.55)"),
        legend=dict(font=dict(size=11, color="rgba(255,255,255,.55)")),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Private helpers — return Plotly figures
# ═══════════════════════════════════════════════════════════════════════════════

def _build_roc_curve(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, fill="tozeroy",
        fillcolor="rgba(55,138,221,0.1)",
        line=dict(color=ACCENT, width=2),
        name=f"LSTM-AE (AUC={roc_auc:.4f})",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color="#8B949E", dash="dash"), name="Random",
    ))
    fig.update_layout(**_dark_layout("ROC Curve", height=380))
    fig.update_xaxes(title_text="False Positive Rate")
    fig.update_yaxes(title_text="True Positive Rate")
    return fig, roc_auc


def _build_pr_curve(y_true, scores):
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    pr_auc_val = auc(rec_arr, prec_arr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rec_arr, y=prec_arr, fill="tozeroy",
        fillcolor="rgba(127,119,221,0.1)",
        line=dict(color=PURPLE, width=2),
        name=f"LSTM-AE (PR-AUC={pr_auc_val:.4f})",
    ))
    fig.update_layout(**_dark_layout("Precision-Recall Curve", height=380))
    fig.update_xaxes(title_text="Recall")
    fig.update_yaxes(title_text="Precision")
    return fig, pr_auc_val


def _build_f1_sweep(y_true, scores, thr):
    sweep_thrs = np.percentile(scores, np.arange(50, 100, 0.5))
    f1_vals = [
        f1_score(y_true, (scores > t).astype(int), zero_division=0)
        for t in sweep_thrs
    ]
    best_idx = int(np.argmax(f1_vals))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_thrs, y=f1_vals,
        line=dict(color=ACCENT, width=2), name="F1",
    ))
    fig.add_vline(
        x=sweep_thrs[best_idx], line_dash="dash", line_color=GREEN,
        annotation_text=f"best F1={f1_vals[best_idx]:.3f}",
        annotation_font_color=GREEN,
    )
    fig.add_vline(
        x=thr, line_dash="dot", line_color=ORANGE,
        annotation_text="p95 thr", annotation_font_color=ORANGE,
    )
    fig.update_layout(**_dark_layout("F1-Score vs Threshold", height=380))
    fig.update_xaxes(title_text="Threshold")
    fig.update_yaxes(title_text="F1")
    return fig, float(sweep_thrs[best_idx]), float(f1_vals[best_idx])


def _build_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Pred Normal", "Pred Fault"],
        y=["True Normal", "True Fault"],
        colorscale=[[0, "#161B22"], [1, ACCENT]],
        showscale=False,
    )
    fig.update_layout(**_dark_layout("Confusion Matrix", height=380))
    return fig, cm


def _build_error_distribution(scores, thr, y_true=None):
    fig = go.Figure()
    if y_true is not None:
        n_scores = scores[y_true == 0]
        f_scores = scores[y_true == 1]
        fig.add_trace(go.Histogram(
            x=n_scores, name="Normal",
            marker_color=GREEN, opacity=0.7,
            histnorm="density", nbinsx=80,
        ))
        fig.add_trace(go.Histogram(
            x=f_scores, name="Fault",
            marker_color=RED, opacity=0.7,
            histnorm="density", nbinsx=80,
        ))
        fig.update_layout(barmode="overlay")
    else:
        fig.add_trace(go.Histogram(
            x=scores, name="All samples",
            marker_color=ACCENT, opacity=0.7,
            histnorm="density", nbinsx=80,
        ))
    fig.add_vline(
        x=thr, line_dash="dash", line_color=ORANGE,
        annotation_text=f"thr={thr:.4f}", annotation_font_color=ORANGE,
    )
    fig.update_layout(**_dark_layout("Reconstruction Error Distribution", height=380))
    return fig


def _build_per_sensor_errors(result_df, features=None):
    """Build a per-sensor error subplot from err_* columns in result_df."""
    features = features or FEATURES
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        subplot_titles=[f.title() for f in features],
        vertical_spacing=0.10, horizontal_spacing=0.06,
    )
    ts = result_df["timestamp"] if "timestamp" in result_df.columns else result_df.index
    for idx, feat in enumerate(features):
        col_name = f"err_{feat}"
        if col_name not in result_df.columns:
            continue
        r, c = divmod(idx, 2)
        color = SENSOR_COLORS.get(feat, ACCENT)
        vals = result_df[col_name].fillna(0)
        fig.add_trace(go.Scatter(
            x=ts, y=vals, name=feat.title(),
            line=dict(color=color, width=1.2),
            fill="tozeroy", fillcolor=f"rgba{tuple(int(color[i:i+2], 16) for i in (1,3,5)) + (0.08,)}",
            showlegend=True,
        ), row=r + 1, col=c + 1)
    fig.update_layout(**_dark_layout("Per-Sensor Reconstruction Error", height=480))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", title_text="MAE")
    return fig


def _build_score_timeline(result_df, thr):
    valid = result_df.dropna(subset=["anomaly_score"])
    ts = valid["timestamp"] if "timestamp" in valid.columns else valid.index
    scores = valid["anomaly_score"].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts, y=scores, name="Score",
        line=dict(color=PURPLE, width=1.2),
        fill="tozeroy", fillcolor="rgba(127,119,221,0.08)",
    ))
    fig.add_hline(
        y=thr, line_dash="dash", line_color=RED,
        annotation_text=f"threshold ({thr:.4f})",
        annotation_font_color=RED,
    )
    fig.update_layout(**_dark_layout("Anomaly Score Over Time", height=350))
    fig.update_xaxes(title_text="Time", showgrid=False)
    fig.update_yaxes(title_text="Anomaly Score")
    return fig


def _build_signal_view(result_df, fault_log=None):
    from plotly.subplots import make_subplots
    features = FEATURES
    fig = make_subplots(
        rows=len(features), cols=1, shared_xaxes=True,
        subplot_titles=[f.title() for f in features],
        vertical_spacing=0.05,
    )
    ts = result_df["timestamp"] if "timestamp" in result_df.columns else result_df.index
    for idx, feat in enumerate(features):
        if feat not in result_df.columns:
            continue
        color = SENSOR_COLORS.get(feat, ACCENT)
        fig.add_trace(go.Scatter(
            x=ts, y=result_df[feat], name=feat.title(),
            line=dict(color=color, width=0.8),
        ), row=idx + 1, col=1)

    # Shade fault regions if fault_log available
    if fault_log:
        for fl in fault_log:
            s = fl["start_sample"]
            e = min(s + fl["duration_samples"], len(result_df) - 1)
            if "timestamp" in result_df.columns:
                x0 = result_df.iloc[s]["timestamp"]
                x1 = result_df.iloc[e]["timestamp"]
            else:
                x0, x1 = s, e
            for r in range(1, len(features) + 1):
                fig.add_vrect(
                    x0=x0, x1=x1, fillcolor=RED, opacity=0.07,
                    line_width=0, row=r, col=1,
                )
    fig.update_layout(**_dark_layout("Sensor Signals", height=500))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# Public API — called by api.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    result_df: pd.DataFrame,
    threshold: float,
    fault_log: list[dict] | None = None,
) -> dict:
    """
    Build all evaluation Plotly figures and compute metrics.

    Parameters
    ----------
    result_df : DataFrame with at least 'anomaly_score' and sensor columns.
                Optionally 'label' (0/1) for classification metrics.
                Optionally 'err_voltage', 'err_current', etc. for per-sensor view.
    threshold : float — the anomaly detection threshold.
    fault_log : optional list of fault dicts from simulator.

    Returns
    -------
    dict with keys:
      Always present:
        'score_timeline_fig', 'per_sensor_fig', 'signal_fig', 'error_dist_fig'
      Present only if labels available:
        'roc_fig', 'roc_auc', 'pr_fig', 'pr_auc',
        'cm_fig', 'confusion_matrix',
        'f1_fig', 'best_f1_thr', 'best_f1',
        'metrics' (dict with precision, recall, f1, roc_auc, pr_auc)
      'has_labels': bool
    """
    results: dict = {}
    valid = result_df.dropna(subset=["anomaly_score"])
    scores = valid["anomaly_score"].values

    # Robust check for classification metrics: requires both classes (0 and 1)
    unique_labels = valid["label"].unique() if "label" in valid.columns else []
    has_labels = len(unique_labels) > 1
    results["has_labels"] = has_labels
    
    if "label" in valid.columns and not has_labels and len(unique_labels) == 1:
        log.warning(f"Evaluation set contains only one class ({unique_labels[0]}). Skipping classification metrics.")

    # Always-available charts
    results["score_timeline_fig"] = _build_score_timeline(result_df, threshold)
    results["signal_fig"] = _build_signal_view(result_df, fault_log)

    # Per-sensor errors (if err_* columns exist)
    has_per_sensor = any(f"err_{f}" in result_df.columns for f in FEATURES)
    if has_per_sensor:
        results["per_sensor_fig"] = _build_per_sensor_errors(result_df)

    if has_labels:
        y_true = valid["label"].values
        y_pred = (scores > threshold).astype(int)

        fig_roc, roc_auc_val = _build_roc_curve(y_true, scores)
        results["roc_fig"] = fig_roc
        results["roc_auc"] = roc_auc_val

        fig_pr, pr_auc_val = _build_pr_curve(y_true, scores)
        results["pr_fig"] = fig_pr
        results["pr_auc"] = pr_auc_val

        fig_f1, best_thr, best_f1 = _build_f1_sweep(y_true, scores, threshold)
        results["f1_fig"] = fig_f1
        results["best_f1_thr"] = best_thr
        results["best_f1"] = best_f1

        fig_cm, cm_array = _build_confusion_matrix(y_true, y_pred)
        results["cm_fig"] = fig_cm
        results["confusion_matrix"] = cm_array

        results["error_dist_fig"] = _build_error_distribution(scores, threshold, y_true)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        results["metrics"] = {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc_val), 4),
            "pr_auc": round(float(pr_auc_val), 4),
            "threshold": round(float(threshold), 6),
            "best_f1_thr": round(float(best_thr), 6),
            "best_f1": round(float(best_f1), 4),
            "n_samples": int(len(result_df)),
            "n_valid": int(len(valid)),
            "n_normal": int((y_true == 0).sum()),
            "n_fault": int((y_true == 1).sum()),
        }
    else:
        results["error_dist_fig"] = _build_error_distribution(scores, threshold)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entrypoint — original behavior preserved
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from model import CableFaultDetector
    from simulator import generate_dataset
    from utils import make_sequences

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # ── plot style ────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor":  "#0E1117",
        "axes.facecolor":    "#161B22",
        "axes.edgecolor":    "#30363D",
        "axes.labelcolor":   "#C9D1D9",
        "text.color":        "#C9D1D9",
        "xtick.color":       "#8B949E",
        "ytick.color":       "#8B949E",
        "grid.color":        "#21262D",
        "grid.alpha":        0.6,
        "font.family":       "sans-serif",
        "font.size":         11,
        "legend.facecolor":  "#161B22",
        "legend.edgecolor":  "#30363D",
        "figure.dpi":        150,
    })

    SENSOR_CLR = {
        "voltage": ACCENT, "current": GREEN,
        "temperature": ORANGE, "vibration": RED,
    }

    OUT_DIR = "evaluation_plots"
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load or train model ───────────────────────────────────────────────
    log.info("Loading model…")
    detector = CableFaultDetector()
    if os.path.exists("saved_model/lstm_ae.keras"):
        detector.load()
    else:
        log.info("No saved model found — training now…")
        df_train, _ = generate_dataset(n_seconds=400, fault_count=0, seed=1)
        detector.train(df_train)
        detector.save()

    log.info("Generating test data…")
    df_test, fault_log = generate_dataset(n_seconds=300, fault_count=5, seed=42)
    result = detector.predict(df_test)

    valid  = result.dropna(subset=["anomaly_score"])
    y_true = valid["label"].values
    scores = valid["anomaly_score"].values
    thr    = detector.threshold
    y_pred = (scores > thr).astype(int)

    def _shade_faults_mpl(ax, timestamps=None):
        for fl in fault_log:
            if timestamps is not None:
                s_ts = df_test.iloc[fl["start_sample"]]["timestamp"]
                e_ts = df_test.iloc[
                    min(fl["start_sample"] + fl["duration_samples"], len(df_test) - 1)
                ]["timestamp"]
                ax.axvspan(s_ts, e_ts, alpha=0.10, color=RED)
            else:
                s = max(0, fl["start_sample"] - SEQ_LEN)
                e = min(ax.get_xlim()[1],
                        fl["start_sample"] + fl["duration_samples"] - SEQ_LEN)
                if e > s:
                    ax.axvspan(s, e, alpha=0.10, color=RED)

    # ── 1. ROC ────────────────────────────────────────────────────────────
    log.info("Plotting ROC curve…")
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.fill_between(fpr, tpr, alpha=0.12, color=ACCENT)
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"LSTM-AE  (AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#8B949E", lw=1, label="Random baseline")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("ROC Curve — LSTM Autoencoder", fontweight="bold", fontsize=13)
    ax.legend(loc="lower right"); ax.grid(True, ls="--")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/roc_curve.png"); plt.close(fig)
    log.info("→ %s/roc_curve.png  (AUC = %.4f)", OUT_DIR, roc_auc_val)

    # ── 2. Precision–Recall ───────────────────────────────────────────────
    log.info("Plotting Precision-Recall curve…")
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, scores)
    pr_auc_val = auc(rec_arr, prec_arr)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.fill_between(rec_arr, prec_arr, alpha=0.12, color=PURPLE)
    ax.plot(rec_arr, prec_arr, color=PURPLE, lw=2,
            label=f"LSTM-AE  (PR-AUC = {pr_auc_val:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=13)
    ax.legend(loc="lower left"); ax.grid(True, ls="--")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/precision_recall_curve.png"); plt.close(fig)
    log.info("→ %s/precision_recall_curve.png  (PR-AUC = %.4f)", OUT_DIR, pr_auc_val)

    # ── 3. F1 vs Threshold ────────────────────────────────────────────────
    log.info("Plotting F1-vs-threshold sweep…")
    sweep_thrs = np.percentile(scores, np.arange(50, 100, 0.5))
    f1_vals = [f1_score(y_true, (scores > t).astype(int), zero_division=0) for t in sweep_thrs]
    best_idx = int(np.argmax(f1_vals))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep_thrs, f1_vals, color=ACCENT, lw=2)
    ax.axvline(sweep_thrs[best_idx], color=GREEN, ls="--", lw=1.5,
               label=f"Best F1 = {f1_vals[best_idx]:.4f}  (thr = {sweep_thrs[best_idx]:.5f})")
    ax.axvline(thr, color=ORANGE, ls=":", lw=1.5, label=f"p95 threshold = {thr:.5f}")
    ax.set(xlabel="Threshold", ylabel="F1-Score")
    ax.set_title("F1-Score vs Detection Threshold", fontweight="bold", fontsize=13)
    ax.legend(); ax.grid(True, ls="--")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/f1_threshold_sweep.png"); plt.close(fig)
    log.info("→ %s/f1_threshold_sweep.png", OUT_DIR)

    # ── 4. Confusion Matrix ───────────────────────────────────────────────
    log.info("Plotting confusion matrix…")
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Fault"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    dark_cmap = LinearSegmentedColormap.from_list("dkblue", ["#161B22", ACCENT])
    im = ax.imshow(cm, interpolation="nearest", cmap=dark_cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            clr = "#fff" if val > cm.max() / 2 else "#C9D1D9"
            ax.text(j, i, f"{val:,}", ha="center", va="center",
                    fontsize=18, fontweight="bold", color=clr)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set(xlabel="Predicted", ylabel="Actual")
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/confusion_matrix.png"); plt.close(fig)
    log.info("→ %s/confusion_matrix.png", OUT_DIR)

    # ── 5. Error Distribution ─────────────────────────────────────────────
    log.info("Plotting error distribution…")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores[y_true == 0], bins=80, alpha=0.65, color=GREEN, label="Normal", density=True)
    ax.hist(scores[y_true == 1], bins=80, alpha=0.65, color=RED, label="Fault", density=True)
    ax.axvline(thr, color=ORANGE, ls="--", lw=2, label=f"p95 Threshold ({thr:.4f})")
    ax.set(xlabel="Reconstruction Error (MAE)", ylabel="Density")
    ax.set_title("Reconstruction Error Distribution", fontweight="bold", fontsize=13)
    ax.legend(); ax.grid(True, ls="--")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/error_distribution.png"); plt.close(fig)
    log.info("→ %s/error_distribution.png", OUT_DIR)

    # ── 6. Per-Sensor Error ───────────────────────────────────────────────
    log.info("Plotting per-sensor reconstruction errors…")
    per_feat = detector.reconstruction_errors_per_feature(df_test)
    ts_windowed = df_test["timestamp"].iloc[SEQ_LEN:].reset_index(drop=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, (feat, ax) in enumerate(zip(FEATURES, axes.flat)):
        err = per_feat[:, idx]
        color = SENSOR_CLR[feat]
        ax.plot(ts_windowed, err, color=color, lw=0.6, alpha=0.8)
        ax.fill_between(ts_windowed, err, alpha=0.10, color=color)
        ax.set_title(feat.title(), fontweight="bold", fontsize=12, color=color)
        ax.set_ylabel("MAE"); ax.grid(True, ls="--")
        _shade_faults_mpl(ax, timestamps=True)
    axes[1, 0].set_xlabel("Time"); axes[1, 1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=20)
    fig.suptitle("Per-Sensor Reconstruction Error", fontweight="bold", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/per_sensor_error.png", bbox_inches="tight"); plt.close(fig)
    log.info("→ %s/per_sensor_error.png", OUT_DIR)

    # ── 7. Anomaly Timeline ───────────────────────────────────────────────
    log.info("Plotting anomaly score timeline…")
    ts = valid["timestamp"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(ts, scores, alpha=0.12, color=PURPLE)
    ax.plot(ts, scores, color=PURPLE, lw=0.7)
    ax.axhline(thr, color=RED, ls="--", lw=1.5, label=f"Threshold ({thr:.4f})")
    _shade_faults_mpl(ax, timestamps=True)
    ax.set(xlabel="Time", ylabel="Anomaly Score")
    ax.set_title("Anomaly Score Over Time", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right"); ax.grid(True, ls="--")
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/anomaly_timeline.png"); plt.close(fig)
    log.info("→ %s/anomaly_timeline.png", OUT_DIR)

    # ── 8. Sensor Signals ─────────────────────────────────────────────────
    log.info("Plotting sensor signals with fault zones…")
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for idx, (feat, ax) in enumerate(zip(FEATURES, axes)):
        color = SENSOR_CLR[feat]
        ax.plot(df_test["timestamp"], df_test[feat], color=color, lw=0.5, alpha=0.85)
        ax.set_ylabel(feat.title(), fontweight="bold", color=color)
        ax.grid(True, ls="--"); _shade_faults_mpl(ax, timestamps=True)
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate(rotation=20)
    fig.suptitle("Sensor Signals — fault zones shaded red",
                 fontweight="bold", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/sensor_signals.png", bbox_inches="tight"); plt.close(fig)
    log.info("→ %s/sensor_signals.png", OUT_DIR)

    # ── 9. Metrics JSON ───────────────────────────────────────────────────
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    metrics = {
        "threshold": round(float(thr), 6),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_val), 4),
        "pr_auc": round(float(pr_auc_val), 4),
        "best_f1_thr": round(float(sweep_thrs[best_idx]), 6),
        "best_f1": round(float(f1_vals[best_idx]), 4),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(result)),
        "n_valid": int(len(valid)),
        "n_normal": int((y_true == 0).sum()),
        "n_fault": int((y_true == 1).sum()),
    }
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("→ %s/metrics.json", OUT_DIR)

    # ── 10. Summary Report ────────────────────────────────────────────────
    report_txt = f"""
+--------------------------------------------------------------+
|         LSTM Autoencoder — Evaluation Summary                |
+--------------------------------------------------------------+
|  Total samples          : {len(result):>8,}                          |
|  Valid (windowed)        : {len(valid):>8,}                          |
|  Normal samples          : {int((y_true==0).sum()):>8,}                          |
|  Fault samples           : {int((y_true==1).sum()):>8,}                          |
|                                                              |
|  Threshold (p{str(95).ljust(2)})        : {thr:>10.5f}                        |
|  Best F1 threshold       : {sweep_thrs[best_idx]:>10.5f}                        |
|                                                              |
|  — Classification ─────────────────────────────────         |
|  Precision               : {prec:>10.4f}                        |
|  Recall                  : {rec:>10.4f}                        |
|  F1-Score                : {f1:>10.4f}                        |
|  Best possible F1        : {f1_vals[best_idx]:>10.4f}                        |
|  ROC-AUC                 : {roc_auc_val:>10.4f}                        |
|  PR-AUC                  : {pr_auc_val:>10.4f}                        |
|                                                              |
|  — Confusion Matrix ───────────────────────────────         |
|           Pred Normal  Pred Fault                            |
|  Normal   {cm[0,0]:>8,}    {cm[0,1]:>8,}                            |
|  Fault    {cm[1,0]:>8,}    {cm[1,1]:>8,}                            |
|                                                              |
|  — Fault Log ──────────────────────────────────             |
"""
    for fl in fault_log:
        ft = fl["fault_type"].replace("_", " ").title()
        report_txt += (
            f"|  {ft:<22}  start={fl['start_sample']:>5}  "
            f"dur={fl['duration_samples']:>4}  "
            f"pos={fl['fault_distance_m']:.0f} m  |\n"
        )
    report_txt += "+--------------------------------------------------------------+\n"

    print(report_txt)
    with open(f"{OUT_DIR}/summary_report.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)
    log.info("→ %s/summary_report.txt", OUT_DIR)
    log.info("\n[OK] All outputs saved to '%s/'", OUT_DIR)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fault"]))
