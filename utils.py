"""
utils.py
Shared utility functions used across model.py, evaluate.py, and dashboard.py.
Keeping them here avoids circular imports and mid-file import hacks.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


def make_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Sliding-window segmentation.
    Input shape : (n_samples, n_features)
    Output shape: (n_windows, seq_len, n_features)
    """
    n = len(data) - seq_len
    if n <= 0:
        raise ValueError(
            f"Data length {len(data)} is shorter than seq_len {seq_len}."
        )
    return np.array([data[i : i + seq_len] for i in range(n)])


def clip_to_scaler_bounds(data: np.ndarray, scaler) -> np.ndarray:
    """
    Clips inference data to the Min/Max range seen during training.
    Prevents the Scaler from extrapolating, which often produces
    nonsensical inputs for the model during extreme sensor spikes.
    """
    lo, hi = scaler.data_min_, scaler.data_max_
    
    # Optional: Log warning if we're clipping a significant amount
    # (e.g. more than 10% beyond the original range)
    for i in range(data.shape[1]):
        col_min, col_max = data[:, i].min(), data[:, i].max()
        if col_min < lo[i] * 0.9 or col_max > hi[i] * 1.1:
            log.warning(
                "Feature %d has extreme values [%.4f, %.4f] outside training range [%.4f, %.4f]. Clipping.",
                i, col_min, col_max, lo[i], hi[i]
            )
            
    return np.clip(data, lo, hi)


def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Sweep candidate thresholds and return the one that maximises F1-score.
    Falls back gracefully when all predictions are one class.
    """
    from sklearn.metrics import f1_score

    candidates = np.percentile(scores, np.arange(50, 100, 0.5))
    best_f1, best_thr = -1.0, candidates[0]

    for thr in candidates:
        preds = (scores > thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    log.info("Optimal threshold: %.5f  (F1 = %.4f)", best_thr, best_f1)
    return float(best_thr)


def ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average — smoother health metric than a simple mean."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out
