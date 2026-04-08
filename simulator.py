"""
simulator.py
Generates synthetic undersea cable sensor data.

Key improvements over v1:
  - Each fault is assigned a true random cable position (metres); no more
    wrap-around modulo nonsense. TDR reports this value directly.
  - Uses a local np.random.RandomState instead of the global seed, so
    changing fault_count no longer shifts unrelated draws.
  - Minimum fault duration is 2 × SEQ_LEN to guarantee full LSTM windows
    are available over every fault region.
  - Removed the redundant `add_faults` flag; use fault_count=0 instead.
"""

import logging
import numpy as np
import pandas as pd

from config import (
    SAMPLE_RATE, CABLE_LENGTH, SIGNAL_SPEED,
    NORMAL_PROFILES, FAULT_TYPES, SEQ_LEN,
)

log = logging.getLogger(__name__)


# ── fault injection ──────────────────────────────────────────────────────────
def _inject_fault(
    df: pd.DataFrame,
    fault_type: str,
    start: int,
    duration: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Modify a slice of the dataframe to simulate a fault."""
    end = min(start + duration, len(df))
    n   = end - start

    if fault_type == "cable_cut":
        df.iloc[start:end, df.columns.get_loc("voltage")]   = rng.uniform(0, 10,   n)
        df.iloc[start:end, df.columns.get_loc("current")]   = rng.uniform(0, 0.1,  n)
        df.iloc[start:end, df.columns.get_loc("vibration")] = rng.normal(0.8, 0.3, n)

    elif fault_type == "anchor_drag":
        t = np.linspace(0, np.pi, n)
        df.iloc[start:end, df.columns.get_loc("vibration")] += (
            np.sin(t) * rng.uniform(1.5, 3.0)
        )
        df.iloc[start:end, df.columns.get_loc("voltage")] -= (
            np.sin(t) * rng.uniform(5, 15)
        )

    elif fault_type == "overheating":
        ramp = np.linspace(0, 1, n)
        df.iloc[start:end, df.columns.get_loc("temperature")] += (
            ramp * rng.uniform(15, 35)
        )
        df.iloc[start:end, df.columns.get_loc("current")] += (
            ramp * rng.uniform(1, 3)
        )

    elif fault_type == "insulation_failure":
        ramp = np.linspace(0, 1, n)
        df.iloc[start:end, df.columns.get_loc("voltage")] -= (
            ramp * rng.uniform(20, 60)
        )
        df.iloc[start:end, df.columns.get_loc("temperature")] += (
            ramp * rng.uniform(5, 20)
        )
        df.iloc[start:end, df.columns.get_loc("current")] += (
            ramp * rng.uniform(0.5, 2)
        )

    return df


# ── main generation function ─────────────────────────────────────────────────
def generate_dataset(
    n_seconds: int   = 600,
    fault_count: int = 6,
    seed: int        = 42,
    env_noise: bool  = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Generate a synthetic cable dataset.

    Returns
    -------
    df : DataFrame
        Columns: timestamp, voltage, current, temperature, vibration,
                 label (0/1), fault_type
    fault_log : list[dict]
        One entry per injected fault with keys:
          fault_type, start_sample, duration_samples, fault_distance_m
        `fault_distance_m` is the true cable position (0–CABLE_LENGTH)
        that a TDR system would report.
    """
    rng       = np.random.RandomState(seed)
    n_samples = n_seconds * SAMPLE_RATE

    # ── base normal signal ────────────────────────────────────────────────────
    t = np.arange(n_samples) / SAMPLE_RATE
    df = pd.DataFrame({
        "timestamp":   pd.to_datetime(t, unit="s", origin="2025-01-01"),
        "voltage":     rng.normal(*NORMAL_PROFILES["voltage"],     n_samples),
        "current":     rng.normal(*NORMAL_PROFILES["current"],     n_samples),
        "temperature": rng.normal(*NORMAL_PROFILES["temperature"], n_samples),
        "vibration":   rng.normal(*NORMAL_PROFILES["vibration"],   n_samples),
        "label":       0,
        "fault_type":  "none",
    })

    # Realistic low-frequency drift
    df["voltage"]     += 0.5 * np.sin(2 * np.pi * t / 120)
    df["temperature"] += 0.3 * np.sin(2 * np.pi * t / 300)

    if env_noise:
        # Deep-sea ocean currents and thermal gradients
        df["vibration"]   += 0.15 * np.sin(2 * np.pi * t / 45) * np.sin(2 * np.pi * t / 150)
        df["temperature"] += 0.4 * np.cos(2 * np.pi * t / 800)
        df["current"]     += 0.08 * np.sin(2 * np.pi * t / 200)


    fault_log: list[dict] = []
    if fault_count <= 0:
        return df, fault_log

    # Minimum fault duration: 2 × SEQ_LEN so the LSTM always has full windows
    min_dur_samples = int(2 * SEQ_LEN)

    used_ranges: list[tuple[int, int]] = []

    for _ in range(fault_count):
        ftype    = rng.choice(FAULT_TYPES)
        duration = max(
            min_dur_samples,
            int(rng.uniform(20, 80) * SAMPLE_RATE),  # 20–80 s
        )

        # Find a non-overlapping window, with a safety margin of SEQ_LEN
        margin = SEQ_LEN
        for _attempt in range(100):
            start = int(rng.uniform(0.05, 0.90) * n_samples)
            end   = start + duration
            if end >= n_samples - margin:
                continue
            if any(s - margin <= end and e + margin >= start
                   for s, e in used_ranges):
                continue
            break
        else:
            log.warning("Could not place fault %s without overlap; skipping.", ftype)
            continue

        used_ranges.append((start, start + duration))
        df = _inject_fault(df, ftype, start, duration, rng)
        df.iloc[start : start + duration, df.columns.get_loc("label")]      = 1
        df.iloc[start : start + duration, df.columns.get_loc("fault_type")] = ftype

        # True cable position: random point along the cable.
        # This is what a real TDR system would measure (round-trip echo delay).
        fault_distance_m = round(float(rng.uniform(0, CABLE_LENGTH)), 1)

        fault_log.append({
            "fault_type":       ftype,
            "start_sample":     start,
            "duration_samples": duration,
            "fault_distance_m": fault_distance_m,
        })
        log.debug(
            "Injected %s at sample %d (%.1f m)",
            ftype, start, fault_distance_m,
        )

    log.info(
        "Dataset generated: %d samples, %d faults injected",
        n_samples, len(fault_log),
    )
    return df, fault_log


# ── quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, log_ = generate_dataset(n_seconds=300, fault_count=4)
    print(df.head())
    print(f"\n{len(df):,} samples | {df['label'].mean()*100:.1f}% fault")
    print("\nFault log:")
    for f in log_:
        print(f)
