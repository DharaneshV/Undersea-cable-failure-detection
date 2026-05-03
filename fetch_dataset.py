"""
fetch_dataset.py
Download and adapt real-world fault detection datasets for the
Undersea Cable Fault Detection system.

Supported datasets (from Kaggle):
  1. Smart Grid Terminal Fault Dataset
     - 1,000 records with Voltage, Current, Temperature, Vibration, Fault label
     - CC0 Public Domain
  2. Digital Asset Fault Detection Grid Management Data
     - 3,000 records with Voltage, Current, Temperature, Fault types
     - Includes overload, short circuit, voltage surge, temperature rise

Usage:
  # Option A: With Kaggle API key configured
  python fetch_dataset.py --source kaggle

  # Option B: Download manually from Kaggle, then adapt
  python fetch_dataset.py --file path/to/downloaded.csv --format smart_grid

  # Option C: Generate realistic dataset based on published distributions
  python fetch_dataset.py --source generate --samples 10000
"""

import argparse
import logging
import os
import sys
import zipfile

import numpy as np
import pandas as pd

from config import FEATURES, FAULT_TYPES, NORMAL_PROFILES, SAMPLE_RATE

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── column mappings for known datasets ─────────────────────────────────────────
DATASET_CONFIGS = {
    "smart_grid": {
        "kaggle_slug": "zara2099/smart-grid-terminal-fault-dataset",
        "column_map": {
            "Voltage(V)":       "voltage",
            "Current(A)":       "current",
            "Temperature(C)":   "temperature",
            "Vibration_Level":  "vibration",
        },
        "label_column": "Fault_Status",
        "label_map": {0: 0, 1: 1},  # already binary
        "description": "Smart Grid Terminal Fault Dataset (1,000 records)",
    },
    "grid_fault": {
        "kaggle_slug": "sergiodelarosa/digital-asset-fault-detection-grid-management-data",
        "column_map": {
            "Voltage":     "voltage",
            "Current":     "current",
            "Temperature": "temperature",
        },
        "label_column": "Fault_Type",
        "label_map": {
            "No Fault": 0, "Normal": 0,
            "Overload": 1, "Short Circuit": 1,
            "Voltage Surge": 1, "Temperature Rise": 1,
        },
        "fault_type_column": "Fault_Type",
        "fault_type_map": {
            "No Fault": "none", "Normal": "none",
            "Overload": "overheating",
            "Short Circuit": "cable_cut",
            "Voltage Surge": "insulation_failure",
            "Temperature Rise": "overheating",
        },
        "description": "Digital Asset Fault Detection (3,000 records)",
    },
}


# ── Kaggle download ───────────────────────────────────────────────────────────
def download_from_kaggle(slug: str, dest_dir: str = "datasets") -> str:
    """Download a dataset from Kaggle. Requires kaggle API credentials."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        log.error(
            "Kaggle package not installed. Run: pip install kaggle\n"
            "Then set up your API key: https://www.kaggle.com/docs/api"
        )
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    os.makedirs(dest_dir, exist_ok=True)
    log.info("Downloading %s from Kaggle...", slug)
    api.dataset_download_files(slug, path=dest_dir, unzip=True)

    # Find the CSV file
    for f in os.listdir(dest_dir):
        if f.endswith(".csv"):
            path = os.path.join(dest_dir, f)
            log.info("Downloaded: %s", path)
            return path

    raise FileNotFoundError(f"No CSV found in {dest_dir} after download.")


# ── adapt dataset ─────────────────────────────────────────────────────────────
def adapt_dataset(
    csv_path: str,
    dataset_format: str,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Read a real-world CSV and adapt it to our model's expected format.

    Returns (df, fault_log) matching the simulator output format.
    """
    config = DATASET_CONFIGS[dataset_format]
    raw = pd.read_csv(csv_path)
    log.info(
        "Loaded %s: %d rows, columns: %s",
        config["description"], len(raw), list(raw.columns),
    )

    # ── map columns ───────────────────────────────────────────────────────────
    df = pd.DataFrame()

    for src_col, dst_col in config["column_map"].items():
        if src_col in raw.columns:
            df[dst_col] = raw[src_col].astype(float)
        else:
            log.warning(
                "Column '%s' not found in data. Using synthetic values for '%s'.",
                src_col, dst_col,
            )
            mean, std = NORMAL_PROFILES[dst_col]
            df[dst_col] = np.random.normal(mean, std, len(raw))

    # Fill missing features with synthetic normal data
    for feat in FEATURES:
        if feat not in df.columns:
            mean, std = NORMAL_PROFILES[feat]
            df[feat] = np.random.normal(mean, std, len(raw))
            log.info("Synthesised '%s' column (not present in source data).", feat)

    # ── map labels ────────────────────────────────────────────────────────────
    label_col = config["label_column"]
    if label_col in raw.columns:
        df["label"] = raw[label_col].map(config["label_map"]).fillna(0).astype(int)
    else:
        log.warning("Label column '%s' not found. Defaulting all to normal.", label_col)
        df["label"] = 0

    # ── map fault types ───────────────────────────────────────────────────────
    ft_col = config.get("fault_type_column")
    ft_map = config.get("fault_type_map")
    if ft_col and ft_col in raw.columns and ft_map:
        df["fault_type"] = raw[ft_col].map(ft_map).fillna("none")
    else:
        df["fault_type"] = np.where(df["label"] == 1, "cable_cut", "none")

    # ── add timestamp ─────────────────────────────────────────────────────────
    t = np.arange(len(df)) / SAMPLE_RATE
    df["timestamp"] = pd.to_datetime(t, unit="s", origin="2025-01-01")

    # ── pad missing modalities and set domain id ──────────────────────────────
    df["acoustic_strain"] = 0.0
    df["optical_osnr"] = 0.0
    df["optical_ber"] = 0.0
    df["optical_power"] = 0.0
    df["cable_domain_id"] = 0  # 0 = Electrical Domain

    # Reorder columns to match simulator output
    df = df[["timestamp"] + FEATURES + ["cable_domain_id", "label", "fault_type"]]

    # ── build fault log ───────────────────────────────────────────────────────
    fault_log = _extract_fault_log(df)
    log.info(
        "Adapted dataset: %d samples, %d faults, %.1f%% fault rate",
        len(df), len(fault_log), df["label"].mean() * 100,
    )
    return df, fault_log


def _extract_fault_log(df: pd.DataFrame) -> list[dict]:
    """Extract contiguous fault regions from the label column."""
    fault_log = []
    in_fault = False
    start = 0
    current_type = "none"

    for i, row in df.iterrows():
        if row["label"] == 1 and not in_fault:
            in_fault = True
            start = i
            current_type = row["fault_type"]
        elif row["label"] == 0 and in_fault:
            in_fault = False
            fault_log.append({
                "fault_type":       current_type,
                "start_sample":     start,
                "duration_samples": i - start,
                "fault_distance_m": round(np.random.uniform(0, 500), 1),
            })

    # Handle fault at end of data
    if in_fault:
        fault_log.append({
            "fault_type":       current_type,
            "start_sample":     start,
            "duration_samples": len(df) - start,
            "fault_distance_m": round(np.random.uniform(0, 500), 1),
        })

    return fault_log


# ── generate realistic dataset ────────────────────────────────────────────────
def generate_realistic_dataset(
    n_samples: int = 10000,
    fault_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Generate a dataset based on published statistical distributions
    from real submarine cable monitoring literature.

    This uses realistic value ranges from:
    - ITU-T G.977.1 (submarine cable system parameters)
    - Real power cable monitoring standards (IEC 60287)
    - Published DAS monitoring research
    """
    rng = np.random.RandomState(seed)
    log.info(
        "Generating realistic dataset: %d samples, %.0f%% fault ratio",
        n_samples, fault_ratio * 100,
    )

    t = np.arange(n_samples) / SAMPLE_RATE

    # ── realistic base signals with environmental noise ───────────────────────
    # Voltage: submarine repeaters typically operate at 5-15 kV DC
    # Scaled to monitoring sensor range ~220V nominal
    voltage = rng.normal(220.0, 1.5, n_samples)
    voltage += 0.8 * np.sin(2 * np.pi * t / 120)          # slow drift
    voltage += 0.3 * np.sin(2 * np.pi * t / 3600)         # diurnal cycle
    voltage += rng.normal(0, 0.15, n_samples)               # sensor noise

    # Current: line current monitoring
    current = rng.normal(5.0, 0.25, n_samples)
    current += 0.12 * np.sin(2 * np.pi * t / 200)
    current += 0.05 * np.sin(2 * np.pi * t / 86400)       # daily load pattern
    current += rng.normal(0, 0.03, n_samples)

    # Temperature: deep-sea is ~2-4°C, but equipment runs warmer
    temperature = rng.normal(18.0, 0.6, n_samples)
    temperature += 0.5 * np.sin(2 * np.pi * t / 300)      # thermal cycling
    temperature += 0.8 * np.cos(2 * np.pi * t / 43200)    # tidal influence
    temperature += rng.normal(0, 0.08, n_samples)

    # Vibration: seabed vibration from ocean currents, marine life
    vibration = np.abs(rng.normal(0.0, 0.06, n_samples))
    vibration += 0.02 * np.sin(2 * np.pi * t / 45)        # wave period
    vibration += 0.015 * np.sin(2 * np.pi * t / 150)      # swell
    # Occasional micro-seismic events (realistic)
    n_micro = int(n_samples * 0.002)
    micro_idx = rng.choice(n_samples, n_micro, replace=False)
    vibration[micro_idx] += rng.uniform(0.1, 0.3, n_micro)

    df = pd.DataFrame({
        "timestamp":       pd.to_datetime(t, unit="s", origin="2025-01-01"),
        "voltage":         voltage,
        "current":         current,
        "temperature":     temperature,
        "vibration":       vibration,
        "acoustic_strain": 0.0,
        "optical_osnr":    0.0,
        "optical_ber":     0.0,
        "optical_power":   0.0,
        "cable_domain_id": 0,
        "label":           0,
        "fault_type":      "none",
    })

    # ── inject realistic faults ───────────────────────────────────────────────
    n_fault_samples = int(n_samples * fault_ratio)
    fault_types_available = [
        ("cable_cut",          0.15),  # rare but severe
        ("anchor_drag",        0.35),  # most common cause (ICPC data)
        ("overheating",        0.25),  # component degradation
        ("insulation_failure", 0.25),  # age-related
    ]

    fault_log = []
    remaining = n_fault_samples
    used_ranges = []

    for ftype, proportion in fault_types_available:
        target_samples = int(n_fault_samples * proportion)
        if target_samples < 50:
            continue

        # Split into multiple fault events
        n_events = max(1, target_samples // rng.randint(200, 600))
        for _ in range(n_events):
            if remaining <= 0:
                break

            duration = min(remaining, rng.randint(100, 500))
            # Find non-overlapping position
            for _attempt in range(50):
                start = rng.randint(int(n_samples * 0.05), int(n_samples * 0.90))
                end = start + duration
                if end >= n_samples:
                    continue
                if any(s - 50 <= end and e + 50 >= start for s, e in used_ranges):
                    continue
                break
            else:
                continue

            used_ranges.append((start, end))

            # Apply fault signatures
            if ftype == "cable_cut":
                df.iloc[start:end, df.columns.get_loc("voltage")] = rng.uniform(2, 15, duration)
                df.iloc[start:end, df.columns.get_loc("current")] = rng.uniform(0, 0.2, duration)
                df.iloc[start:end, df.columns.get_loc("vibration")] += rng.normal(0.7, 0.25, duration)

            elif ftype == "anchor_drag":
                drag_profile = np.sin(np.linspace(0, np.pi, duration))
                df.iloc[start:end, df.columns.get_loc("vibration")] += drag_profile * rng.uniform(1.5, 4.0)
                df.iloc[start:end, df.columns.get_loc("voltage")] -= drag_profile * rng.uniform(8, 20)

            elif ftype == "overheating":
                ramp = np.linspace(0, 1, duration)
                df.iloc[start:end, df.columns.get_loc("temperature")] += ramp * rng.uniform(15, 40)
                df.iloc[start:end, df.columns.get_loc("current")] += ramp * rng.uniform(1.5, 4)

            elif ftype == "insulation_failure":
                ramp = np.linspace(0, 1, duration)
                df.iloc[start:end, df.columns.get_loc("voltage")] -= ramp * rng.uniform(25, 70)
                df.iloc[start:end, df.columns.get_loc("temperature")] += ramp * rng.uniform(5, 25)
                df.iloc[start:end, df.columns.get_loc("current")] += ramp * rng.uniform(0.5, 2.5)

            df.iloc[start:end, df.columns.get_loc("label")] = 1
            df.iloc[start:end, df.columns.get_loc("fault_type")] = ftype
            remaining -= duration

            fault_log.append({
                "fault_type":       ftype,
                "start_sample":     start,
                "duration_samples": duration,
                "fault_distance_m": round(rng.uniform(0, 500), 1),
            })

    log.info(
        "Generated: %d samples, %d fault events, %.1f%% fault rate",
        len(df), len(fault_log), df["label"].mean() * 100,
    )
    return df, fault_log


# ── save adapted dataset ─────────────────────────────────────────────────────
def save_dataset(
    df: pd.DataFrame,
    fault_log: list[dict],
    output_dir: str = "datasets",
    name: str = "real_data",
):
    """Save dataset to CSV for use with the dashboard."""
    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, f"{name}.csv")
    df.to_csv(data_path, index=False)
    log.info("Saved dataset to %s (%d rows)", data_path, len(df))

    # Save fault log
    if fault_log:
        fl_path = os.path.join(output_dir, f"{name}_fault_log.csv")
        pd.DataFrame(fault_log).to_csv(fl_path, index=False)
        log.info("Saved fault log to %s (%d faults)", fl_path, len(fault_log))

    return data_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fetch and adapt real-world datasets for cable fault detection."
    )
    parser.add_argument(
        "--source",
        choices=["kaggle", "generate", "file"],
        default="generate",
        help="Data source: 'kaggle' downloads from Kaggle, "
             "'generate' creates realistic synthetic data, "
             "'file' adapts a local CSV.",
    )
    parser.add_argument(
        "--format",
        choices=list(DATASET_CONFIGS.keys()),
        default="smart_grid",
        help="Dataset format for Kaggle/file sources.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to local CSV file (with --source file).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of samples for generated dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets",
        help="Output directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="real_data",
        help="Output file name prefix.",
    )

    args = parser.parse_args()

    if args.source == "kaggle":
        config = DATASET_CONFIGS[args.format]
        csv_path = download_from_kaggle(config["kaggle_slug"], args.output)
        df, fault_log = adapt_dataset(csv_path, args.format)

    elif args.source == "file":
        if not args.file:
            parser.error("--file is required when --source is 'file'.")
        df, fault_log = adapt_dataset(args.file, args.format)

    elif args.source == "generate":
        df, fault_log = generate_realistic_dataset(
            n_samples=args.samples, seed=42,
        )
        args.name = "realistic_data"

    save_dataset(df, fault_log, args.output, args.name)

    # Print summary
    print("\n" + "=" * 60)
    print("  DATASET READY")
    print("=" * 60)
    print(f"  Samples  : {len(df):,}")
    print(f"  Features : {FEATURES}")
    print(f"  Faults   : {len(fault_log)} events ({df['label'].mean()*100:.1f}%)")
    print(f"  Types    : {df[df['label']==1]['fault_type'].value_counts().to_dict()}")
    print(f"  Saved to : {args.output}/")
    print("=" * 60)
    print("\n  To use in the dashboard:")
    print("    1. Start the API:       make run-api")
    print("    2. Start the frontend:  cd frontend && npm run dev")
    print("    2. Go to 'Upload CSV' tab")
    print(f"    3. Upload '{args.name}.csv'")
    print()


if __name__ == "__main__":
    main()
