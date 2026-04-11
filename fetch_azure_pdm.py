"""
fetch_azure_pdm.py
Download the Microsoft Azure Predictive Maintenance dataset (no Kaggle login
required), merge telemetry + failures, and output a fully-labelled CSV that
plugs directly into the undersea cable fault detection pipeline.

Source files (publicly hosted on Azure Blob):
  PdM_telemetry.csv  – hourly voltage, rotation, pressure, vibration (876,100 rows)
  PdM_failures.csv   – timestamps of component failures (component1–4)

Output (in datasets/):
  azure_pdm.csv          – merged, labelled, resampled to pipeline format
  azure_pdm_fault_log.csv – fault regions for the cable diagram / TDR display

Column mapping:
  voltage   ← volt    (directly usable)
  vibration ← vibration (directly usable)
  current   ← rotate  (scaled – rotational current proxy)
  temperature ← pressure (scaled – thermal pressure proxy)

Usage:
  python fetch_azure_pdm.py                  # download + process
  python fetch_azure_pdm.py --samples 50000  # limit output rows
  python fetch_azure_pdm.py --no-download --file datasets/PdM_telemetry.csv
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ── Dataset URLs (GitHub mirrors of the Azure PdM dataset) ───────────────────
_BASE = "https://raw.githubusercontent.com/ashishpatel26/Predictive_Maintenance_using_Machine-Learning_Microsoft_Casestudy/master/data"
TELEMETRY_URL = f"{_BASE}/PdM_telemetry.csv"
FAILURES_URL  = f"{_BASE}/PdM_failures.csv"

# Fault-type mapping: Azure component name → our fault taxonomy
FAULT_TYPE_MAP = {
    "comp1": "cable_cut",           # immediate catastrophic failure
    "comp2": "insulation_failure",  # insulation / electrical breakdown
    "comp3": "overheating",         # thermal component failure
    "comp4": "anchor_drag",         # mechanical / vibration failure
}

CABLE_LENGTH_M = 500  # for TDR distance simulation


# ── download helper ───────────────────────────────────────────────────────────
def _download(url: str, dest: str) -> str:
    """Download a URL to dest if not already cached. Returns local path."""
    if os.path.exists(dest):
        log.info("Using cached file: %s", dest)
        return dest

    log.info("Downloading %s ...", url)
    try:
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        log.info("Saved to %s", dest)
    except Exception as e:
        log.error("Download failed: %s", e)
        sys.exit(1)
    return dest


# ── load & merge ──────────────────────────────────────────────────────────────
def load_telemetry(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp"})
    df = df.sort_values(["machineID", "timestamp"]).reset_index(drop=True)
    log.info("Telemetry: %d rows, machines: %s", len(df), df["machineID"].unique()[:5])
    return df


def load_failures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.rename(columns={"datetime": "timestamp", "failure": "component"})
    df["fault_type"] = df["component"].map(FAULT_TYPE_MAP).fillna("cable_cut")
    log.info("Failures: %d records", len(df))
    return df


# ── label windows ─────────────────────────────────────────────────────────────
def label_fault_windows(
    tel: pd.DataFrame,
    fail: pd.DataFrame,
    pre_fault_hours: int = 12,
) -> pd.DataFrame:
    """
    For each failure event, mark the N hours before the failure timestamp
    as fault=1. This simulates the degrading signal window the model should
    learn to detect.
    """
    tel = tel.copy()
    tel["label"]      = 0
    tel["fault_type"] = "none"

    for _, row in fail.iterrows():
        mid    = row["machineID"]
        ts     = row["timestamp"]
        ftype  = row["fault_type"]
        window_start = ts - pd.Timedelta(hours=pre_fault_hours)

        mask = (
            (tel["machineID"] == mid) &
            (tel["timestamp"] >= window_start) &
            (tel["timestamp"] <= ts)
        )
        tel.loc[mask, "label"]      = 1
        tel.loc[mask, "fault_type"] = ftype

    log.info(
        "Labelled: %.1f%% fault rows (%d failure events)",
        tel["label"].mean() * 100, len(fail),
    )
    return tel


# ── feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Azure PdM sensor columns onto our 4-channel pipeline schema.

    Azure columns:
      volt     → voltage      (V)   ←  direct, already in similar range
      rotate   → current      (A)   ←  scaled: rotational speed ~ electrical load
      pressure → temperature  (°C)  ←  scaled: pressure correlates with thermal
      vibration→ vibration    (g)   ←  direct, rescaled to our range

    Scaling applied to keep values within NORMAL_PROFILES ranges so the
    pre-trained scaler doesn't extrapolate excessively:
      voltage:     volt clamped/scaled to [190, 260] V
      current:     rotate normalised to [2, 9] A
      temperature: pressure normalised to [15, 60] °C
      vibration:   vibration normalised to [0, 3.5] g
    """
    out = pd.DataFrame()

    # voltage: volt is already ~150–250 range in the dataset
    v = df["volt"].clip(lower=100, upper=350)
    out["voltage"] = 210 + (v - v.mean()) / v.std() * 12  # μ=210, σ≈12

    # current: rotate ~100–600 rpm → scale to 2–9 A
    r = df["rotate"].clip(lower=0, upper=800)
    out["current"] = 2 + (r - r.min()) / (r.max() - r.min()) * 7

    # temperature: pressure ~80–110 → scale to 15–60 °C
    p = df["pressure"].clip(lower=0, upper=200)
    out["temperature"] = 15 + (p - p.min()) / (p.max() - p.min()) * 45

    # vibration: already 0–80 range → scale to 0–3.5 g
    vib = df["vibration"].clip(lower=0, upper=200)
    out["vibration"] = vib / vib.max() * 3.5

    out["timestamp"]  = df["timestamp"].values
    out["machineID"]  = df["machineID"].values
    out["label"]      = df["label"].values
    out["fault_type"] = df["fault_type"].values

    return out


# ── fault log extraction ──────────────────────────────────────────────────────
def extract_fault_log(df: pd.DataFrame, seed: int = 42) -> list[dict]:
    """Extract contiguous fault windows into the fault_log format."""
    rng       = np.random.RandomState(seed)
    fault_log = []
    in_fault  = False
    start_idx = 0
    cur_type  = "none"

    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        if row["label"] == 1 and not in_fault:
            in_fault  = True
            start_idx = i
            cur_type  = row["fault_type"]
        elif row["label"] == 0 and in_fault:
            in_fault = False
            fault_log.append({
                "fault_type":       cur_type,
                "start_sample":     int(start_idx),
                "duration_samples": int(i - start_idx),
                "fault_distance_m": round(float(rng.uniform(0, CABLE_LENGTH_M)), 1),
            })

    if in_fault:  # trailing fault at end
        fault_log.append({
            "fault_type":       cur_type,
            "start_sample":     int(start_idx),
            "duration_samples": int(len(df) - start_idx),
            "fault_distance_m": round(float(rng.uniform(0, CABLE_LENGTH_M)), 1),
        })

    log.info("Extracted %d fault windows", len(fault_log))
    return fault_log


# ── save ──────────────────────────────────────────────────────────────────────
def save(
    df: pd.DataFrame,
    fault_log: list[dict],
    output_dir: str = "datasets",
    name: str = "azure_pdm",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # Final column order expected by the pipeline
    cols = ["timestamp", "voltage", "current", "temperature", "vibration",
            "label", "fault_type"]
    out = df[cols]

    data_path = os.path.join(output_dir, f"{name}.csv")
    out.to_csv(data_path, index=False)
    log.info("Saved dataset → %s  (%d rows)", data_path, len(out))

    if fault_log:
        fl_path = os.path.join(output_dir, f"{name}_fault_log.csv")
        pd.DataFrame(fault_log).to_csv(fl_path, index=False)
        log.info("Saved fault log → %s  (%d events)", fl_path, len(fault_log))

    return data_path


# ── main pipeline ─────────────────────────────────────────────────────────────
def run(
    tel_path: str | None = None,
    fail_path: str | None = None,
    samples: int | None = None,
    machine_id: int = 1,
    pre_fault_hours: int = 12,
    output_dir: str = "datasets",
    name: str = "azure_pdm",
    seed: int = 42,
):
    cache_dir = output_dir
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1: Download or use provided files
    tel_file  = tel_path  or _download(TELEMETRY_URL,  os.path.join(cache_dir, "_pdm_telemetry_raw.csv"))
    fail_file = fail_path or _download(FAILURES_URL,   os.path.join(cache_dir, "_pdm_failures_raw.csv"))

    # Step 2: Load
    tel  = load_telemetry(tel_file)
    fail = load_failures(fail_file)

    # Step 3: Optionally filter to a single machine for a clean time-series
    if machine_id is not None:
        log.info("Filtering to machineID=%d", machine_id)
        tel  = tel[tel["machineID"] == machine_id].reset_index(drop=True)
        fail = fail[fail["machineID"] == machine_id].reset_index(drop=True)

    # Step 4: Label fault windows
    tel = label_fault_windows(tel, fail, pre_fault_hours=pre_fault_hours)

    # Step 5: Build feature columns
    tel = build_features(tel)

    # Step 6: Optionally subsample
    if samples and len(tel) > samples:
        # Stratified sample — preserve fault ratio
        normal = tel[tel["label"] == 0].sample(
            int(samples * (1 - tel["label"].mean())), random_state=seed)
        fault  = tel[tel["label"] == 1].sample(
            int(samples * tel["label"].mean()), random_state=seed)
        tel = pd.concat([normal, fault]).sort_values("timestamp").reset_index(drop=True)
        log.info("Sampled to %d rows (fault rate %.1f%%)",
                 len(tel), tel["label"].mean() * 100)

    # Step 7: Extract fault log
    fault_log = extract_fault_log(tel, seed=seed)

    # Step 8: Save
    path = save(tel, fault_log, output_dir=output_dir, name=name)

    # Step 9: Summary report
    print("\n" + "=" * 62)
    print("  AZURE PdM DATASET READY")
    print("=" * 62)
    print(f"  Rows        : {len(tel):,}")
    print(f"  Fault rate  : {tel['label'].mean()*100:.1f}%  ({tel['label'].sum():,} fault rows)")
    print(f"  Fault events: {len(fault_log)}")
    print(f"  Fault types : {tel[tel['label']==1]['fault_type'].value_counts().to_dict()}")
    print(f"  Machine     : {machine_id}")
    print(f"  Saved to    : {path}")
    print("=" * 62)
    print()
    print("  ► To train the model on this dataset:")
    print(f"    python model.py --dataset datasets/{name}.csv")
    print()
    print("  ► To stream it in the API / React dashboard:")
    print(f"    Select '{name}.csv' from the dataset dropdown and press Start Stream")
    print()

    return path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download & prepare the Azure Predictive Maintenance dataset."
    )
    parser.add_argument(
        "--tel-file", type=str, default=None,
        help="Path to locally downloaded PdM_telemetry.csv (skips download)."
    )
    parser.add_argument(
        "--fail-file", type=str, default=None,
        help="Path to locally downloaded PdM_failures.csv (skips download)."
    )
    parser.add_argument(
        "--machine", type=int, default=1,
        help="Which machine ID to extract (1–100). Default: 1."
    )
    parser.add_argument(
        "--all-machines", action="store_true",
        help="Use all 100 machines (produces a very large dataset ~870K rows)."
    )
    parser.add_argument(
        "--pre-fault-hours", type=int, default=12,
        help="Hours before a failure to mark as fault=1. Default: 12."
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Limit total output rows (stratified). Default: no limit."
    )
    parser.add_argument(
        "--output", type=str, default="datasets",
        help="Output directory. Default: datasets/"
    )
    parser.add_argument(
        "--name", type=str, default="azure_pdm",
        help="Output filename prefix. Default: azure_pdm"
    )

    args = parser.parse_args()

    run(
        tel_path        = args.tel_file,
        fail_path       = args.fail_file,
        samples         = args.samples,
        machine_id      = None if args.all_machines else args.machine,
        pre_fault_hours = args.pre_fault_hours,
        output_dir      = args.output,
        name            = args.name,
    )


if __name__ == "__main__":
    main()
