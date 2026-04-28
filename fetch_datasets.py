"""
Dataset fetcher for undersea cable fault detection training.

Downloads publicly available datasets for predictive maintenance and cable fault detection.
"""

import os
import csv
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import (
    NORMAL_PROFILES, FAULT_TYPES, CABLE_LENGTH, SIGNAL_SPEED, SAMPLE_RATE
)

DATASETS_DIR = "datasets"


def generate_synthetic_cable_dataset(
    n_samples: int = 10000,
    fault_probability: float = 0.1,
    seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic cable telemetry with realistic faults."""
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    current_fault = None
    fault_start = None
    
    for i in range(n_samples):
        t = datetime(2025, 1, 1) + timedelta(seconds=i / SAMPLE_RATE)
        
        is_fault = random.random() < fault_probability
        
        if is_fault and current_fault is None:
            current_fault = random.choice(FAULT_TYPES)
            fault_start = i
            fault_duration = random.randint(50, 500)
        
        if current_fault is not None and i > fault_start + fault_duration:
            current_fault = None
        
        if current_fault is None:
            voltage = np.random.normal(NORMAL_PROFILES["voltage"][0], NORMAL_PROFILES["voltage"][1])
            current = np.random.normal(NORMAL_PROFILES["current"][0], NORMAL_PROFILES["current"][1])
            temperature = np.random.normal(NORMAL_PROFILES["temperature"][0], NORMAL_PROFILES["temperature"][1])
            vibration = max(0, np.random.normal(NORMAL_PROFILES["vibration"][0], NORMAL_PROFILES["vibration"][1]))
            label = 0
            fault_type = "none"
        else:
            label = 1
            fault_type = current_fault
            
            if current_fault == "cable_cut":
                voltage = np.random.uniform(0, 50)
                current = np.random.uniform(0, 0.5)
                temperature = np.random.normal(18, 0.5)
                vibration = np.random.uniform(0, 0.3)
            elif current_fault == "anchor_drag":
                voltage = np.random.normal(220, 5)
                current = np.random.normal(5, 0.5)
                temperature = np.random.normal(18, 0.5)
                vibration = np.random.uniform(0.5, 2.0)
            elif current_fault == "overheating":
                voltage = np.random.normal(220, 5)
                current = np.random.normal(5, 0.5)
                temperature = np.random.uniform(40, 80)
                vibration = np.random.normal(0, 0.05)
            elif current_fault == "insulation_failure":
                voltage = np.random.uniform(50, 150)
                current = np.random.uniform(2, 8)
                temperature = np.random.normal(18, 1)
                vibration = np.random.normal(0, 0.05)
        
        data.append({
            "timestamp": t.isoformat(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "temperature": round(temperature, 2),
            "vibration": round(vibration, 4),
            "label": label,
            "fault_type": fault_type
        })
    
    return pd.DataFrame(data)


def generate_industrial_pump_dataset(n_samples: int = 15000, seed: int = 43) -> pd.DataFrame:
    """Generate industrial pump/motor dataset for predictive maintenance."""
    np.random.seed(seed)
    
    timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="1min")
    
    base_voltage = 400
    base_current = 10
    base_temp = 65
    base_vibration = 0.5
    
    data = []
    degradation_state = 0
    
    for i, t in enumerate(timestamps):
        trend = min(i / n_samples, 1.0)
        
        if random.random() < 0.02:
            degradation_state = random.choice([0, 1, 2, 3])
        
        if degradation_state == 0:
            label, fault_type = 0, "none"
        elif degradation_state == 1:
            label, fault_type = 1, "bearing_wear"
        elif degradation_state == 2:
            label, fault_type = 1, "winding_short"
        else:
            label, fault_type = 1, " Insulation_failure"
        
        voltage = base_voltage + np.random.normal(0, 5)
        current = base_current + trend * 2 + np.random.normal(0, 0.5)
        
        if fault_type == "bearing_wear":
            current += np.random.uniform(1, 3)
            temperature = base_temp + trend * 15 + np.random.normal(0, 3)
            vibration = base_vibration + np.random.uniform(0.5, 2.0)
        elif fault_type == "winding_short":
            voltage -= np.random.uniform(10, 30)
            current += np.random.uniform(2, 5)
            temperature = base_temp + np.random.uniform(10, 25)
            vibration = base_vibration + np.random.uniform(0.2, 0.8)
        else:
            temperature = base_temp + np.random.normal(0, 2)
            vibration = base_vibration + np.random.normal(0, 0.1)
        
        temperature = max(20, min(120, temperature))
        vibration = max(0, vibration)
        
        data.append({
            "timestamp": t.isoformat(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "temperature": round(temperature, 2),
            "vibration": round(vibration, 4),
            "label": label,
            "fault_type": fault_type
        })
    
    return pd.DataFrame(data)


def generate_grid_stability_dataset(n_samples: int = 20000, seed: int = 44) -> pd.DataFrame:
    """Generate power grid stability data with grid disturbances."""
    np.random.seed(seed)
    
    timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="100ms")
    
    data = []
    
    for i, t in enumerate(timestamps):
        is_disturbance = random.random() < 0.05
        
        if is_disturbance:
            disturbance_type = random.choice(["voltage_sag", "voltage_swell", "frequency_deviation", "harmonic_distortion"])
            label = 1
            fault_type = disturbance_type
        else:
            disturbance_type = None
            label = 0
            fault_type = "none"
        
        if disturbance_type == "voltage_sag":
            voltage = np.random.uniform(180, 210)
            current = np.random.normal(50, 5)
            temperature = np.random.normal(25, 2)
            vibration = np.random.normal(0.1, 0.02)
        elif disturbance_type == "voltage_swell":
            voltage = np.random.uniform(250, 280)
            current = np.random.normal(50, 5)
            temperature = np.random.normal(25, 2)
            vibration = np.random.normal(0.1, 0.02)
        elif disturbance_type == "frequency_deviation":
            voltage = np.random.normal(230, 3)
            current = np.random.normal(50, 5)
            temperature = np.random.normal(25, 2)
            vibration = np.random.uniform(0.2, 0.5)
        elif disturbance_type == "harmonic_distortion":
            voltage = np.random.normal(230, 5)
            current = np.random.normal(50, 5)
            temperature = np.random.normal(30, 5)
            vibration = np.random.normal(0.1, 0.02)
        else:
            voltage = np.random.normal(230, 3)
            current = np.random.normal(50, 2)
            temperature = np.random.normal(25, 2)
            vibration = np.random.normal(0.1, 0.02)
        
        data.append({
            "timestamp": t.isoformat(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "temperature": round(temperature, 2),
            "vibration": round(vibration, 4),
            "label": label,
            "fault_type": fault_type
        })
    
    return pd.DataFrame(data)


def generate_high_freq_sensing(n_samples: int = 30000, seed: int = 45) -> pd.DataFrame:
    """Generate high-frequency sensing data for OTDR simulation."""
    np.random.seed(seed)
    
    timestamps = pd.date_range("2025-01-01", periods=n_samples, freq="10ms")
    
    data = []
    position = 0
    
    for i, t in enumerate(timestamps):
        event_type = random.choice(["normal", "bending_loss", "connector_fault", "splice_loss"])
        
        if event_type == "normal":
            voltage = np.random.normal(220, 1)
            current = np.random.normal(5, 0.2)
            temp = np.random.normal(20, 0.5)
            vib = np.random.normal(0, 0.01)
            label = 0
            fault_type = "none"
        elif event_type == "bending_loss":
            voltage = np.random.normal(215, 2)
            current = np.random.normal(5, 0.3)
            temp = np.random.normal(20, 0.5)
            vib = np.random.normal(0, 0.02)
            label = 1
            fault_type = "bending_loss"
        elif event_type == "connector_fault":
            voltage = np.random.uniform(180, 200)
            current = np.random.normal(5, 0.5)
            temp = np.random.normal(22, 1)
            vib = np.random.normal(0, 0.01)
            label = 1
            fault_type = "connector_fault"
        else:
            voltage = np.random.normal(210, 3)
            current = np.random.normal(5, 0.4)
            temp = np.random.normal(25, 2)
            vib = np.random.normal(0, 0.03)
            label = 1
            fault_type = "splice_loss"
        
        data.append({
            "timestamp": t.isoformat(),
            "voltage": round(voltage, 2),
            "current": round(current, 3),
            "temperature": round(temp, 2),
            "vibration": round(vib, 4),
            "label": label,
            "fault_type": fault_type
        })
    
    return pd.DataFrame(data)


def main():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    
    datasets = [
        ("synthetic_cable_10k.csv", lambda: generate_synthetic_cable_dataset(10000, 0.1, 42)),
        ("synthetic_cable_50k.csv", lambda: generate_synthetic_cable_dataset(50000, 0.08, 43)),
        ("industrial_pump.csv", lambda: generate_industrial_pump_dataset(15000, 43)),
        ("grid_stability.csv", lambda: generate_grid_stability_dataset(20000, 44)),
        ("high_freq_sensing.csv", lambda: generate_high_freq_sensing(30000, 45)),
    ]
    
    print("Generating datasets...")
    
    for filename, generator in datasets:
        filepath = os.path.join(DATASETS_DIR, filename)
        
        if os.path.exists(filepath):
            print(f"  Skipping {filename} (exists)")
            continue
        
        print(f"  Generating {filename}...")
        df = generator()
        
        fault_log = df[df["label"] == 1][["timestamp", "fault_type"]].copy()
        fault_log["severity"] = "Medium"
        
        df.to_csv(filepath, index=False)
        
        log_path = filepath.replace(".csv", "_fault_log.csv")
        fault_log.to_csv(log_path, index=False)
        
        print(f"    Saved {len(df)} rows, {len(fault_log)} faults")
    
    print("\nDone! Datasets in /datasets:")
    for f in os.listdir(DATASETS_DIR):
        fpath = os.path.join(DATASETS_DIR, f)
        if os.path.isfile(fpath) and f.endswith(".csv"):
            size = os.path.getsize(fpath) / 1024
            print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    main()