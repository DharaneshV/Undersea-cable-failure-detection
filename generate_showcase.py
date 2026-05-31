import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATASETS_DIR = "datasets"

def create_electrical_showcase():
    """Generate a highly structured 1,200-sample electrical showcase dataset."""
    np.random.seed(42)
    n_samples = 1200
    
    # Base normal profiles
    timestamps = [datetime(2026, 6, 1, 12, 0, 0) + timedelta(milliseconds=i * 100) for i in range(n_samples)]
    
    voltages = []
    currents = []
    temperatures = []
    vibrations = []
    labels = []
    fault_types = []
    cable_distance_norms = []
    
    for i in range(n_samples):
        # 1. Normal Phase (0 - 400)
        if i < 400:
            voltage = np.random.normal(220.0, 0.8)
            current = np.random.normal(5.0, 0.1)
            temp = np.random.normal(18.0, 0.3)
            vib = max(0.01, np.random.normal(0.05, 0.01))
            label = 0
            ftype = "none"
            dist_norm = 0.0
            
        # 2. Degradation/Overheating Phase (400 - 700)
        elif 400 <= i < 700:
            # Temperature rises gradually from 18 to 45
            progress = (i - 400) / 300.0
            voltage = np.random.normal(220.0 - (progress * 5.0), 1.0)
            current = np.random.normal(5.0 + (progress * 1.5), 0.2)
            temp = 18.0 + (progress * 27.0) + np.random.normal(0, 0.5)
            vib = max(0.01, np.random.normal(0.05 + (progress * 0.05), 0.01))
            label = 1
            ftype = "overheating"
            dist_norm = 0.35  # Constant estimated fault distance norm
            
        # 3. Insulation Failure / Shunt Short (700 - 1000)
        elif 700 <= i < 1000:
            # High current draw, voltage drops significantly
            voltage = np.random.normal(140.0, 2.0)
            current = np.random.normal(11.5, 0.3)
            temp = np.random.normal(55.0, 0.8)
            vib = max(0.01, np.random.normal(0.12, 0.02))
            label = 1
            ftype = "insulation_failure"
            dist_norm = 0.35
            
        # 4. Catastrophic Cable Cut / Short (1000 - 1200)
        else:
            voltage = np.random.uniform(5.0, 15.0)
            current = np.random.uniform(0.05, 0.15)
            temp = np.random.normal(22.0, 0.4)
            vib = np.random.uniform(0.01, 0.04)
            label = 1
            ftype = "cable_cut"
            dist_norm = 0.35

        voltages.append(round(voltage, 2))
        currents.append(round(current, 3))
        temperatures.append(round(temp, 2))
        vibrations.append(round(vib, 4))
        labels.append(label)
        fault_types.append(ftype)
        cable_distance_norms.append(round(dist_norm, 4))
        
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "voltage": voltages,
        "current": currents,
        "temperature": temperatures,
        "vibration": vibrations,
        "acoustic_strain": [0.0] * n_samples,
        "optical_osnr": [25.0] * n_samples,
        "optical_ber": [-12.0] * n_samples,
        "optical_power": [1.5] * n_samples,
        "cable_distance_norm": cable_distance_norms,
        "cable_domain_id": [0] * n_samples,  # Electrical Domain
        "label": labels,
        "fault_type": fault_types
    })
    
    # Save datasets
    csv_path = os.path.join(DATASETS_DIR, "showcase_electrical_3min.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate companion fault log
    fault_log_data = [
        {"fault_type": "overheating", "start_sample": 400, "duration_samples": 300, "fault_distance_m": 1330000.0, "severity": "Warning"},
        {"fault_type": "insulation_failure", "start_sample": 700, "duration_samples": 300, "fault_distance_m": 1330000.0, "severity": "High"},
        {"fault_type": "cable_cut", "start_sample": 1000, "duration_samples": 200, "fault_distance_m": 1330000.0, "severity": "Critical"}
    ]
    log_path = os.path.join(DATASETS_DIR, "showcase_electrical_3min_fault_log.csv")
    pd.DataFrame(fault_log_data).to_csv(log_path, index=False)
    print(f"Generated: {csv_path} and its fault log.")


def create_optical_showcase():
    """Generate a highly structured 1,200-sample optical showcase dataset."""
    np.random.seed(84)
    n_samples = 1200
    
    timestamps = [datetime(2026, 6, 1, 12, 0, 0) + timedelta(milliseconds=i * 100) for i in range(n_samples)]
    
    vibrations = []
    acoustic_strains = []
    osnrs = []
    bers = []
    powers = []
    labels = []
    fault_types = []
    cable_distance_norms = []
    
    for i in range(n_samples):
        # 1. Normal Phase (0 - 400)
        if i < 400:
            vib = max(0.01, np.random.normal(0.02, 0.005))
            strain = np.random.normal(0.0, 0.05)
            osnr = np.random.normal(25.0, 0.2)
            ber = -12.0 + np.random.normal(0, 0.1)
            power = np.random.normal(1.5, 0.05)
            label = 0
            ftype = "none"
            dist_norm = 0.0
            
        # 2. Warning Phase: Bending Loss (400 - 700)
        elif 400 <= i < 700:
            progress = (i - 400) / 300.0
            vib = max(0.01, np.random.normal(0.02 + (progress * 0.05), 0.005))
            strain = np.random.normal(0.0 + (progress * 0.5), 0.1)
            osnr = 25.0 - (progress * 7.0) + np.random.normal(0, 0.3)
            ber = -12.0 + (progress * 4.0) + np.random.normal(0, 0.2)
            power = 1.5 - (progress * 2.5) + np.random.normal(0, 0.05)
            label = 1
            ftype = "insulation_failure"  # Map to soft degradation/insulation fault
            dist_norm = 0.72  # 72% down the cable
            
        # 3. Anchor Drag / Intense Mechanical Stress (700 - 1000)
        elif 700 <= i < 1000:
            vib = np.random.normal(1.8, 0.2)  # Extreme vibration
            strain = np.random.normal(8.0, 0.5)
            osnr = np.random.normal(12.0, 0.5)
            ber = np.random.normal(-6.0, 0.3)
            power = np.random.normal(-2.0, 0.1)
            label = 1
            ftype = "anchor_drag"
            dist_norm = 0.72
            
        # 4. Physical Fiber Cut (1000 - 1200)
        else:
            vib = np.random.uniform(0.01, 0.03)
            strain = np.random.normal(0.0, 0.05)
            osnr = np.random.uniform(3.0, 5.0)
            ber = np.random.uniform(-3.0, -2.0)
            power = np.random.uniform(-9.0, -12.0)
            label = 1
            ftype = "cable_cut"
            dist_norm = 0.72

        vibrations.append(round(vib, 4))
        acoustic_strains.append(round(strain, 4))
        osnrs.append(round(osnr, 2))
        bers.append(round(ber, 2))
        powers.append(round(power, 3))
        labels.append(label)
        fault_types.append(ftype)
        cable_distance_norms.append(round(dist_norm, 4))
        
    df = pd.DataFrame({
        "timestamp": [t.isoformat() for t in timestamps],
        "voltage": [0.0] * n_samples,
        "current": [0.0] * n_samples,
        "temperature": [12.0] * n_samples,  # Deep sea constant temperature
        "vibration": vibrations,
        "acoustic_strain": acoustic_strains,
        "optical_osnr": osnrs,
        "optical_ber": bers,
        "optical_power": powers,
        "cable_distance_norm": cable_distance_norms,
        "cable_domain_id": [1] * n_samples,  # Optical Domain
        "label": labels,
        "fault_type": fault_types
    })
    
    # Save datasets
    csv_path = os.path.join(DATASETS_DIR, "showcase_optical_3min.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate companion fault log
    fault_log_data = [
        {"fault_type": "insulation_failure", "start_sample": 400, "duration_samples": 300, "fault_distance_m": 2736000.0, "severity": "Warning"},
        {"fault_type": "anchor_drag", "start_sample": 700, "duration_samples": 300, "fault_distance_m": 2736000.0, "severity": "High"},
        {"fault_type": "cable_cut", "start_sample": 1000, "duration_samples": 200, "fault_distance_m": 2736000.0, "severity": "Critical"}
    ]
    log_path = os.path.join(DATASETS_DIR, "showcase_optical_3min_fault_log.csv")
    pd.DataFrame(fault_log_data).to_csv(log_path, index=False)
    print(f"Generated: {csv_path} and its fault log.")


if __name__ == "__main__":
    os.makedirs(DATASETS_DIR, exist_ok=True)
    create_electrical_showcase()
    create_optical_showcase()
    print("Showcase datasets successfully built inside /datasets directory.")
