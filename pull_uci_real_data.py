import pandas as pd
import numpy as np
import os

# Create the output directory
os.makedirs("datasets", exist_ok=True)

# Load the downloaded UCI AI4I 2020 Predictive Maintenance Dataset
df = pd.read_csv("ai4i.csv")

# We will map the real-world manufacturing data to our pipeline's "Undersea Cable" schema.
# 
# Mapping logic:
# Air temperature [K]      -> Temperature (Celsius)
# Rotational speed [rpm]   -> Current (scaled proxy for load)
# Torque [Nm]              -> Vibration (scaled proxy for mechanical stress)
# Tool wear [min]          -> Voltage (inverse proxy: as wear increases, voltage drops)

out = pd.DataFrame()

# Convert Kelvin to Celsius and scale it so it fits our normal temperature profile (~15-30C)
# Mean air temp in dataset is ~300K (27C). We'll subtract 273.15.
out["temperature"] = df["Air temperature [K]"] - 273.15

# Rotational speed (avg 1538 rpm) -> Current (scale to 2-10A)
min_rpm = df["Rotational speed [rpm]"].min()
max_rpm = df["Rotational speed [rpm]"].max()
out["current"] = 2.0 + (df["Rotational speed [rpm]"] - min_rpm) / (max_rpm - min_rpm) * 8.0

# Torque (avg 40 Nm) -> Vibration (scale to 0-3g)
min_torque = df["Torque [Nm]"].min()
max_torque = df["Torque [Nm]"].max()
out["vibration"] = (df["Torque [Nm]"] - min_torque) / (max_torque - min_torque) * 3.0

# Tool wear (0-250 mins) -> Voltage (starts at 240V, drops to 200V as wear increases)
out["voltage"] = 240.0 - (df["Tool wear [min]"] / 250.0) * 40.0

# Add baseline missing modalities
out["acoustic_strain"] = 0.0
out["optical_osnr"] = 20.0
out["optical_ber"] = 0.0
out["optical_power"] = 0.0
out["cable_domain_id"] = 0  # Electrical domain
out["cable_distance_norm"] = 0.0

# Define labels
out["label"] = df["Machine failure"]

# Determine fault type based on the individual failure flags
def determine_fault(row):
    if row["TWF"] == 1: return "anchor_drag"        # Tool Wear -> mechanical degradation
    if row["HDF"] == 1: return "overheating"        # Heat Dissipation -> overheating
    if row["PWF"] == 1: return "insulation_failure" # Power Failure -> short circuit
    if row["OSF"] == 1: return "cable_cut"          # Overstrain -> complete break
    if row["RNF"] == 1: return "cable_cut"          # Random Failure
    if row["Machine failure"] == 1: return "insulation_failure" # Catch-all
    return "none"

out["fault_type"] = df.apply(determine_fault, axis=1)

# Sort out timestamps (assume 100ms intervals, typical for our streaming pipeline)
timestamps = pd.date_range("2025-01-01", periods=len(df), freq="100ms")
out.insert(0, "timestamp", timestamps)

# Save the dataset
csv_path = "datasets/real_uci_ai4i.csv"
out.to_csv(csv_path, index=False)
print(f"Saved real dataset to {csv_path} ({len(out)} rows)")

# Generate the fault log required for the TDR Localisation UI
fault_log = []
in_fault = False
start_idx = 0
cur_type = "none"

for i, row in out.iterrows():
    if row["label"] == 1 and not in_fault:
        in_fault = True
        start_idx = i
        cur_type = row["fault_type"]
    elif row["label"] == 0 and in_fault:
        in_fault = False
        fault_log.append({
            "fault_type": cur_type,
            "start_sample": start_idx,
            "duration_samples": i - start_idx,
            "fault_distance_m": round(np.random.uniform(5000, 200000), 1)
        })

if in_fault:
    fault_log.append({
        "fault_type": cur_type,
        "start_sample": start_idx,
        "duration_samples": len(out) - start_idx,
        "fault_distance_m": round(np.random.uniform(5000, 200000), 1)
    })

log_path = "datasets/real_uci_ai4i_fault_log.csv"
pd.DataFrame(fault_log).to_csv(log_path, index=False)
print(f"Saved fault log to {log_path} ({len(fault_log)} fault events)")
