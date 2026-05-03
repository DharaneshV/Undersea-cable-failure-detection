"""
config.py
Single source of truth for all project hyper-parameters and constants.
Change values here — they propagate everywhere automatically.
"""

# ── simulator ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 10                # samples per second
CABLE_LENGTH  = 3_800_000         # metres — 3,800 km (India–Singapore regional cable)
SIGNAL_SPEED  = 2e8              # m/s (≈ 2/3 speed of light in fibre)

NORMAL_PROFILES = {
    "voltage":     (220.0, 1.2),   # (mean, std)
    "current":     (  5.0, 0.2),
    "temperature": ( 18.0, 0.5),
    "vibration":   (  0.0, 0.05),
}

# ── cable domain metadata ─────────────────────────────────────────────────────
# Maps cable_domain_id integer → human-readable cable type label
CABLE_DOMAIN_NAMES = {
    0: "Electrical (Copper)",
    1: "Optical (Fibre)",
    2: "Hybrid (Electro-Optical)",
    3: "Acoustic (Piezo-Array)",
}

FAULT_TYPES = ["cable_cut", "anchor_drag", "overheating", "insulation_failure"]

# Mapping from all dataset fault type strings to model class indices
CLASSIFICATION_MAP = {
    # Core simulator types
    "none":                0,  # Normal
    "insulation_failure":  1,  # Short Circuit
    "Insulation_failure":  1,  # alias (capitalised, seen in azure_pdm)
    " Insulation_failure": 1,  # alias (leading space, seen in azure_pdm)
    "cable_cut":           2,  # Open Circuit
    "overheating":         3,  # High-Z / Degradation
    "anchor_drag":         3,  # High-Z / Degradation (mechanical proxy)
    # industrial_pump / azure_pdm extras
    "bearing_wear":        3,  # High-Z / Degradation (mechanical wear)
    "winding_short":       1,  # Short Circuit (winding insulation failure)
    "seal_leak":           3,  # High-Z / Degradation (fluid ingress)
    "corrosion":           3,  # High-Z / Degradation
    "mechanical_wear":     3,  # High-Z / Degradation
}

CLASS_NAMES = ["Normal", "Short Circuit", "Open Circuit", "High-Impedance"]
NUM_CLASSES = 4

# ── model ─────────────────────────────────────────────────────────────────────
# Unified Multi-Modal Feature Space (9 features)
FEATURES = [
    # Electrical Domain
    "voltage", "current", "temperature",
    # Mechanical / Acoustic Domain
    "vibration", "acoustic_strain",
    # Optical Domain
    "optical_osnr", "optical_ber", "optical_power",
    # Spatial Domain — normalised distance along cable [0.0–1.0]
    # 0.0 = no active fault located; >0 = fault position / CABLE_LENGTH
    "cable_distance_norm",
]
USE_DERIVED_FEATURES = False  # If True, feature_engineering.py will be invoked

# ── transformer / encoder hyper-parameters ────────────────────────────────────
SEQ_LEN            = 60    # timesteps per input window
TRANSFORMER_HEADS  = 8     # ↑ from 4 — richer attention heads
TRANSFORMER_FF_DIM = 256   # ↑ from 128 — wider feed-forward sub-layer
TRANSFORMER_BLOCKS = 3     # ↑ from 2 — deeper encoder
LSTM_UNITS         = 128   # ↑ from 64 — more conv filters
LATENT_UNITS       = 64    # ↑ from 32 — richer bottleneck
DROPOUT_RATE       = 0.15  # ↓ from 0.20 — less regularisation for deeper net
EPOCHS             = 30    # ↑ from 5 — EarlyStopping will halt when plateau
BATCH_SIZE         = 32    # ↓ from 64 — finer gradient estimates
THRESHOLD_PCT      = 97    # ↑ from 95 — tighter normal boundary
USE_OPTIMAL_THRESHOLD = True  # F1-sweep calibration ON for better threshold

# fraction of normal data held-out for threshold calibration (not trained on)
THRESHOLD_VAL_SPLIT = 0.10  # ↑ from 0.05 — larger calibration set

# ── class weights (fault classes are minority) ────────────────────────────────
# Passed to model.fit() to counter imbalanced normal vs fault samples.
CLASS_WEIGHTS = {
    0: 1.0,   # Normal  — majority class
    1: 4.0,   # Short Circuit (insulation_failure)
    2: 4.0,   # Open Circuit  (cable_cut)
    3: 3.0,   # High-Impedance (overheating / anchor_drag)
}

# ── dashboard colours ─────────────────────────────────────────────────────────
FAULT_COLORS = {
    "cable_cut":          "#E24B4A",
    "anchor_drag":        "#EF9F27",
    "overheating":        "#D85A30",
    "insulation_failure": "#7F77DD",
    "none":               "#1D9E75",
}

SENSOR_COLORS = {
    "voltage":     "#3B82F6",
    "current":     "#10B981",
    "temperature": "#F59E0B",
    "vibration":   "#EF5941",
}
