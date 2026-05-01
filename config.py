"""
config.py
Single source of truth for all project hyper-parameters and constants.
Change values here — they propagate everywhere automatically.
"""

# ── simulator ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 10          # samples per second
CABLE_LENGTH  = 500         # metres (simulated cable)
SIGNAL_SPEED  = 2e8         # m/s (≈ 2/3 speed of light in fibre)

NORMAL_PROFILES = {
    "voltage":     (220.0, 1.2),   # (mean, std)
    "current":     (  5.0, 0.2),
    "temperature": ( 18.0, 0.5),
    "vibration":   (  0.0, 0.05),
}

FAULT_TYPES = ["cable_cut", "anchor_drag", "overheating", "insulation_failure"]

# Mapping from simulator physical events to electrical fault classes
CLASSIFICATION_MAP = {
    "none":               0,  # "Normal"
    "insulation_failure": 1,  # "Short Circuit"
    "cable_cut":          2,  # "Open Circuit"
    "overheating":        3,  # "High-Z / Degradation"
    "anchor_drag":        3,  # "High-Z / Degradation" (Mechanical proxy)
}

CLASS_NAMES = ["Normal", "Short Circuit", "Open Circuit", "High-Impedance"]
NUM_CLASSES = 4

# ── model ─────────────────────────────────────────────────────────────────────
# Unified Multi-Modal Feature Space
FEATURES = [
    # Electrical Domain
    "voltage", "current", "temperature",
    # Mechanical / Acoustic Domain
    "vibration", "acoustic_strain",
    # Optical Domain
    "optical_osnr", "optical_ber", "optical_power"
]
USE_DERIVED_FEATURES = False  # If True, feature_engineering.py will be invoked

SEQ_LEN         = 60        # 60 timesteps for transformer sequence
TRANSFORMER_HEADS  = 4
TRANSFORMER_FF_DIM = 128
TRANSFORMER_BLOCKS = 2
LSTM_UNITS      = 64        # encoder outer layer units
LATENT_UNITS    = 32        # bottleneck units
DROPOUT_RATE    = 0.20      # applied after each LSTM layer
EPOCHS          = 5
BATCH_SIZE      = 64
THRESHOLD_PCT   = 95        # percentile of validation reconstruction error
USE_OPTIMAL_THRESHOLD = False # set True only when calibrating with labeled eval data

# fraction of normal data held-out for threshold calibration (not trained on)
THRESHOLD_VAL_SPLIT = 0.05

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
