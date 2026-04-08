"""
model.py
LSTM Autoencoder for unsupervised anomaly detection on cable sensor data.

Improvements over v1
────────────────────
1. Fixed encoder: uses return_sequences=True, eliminating the Reshape hack.
2. Dropout regularisation after every LSTM layer (configurable in config.py).
3. Threshold is calibrated on a held-out validation split of NORMAL data
   (not the same data the model was trained on) — prevents leakage.
4. Optional F1-optimal threshold sweep via utils.find_optimal_threshold().
5. Scaler out-of-range warning at inference time.
6. Python logging instead of bare print() — control verbosity externally.
7. All hyper-parameters imported from config.py.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, RepeatVector, TimeDistributed,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from config import (
    BATCH_SIZE, DROPOUT_RATE, EPOCHS, FEATURES,
    LATENT_UNITS, LSTM_UNITS, SEQ_LEN,
    THRESHOLD_PCT, THRESHOLD_VAL_SPLIT,
)
from utils import check_scaler_bounds, find_optimal_threshold, make_sequences

log = logging.getLogger(__name__)


# ── model definition ─────────────────────────────────────────────────────────
def build_lstm_autoencoder(seq_len: int, n_features: int) -> Model:
    """
    Encoder  : LSTM(64, seq) → Dropout → LSTM(32, no-seq) → bottleneck
    Decoder  : RepeatVector → LSTM(32, seq) → Dropout → LSTM(64, seq) → Dense

    No Reshape hacks — return_sequences=True feeds the second encoder LSTM
    directly from the first.
    """
    inp = Input(shape=(seq_len, n_features), name="input")

    # ── Encoder ──────────────────────────────────────────────────────────────
    x = LSTM(LSTM_UNITS, activation="tanh",
             return_sequences=True, name="enc_lstm1")(inp)
    x = Dropout(DROPOUT_RATE, name="enc_drop1")(x)
    encoded = LSTM(LATENT_UNITS, activation="tanh",
                   return_sequences=False, name="enc_lstm2")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = RepeatVector(seq_len, name="repeat")(encoded)
    x = LSTM(LATENT_UNITS, activation="tanh",
             return_sequences=True, name="dec_lstm1")(x)
    x = Dropout(DROPOUT_RATE, name="dec_drop1")(x)
    x = LSTM(LSTM_UNITS, activation="tanh",
             return_sequences=True, name="dec_lstm2")(x)
    out = TimeDistributed(Dense(n_features), name="output")(x)

    model = Model(inp, out, name="lstm_autoencoder")
    model.compile(optimizer="adam", loss="mae")
    return model


# ── detector class ────────────────────────────────────────────────────────────
class CableFaultDetector:
    def __init__(self):
        self.scaler    = MinMaxScaler()
        self.model     = None
        self.threshold = None

    # ── internal helpers ──────────────────────────────────────────────────────
    def _scale(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        values = df[FEATURES].values.astype(np.float32)
        if fit:
            return self.scaler.fit_transform(values)
        check_scaler_bounds(values, self.scaler)
        return self.scaler.transform(values)

    # ── train ─────────────────────────────────────────────────────────────────
    def train(self, df: pd.DataFrame, use_optimal_threshold: bool = False):
        """
        Train exclusively on normal (label == 0) samples.

        The normal data is split:
          • (1 - THRESHOLD_VAL_SPLIT) fraction → model training
          • THRESHOLD_VAL_SPLIT fraction        → threshold calibration
        This prevents the threshold from being set on data the model has
        already memorised, giving a more honest anomaly boundary.

        Parameters
        ----------
        df : DataFrame with a 'label' column (0 = normal, 1 = fault).
        use_optimal_threshold : if True, sweep thresholds and pick the one
            maximising F1 on the calibration split (requires labels there).
            Defaults to False (use p95 of calibration errors).
        """
        normal_df = df[df["label"] == 0].reset_index(drop=True)
        n         = len(normal_df)
        split_at  = int(n * (1 - THRESHOLD_VAL_SPLIT))

        train_df = normal_df.iloc[:split_at]
        cal_df   = normal_df.iloc[split_at:]

        log.info(
            "Training on %d normal samples; calibrating threshold on %d.",
            len(train_df), len(cal_df),
        )

        scaled_train = self._scale(train_df, fit=True)
        scaled_cal   = self._scale(cal_df, fit=False)

        X_train = make_sequences(scaled_train, SEQ_LEN)
        X_cal   = make_sequences(scaled_cal,   SEQ_LEN)

        self.model = build_lstm_autoencoder(SEQ_LEN, len(FEATURES))
        self.model.summary(print_fn=log.debug)

        callbacks = [
            EarlyStopping(
                monitor="val_loss", patience=7,
                restore_best_weights=True, verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-5, verbose=0,
            ),
        ]

        self.model.fit(
            X_train, X_train,
            epochs           = EPOCHS,
            batch_size       = BATCH_SIZE,
            validation_split = 0.10,
            callbacks        = callbacks,
            verbose          = 1,
        )

        # ── calibrate threshold on held-out normal data ───────────────────
        X_cal_pred = self.model.predict(X_cal, verbose=0)
        cal_errors = np.mean(np.abs(X_cal - X_cal_pred), axis=(1, 2))

        if use_optimal_threshold:
            # All calibration samples are normal (label == 0)
            cal_labels = np.zeros(len(cal_errors), dtype=int)
            self.threshold = find_optimal_threshold(cal_errors, cal_labels)
        else:
            self.threshold = float(np.percentile(cal_errors, THRESHOLD_PCT))

        log.info(
            "Threshold set to %.5f (p%d of held-out normal errors).",
            self.threshold, THRESHOLD_PCT,
        )

    # ── predict ───────────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference and return df augmented with:
          anomaly_score   — per-window MAE reconstruction error
          predicted_label — 1 if score > threshold, else 0
          err_*          — per-feature MAE reconstruction error (for XAI)
        """
        scaled  = self._scale(df)
        X       = make_sequences(scaled, SEQ_LEN)
        X_pred  = self.model.predict(X, verbose=0)
        
        per_feature_errors = np.mean(np.abs(X - X_pred), axis=1)
        errors = np.mean(per_feature_errors, axis=1)

        pad    = np.full(SEQ_LEN, np.nan)
        scores = np.concatenate([pad, errors])

        result = df.copy().reset_index(drop=True)
        result["anomaly_score"]   = scores
        result["predicted_label"] = (scores > self.threshold).astype(int)
        result.loc[result["anomaly_score"].isna(), "predicted_label"] = 0
        
        # Add per-feature errors for XAI feature attribution
        for i, feature in enumerate(FEATURES):
            feat_pad = np.full(SEQ_LEN, np.nan)
            feat_scores = np.concatenate([feat_pad, per_feature_errors[:, i]])
            result[f"err_{feature}"] = feat_scores
            
        return result

    # ── evaluate ──────────────────────────────────────────────────────────────
    def evaluate(self, result: pd.DataFrame) -> dict:
        """Compute and log classification metrics (requires true labels)."""
        valid  = result.dropna(subset=["anomaly_score"])
        y_true = valid["label"].values
        y_pred = valid["predicted_label"].values.astype(int)
        scores = valid["anomaly_score"].values

        report = classification_report(
            y_true, y_pred,
            target_names=["normal", "fault"],
            output_dict=True,
        )
        auc = roc_auc_score(y_true, scores)

        log.info("\n%s", classification_report(
            y_true, y_pred, target_names=["normal", "fault"]
        ))
        log.info("ROC-AUC: %.4f", auc)
        return {"report": report, "roc_auc": auc, "threshold": self.threshold}

    # ── per-feature reconstruction error ──────────────────────────────────────
    def reconstruction_errors_per_feature(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Returns array of shape (n_windows, n_features) — useful for
        per-sensor analysis in evaluate.py.
        """
        scaled = self._scale(df)
        X      = make_sequences(scaled, SEQ_LEN)
        X_pred = self.model.predict(X, verbose=0)
        return np.mean(np.abs(X - X_pred), axis=1)   # average over time axis

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path: str = "saved_model"):
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/lstm_ae.keras")
        with open(f"{path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(f"{path}/threshold.pkl", "wb") as f:
            pickle.dump(self.threshold, f)
        log.info("Model saved to %s/", path)

    def load(self, path: str = "saved_model"):
        self.model = tf.keras.models.load_model(f"{path}/lstm_ae.keras")
        with open(f"{path}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        with open(f"{path}/threshold.pkl", "rb") as f:
            self.threshold = pickle.load(f)
        log.info("Model loaded from %s/", path)


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from simulator import generate_dataset

    log.info("Generating training data (normal only)…")
    df_train, _ = generate_dataset(n_seconds=300, fault_count=0, seed=1)

    log.info("Generating test data (with faults)…")
    df_test, fault_log = generate_dataset(n_seconds=300, fault_count=5, seed=42)

    detector = CableFaultDetector()
    log.info("Training LSTM autoencoder…")
    detector.train(df_train)

    log.info("Running inference…")
    result = detector.predict(df_test)

    metrics = detector.evaluate(result)
    detector.save()
    log.info("Fault log: %s", fault_log)
