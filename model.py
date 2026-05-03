"""
model.py
Multi-task Conv-Transformer Hybrid for fault diagnosis and anomaly detection.
"""

from __future__ import annotations
import logging
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import (
    Conv1D, Dense, Dropout, GlobalAveragePooling1D, Layer,
    LayerNormalization, MultiHeadAttention, Reshape, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from config import (
    BATCH_SIZE, CABLE_DOMAIN_NAMES, CLASSIFICATION_MAP, CLASS_NAMES,
    CLASS_WEIGHTS, DROPOUT_RATE, EPOCHS, FEATURES, LATENT_UNITS,
    LSTM_UNITS, NUM_CLASSES, SEQ_LEN, THRESHOLD_PCT, THRESHOLD_VAL_SPLIT,
    TRANSFORMER_BLOCKS, TRANSFORMER_FF_DIM, TRANSFORMER_HEADS,
    USE_OPTIMAL_THRESHOLD,
)
from utils import clip_to_scaler_bounds, find_optimal_threshold, make_sequences

log = logging.getLogger(__name__)

CONV_FILTERS  : int = LSTM_UNITS
LATENT_DIM    : int = LATENT_UNITS
CONV_STRIDE   : int = 2

# ── Custom Layers ─────────────────────────────────────────────────────────────

class SinePositionalEncoding(Layer):
    """Sinusoidal positional encoding compatible with Keras 3.x static shape tracing."""
    def __init__(self, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape: tuple):
        d_model = input_shape[-1]
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10_000.0, (2 * (dims // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Store as a non-trainable weight so Keras can trace it properly
        self._pe_table = self.add_weight(
            name="pe_table",
            shape=(1, self.max_len, d_model),
            initializer=tf.keras.initializers.Constant(angles[np.newaxis, :, :]),
            trainable=False,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self._pe_table[:, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["max_len"] = self.max_len
        return cfg

class TransformerEncoderBlock(Layer):
    """
    Transformer encoder block.
    build_from_config() is implemented for Keras 3 deserialization.
    """
    def __init__(self, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout

    def build(self, input_shape: tuple):
        d_model = int(input_shape[-1])
        seq_len = input_shape[-2]  # may be None
        self._stored_input_shape = list(input_shape)  # saved for get_build_config

        self._mha       = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=max(1, d_model // self.num_heads),
            dropout=self.dropout,
        )
        self._drop_attn = Dropout(self.dropout)
        self._norm1     = LayerNormalization(epsilon=1e-6)
        self._ffn1      = Dense(self.ff_dim, activation="relu")
        self._ffn_drop  = Dropout(self.dropout)
        self._ffn2      = Dense(d_model)
        self._norm2     = LayerNormalization(epsilon=1e-6)

        # Explicitly build each sub-layer with the correct input shape
        # so that their weights exist before checkpoint loading.
        mha_shape  = (None, seq_len, d_model)
        ffn1_shape = (None, seq_len, d_model)
        ffn2_shape = (None, seq_len, self.ff_dim)  # ffn2 input is ffn1 output
        norm_shape = (None, seq_len, d_model)

        self._mha.build(mha_shape, mha_shape)   # MHA needs (query_shape, value_shape)
        self._norm1.build(norm_shape)
        self._ffn1.build(ffn1_shape)
        self._ffn2.build(ffn2_shape)
        self._norm2.build(norm_shape)

        super().build(input_shape)

    def get_build_config(self) -> dict:
        """Saves the input shape so build_from_config can reconstruct sub-layers."""
        return {"input_shape": getattr(self, "_stored_input_shape", None)}

    def build_from_config(self, config: dict):
        """Called by Keras 3 during model loading to rebuild sub-layers."""
        self.build(config["input_shape"])

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_out = self._mha(x, x, training=training)
        attn_out = self._drop_attn(attn_out, training=training)
        x        = self._norm1(x + attn_out)
        ffn_out  = self._ffn1(x)
        ffn_out  = self._ffn_drop(ffn_out, training=training)
        ffn_out  = self._ffn2(ffn_out)
        x        = self._norm2(x + ffn_out)
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "ff_dim": self.ff_dim, "dropout": self.dropout})
        return cfg

    def compute_output_shape(self, input_shape):
        return input_shape

# ── Model Factory ─────────────────────────────────────────────────────────────

def build_conv_transformer_autoencoder(seq_len: int, n_features: int, num_classes: int = 4) -> Model:
    """
    Single-input Conv-Transformer autoencoder.

    `n_features` includes the cable_domain one-hot channels (appended by the
    detector's _scale method), so this model sees cable type as a feature —
    equivalent to domain conditioning but compatible with Keras 3.x.
    """
    downsampled_len = seq_len // CONV_STRIDE

    # Single sensor input — domain identity is baked into the feature vector
    inp = Input(shape=(seq_len, n_features), name="sensor_input")

    # Encoder
    x = Conv1D(filters=CONV_FILTERS, kernel_size=3, strides=CONV_STRIDE,
               padding="same", activation="relu")(inp)
    x = SinePositionalEncoding(max_len=seq_len + 8)(x)
    for _ in range(TRANSFORMER_BLOCKS):
        x = TransformerEncoderBlock(
            num_heads=TRANSFORMER_HEADS, ff_dim=TRANSFORMER_FF_DIM, dropout=DROPOUT_RATE
        )(x)
    x = GlobalAveragePooling1D()(x)

    # Bottleneck
    encoded = Dense(LATENT_DIM, activation="relu", name="bottleneck")(x)

    # ── Branch 1: Reconstruction ───────────────────────────────────────────────
    rec = Dense(downsampled_len * CONV_FILTERS, activation="relu")(encoded)
    rec = Reshape((downsampled_len, CONV_FILTERS))(rec)
    for _ in range(TRANSFORMER_BLOCKS):
        rec = TransformerEncoderBlock(
            num_heads=TRANSFORMER_HEADS, ff_dim=TRANSFORMER_FF_DIM, dropout=DROPOUT_RATE
        )(rec)
    rec = layers.Conv1DTranspose(
        filters=CONV_FILTERS, kernel_size=3, strides=CONV_STRIDE, padding="same", activation="relu"
    )(rec)
    out_rec = TimeDistributed(Dense(n_features), name="reconstruction")(rec)

    # ── Branch 2: Classification (deeper head for better fault discrimination) ─
    cls_x = Dense(64, activation="relu")(encoded)
    cls_x = Dropout(DROPOUT_RATE)(cls_x)
    cls_x = Dense(32, activation="relu")(cls_x)
    cls_x = Dropout(DROPOUT_RATE / 2)(cls_x)
    out_cls = Dense(num_classes, activation="softmax", name="classification")(cls_x)

    model = Model(inputs=inp, outputs=[out_rec, out_cls])

    # Cosine-decay LR: starts at 3e-4, decays over the full 30 epochs.
    # steps_per_epoch ≈ n_train_samples / batch_size ≈ 90k / 32 ≈ 2814
    # Total steps = 2814 × 30 ≈ 85_000  →  decay_steps set to 84_420
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=3e-4, decay_steps=84_420, alpha=1e-6
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    # Use list/tuple losses instead of dict — Keras 3.14 dict path triggers
    # a squeeze() shape bug (Cannot squeeze axis=-1) on multi-output models.
    model.compile(
        optimizer=optimizer,
        loss=["mae", tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)],
        loss_weights=[1.0, 2.0],
        metrics={"classification": "accuracy"},
    )
    return model

# ── Detector Class ────────────────────────────────────────────────────────────

class CableFaultDetector:
    # Number of cable domain one-hot channels appended to sensor features
    _N_DOMAIN_CHANNELS: int = 10

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None

    def _scale(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale sensor features and append cable_domain one-hot channels."""
        values = df.reindex(columns=FEATURES).fillna(0.0)[FEATURES].values.astype(np.float32)
        if fit:
            scaled = self.scaler.fit_transform(values)
        else:
            scaled = self.scaler.transform(clip_to_scaler_bounds(values, self.scaler))
        # Append one-hot domain encoding so the model learns cable-type-specific patterns
        domain_ids = df["cable_domain_id"].values.astype(int) if "cable_domain_id" in df.columns else np.zeros(len(df), dtype=int)
        domain_onehot = np.zeros((len(df), self._N_DOMAIN_CHANNELS), dtype=np.float32)
        domain_onehot[np.arange(len(df)), np.clip(domain_ids, 0, self._N_DOMAIN_CHANNELS - 1)] = 1.0
        return np.concatenate([scaled, domain_onehot], axis=1)

    @property
    def n_input_features(self) -> int:
        """Total feature width passed to model (sensor features + one-hot domain channels)."""
        return len(FEATURES) + self._N_DOMAIN_CHANNELS

    def train(self, df: pd.DataFrame, use_optimal_threshold: bool = USE_OPTIMAL_THRESHOLD, resume: bool = False) -> None:
        normal_df = df[df["label"] == 0].reset_index(drop=True)
        fault_df  = df[df["label"] == 1].reset_index(drop=True)

        min_normal_for_cal = SEQ_LEN + 100
        if len(normal_df) < min_normal_for_cal:
            cal_split = 0.0
            log.warning("Only %d normal samples — using reconstruction error for threshold", len(normal_df))
        else:
            cal_split = THRESHOLD_VAL_SPLIT

        split_at = int(len(normal_df) * (1 - cal_split))
        # Fit scaler on ALL normal rows from the combined dataset so it covers
        # the full value range across electrical, optical, and industrial data.
        # Previously we only fit on normal_df.iloc[:split_at] which was too narrow.
        self._scale(normal_df, fit=True)

        # ── Oversample minority fault rows before sequencing ──────────────────
        # Avoids sample_weight API differences across Keras versions.
        # Target: fault rows ≈ 50 % of normal rows so training is balanced.
        target_fault = max(len(fault_df), len(normal_df) // 2)
        if len(fault_df) > 0 and len(fault_df) < target_fault:
            rng = np.random.default_rng(seed=42)
            repeat = int(np.ceil(target_fault / len(fault_df)))
            idx = np.tile(np.arange(len(fault_df)), repeat)[:target_fault]
            idx = rng.permutation(idx)
            fault_df_aug = fault_df.iloc[idx].reset_index(drop=True)
            train_df = pd.concat([normal_df, fault_df_aug], ignore_index=True)
            log.info("Oversampled: %d fault rows → %d  |  training set: %d rows",
                     len(fault_df), target_fault, len(train_df))
        else:
            train_df = df.reset_index(drop=True)
            log.info("Training set: %d rows (no oversampling needed)", len(train_df))

        X = make_sequences(self._scale(train_df), SEQ_LEN)

        # Strip whitespace from fault_type strings (some datasets have leading spaces)
        # and map to class index; unknown types default to 0 (Normal).
        raw_types = train_df["fault_type"].iloc[SEQ_LEN:].astype(str).str.strip().values
        y_cls = np.array(
            [CLASSIFICATION_MAP.get(t, 0) for t in raw_types], dtype=np.int32
        )

        # ── Model Initialisation ──────────────────────────────────────────────
        checkpoint_path = "checkpoints/best_model.keras"
        saved_path = "saved_model/conv_transformer_multitask.keras"
        
        if resume and (os.path.exists(checkpoint_path) or os.path.exists(saved_path)):
            load_path = checkpoint_path if os.path.exists(checkpoint_path) else saved_path
            log.info(f"Resuming training from: {load_path}")
            try:
                self.load(os.path.dirname(load_path))
                # Re-compile to ensure LR schedule and optimizer state are fresh or continued
                # (load_model usually restores optimizer state, but we might want to tweak LR)
            except Exception as e:
                log.error(f"Failed to load model for resume: {e}. Starting fresh.")
                self.model = build_conv_transformer_autoencoder(SEQ_LEN, self.n_input_features, NUM_CLASSES)
        else:
            log.info("Starting fresh training run...")
            self.model = build_conv_transformer_autoencoder(SEQ_LEN, self.n_input_features, NUM_CLASSES)
        os.makedirs("checkpoints", exist_ok=True)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
            # Note: ReduceLROnPlateau removed — incompatible with CosineDecay schedule
            ModelCheckpoint(
                filepath="checkpoints/best_model.keras",
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            )
        ]

        # Manual 85/15 train/val split — Keras validation_split triggers a graph
        # shape inference squeeze bug in Keras 3.14 on large (>50k) datasets.
        _rng = np.random.default_rng(seed=7)
        _val_size = max(1, int(len(X) * 0.15))
        _val_idx = _rng.choice(len(X), size=_val_size, replace=False)
        _train_idx = np.setdiff1d(np.arange(len(X)), _val_idx)
        X_tr, X_vl = X[_train_idx], X[_val_idx]
        y_tr, y_vl = y_cls[_train_idx], y_cls[_val_idx]

        self.model.fit(
            X_tr,
            [X_tr, y_tr],
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_vl, [X_vl, y_vl]),
            callbacks=callbacks,
            verbose=1,
        )

        if len(normal_df) >= min_normal_for_cal:
            cal_df = normal_df.iloc[split_at:]
            X_cal = make_sequences(self._scale(cal_df), SEQ_LEN)
            X_cal_pred, _ = self.model.predict(X_cal, verbose=0)
            cal_errors = np.mean(np.abs(X_cal - X_cal_pred), axis=(1, 2))
            self.threshold = float(np.percentile(cal_errors, THRESHOLD_PCT))
        else:
            X_pred, _ = self.model.predict(X[:500], verbose=0)
            errors = np.mean(np.abs(X[:500] - X_pred), axis=(1, 2))
            self.threshold = float(np.percentile(errors, THRESHOLD_PCT + 3))
            
        log.info(f"Threshold: {self.threshold:.6f}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = make_sequences(self._scale(df), SEQ_LEN)
        X_pred, C_pred = self.model.predict(X, verbose=0)

        # ── Reconstruction error (kept as secondary signal) ──────────────────
        per_feature_errors = np.mean(np.abs(X - X_pred), axis=1)
        sensor_errors = per_feature_errors[:, :len(FEATURES)]  # strip domain one-hot
        recon_errors  = np.mean(sensor_errors, axis=1)

        # ── Classification-based anomaly signal (PRIMARY) ────────────────────
        # anomaly_score = 1 - P(Normal)  →  ranges [0, 1]
        # 0.0  = model is certain this is Normal
        # 1.0  = model is certain this is a fault
        normal_prob   = C_pred[:, 0]          # P(class == 0 == Normal)
        fault_prob    = 1.0 - normal_prob      # P(any fault class)
        pred_classes  = np.argmax(C_pred, axis=1)   # 0=Normal,1=Short,2=Open,3=High-Z
        class_names   = [CLASS_NAMES[idx] for idx in pred_classes]

        # Pad the first SEQ_LEN rows (no prediction window yet) with NaN / Normal
        pad = SEQ_LEN
        scores_full  = np.concatenate([np.full(pad, np.nan), fault_prob])
        recon_full   = np.concatenate([np.full(pad, np.nan), recon_errors])
        labels_full  = np.concatenate([np.zeros(pad, dtype=int),
                                        (pred_classes != 0).astype(int)])
        classes_full = (["Normal"] * pad) + class_names

        result = df.copy().reset_index(drop=True)
        result["anomaly_score"]   = scores_full          # 1 - P(Normal)  ← primary signal
        result["recon_error"]     = recon_full            # MAE reconstruction ← secondary
        result["predicted_label"] = labels_full
        result["fault_diagnosis"] = classes_full
        result.loc[result["anomaly_score"].isna(), "predicted_label"] = 0

        # Human-readable cable type label
        if "cable_domain_id" in result.columns:
            result["cable_type"] = result["cable_domain_id"].map(
                lambda d: CABLE_DOMAIN_NAMES.get(int(d), f"Domain-{int(d)}")
            )
        else:
            result["cable_type"] = CABLE_DOMAIN_NAMES[0]

        for i, feature in enumerate(FEATURES):
            result[f"err_{feature}"] = np.concatenate([np.full(pad, np.nan), sensor_errors[:, i]])
        return result

    def evaluate(self, result: pd.DataFrame) -> dict:
        valid = result.dropna(subset=["anomaly_score"])
        y_true, y_pred, scores = valid["label"].values, valid["predicted_label"].values.astype(int), valid["anomaly_score"].values
        report = classification_report(y_true, y_pred, target_names=["normal", "fault"], output_dict=True)
        auc = roc_auc_score(y_true, scores)
        log.info(f"ROC-AUC: {auc:.4f}")
        return {"report": report, "roc_auc": auc, "threshold": self.threshold}

    def calibrate_threshold(self, eval_df: pd.DataFrame) -> float:
        """Refine threshold using F1-sweep on labeled evaluation data.
        Call this AFTER training, with held-out data that has both normal + fault labels.
        Returns the optimised threshold and updates self.threshold in-place.
        """
        result = self.predict(eval_df)
        valid = result.dropna(subset=["anomaly_score"])
        scores = valid["anomaly_score"].values
        labels = valid["label"].values
        if len(np.unique(labels)) < 2:
            log.warning("Calibration requires both normal and fault labels — keeping p95 threshold")
            return self.threshold
        new_thr = find_optimal_threshold(scores, labels)
        log.info(f"Threshold calibrated: {self.threshold:.6f} → {new_thr:.6f}")
        self.threshold = new_thr
        return new_thr

    def reconstruction_errors_per_feature(self, df: pd.DataFrame) -> np.ndarray:
        X = make_sequences(self._scale(df), SEQ_LEN)
        preds = self.model.predict(X, verbose=0)
        # Return per-feature errors for sensor features only (not one-hot domain channels)
        return np.mean(np.abs(X - preds[0]), axis=1)[:, :len(FEATURES)]


    def save(self, path: str = "saved_model") -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/conv_transformer_multitask.keras")
        with open(f"{path}/scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        with open(f"{path}/threshold.pkl", "wb") as f: pickle.dump(self.threshold, f)

    def load(self, path: str = "saved_model") -> None:
        custom = {"SinePositionalEncoding": SinePositionalEncoding, "TransformerEncoderBlock": TransformerEncoderBlock}
        # Try both common filenames
        model_file = f"{path}/conv_transformer_multitask.keras"
        if not os.path.exists(model_file):
            model_file = f"{path}/best_model.keras"
            
        self.model = tf.keras.models.load_model(model_file, custom_objects=custom)
        
        scaler_file = f"{path}/scaler.pkl"
        if os.path.exists(scaler_file):
            with open(scaler_file, "rb") as f: self.scaler = pickle.load(f)
            
        threshold_file = f"{path}/threshold.pkl"
        if os.path.exists(threshold_file):
            with open(threshold_file, "rb") as f: self.threshold = pickle.load(f)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="datasets/azure_pdm.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.dataset)
    detector = CableFaultDetector()
    detector.train(df)
    
    # Save ROC-AUC for the dashboard
    result = detector.predict(df)
    metrics = detector.evaluate(result)
    
    detector.save()
    with open("saved_model/roc_auc.pkl", "wb") as f:
        pickle.dump(metrics["roc_auc"], f)
    print("Training complete.")
