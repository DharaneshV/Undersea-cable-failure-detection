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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from config import (
    BATCH_SIZE, CLASSIFICATION_MAP, CLASS_NAMES, DROPOUT_RATE, EPOCHS,
    FEATURES, LATENT_UNITS, LSTM_UNITS, NUM_CLASSES, SEQ_LEN,
    THRESHOLD_PCT, THRESHOLD_VAL_SPLIT, TRANSFORMER_BLOCKS,
    TRANSFORMER_FF_DIM, TRANSFORMER_HEADS,
)
from utils import clip_to_scaler_bounds, find_optimal_threshold, make_sequences

log = logging.getLogger(__name__)

CONV_FILTERS  : int = LSTM_UNITS
LATENT_DIM    : int = LATENT_UNITS
CONV_STRIDE   : int = 2

# ── Custom Layers ─────────────────────────────────────────────────────────────

class SinePositionalEncoding(Layer):
    def __init__(self, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape: tuple):
        _, seq_len, d_model = input_shape
        positions = np.arange(self.max_len)[:, np.newaxis]
        dims      = np.arange(d_model)[np.newaxis, :]
        angles    = positions / np.power(10_000.0, (2 * (dims // 2)) / np.float32(d_model))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self._pe_table = tf.constant(angles[np.newaxis, :, :], dtype=tf.float32, name="pe_table")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        seq_len = tf.shape(x)[1]
        return x + self._pe_table[:, :seq_len, :]

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["max_len"] = self.max_len
        return cfg

class TransformerEncoderBlock(Layer):
    def __init__(self, num_heads: int, ff_dim: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout

    def build(self, input_shape: tuple):
        d_model = input_shape[-1]
        self._mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=max(1, d_model // self.num_heads), dropout=self.dropout)
        self._drop_attn = Dropout(self.dropout)
        self._norm1     = LayerNormalization(epsilon=1e-6)
        self._ffn1      = Dense(self.ff_dim, activation="relu")
        self._ffn_drop  = Dropout(self.dropout)
        self._ffn2      = Dense(d_model)
        self._norm2     = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

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

# ── Model Factory ─────────────────────────────────────────────────────────────

def build_conv_transformer_autoencoder(seq_len: int, n_features: int, num_classes: int = 4) -> Model:
    downsampled_len = seq_len // CONV_STRIDE
    inp = Input(shape=(seq_len, n_features), name="input")

    # Encoder
    x = Conv1D(filters=CONV_FILTERS, kernel_size=3, strides=CONV_STRIDE, padding="same", activation="relu")(inp)
    x = SinePositionalEncoding(max_len=downsampled_len + 8)(x)
    for i in range(TRANSFORMER_BLOCKS):
        x = TransformerEncoderBlock(num_heads=TRANSFORMER_HEADS, ff_dim=TRANSFORMER_FF_DIM, dropout=DROPOUT_RATE)(x)
    x = GlobalAveragePooling1D()(x)

    # Bottleneck
    encoded = Dense(LATENT_DIM, activation="relu", name="bottleneck")(x)

    # Branch 1: Reconstruction
    rec = Dense(downsampled_len * CONV_FILTERS, activation="relu")(encoded)
    rec = Reshape((downsampled_len, CONV_FILTERS))(rec)
    for i in range(TRANSFORMER_BLOCKS):
        rec = TransformerEncoderBlock(num_heads=TRANSFORMER_HEADS, ff_dim=TRANSFORMER_FF_DIM, dropout=DROPOUT_RATE)(rec)
    rec = layers.Conv1DTranspose(filters=CONV_FILTERS, kernel_size=3, strides=CONV_STRIDE, padding="same", activation="relu")(rec)
    out_rec = TimeDistributed(Dense(n_features), name="reconstruction")(rec)

    # Branch 2: Classification
    cls_x = Dense(32, activation="relu")(encoded)
    cls_x = Dropout(DROPOUT_RATE)(cls_x)
    out_cls = Dense(num_classes, activation="softmax", name="classification")(cls_x)

    model = Model(inputs=inp, outputs=[out_rec, out_cls])
    model.compile(
        optimizer="adam",
        loss={"reconstruction": "mae", "classification": "sparse_categorical_crossentropy"},
        loss_weights={"reconstruction": 1.0, "classification": 1.0}
    )
    return model

# ── Detector Class ────────────────────────────────────────────────────────────

class CableFaultDetector:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.threshold = None

    def _scale(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        values = df[FEATURES].values.astype(np.float32)
        if fit: return self.scaler.fit_transform(values)
        return self.scaler.transform(clip_to_scaler_bounds(values, self.scaler))

    def train(self, df: pd.DataFrame, use_optimal_threshold: bool = False) -> None:
        normal_df = df[df["label"] == 0].reset_index(drop=True)
        split_at  = int(len(normal_df) * (1 - THRESHOLD_VAL_SPLIT))
        self._scale(normal_df.iloc[:split_at], fit=True)
        
        X = make_sequences(self._scale(df), SEQ_LEN)
        y_cls = np.array([CLASSIFICATION_MAP.get(t, 0) for t in df["fault_type"].iloc[SEQ_LEN:].values])

        self.model = build_conv_transformer_autoencoder(SEQ_LEN, len(FEATURES), NUM_CLASSES)
        callbacks = [EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)]
        self.model.fit(X, {"reconstruction": X, "classification": y_cls}, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.15, callbacks=callbacks, verbose=1)

        cal_df = normal_df.iloc[split_at:]
        X_cal = make_sequences(self._scale(cal_df), SEQ_LEN)
        X_cal_pred, _ = self.model.predict(X_cal, verbose=0)
        cal_errors = np.mean(np.abs(X_cal - X_cal_pred), axis=(1, 2))
        self.threshold = float(np.percentile(cal_errors, THRESHOLD_PCT)) if not use_optimal_threshold else find_optimal_threshold(cal_errors, np.zeros(len(cal_errors)))
        log.info(f"Threshold: {self.threshold:.6f}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = self._scale(df)
        X = make_sequences(scaled, SEQ_LEN)
        X_pred, C_pred = self.model.predict(X, verbose=0)
        
        per_feature_errors = np.mean(np.abs(X - X_pred), axis=1)
        errors = np.mean(per_feature_errors, axis=1)
        
        class_names = [CLASS_NAMES[idx] for idx in np.argmax(C_pred, axis=1)]
        scores = np.concatenate([np.full(SEQ_LEN, np.nan), errors])
        final_classes = (["Normal"] * SEQ_LEN) + class_names

        result = df.copy().reset_index(drop=True)
        result["anomaly_score"] = scores
        result["predicted_label"] = (scores > self.threshold).astype(int)
        result["fault_diagnosis"] = final_classes
        result.loc[result["anomaly_score"].isna(), "predicted_label"] = 0
        
        for i, feature in enumerate(FEATURES):
            result[f"err_{feature}"] = np.concatenate([np.full(SEQ_LEN, np.nan), per_feature_errors[:, i]])
        return result

    def evaluate(self, result: pd.DataFrame) -> dict:
        valid = result.dropna(subset=["anomaly_score"])
        y_true, y_pred, scores = valid["label"].values, valid["predicted_label"].values.astype(int), valid["anomaly_score"].values
        report = classification_report(y_true, y_pred, target_names=["normal", "fault"], output_dict=True)
        auc = roc_auc_score(y_true, scores)
        log.info(f"ROC-AUC: {auc:.4f}")
        return {"report": report, "roc_auc": auc, "threshold": self.threshold}

    def reconstruction_errors_per_feature(self, df: pd.DataFrame) -> np.ndarray:
        X = make_sequences(self._scale(df), SEQ_LEN)
        preds = self.model.predict(X, verbose=0)
        return np.mean(np.abs(X - preds[0]), axis=1)

    def save(self, path: str = "saved_model") -> None:
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/conv_transformer_multitask.keras")
        with open(f"{path}/scaler.pkl", "wb") as f: pickle.dump(self.scaler, f)
        with open(f"{path}/threshold.pkl", "wb") as f: pickle.dump(self.threshold, f)

    def load(self, path: str = "saved_model") -> None:
        custom = {"SinePositionalEncoding": SinePositionalEncoding, "TransformerEncoderBlock": TransformerEncoderBlock}
        self.model = tf.keras.models.load_model(f"{path}/conv_transformer_multitask.keras", custom_objects=custom)
        with open(f"{path}/scaler.pkl", "rb") as f: self.scaler = pickle.load(f)
        with open(f"{path}/threshold.pkl", "rb") as f: self.threshold = pickle.load(f)

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
