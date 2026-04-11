"""
model.py
Conv-Transformer Hybrid Autoencoder for unsupervised anomaly detection
on undersea cable sensor data.

Upgrade summary (LSTM Autoencoder → Conv-Transformer Hybrid)
─────────────────────────────────────────────────────────────
1. Conv1D (stride-2) precedes every Attention block, halving the sequence
   length that Multi-Head Attention must process.  Self-attention cost drops
   from O(L²) to O((L/2)²) = O(L²/4) — live dashboard inference is ~4×
   faster on long sequences, with negligible accuracy loss because the conv
   has already extracted the fine-grained local patterns.

2. SinePositionalEncoding (Vaswani et al., 2017) injects fixed sine/cosine
   position signals so attention can distinguish 'earlier' vs 'later'
   timesteps — attention is permutation-invariant without this.

3. TransformerEncoderBlock stacks Multi-Head Self-Attention + FFN with
   Pre-residual Add & LayerNorm, matching the canonical architecture.

4. Symmetric decoder: TransformerEncoderBlock(s) + Conv1DTranspose restores
   the original (SEQ_LEN, n_features) shape — no Reshape hacks.

5. Strict interface compliance: CableFaultDetector's public surface
   (train, predict, evaluate, save, load, _scale,
    reconstruction_errors_per_feature) is identical to the LSTM version.
   dashboard.py and evaluate.py require zero changes.

6. XAI feature attribution: per-feature MAE columns (err_voltage,
   err_current, err_temperature, err_vibration) are propagated exactly as
   before so the dashboard's explainability panel continues to work.

Required additions to config.py
────────────────────────────────
    SEQ_LEN            = 60
    TRANSFORMER_HEADS  = 4
    TRANSFORMER_FF_DIM = 128
    TRANSFORMER_BLOCKS = 2

All other hyper-parameters (LSTM_UNITS, LATENT_UNITS, DROPOUT_RATE,
EPOCHS, BATCH_SIZE, THRESHOLD_PCT, THRESHOLD_VAL_SPLIT) are reused as-is.
LSTM_UNITS  → CONV_FILTERS  (same role: feature width of the hidden repr.)
LATENT_UNITS → LATENT_DIM   (same role: bottleneck compression dimension)
"""

from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd

# Silence TF C++ noise before any keras imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import Input, Model, layers
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
    Reshape,
    TimeDistributed,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    FEATURES,
    LATENT_UNITS,       # repurposed: bottleneck Dense dimension
    LSTM_UNITS,         # repurposed: Conv1D filter count
    SEQ_LEN,
    THRESHOLD_PCT,
    THRESHOLD_VAL_SPLIT,
    TRANSFORMER_BLOCKS,
    TRANSFORMER_FF_DIM,
    TRANSFORMER_HEADS,
)
from utils import check_scaler_bounds, find_optimal_threshold, make_sequences

log = logging.getLogger(__name__)

# ── architecture constants ────────────────────────────────────────────────────
# LSTM_UNITS (64) carries the same semantic role it had in the LSTM version:
#   it controls the hidden feature width.  Here it becomes the Conv1D filter
#   count, so the encoder output is still (downsampled_len, 64).
# LATENT_UNITS (32) still means the bottleneck compression dimension.
CONV_FILTERS  : int = LSTM_UNITS    # 64  — Conv1D output channels
LATENT_DIM    : int = LATENT_UNITS  # 32  — bottleneck Dense units
CONV_STRIDE   : int = 2             # sequence downsampling factor (60 → 30)


# ═════════════════════════════════════════════════════════════════════════════
# Custom Keras Layers
# ═════════════════════════════════════════════════════════════════════════════

class SinePositionalEncoding(Layer):
    """
    Fixed sine/cosine positional encoding (Vaswani et al., "Attention Is All
    You Need", 2017).

    Multi-Head Attention is permutation-invariant: without position signals
    it cannot tell step 0 from step 29.  This layer adds a deterministic,
    non-trainable position signature to every timestep so downstream
    attention can exploit temporal order.

    Encoding formula for position pos and feature dimension i:
        PE(pos, 2i)   = sin( pos / 10_000^(2i / d_model) )
        PE(pos, 2i+1) = cos( pos / 10_000^(2i / d_model) )

    The table is computed once at build() time and sliced to the actual
    runtime sequence length at call() time, so the layer handles variable-
    length inputs without recomputation.

    Parameters
    ----------
    max_len : int
        Pre-computed table length.  Must be ≥ the longest sequence this
        layer will ever see.  Default 512 is well above any cable window.
    """

    def __init__(self, max_len: int = 512, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len

    def build(self, input_shape: tuple):
        _, seq_len, d_model = input_shape

        # ── Build the (max_len, d_model) PE table once ────────────────────────
        positions = np.arange(self.max_len)[:, np.newaxis]              # (L, 1)
        dims      = np.arange(d_model)[np.newaxis, :]                   # (1, d)

        # Angle matrix: (L, d) — same value used for both sin and cos branches
        angles = positions / np.power(10_000.0, (2 * (dims // 2)) / np.float32(d_model))

        # Apply sin to even indices, cos to odd indices (in-place)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])

        # Store as a non-trainable float32 constant: (1, max_len, d_model)
        self._pe_table = tf.constant(
            angles[np.newaxis, :, :],
            dtype = tf.float32,
            name  = "pe_table",
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Slice to actual runtime sequence length and broadcast-add
        seq_len = tf.shape(x)[1]
        return x + self._pe_table[:, :seq_len, :]

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg["max_len"] = self.max_len
        return cfg


class TransformerEncoderBlock(Layer):
    """
    One canonical Transformer encoder block (Post-LN variant):

        ┌─ input ─────────────────────────────────────────────────┐
        │  MultiHeadAttention(Q=x, K=x, V=x)  ← self-attention   │
        │  Dropout                                                  │
        │  Add(input, attn_output)  → LayerNorm       ← sublayer 1│
        │                                                           │
        │  Dense(ff_dim, relu)  → Dropout → Dense(d_model)        │
        │  Add(sublayer1_out, ffn_output)  → LayerNorm  ← sublayer 2│
        └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    num_heads : int
        Number of parallel attention heads.
    ff_dim : int
        Inner dimension of the point-wise feed-forward network.
        Typically 2–4× the model dimension.
    dropout : float
        Dropout probability applied after attention and inside the FFN.
    """

    def __init__(
        self,
        num_heads : int,
        ff_dim    : int,
        dropout   : float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout

    def build(self, input_shape: tuple):
        d_model = input_shape[-1]       # infer model dimension from input

        # ── Sublayer 1: Multi-Head Self-Attention ─────────────────────────────
        self._mha = MultiHeadAttention(
            num_heads = self.num_heads,
            key_dim   = max(1, d_model // self.num_heads),  # head dimension
            dropout   = self.dropout,
            name      = f"{self.name}_mha",
        )
        self._drop_attn = Dropout(self.dropout, name=f"{self.name}_drop_attn")
        self._norm1     = LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm1")

        # ── Sublayer 2: Position-wise Feed-Forward Network ────────────────────
        # Two linear transforms with one ReLU non-linearity in between.
        # The output Dense projects back to d_model so the residual connection
        # dimensions always match (no projection layer needed).
        self._ffn1     = Dense(self.ff_dim,  activation="relu", name=f"{self.name}_ffn1")
        self._ffn_drop = Dropout(self.dropout, name=f"{self.name}_ffn_drop")
        self._ffn2     = Dense(d_model, name=f"{self.name}_ffn2")
        self._norm2    = LayerNormalization(epsilon=1e-6, name=f"{self.name}_norm2")

        super().build(input_shape)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # ── Sublayer 1: Self-attention ────────────────────────────────────────
        attn_out = self._mha(x, x, training=training)          # Q = K = V = x
        attn_out = self._drop_attn(attn_out, training=training)
        x        = self._norm1(x + attn_out)                   # residual + norm

        # ── Sublayer 2: FFN ───────────────────────────────────────────────────
        ffn_out = self._ffn1(x)
        ffn_out = self._ffn_drop(ffn_out, training=training)
        ffn_out = self._ffn2(ffn_out)
        x       = self._norm2(x + ffn_out)                     # residual + norm
        return x

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "num_heads" : self.num_heads,
            "ff_dim"    : self.ff_dim,
            "dropout"   : self.dropout,
        })
        return cfg


# ═════════════════════════════════════════════════════════════════════════════
# Model factory
# ═════════════════════════════════════════════════════════════════════════════

def build_conv_transformer_autoencoder(seq_len: int, n_features: int) -> Model:
    """
    Build and compile the Conv-Transformer Hybrid Autoencoder.

    Data flow
    ─────────

    ENCODER
      Input        (seq_len, n_features)              e.g. (60, 4)
        ↓ Conv1D(CONV_FILTERS, k=3, s=2, 'same', relu)
      LocalFeatures (seq_len/2, CONV_FILTERS)         e.g. (30, 64)
        ↓ SinePositionalEncoding
      Positioned    (30, 64)
        ↓ [TransformerEncoderBlock] × TRANSFORMER_BLOCKS
      Contextualised (30, 64)
        ↓ GlobalAveragePooling1D
      Temporal mean  (64,)
        ↓ Dense(LATENT_DIM, relu)
      Bottleneck     (32,)                            ← compressed repr.

    DECODER
      Bottleneck     (32,)
        ↓ Dense(30 × 64, relu)
        ↓ Reshape(30, 64)
      Upspace        (30, 64)
        ↓ [TransformerEncoderBlock] × TRANSFORMER_BLOCKS   (symmetric)
      Contextualised (30, 64)
        ↓ Conv1DTranspose(CONV_FILTERS, k=3, s=2, 'same', relu)
      Upsampled      (60, 64)                         ← original seq_len
        ↓ TimeDistributed(Dense(n_features))
      Reconstruction (60, 4)                          ← same shape as input

    Loss: MAE (identical to LSTM version — threshold values remain comparable).

    Parameters
    ----------
    seq_len    : number of timesteps per window (e.g. 60)
    n_features : number of sensor channels (len(FEATURES), e.g. 4)

    Returns
    -------
    Compiled tf.keras.Model
    """
    downsampled_len = seq_len // CONV_STRIDE    # 60 // 2 = 30

    # ── INPUT ─────────────────────────────────────────────────────────────────
    inp = Input(shape=(seq_len, n_features), name="input")

    # ═════════════════════════════════════════════════════════════════════════
    # ENCODER
    # ═════════════════════════════════════════════════════════════════════════

    # Step 1 ── Local feature extraction + sequence compression
    #   kernel=3 creates a receptive field over 3 consecutive timesteps,
    #   capturing short-range dependencies (voltage spikes, current transients)
    #   before the global attention mechanism takes over.
    #   stride=2 halves L: attention now operates on (L/2)² pairs instead of
    #   L², cutting quadratic cost by 75% with SEQ_LEN=60.
    x = Conv1D(
        filters     = CONV_FILTERS,
        kernel_size = 3,
        strides     = CONV_STRIDE,
        padding     = "same",
        activation  = "relu",
        name        = "enc_conv1d",
    )(inp)                                          # → (30, 64)

    # Step 2 ── Inject positional information
    #   Without this, the self-attention treats the 30 timesteps as an
    #   unordered set.  The sine/cosine encoding disambiguates them.
    x = SinePositionalEncoding(
        max_len = downsampled_len + 8,              # small buffer over exact length
        name    = "enc_pos_enc",
    )(x)

    # Step 3 ── Transformer encoder blocks (contextual feature extraction)
    #   Each block lets every timestep attend to every other timestep,
    #   building a global view of the cable's temporal state.
    for i in range(TRANSFORMER_BLOCKS):
        x = TransformerEncoderBlock(
            num_heads = TRANSFORMER_HEADS,
            ff_dim    = TRANSFORMER_FF_DIM,
            dropout   = DROPOUT_RATE,
            name      = f"enc_transformer_block_{i}",
        )(x)                                        # → (30, 64) each

    # Step 4 ── Temporal aggregation
    #   GlobalAveragePooling collapses the time dimension: the bottleneck
    #   must encode the entire window's anomaly-relevant information into a
    #   single fixed-size vector — forcing the decoder to learn a compact,
    #   generalisable reconstruction rather than a trivial copy.
    x = GlobalAveragePooling1D(name="enc_gap")(x)  # → (64,)

    # Step 5 ── Bottleneck (information compression)
    encoded = Dense(
        LATENT_DIM,
        activation = "relu",
        name       = "bottleneck",
    )(x)                                            # → (32,)

    # ═════════════════════════════════════════════════════════════════════════
    # DECODER  (symmetric to encoder)
    # ═════════════════════════════════════════════════════════════════════════

    # Step 6 ── Project latent vector back into the downsampled sequence space
    #   We expand the scalar bottleneck into a full (30, 64) feature map so
    #   the Transformer blocks can refine it timestep-by-timestep.
    x = Dense(
        downsampled_len * CONV_FILTERS,
        activation = "relu",
        name       = "dec_project",
    )(encoded)                                          # → (30 × 64,)
    x = Reshape(
        (downsampled_len, CONV_FILTERS),
        name = "dec_reshape",
    )(x)                                                # → (30, 64)

    # Step 7 ── Transformer decoder blocks (reconstruction context)
    #   Symmetric stack: each block refines the reconstructed feature map
    #   using self-attention.  Using the same block type as the encoder keeps
    #   the design clean; a full cross-attention decoder is unnecessary for
    #   the autoencoder paradigm.
    for i in range(TRANSFORMER_BLOCKS):
        x = TransformerEncoderBlock(
            num_heads = TRANSFORMER_HEADS,
            ff_dim    = TRANSFORMER_FF_DIM,
            dropout   = DROPOUT_RATE,
            name      = f"dec_transformer_block_{i}",
        )(x)                                            # → (30, 64) each

    # Step 8 ── Upsample back to original sequence length
    #   Conv1DTranspose with stride=2 doubles L: (30, 64) → (60, 64).
    #   padding='same' guarantees the output length is exactly seq_len
    #   (input_length × stride = 30 × 2 = 60), regardless of kernel_size.
    x = layers.Conv1DTranspose(
        filters     = CONV_FILTERS,
        kernel_size = 3,
        strides     = CONV_STRIDE,
        padding     = "same",
        activation  = "relu",
        name        = "dec_conv1d_transpose",
    )(x)                                                # → (60, 64)

    # Step 9 ── Project each reconstructed timestep to the original feature space
    #   TimeDistributed applies the same Dense(n_features) weight matrix
    #   independently at every timestep, producing the (60, 4) reconstruction.
    #   This is the layer whose output is compared against the input to
    #   compute the MAE reconstruction error used for anomaly scoring.
    out = TimeDistributed(
        Dense(n_features),
        name = "output",
    )(x)                                                # → (60, 4)

    model = Model(inp, out, name="conv_transformer_autoencoder")
    model.compile(optimizer="adam", loss="mae")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# CableFaultDetector
# ═════════════════════════════════════════════════════════════════════════════

class CableFaultDetector:
    """
    Undersea cable fault detector — Conv-Transformer Hybrid backend.

    Strict drop-in replacement for the LSTM Autoencoder version.
    The public interface is byte-for-byte identical so dashboard.py and
    evaluate.py require zero changes.

    Public API
    ──────────
    .train(df, use_optimal_threshold=False)
    .predict(df)               → augmented DataFrame (same columns as before)
    .evaluate(result)          → {"report", "roc_auc", "threshold"}
    .reconstruction_errors_per_feature(df)   → ndarray (n_windows, n_features)
    .save(path)
    .load(path)
    ._scale(df, fit=False)     → scaled ndarray
    """

    def __init__(self):
        self.scaler    : MinMaxScaler = MinMaxScaler()
        self.model     : Model | None = None
        self.threshold : float | None = None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _scale(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        MinMax-scale the FEATURES columns of df.

        Parameters
        ----------
        df  : input DataFrame (must contain all FEATURES columns)
        fit : if True, fit the scaler on this data before transforming.
              Only set True once, on the training partition.

        Returns
        -------
        float32 ndarray of shape (len(df), len(FEATURES))
        """
        values = df[FEATURES].values.astype(np.float32)
        if fit:
            return self.scaler.fit_transform(values)
        # Warn if inference data falls outside the training distribution
        check_scaler_bounds(values, self.scaler)
        return self.scaler.transform(values)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, use_optimal_threshold: bool = False) -> None:
        """
        Train exclusively on normal (label == 0) samples.

        The normal pool is split into two non-overlapping partitions:
          • Training partition   (1 − THRESHOLD_VAL_SPLIT) of normal rows
              → fed into model.fit()
          • Calibration partition  THRESHOLD_VAL_SPLIT of normal rows
              → held out; used ONLY to set the anomaly threshold

        This separation prevents the threshold from being calibrated on data
        the model has already memorised, giving an honest anomaly boundary.

        Parameters
        ----------
        df : DataFrame with a 'label' column (0 = normal, 1 = fault).
             Must contain all FEATURES columns.
        use_optimal_threshold : if True, sweep candidate thresholds on the
             calibration split and choose the F1-maximising value.
             Defaults to False → THRESHOLD_PCT-th percentile of cal. errors.
        """
        normal_df = df[df["label"] == 0].reset_index(drop=True)
        n         = len(normal_df)
        split_at  = int(n * (1 - THRESHOLD_VAL_SPLIT))

        train_df  = normal_df.iloc[:split_at]
        cal_df    = normal_df.iloc[split_at:]

        log.info(
            "Training partition: %d normal samples | "
            "Calibration partition: %d normal samples.",
            len(train_df), len(cal_df),
        )

        # ── Scale ──────────────────────────────────────────────────────────────
        scaled_train = self._scale(train_df, fit=True)   # fits the scaler
        scaled_cal   = self._scale(cal_df,   fit=False)

        # ── Sliding-window segmentation ────────────────────────────────────────
        X_train = make_sequences(scaled_train, SEQ_LEN)  # (n_win, SEQ_LEN, 4)
        X_cal   = make_sequences(scaled_cal,   SEQ_LEN)

        # ── Build model ────────────────────────────────────────────────────────
        self.model = build_conv_transformer_autoencoder(SEQ_LEN, len(FEATURES))
        self.model.summary(print_fn=log.debug)

        # ── Callbacks ──────────────────────────────────────────────────────────
        callbacks = [
            EarlyStopping(
                monitor              = "val_loss",
                patience             = 7,
                restore_best_weights = True,
                verbose              = 0,
            ),
            ReduceLROnPlateau(
                monitor  = "val_loss",
                factor   = 0.5,
                patience = 3,
                min_lr   = 1e-5,
                verbose  = 0,
            ),
        ]

        # ── Fit ────────────────────────────────────────────────────────────────
        # Autoencoder training: input == target (reconstruct the input)
        self.model.fit(
            X_train, X_train,
            epochs           = EPOCHS,
            batch_size       = BATCH_SIZE,
            validation_split = 0.10,
            callbacks        = callbacks,
            verbose          = 1,
        )

        # ── Threshold calibration on held-out normal data ──────────────────────
        X_cal_pred = self.model.predict(X_cal, verbose=0)

        # Per-window scalar MAE: mean over (time, features) axes
        cal_errors = np.mean(np.abs(X_cal - X_cal_pred), axis=(1, 2))  # (n_cal_windows,)

        if use_optimal_threshold:
            # Calibration split is entirely normal (label == 0)
            cal_labels     = np.zeros(len(cal_errors), dtype=int)
            self.threshold = find_optimal_threshold(cal_errors, cal_labels)
        else:
            self.threshold = float(np.percentile(cal_errors, THRESHOLD_PCT))

        log.info(
            "Anomaly threshold → %.6f  (p%d of held-out normal reconstruction errors).",
            self.threshold, THRESHOLD_PCT,
        )

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference over the full df and return an augmented copy.

        Added columns (identical names to the LSTM version)
        ────────────────────────────────────────────────────
        anomaly_score   : float  — per-window MAE reconstruction error.
                          NaN for the first SEQ_LEN rows (no full window).
        predicted_label : int    — 1 if score > threshold, else 0.
                          0 for NaN rows.
        err_<feature>   : float  — per-sensor MAE for XAI attribution.
                          One column per entry in FEATURES.
                          NaN for the first SEQ_LEN rows.

        The NaN-padding of the first SEQ_LEN rows matches the LSTM version's
        behaviour exactly, so evaluate.py's dropna() call works unchanged.

        Parameters
        ----------
        df : DataFrame with all FEATURES columns (does not need 'label').

        Returns
        -------
        pd.DataFrame  (copy of df with the columns above appended)
        """
        scaled = self._scale(df)
        X      = make_sequences(scaled, SEQ_LEN)    # (n_windows, SEQ_LEN, n_feat)
        X_pred = self.model.predict(X, verbose=0)   # same shape

        # ── Reconstruction errors ──────────────────────────────────────────────
        # Average over the time axis (axis=1) to get (n_windows, n_features)
        # — i.e. one scalar error per feature per window, used for XAI.
        per_feature_errors = np.mean(np.abs(X - X_pred), axis=1)   # (n_win, n_feat)
        # Scalar window-level score: mean over features
        errors = np.mean(per_feature_errors, axis=1)                # (n_win,)

        # ── Pad the first SEQ_LEN rows with NaN ───────────────────────────────
        # There are no complete sliding windows for these rows, so their score
        # is undefined.  NaN is preserved so downstream code can dropna().
        pad    = np.full(SEQ_LEN, np.nan)
        scores = np.concatenate([pad, errors])                      # (len(df),)

        # ── Build result DataFrame ─────────────────────────────────────────────
        result                    = df.copy().reset_index(drop=True)
        result["anomaly_score"]   = scores
        result["predicted_label"] = (scores > self.threshold).astype(int)
        # NaN rows must be labelled 0, not 1 (threshold comparison of NaN is False
        # in numpy, but be explicit to avoid any edge-case surprises)
        result.loc[result["anomaly_score"].isna(), "predicted_label"] = 0

        # ── Per-feature error columns (XAI) ───────────────────────────────────
        # These columns let the dashboard and evaluate.py show which sensor
        # contributed most to each anomaly score — identical contract to the
        # LSTM version's err_* columns.
        for i, feature in enumerate(FEATURES):
            feat_pad              = np.full(SEQ_LEN, np.nan)
            feat_scores           = np.concatenate([feat_pad, per_feature_errors[:, i]])
            result[f"err_{feature}"] = feat_scores

        return result

    # ── evaluate ──────────────────────────────────────────────────────────────

    def evaluate(self, result: pd.DataFrame) -> dict:
        """
        Compute classification metrics against ground-truth labels.

        Requires the 'label' column to be present in result (0 = normal,
        1 = fault).  Drops NaN anomaly_score rows before evaluation, exactly
        as the LSTM version does.

        Returns
        -------
        dict with keys: 'report' (sklearn dict), 'roc_auc' (float),
                        'threshold' (float)
        """
        valid     = result.dropna(subset=["anomaly_score"])
        y_true    = valid["label"].values
        y_pred    = valid["predicted_label"].values.astype(int)
        scores    = valid["anomaly_score"].values

        report = classification_report(
            y_true, y_pred,
            target_names = ["normal", "fault"],
            output_dict  = True,
        )
        auc_score = roc_auc_score(y_true, scores)

        log.info(
            "\n%s",
            classification_report(y_true, y_pred, target_names=["normal", "fault"]),
        )
        log.info("ROC-AUC: %.4f", auc_score)
        return {
            "report"    : report,
            "roc_auc"   : auc_score,
            "threshold" : self.threshold,
        }

    # ── per-feature reconstruction error ──────────────────────────────────────

    def reconstruction_errors_per_feature(
        self, df: pd.DataFrame
    ) -> np.ndarray:
        """
        Return per-sensor reconstruction error for every sliding window.

        Used by evaluate.py's per-sensor error subplot and by the dashboard's
        XAI attribution panel.

        Parameters
        ----------
        df : DataFrame with all FEATURES columns.

        Returns
        -------
        ndarray of shape (n_windows, n_features)
            n_windows = len(df) − SEQ_LEN
        """
        scaled = self._scale(df)
        X      = make_sequences(scaled, SEQ_LEN)
        X_pred = self.model.predict(X, verbose=0)
        # Mean over the time axis → (n_windows, n_features)
        return np.mean(np.abs(X - X_pred), axis=1)

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str = "saved_model") -> None:
        """
        Persist model weights, MinMaxScaler, and detection threshold to disk.

        Custom layers (SinePositionalEncoding, TransformerEncoderBlock) are
        serialised automatically via their get_config() implementations.
        The .keras format includes the full architecture so load() does not
        need a separate model-building call.

        Parameters
        ----------
        path : directory to write into.  Created if it does not exist.
        """
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/conv_transformer_ae.keras")
        with open(f"{path}/scaler.pkl",    "wb") as f:
            pickle.dump(self.scaler,    f)
        with open(f"{path}/threshold.pkl", "wb") as f:
            pickle.dump(self.threshold, f)
        log.info("Model artifacts saved to %s/", path)

    def load(self, path: str = "saved_model") -> None:
        """
        Restore model, scaler, and threshold from disk.

        The custom_objects map tells Keras how to re-instantiate the custom
        layers from their stored configs.  Without it, load_model() would
        raise an 'Unknown layer' error.

        Parameters
        ----------
        path : directory containing conv_transformer_ae.keras,
               scaler.pkl, and threshold.pkl.
        """
        custom_objects = {
            "SinePositionalEncoding"  : SinePositionalEncoding,
            "TransformerEncoderBlock" : TransformerEncoderBlock,
        }
        self.model = tf.keras.models.load_model(
            f"{path}/conv_transformer_ae.keras",
            custom_objects = custom_objects,
        )
        with open(f"{path}/scaler.pkl",    "rb") as f:
            self.scaler    = pickle.load(f)
        with open(f"{path}/threshold.pkl", "rb") as f:
            self.threshold = pickle.load(f)
        log.info("Model artifacts loaded from %s/", path)


# ═════════════════════════════════════════════════════════════════════════════
# CLI smoke-test  (python model.py)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train the Conv-Transformer Autoencoder on a labelled cable dataset."
    )
    parser.add_argument(
        "--dataset",
        type    = str,
        default = "datasets/azure_pdm.csv",
        help    = "Path to the labelled CSV dataset (default: datasets/azure_pdm.csv).",
    )
    parser.add_argument(
        "--use-optimal-threshold",
        action  = "store_true",
        help    = "Sweep F1-maximising threshold on calibration split instead of percentile.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        log.error(
            "Dataset not found at '%s'. "
            "Run fetch_azure_pdm.py or fetch_dataset.py first.",
            args.dataset,
        )
        raise SystemExit(1)

    log.info("Loading dataset: %s", args.dataset)
    df = pd.read_csv(args.dataset)
    log.info("Rows: %d | Fault rate: %.1f%%", len(df), df["label"].mean() * 100)

    detector = CableFaultDetector()

    log.info("Building and training Conv-Transformer Autoencoder…")
    detector.train(df, use_optimal_threshold=args.use_optimal_threshold)

    log.info("Running inference on full dataset…")
    result = detector.predict(df)

    metrics = detector.evaluate(result)
    log.info("ROC-AUC: %.4f | threshold: %.6f", metrics["roc_auc"], metrics["threshold"])

    detector.save()
    log.info("Done. Artifacts saved to saved_model/.")
