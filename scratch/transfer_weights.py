"""
Direct weight loading — bypasses Keras config mismatch by:
1. Building a fresh model with the EXACT same architecture as the checkpoint
2. Loading weights via set_weights() after matching layer by layer
"""
import sys, os, zipfile, json
sys.path.insert(0, os.getcwd())

import numpy as np
import tensorflow as tf
from model import build_conv_transformer_autoencoder, SinePositionalEncoding, TransformerEncoderBlock

SEQ_LEN = 60
N_FEATURES = 19  # 9 sensor + 10 domain one-hot (from checkpoint build_config)
NUM_CLASSES = 4

print("Building fresh model with checkpoint architecture...")
model = build_conv_transformer_autoencoder(SEQ_LEN, N_FEATURES, NUM_CLASSES)
# Force build
dummy = tf.zeros([1, SEQ_LEN, N_FEATURES])
_ = model(dummy, training=False)
print(f"Fresh model params: {model.count_params():,}")

print("\nLoading weights from checkpoint...")
try:
    model.load_weights("checkpoints/best_model.keras")
    print("SUCCESS via load_weights!")
except Exception as e:
    print(f"load_weights failed: {e}")
    # Try loading via custom_objects
    custom = {
        "SinePositionalEncoding": SinePositionalEncoding,
        "TransformerEncoderBlock": TransformerEncoderBlock,
    }
    try:
        ckpt_model = tf.keras.models.load_model(
            "checkpoints/best_model.keras",
            custom_objects=custom,
        )
        print("load_model succeeded, transferring weights...")
        model.set_weights(ckpt_model.get_weights())
        print("Weights transferred!")
    except Exception as e2:
        print(f"load_model also failed: {e2}")
        sys.exit(1)

print("\nSaving loadable model to saved_model/...")
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/conv_transformer_multitask.keras")
print("Done. Model saved for resume training.")
