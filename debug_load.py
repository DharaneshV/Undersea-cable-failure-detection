import os, sys
import tensorflow as tf
from model import SinePositionalEncoding, TransformerEncoderBlock

# Add current dir to sys.path
sys.path.insert(0, os.getcwd())

custom = {
    "SinePositionalEncoding": SinePositionalEncoding, 
    "TransformerEncoderBlock": TransformerEncoderBlock
}

path = "checkpoints/best_model.keras"
print(f"Attempting to load: {path}")
try:
    model = tf.keras.models.load_model(path, custom_objects=custom)
    print("Success!")
    model.summary()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
