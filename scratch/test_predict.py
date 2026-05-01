import sys
import os
sys.path.append("d:\\undersea cable failure detection")

import pandas as pd
from model import CableFaultDetector

print("Loading model...")
detector = CableFaultDetector()
detector.load()

print("Loading dataset...")
df_full = pd.read_csv("datasets/optical_240km.csv")

print("Predicting...")
try:
    res = detector.predict(df_full)
    print("Success! Length:", len(res))
except Exception as e:
    import traceback
    traceback.print_exc()
