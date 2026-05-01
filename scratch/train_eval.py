"""Quick training + evaluation on simulator data to validate model improvements."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging, pickle
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from simulator import generate_dataset
from model import CableFaultDetector

print("Generating training data...")
df_train, _ = generate_dataset(n_seconds=600, fault_count=8, seed=42)
print(f"Dataset: {len(df_train)} samples, {(df_train['label']==1).sum()} faults")

detector = CableFaultDetector()
detector.train(df_train)

result = detector.predict(df_train)
metrics = detector.evaluate(result)
print(f"\n=== RESULTS ===")
print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
print(f"Threshold: {metrics['threshold']:.6f}")

detector.save()
with open("saved_model/roc_auc.pkl", "wb") as f:
    pickle.dump(metrics["roc_auc"], f)
print("Model saved.")
