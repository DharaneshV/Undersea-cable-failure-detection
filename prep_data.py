import pandas as pd
from simulator import generate_dataset
import os

print("Generating 10 minutes of labeled sensor data for training...")
df, _ = generate_dataset(n_seconds=600, fault_count=12, seed=1337)

os.makedirs("datasets", exist_ok=True)
df.to_csv("datasets/azure_pdm.csv", index=False)
print(f"Dataset saved: {len(df)} rows, fault rate {df['label'].mean()*100:.1f}%")
