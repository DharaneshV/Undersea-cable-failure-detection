"""Dump all variable shapes from the checkpoint to understand its exact architecture."""
import zipfile, json, sys

path = "checkpoints/best_model.keras"
with zipfile.ZipFile(path, "r") as z:
    names = z.namelist()
    # Find all .weights.h5 or vars
    print("Files in checkpoint:")
    for n in names:
        print(f"  {n}")
    
    # Read the variables file
    if "model.weights.h5" in names:
        print("\n=> Has model.weights.h5")
    
    # Try reading vars config
    for n in names:
        if "config" in n.lower() or "vars" in n.lower():
            try:
                content = z.read(n)
                if len(content) < 50000:  # only print small files
                    print(f"\n=== {n} ===")
                    print(content.decode("utf-8", errors="replace")[:3000])
            except Exception as e:
                print(f"  Error reading {n}: {e}")
