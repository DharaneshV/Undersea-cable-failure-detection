"""
train_universal.py
Universal multi-modal training pipeline.

Loads all available datasets, enriches them with cable_distance_norm and
cable_type if missing, then trains the enhanced Conv-Transformer model.
"""

import logging
import pickle
import os

import numpy as np
import pandas as pd

from config import CABLE_DOMAIN_NAMES, CABLE_LENGTH, CLASS_WEIGHTS
from model import CableFaultDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _enrich(df: pd.DataFrame, domain_id: int) -> pd.DataFrame:
    """Add cable_distance_norm, cable_domain_id and cable_type if missing."""
    if "cable_domain_id" not in df.columns:
        df["cable_domain_id"] = domain_id

    if "cable_type" not in df.columns:
        df["cable_type"] = df["cable_domain_id"].map(
            lambda d: CABLE_DOMAIN_NAMES.get(int(d), f"Domain-{int(d)}")
        )

    if "cable_distance_norm" not in df.columns:
        # Fault rows get a random normalised position along the cable;
        # normal rows stay at 0.0 (no active fault located).
        rng = np.random.default_rng(seed=42)
        dist = np.where(
            df["label"].values == 1,
            rng.uniform(0.0, 1.0, size=len(df)),
            0.0,
        )
        df["cable_distance_norm"] = dist.astype(np.float32)

    return df


def _load(path: str, domain_id: int) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df = _enrich(df, domain_id)
        log.info("Loaded %-45s  %6d rows  (domain=%d — %s)",
                 path, len(df), domain_id, CABLE_DOMAIN_NAMES.get(domain_id, "?"))
        return df
    except FileNotFoundError:
        log.warning("Dataset not found, skipping: %s", path)
        return pd.DataFrame()


# ── main ──────────────────────────────────────────────────────────────────────

def main(resume: bool = False):
    print("\n" + "=" * 60)
    print("  Universal Multi-Modal Training Pipeline")
    print("  Cable length: {:,.0f} m ({:,.0f} km)".format(
        CABLE_LENGTH, CABLE_LENGTH / 1000))
    print("=" * 60 + "\n")

    # ── load all available datasets ──────────────────────────────────────────
    datasets = [
        # (csv_path,                             domain_id)
        ("datasets/realistic_data.csv",          0),   # Electrical (Copper)
        ("datasets/optical_240km.csv",           1),   # Optical (Fibre)
        ("datasets/synthetic_cable_50k.csv",     0),   # Electrical (Copper)
        ("datasets/azure_pdm.csv",               0),   # Electrical (Copper) — industrial proxy
        ("datasets/industrial_pump.csv",         0),   # Electrical (Copper) — industrial proxy
    ]

    frames = [_load(path, did) for path, did in datasets]
    df_combined = pd.concat(
        [f for f in frames if len(f) > 0], ignore_index=True
    )

    if len(df_combined) == 0:
        print("ERROR: No datasets found — check the datasets/ directory.")
        return

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\nCombined dataset : {len(df_combined):,} rows")
    print(f"  Normal rows    : {(df_combined['label']==0).sum():,}")
    print(f"  Fault rows     : {(df_combined['label']==1).sum():,}")
    print(f"  Fault %        : {df_combined['label'].mean()*100:.1f}%\n")

    print("Cable type breakdown:")
    for ctype, grp in df_combined.groupby("cable_type"):
        faults = grp["label"].sum()
        print(f"  {ctype:<30}  {len(grp):>7,} rows  ({faults:>5,} faults)")
    print()

    print("Fault type counts:")
    print(df_combined["fault_type"].value_counts().to_string())
    print()

    # ── train ────────────────────────────────────────────────────────────────
    print("Training Enhanced Conv-Transformer model...")
    detector = CableFaultDetector()
    detector.train(df_combined, resume=resume)

    # ── evaluate on training data (quick sanity check) ───────────────────────
    result = detector.predict(df_combined)
    metrics = detector.evaluate(result)

    print("\n" + "=" * 60)
    print("  Training Complete — Results")
    print("=" * 60)
    print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"  Threshold : {metrics['threshold']:.6f}")
    if "report" in metrics:
        r = metrics["report"]
        fault_metrics = r.get('fault', r.get('1', {}))
        print(f"  Precision : {fault_metrics.get('precision', 0):.4f}")
        print(f"  Recall    : {fault_metrics.get('recall',    0):.4f}")
        print(f"  F1        : {fault_metrics.get('f1-score',  0):.4f}")

    # Print cable_type column values to confirm they're human-readable
    if "cable_type" in result.columns:
        print("\nCable types seen in predictions:")
        print(result["cable_type"].value_counts().to_string())

    # ── save ─────────────────────────────────────────────────────────────────
    detector.save()
    with open("saved_model/roc_auc.pkl", "wb") as f:
        pickle.dump(metrics["roc_auc"], f)

    print("\nModel saved to saved_model/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from existing checkpoint")
    args = parser.parse_args()
    main(resume=args.resume)
