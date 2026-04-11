"""
model_registry.py
Version control system for cable fault detection models.

Tracks model versions, metrics history, and provides easy rollback capability.
"""

import json
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "model_registry"
CURRENT_VERSION_FILE = "current_version"
VERSIONS_DIR = "versions"


class ModelRegistry:
    """
    Version control for trained models.
    
    Each version stores:
      - Model artifacts (keras file, scaler, threshold)
      - Training metadata (date, dataset, metrics)
      - Performance history
      
    Usage:
      registry = ModelRegistry()
      registry.save_version(detector, metrics={"roc_auc": 0.95})
      
      # List all versions
      for version in registry.list_versions():
          print(version)
      
      # Rollback to previous
      detector = registry.load_version("v2")
    """

    def __init__(self, path: str = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)
        self.versions_dir = self.path / VERSIONS_DIR
        self.current_version_file = self.path / CURRENT_VERSION_FILE
        
        self._ensure_directories()

    def _ensure_directories(self):
        """Create registry directories if they don't exist."""
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def _get_current_version_name(self) -> str:
        """Read the current version name."""
        if self.current_version_file.exists():
            return self.current_version_file.read_text().strip()
        return "v1"

    def _increment_version(self, version: str) -> str:
        """Increment version string (v1 -> v2)."""
        if version.startswith("v"):
            num = int(version[1:]) + 1
            return f"v{num}"
        return "v1"

    def list_versions(self) -> list[dict]:
        """List all registered versions."""
        versions = []
        
        if not self.versions_dir.exists():
            return versions
            
        for version_dir in sorted(self.versions_dir.iterdir()):
            if version_dir.is_dir():
                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    versions.append({
                        "name": version_dir.name,
                        "date": metadata.get("date"),
                        "roc_auc": metadata.get("metrics", {}).get("roc_auc"),
                        "threshold": metadata.get("threshold"),
                    })
                else:
                    versions.append({
                        "name": version_dir.name,
                        "date": None,
                        "roc_auc": None,
                        "threshold": None,
                    })
        
        return versions

    def save_version(
        self,
        detector,
        metrics: Optional[dict] = None,
        dataset_info: Optional[dict] = None,
    ) -> str:
        """
        Save current model as a new version.
        
        Parameters
        ----------
        detector : CableFaultDetector
            Trained model to save
        metrics : dict, optional
            Classification metrics (precision, recall, f1, roc_auc)
        dataset_info : dict, optional
            Info about training data (n_samples, n_faults, seed)
            
        Returns
        -------
        str : Version name (e.g., "v2")
        """
        current = self._get_current_version_name()
        new_version = self._increment_version(current)
        version_dir = self.versions_dir / new_version
        
        version_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = version_dir / "conv_transformer_ae.keras"
        detector.model.save(model_path)
        
        import pickle
        with open(version_dir / "scaler.pkl", "wb") as f:
            pickle.dump(detector.scaler, f)
        with open(version_dir / "threshold.pkl", "wb") as f:
            pickle.dump(detector.threshold, f)
        
        metadata = {
            "version": new_version,
            "date": datetime.now().isoformat(),
            "metrics": metrics or {},
            "threshold": detector.threshold,
            "dataset": dataset_info or {},
        }
        
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        with open(self.current_version_file, "w") as f:
            f.write(new_version)
        
        log.info(f"Saved model as {new_version}")
        return new_version

    def load_version(self, version: Optional[str] = None):
        """
        Load a specific model version.
        
        Parameters
        ----------
        version : str, optional
            Version name (e.g., "v2"). If None, loads current.
            
        Returns
        -------
        CableFaultDetector
        """
        from model import CableFaultDetector
        
        if version is None:
            version = self._get_current_version_name()
        
        version_dir = self.versions_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} not found")
        
        detector = CableFaultDetector()
        detector.path = str(version_dir)
        detector.load(str(version_dir))
        
        log.info(f"Loaded model version {version}")
        return detector

    def rollback(self, steps: int = 1):
        """
        Rollback to a previous version.
        
        Parameters
        ----------
        steps : int
            Number of versions to rollback (default: 1)
            
        Returns
        -------
        str : New current version name
        """
        current = self._get_current_version_name()
        current_num = int(current[1:])
        
        if current_num - steps < 1:
            raise ValueError("Cannot rollback beyond v1")
        
        new_version = f"v{current_num - steps}"
        
        with open(self.current_version_file, "w") as f:
            f.write(new_version)
        
        log.info(f"Rolled back to {new_version}")
        return new_version

    def get_current_metrics(self) -> dict:
        """Get metrics for the current version."""
        version = self._get_current_version_name()
        version_dir = self.versions_dir / version
        metadata_file = version_dir / "metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                return json.load(f)
        return {}


def create_registry() -> ModelRegistry:
    """Factory function to create a model registry."""
    return ModelRegistry()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    registry = create_registry()
    
    print("Model Versions:")
    for v in registry.list_versions():
        print(f"  {v}")
    
    if registry.list_versions():
        print(f"\nCurrent: {registry._get_current_version_name()}")
        print(f"Metrics: {registry.get_current_metrics()}")