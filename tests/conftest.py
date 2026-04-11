"""
Pytest configuration and shared fixtures.
"""

import os
import sys
import logging
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import FEATURES, SEQ_LEN
from simulator import generate_dataset
from utils import make_sequences


@pytest.fixture
def normal_df():
    """Generate a normal (no fault) dataset."""
    df, _ = generate_dataset(n_seconds=60, fault_count=0, seed=42)
    return df


@pytest.fixture
def faulty_df():
    """Generate a dataset with faults."""
    df, fault_log = generate_dataset(n_seconds=120, fault_count=5, seed=123)
    return df, fault_log


@pytest.fixture
def sample_data():
    """Generate sample data arrays."""
    np.random.seed(42)
    return {
        "normal": np.random.rand(200, 4),
        "faulty": np.random.rand(200, 4) + np.concatenate([
            np.zeros(100),
            np.ones(100) * 2
        ])[:, np.newaxis],
        "labels": np.concatenate([np.zeros(100), np.ones(100)]),
    }


@pytest.fixture
def temp_dataset(tmp_path):
    """Create a temporary dataset CSV for testing."""
    df, fault_log = generate_dataset(n_seconds=30, fault_count=2, seed=1)
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    log_path = tmp_path / "test_data_fault_log.csv"
    pd.DataFrame(fault_log).to_csv(log_path, index=False)
    
    return str(csv_path), fault_log


@pytest.fixture
def mock_anomaly_scores():
    """Generate mock anomaly scores for testing."""
    np.random.seed(42)
    n = 1000
    scores = np.concatenate([
        np.random.exponential(0.02, n // 2),
        np.random.exponential(0.15, n // 2) + 0.1
    ])
    labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    return scores, labels


@pytest.fixture
def sample_sensor_reading():
    """Generate a single valid sensor reading."""
    return {
        "voltage": 220.0,
        "current": 5.0,
        "temperature": 18.0,
        "vibration": 0.1,
    }


@pytest.fixture
def short_sequence():
    """Generate a short sequence of sensor readings for testing."""
    np.random.seed(42)
    n = SEQ_LEN + 10
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms"),
        "voltage": np.random.normal(220, 1.2, n),
        "current": np.random.normal(5, 0.2, n),
        "temperature": np.random.normal(18, 0.5, n),
        "vibration": np.random.normal(0, 0.05, n),
        "label": 0,
        "fault_type": "none",
    })


@pytest.fixture(autouse=True)
def setup_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    yield