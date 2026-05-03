"""
Tests for model.py - validates model behavior and anomaly detection.
Uses pre-trained model for most tests to avoid slow training.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model import CableFaultDetector, build_conv_transformer_autoencoder
from config import FEATURES, SEQ_LEN, NUM_CLASSES


class TestModelArchitecture:
    """Verify model builds correctly."""

    def test_build_model_returns_model(self):
        model = build_conv_transformer_autoencoder(SEQ_LEN, len(FEATURES), NUM_CLASSES)
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_model_has_reconstruction_output(self):
        model = build_conv_transformer_autoencoder(SEQ_LEN, len(FEATURES), NUM_CLASSES)
        assert 'reconstruction' in model.output_names

    def test_model_has_classification_output(self):
        model = build_conv_transformer_autoencoder(SEQ_LEN, len(FEATURES), NUM_CLASSES)
        assert 'classification' in model.output_names


class TestDetectorBehavior:
    """Verify detector initialization and basic operations."""

    def test_detector_initializes_with_defaults(self):
        detector = CableFaultDetector()
        assert detector.scaler is not None
        assert detector.model is None
        assert detector.threshold is None

    def test_scale_uses_fit_mode(self):
        detector = CableFaultDetector()
        df = pd.DataFrame({f: [1, 2, 3] for f in FEATURES})
        result = detector._scale(df, fit=True)
        assert result.shape == (3, len(FEATURES))

    def test_scale_uses_transform_mode(self):
        detector = CableFaultDetector()
        df = pd.DataFrame({f: [1, 2, 3] for f in FEATURES})
        detector.scaler.fit(df.values)
        result = detector._scale(df, fit=False)
        assert result.shape == (3, len(FEATURES))


class TestPredictOutput:
    """Verify predict() output format using pre-trained model."""

    @pytest.fixture
    def detector(self):
        """Load pre-trained detector."""
        detector = CableFaultDetector()
        if os.path.exists("saved_model"):
            detector.load()
        return detector

    def _create_dummy_df(self, n):
        return pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms"),
            "voltage": np.random.normal(220, 1.2, n),
            "current": np.random.normal(5, 0.2, n),
            "temperature": np.random.normal(18, 0.5, n),
            "vibration": np.random.normal(0, 0.05, n),
            "acoustic_strain": np.random.normal(0, 0.1, n),
            "optical_osnr": np.random.normal(20, 1, n),
            "optical_ber": np.random.normal(0, 0.1, n),
            "optical_power": np.random.normal(0, 0.1, n),
            "cable_distance_norm": np.random.uniform(0, 1, n),
            "cable_domain_id": [0] * n,
            "label": [0] * n,
            "fault_type": ["none"] * n,
        })

    def test_predict_returns_required_columns(self, detector):
        if detector.model is None:
            pytest.skip("No pre-trained model available")
        
        df = self._create_dummy_df(100)
        result = detector.predict(df)
        
        required_cols = ["anomaly_score", "predicted_label", "fault_diagnosis"]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_predict_has_feature_error_columns(self, detector):
        if detector.model is None:
            pytest.skip("No pre-trained model available")
        
        df = self._create_dummy_df(100)
        result = detector.predict(df)
        
        for feature in FEATURES:
            assert f"err_{feature}" in result.columns

    def test_predict_first_rows_have_nan_scores(self, detector):
        if detector.model is None:
            pytest.skip("No pre-trained model available")
        
        df = self._create_dummy_df(100)
        result = detector.predict(df)
        
        assert result["anomaly_score"].iloc[:SEQ_LEN].isna().all()
        assert not result["anomaly_score"].iloc[SEQ_LEN:].isna().all()


class TestPreTrainedModel:
    """Tests that work with the pre-trained model."""

    @pytest.fixture
    def detector(self):
        detector = CableFaultDetector()
        if os.path.exists("saved_model"):
            detector.load()
        return detector

    def test_model_loads_successfully(self, detector):
        """Verify pre-trained model can be loaded."""
        assert detector.model is not None
        assert detector.threshold is not None

    def test_threshold_is_positive(self, detector):
        if detector.threshold is None:
            pytest.skip("No threshold available")
        assert detector.threshold > 0

    def test_evaluate_returns_metrics(self, detector):
        if detector.model is None:
            pytest.skip("No pre-trained model available")
        
        n = 500
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="100ms"),
            "voltage": np.random.normal(220, 1.2, n),
            "current": np.random.normal(5, 0.2, n),
            "temperature": np.random.normal(18, 0.5, n),
            "vibration": np.random.normal(0, 0.05, n),
            "acoustic_strain": np.random.normal(0, 0.1, n),
            "optical_osnr": np.random.normal(20, 1, n),
            "optical_ber": np.random.normal(0, 0.1, n),
            "optical_power": np.random.normal(0, 0.1, n),
            "cable_distance_norm": np.zeros(n),
            "cable_domain_id": [0] * n,
            "label": [0] * n,
            "fault_type": ["none"] * n,
        })
        result = detector.predict(df)
        metrics = detector.evaluate(result)
        
        assert "report" in metrics
        assert "roc_auc" in metrics
        assert "threshold" in metrics