"""
Tests for config.py - validates all configuration values.
"""

import pytest
from config import (
    SAMPLE_RATE, CABLE_LENGTH, SIGNAL_SPEED,
    NORMAL_PROFILES, FAULT_TYPES,
    FEATURES, SEQ_LEN, TRANSFORMER_HEADS, TRANSFORMER_FF_DIM,
    TRANSFORMER_BLOCKS, LSTM_UNITS, LATENT_UNITS, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, THRESHOLD_PCT, THRESHOLD_VAL_SPLIT,
    FAULT_COLORS, SENSOR_COLORS,
)


class TestConfigValues:
    """Verify configuration values are sensible."""

    def test_sample_rate_positive(self):
        assert SAMPLE_RATE > 0

    def test_cable_length_positive(self):
        assert CABLE_LENGTH > 0

    def test_signal_speed_physical(self):
        assert SIGNAL_SPEED > 1e6  # At least 1000 km/s
        assert SIGNAL_SPEED < 3e8  # Speed of light

    def test_normal_profiles_have_mean_std(self):
        for feature, (mean, std) in NORMAL_PROFILES.items():
            assert isinstance(mean, (int, float))
            assert isinstance(std, (int, float))
            assert std > 0

    def test_fault_types_valid(self):
        expected = {"cable_cut", "anchor_drag", "overheating", "insulation_failure"}
        assert set(FAULT_TYPES) == expected

    def test_features_valid(self):
        expected = {"voltage", "current", "temperature", "vibration"}
        assert set(FEATURES) == expected

    def test_seq_len_positive(self):
        assert SEQ_LEN > 0
        assert SEQ_LEN % 2 == 0  # Should be even for transformer downsampling

    def test_transformer_config_positive(self):
        assert TRANSFORMER_HEADS > 0
        assert TRANSFORMER_FF_DIM > 0
        assert TRANSFORMER_BLOCKS > 0

    def test_model_units_positive(self):
        assert LSTM_UNITS > 0
        assert LATENT_UNITS > 0
        assert LATENT_UNITS < LSTM_UNITS  # Bottleneck should be smaller

    def test_dropout_valid(self):
        assert 0 <= DROPOUT_RATE < 1

    def test_training_config_positive(self):
        assert EPOCHS > 0
        assert BATCH_SIZE > 0

    def test_threshold_percentile_valid(self):
        assert 0 < THRESHOLD_PCT < 100

    def test_threshold_split_valid(self):
        assert 0 < THRESHOLD_VAL_SPLIT < 1

    def test_fault_colors_valid(self):
        for fault_type, color in FAULT_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB

    def test_sensor_colors_valid(self):
        for feature, color in SENSOR_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7