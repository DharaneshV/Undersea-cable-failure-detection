"""
tests/test_core.py
Basic smoke-tests that catch the most common regressions.

Run with:  python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import SEQ_LEN, FEATURES, CABLE_LENGTH
from utils import make_sequences, check_scaler_bounds, find_optimal_threshold, ema
from simulator import generate_dataset


# ── utils ─────────────────────────────────────────────────────────────────────
class TestMakeSequences:
    def test_output_shape(self):
        data = np.random.rand(200, 4)
        seqs = make_sequences(data, SEQ_LEN)
        assert seqs.shape == (200 - SEQ_LEN, SEQ_LEN, 4)

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            make_sequences(np.random.rand(10, 4), SEQ_LEN)

    def test_values_preserved(self):
        data = np.arange(100 * 4, dtype=float).reshape(100, 4)
        seqs = make_sequences(data, 5)
        np.testing.assert_array_equal(seqs[0], data[:5])
        np.testing.assert_array_equal(seqs[1], data[1:6])


class TestEMA:
    def test_shape(self):
        v   = np.random.rand(50)
        out = ema(v, alpha=0.1)
        assert out.shape == v.shape

    def test_first_value_unchanged(self):
        v = np.array([3.0, 1.0, 2.0])
        assert ema(v)[0] == 3.0

    def test_smoothed_less_than_raw(self):
        # High-variance signal — EMA should have lower std
        rng = np.random.RandomState(0)
        v   = rng.randn(200)
        assert ema(v, alpha=0.05).std() < v.std()


class TestFindOptimalThreshold:
    def test_returns_float(self):
        scores = np.random.rand(500)
        labels = (scores > 0.7).astype(int)
        thr    = find_optimal_threshold(scores, labels)
        assert isinstance(thr, float)

    def test_thr_in_score_range(self):
        scores = np.random.rand(500)
        labels = (scores > 0.6).astype(int)
        thr    = find_optimal_threshold(scores, labels)
        assert scores.min() <= thr <= scores.max()


# ── simulator ─────────────────────────────────────────────────────────────────
class TestGenerateDataset:
    def setup_method(self):
        self.df, self.log = generate_dataset(n_seconds=60, fault_count=3, seed=7)

    def test_columns(self):
        required = {"timestamp", "voltage", "current", "temperature",
                    "vibration", "label", "fault_type"}
        assert required.issubset(set(self.df.columns))

    def test_length(self):
        assert len(self.df) == 60 * 10      # 10 Hz × 60 s

    def test_label_binary(self):
        assert set(self.df["label"].unique()).issubset({0, 1})

    def test_fault_count(self):
        # At most fault_count faults (some may be skipped if overlap can't be resolved)
        assert len(self.log) <= 3

    def test_fault_distance_in_range(self):
        for fl in self.log:
            assert 0 <= fl["fault_distance_m"] <= CABLE_LENGTH

    def test_fault_windows_non_overlapping(self):
        windows = [(fl["start_sample"], fl["start_sample"] + fl["duration_samples"])
                   for fl in self.log]
        for i, (s1, e1) in enumerate(windows):
            for j, (s2, e2) in enumerate(windows):
                if i != j:
                    assert e1 <= s2 or e2 <= s1, \
                        f"Faults {i} and {j} overlap: [{s1},{e1}) ∩ [{s2},{e2})"

    def test_fault_duration_minimum(self):
        min_dur = 2 * SEQ_LEN
        for fl in self.log:
            assert fl["duration_samples"] >= min_dur, \
                f"Fault duration {fl['duration_samples']} < minimum {min_dur}"

    def test_no_faults(self):
        df0, log0 = generate_dataset(n_seconds=30, fault_count=0, seed=1)
        assert df0["label"].sum() == 0
        assert len(log0) == 0

    def test_seed_reproducibility(self):
        df1, _ = generate_dataset(n_seconds=60, fault_count=2, seed=99)
        df2, _ = generate_dataset(n_seconds=60, fault_count=2, seed=99)
        pd.testing.assert_frame_equal(
            df1.drop(columns=["timestamp"]),
            df2.drop(columns=["timestamp"]),
        )

    def test_different_seeds_differ(self):
        df1, _ = generate_dataset(n_seconds=60, fault_count=2, seed=1)
        df2, _ = generate_dataset(n_seconds=60, fault_count=2, seed=2)
        assert not df1["voltage"].equals(df2["voltage"])


# ── scaler bounds check (integration) ────────────────────────────────────────
class TestScalerBounds:
    def test_warn_on_out_of_range(self, caplog):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        train  = np.random.rand(100, 4)
        scaler.fit(train)
        # Pass data with values far outside training range
        oob = train + 100.0
        import logging
        with caplog.at_level(logging.WARNING):
            check_scaler_bounds(oob, scaler)
        assert "outside training range" in caplog.text

    def test_no_warn_in_range(self, caplog):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        train  = np.random.rand(100, 4)
        scaler.fit(train)
        import logging
        with caplog.at_level(logging.WARNING):
            check_scaler_bounds(train * 0.5 + 0.25, scaler)
        assert "outside training range" not in caplog.text
