"""
Tests for reports/generator.py - validates report generation.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from reports.generator import ReportGenerator


class TestCSVGeneration:
    """Verify CSV report generation."""

    def test_generate_csv_empty_log(self, tmp_path):
        output_path = tmp_path / "report.csv"
        ReportGenerator.generate_csv([], str(output_path))
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "timestamp" in content
        assert "fault_type" in content

    def test_generate_csv_with_faults(self, tmp_path):
        fault_log = [
            {
                "timestamp": "2025-01-01 12:00:00",
                "fault_type": "cable_cut",
                "severity": "High",
                "estimated_distance_m": 125.5,
            },
            {
                "timestamp": "2025-01-01 12:05:00",
                "fault_type": "overheating",
                "severity": "Medium",
                "estimated_distance_m": 340.2,
            },
        ]
        output_path = tmp_path / "report.csv"
        ReportGenerator.generate_csv(fault_log, str(output_path))
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "cable_cut" in content
        assert "overheating" in content


class TestPDFGeneration:
    """Verify PDF report generation."""

    def test_generate_pdf_creates_file(self, tmp_path):
        metadata = {
            "deployment_id": "TEST-001",
            "source": "test",
            "model_version": "1.0.0",
            "threshold": 0.05,
            "total_samples": 1000,
        }
        fault_log = []
        
        output_path = tmp_path / "report.pdf"
        ReportGenerator.generate_pdf(fault_log, metadata, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_pdf_with_faults(self, tmp_path):
        metadata = {
            "deployment_id": "TEST-002",
            "source": "test",
            "model_version": "1.0.0",
            "threshold": 0.05,
            "total_samples": 1000,
        }
        fault_log = [
            {
                "timestamp": "2025-01-01 12:00:00",
                "fault_type": "cable_cut",
                "severity": "High",
                "estimated_distance_m": 125.5,
            },
            {
                "timestamp": "2025-01-01 12:05:00",
                "fault_type": "overheating",
                "severity": "Medium",
                "estimated_distance_m": 340.2,
            },
        ]
        
        output_path = tmp_path / "report.pdf"
        ReportGenerator.generate_pdf(fault_log, metadata, str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_pdf_contains_fault_data(self, tmp_path):
        metadata = {
            "deployment_id": "TEST-003",
            "source": "test",
            "model_version": "1.0.0",
            "threshold": 0.05,
            "total_samples": 100,
        }
        fault_log = [
            {
                "timestamp": "2025-01-01 12:00:00",
                "fault_type": "cable_cut",
                "severity": "Critical",
                "estimated_distance_m": 150.0,
            },
        ]
        
        output_path = tmp_path / "report.pdf"
        ReportGenerator.generate_pdf(fault_log, metadata, str(output_path))
        
        assert output_path.exists()
        
        content = output_path.read_bytes()
        assert len(content) > 1000