"""
Drift Detection Tests
"""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drift import (
    calculate_psi,
    get_drift_status,
    DriftDetector,
    DriftResult
)


class TestPSICalculation:
    """Tests for PSI calculation."""
    
    def test_identical_distributions_zero_psi(self):
        """Identical distributions should have PSI near 0."""
        data = np.random.normal(0, 1, 1000)
        psi, _, _, _ = calculate_psi(data, data)
        
        assert psi < 0.01
    
    def test_similar_distributions_low_psi(self):
        """Similar distributions should have low PSI."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.1, 1, 1000)  # Slight shift
        
        psi, _, _, _ = calculate_psi(baseline, current)
        
        assert psi < 0.1  # Should be in "OK" range
    
    def test_shifted_distributions_high_psi(self):
        """Shifted distributions should have high PSI."""
        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(2, 1, 1000)  # Large shift
        
        psi, _, _, _ = calculate_psi(baseline, current)
        
        assert psi > 0.2  # Should trigger alert
    
    def test_psi_returns_distributions(self):
        """PSI calculation should return distribution info."""
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        psi, baseline_dist, current_dist, bin_edges = calculate_psi(baseline, current)
        
        assert len(baseline_dist) == 10  # Default bins
        assert len(current_dist) == 10
        assert len(bin_edges) == 11
        assert sum(baseline_dist) == pytest.approx(1.0, rel=0.01)


class TestDriftStatus:
    """Tests for drift status mapping."""
    
    def test_low_psi_is_ok(self):
        assert get_drift_status(0.05) == "ok"
    
    def test_medium_psi_is_warning(self):
        assert get_drift_status(0.15) == "warning"
    
    def test_high_psi_is_alert(self):
        assert get_drift_status(0.25) == "alert"
    
    def test_boundary_values(self):
        assert get_drift_status(0.1) == "warning"  # At threshold
        assert get_drift_status(0.2) == "alert"  # At threshold


class TestDriftDetector:
    """Tests for DriftDetector class."""
    
    def test_set_baseline(self):
        """Can set a baseline distribution."""
        detector = DriftDetector()
        baseline = np.random.normal(0, 1, 1000)
        
        detector.set_baseline("feature_a", baseline)
        
        assert "feature_a" in detector.baselines
    
    def test_check_drift_no_baseline(self):
        """Returns None if no baseline exists."""
        detector = DriftDetector()
        result = detector.check_drift("unknown_feature", np.array([1, 2, 3]))
        
        assert result is None
    
    def test_check_drift_returns_result(self):
        """Returns DriftResult with correct structure."""
        detector = DriftDetector()
        np.random.seed(42)
        
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0.1, 1, 1000)
        
        detector.set_baseline("feature_a", baseline)
        result = detector.check_drift("feature_a", current)
        
        assert isinstance(result, DriftResult)
        assert result.feature_name == "feature_a"
        assert result.psi_value >= 0
        assert result.status in ["ok", "warning", "alert"]
    
    def test_check_all_features(self):
        """Can check multiple features at once."""
        detector = DriftDetector()
        np.random.seed(42)
        
        detector.set_baseline("feature_a", np.random.normal(0, 1, 1000))
        detector.set_baseline("feature_b", np.random.normal(0, 1, 1000))
        
        current_data = {
            "feature_a": np.random.normal(0, 1, 1000),
            "feature_b": np.random.normal(0.5, 1, 1000)  # Some drift
        }
        
        results = detector.check_all_features(current_data)
        
        assert len(results) == 2
    
    def test_alert_summary(self):
        """Alert summary groups features by status."""
        detector = DriftDetector()
        np.random.seed(42)
        
        # Feature with no drift
        detector.set_baseline("stable", np.random.normal(0, 1, 1000))
        # Feature with drift
        detector.set_baseline("drifted", np.random.normal(0, 1, 1000))
        
        current_data = {
            "stable": np.random.normal(0, 1, 1000),
            "drifted": np.random.normal(3, 1, 1000)  # Big drift
        }
        
        summary = detector.get_alert_summary(current_data)
        
        assert "ok" in summary
        assert "warnings" in summary
        assert "alerts" in summary
        assert "stable" in summary["ok"]
        assert any("drifted" in alert for alert in summary["alerts"])
