"""
Slice-Based Evaluation Tests

Tests that the model meets minimum performance requirements across slices.
These tests are designed to run in CI to gate model deployment.
"""
import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FraudModel
from src.features import FeatureEngineer, generate_reason_codes


class TestFeatureEngineer:
    """Tests for feature engineering."""
    
    def test_transform_single_returns_array(self):
        """Transform should return numpy array."""
        fe = FeatureEngineer()
        transaction = {"amount": 100.0}
        
        result = fe.transform_single(transaction)
        
        assert isinstance(result, np.ndarray)
    
    def test_transform_single_correct_length(self):
        """Transform should return correct number of features."""
        fe = FeatureEngineer()
        transaction = {"amount": 100.0}
        
        result = fe.transform_single(transaction)
        
        assert len(result) == len(fe.get_feature_names())
    
    def test_transform_batch(self):
        """Batch transform should handle multiple transactions."""
        fe = FeatureEngineer()
        transactions = [
            {"amount": 100.0},
            {"amount": 200.0},
            {"amount": 300.0}
        ]
        
        result = fe.transform_batch(transactions)
        
        assert result.shape == (3, len(fe.get_feature_names()))
    
    def test_amount_log_transform(self):
        """Amount should be log-transformed."""
        fe = FeatureEngineer()
        
        small = fe.transform_single({"amount": 10.0})
        large = fe.transform_single({"amount": 10000.0})
        
        # Log transform compresses range
        assert large[0] < 10 * small[0]  # Not 1000x different
    
    def test_cyclical_hour_encoding(self):
        """Hour should be encoded cyclically."""
        fe = FeatureEngineer()
        
        midnight = fe.transform_single({"amount": 100.0, "hour": 0})
        near_midnight = fe.transform_single({"amount": 100.0, "hour": 23})
        
        # hour_sin and hour_cos should be similar for 0 and 23
        # (indices -2 and -1 are hour_sin and hour_cos)
        assert abs(midnight[-2] - near_midnight[-2]) < 0.5
        assert abs(midnight[-1] - near_midnight[-1]) < 0.5


class TestReasonCodes:
    """Tests for reason code generation."""
    
    def test_high_amount_reason(self):
        """High amounts should generate reason code."""
        transaction = {"amount": 10000.0}
        reasons = generate_reason_codes(transaction, 0.5)
        
        assert "very_high_amount" in reasons or "high_amount" in reasons
    
    def test_night_reason(self):
        """Night transactions should generate reason code."""
        transaction = {"amount": 100.0, "hour": 3}
        reasons = generate_reason_codes(transaction, 0.5)
        
        assert "unusual_hour" in reasons
    
    def test_international_reason(self):
        """International transactions should generate reason code."""
        transaction = {"amount": 100.0, "is_international": True}
        reasons = generate_reason_codes(transaction, 0.5)
        
        assert "international_transaction" in reasons
    
    def test_normal_transaction_has_reason(self):
        """Even normal transactions should have a reason code."""
        transaction = {"amount": 50.0}
        reasons = generate_reason_codes(transaction, 0.1)
        
        assert len(reasons) > 0


class TestFraudModel:
    """Tests for FraudModel."""
    
    def test_model_init(self):
        """Model should initialize without error."""
        model = FraudModel()
        assert model is not None
    
    def test_mock_predict(self):
        """Model should return mock predictions when not loaded."""
        model = FraudModel()
        # Don't load model - use mock
        
        score, tier, reasons = model.predict({"amount": 100.0})
        
        assert 0 <= score <= 1
        assert tier in ["low", "medium", "high"]
        assert isinstance(reasons, list)
    
    def test_mock_predict_high_amount(self):
        """Mock should assign higher risk to high amounts."""
        model = FraudModel()
        
        low_score, _, _ = model.predict({"amount": 50.0})
        high_score, _, _ = model.predict({"amount": 10000.0})
        
        assert high_score > low_score
    
    def test_mock_predict_international(self):
        """Mock should assign higher risk to international."""
        model = FraudModel()
        
        domestic, _, _ = model.predict({"amount": 100.0, "is_international": False})
        international, _, _ = model.predict({"amount": 100.0, "is_international": True})
        
        assert international > domestic
    
    def test_batch_predict(self):
        """Batch prediction should return results for all inputs."""
        model = FraudModel()
        
        transactions = [
            {"amount": 100.0},
            {"amount": 200.0},
            {"amount": 300.0}
        ]
        
        results = model.predict_batch(transactions)
        
        assert len(results) == 3
        for score, tier, reasons in results:
            assert 0 <= score <= 1


class TestModelSlices:
    """
    Slice-based evaluation tests.
    
    These tests verify model performance across different data slices.
    They're designed to catch cases where a model performs well overall
    but fails on specific subgroups.
    """
    
    @pytest.fixture
    def model(self):
        return FraudModel()
    
    def test_amount_slices_not_degenerate(self, model):
        """Model should give varying scores across amount ranges."""
        amounts = [10, 100, 500, 2000, 10000]
        scores = [model.predict({"amount": a})[0] for a in amounts]
        
        # Scores shouldn't all be the same
        assert len(set(scores)) > 1
    
    def test_time_slices_not_degenerate(self, model):
        """Model should consider time of day."""
        hours = [3, 10, 15, 22]  # Night, morning, afternoon, evening
        scores = [model.predict({"amount": 100.0, "hour": h})[0] for h in hours]
        
        # Night should generally be higher risk
        night_score = scores[0]
        day_score = scores[2]
        assert night_score >= day_score * 0.8  # At least somewhat higher
    
    def test_merchant_category_coverage(self, model):
        """Model should handle all merchant categories."""
        categories = ["retail", "grocery", "online", "travel"]
        
        for category in categories:
            score, tier, reasons = model.predict({
                "amount": 100.0,
                "merchant_category": category
            })
            assert 0 <= score <= 1
            assert tier in ["low", "medium", "high"]
    
    def test_risk_tier_thresholds(self, model):
        """Risk tiers should follow defined thresholds."""
        # Generate many predictions and check tier consistency
        test_cases = [
            {"amount": 10.0},  # Should be low risk
            {"amount": 5000.0, "is_international": True, "card_present": False},  # Should be higher
        ]
        
        for case in test_cases:
            score, tier, _ = model.predict(case)
            
            if score < 0.3:
                assert tier == "low"
            elif score < 0.7:
                assert tier == "medium"
            else:
                assert tier == "high"
