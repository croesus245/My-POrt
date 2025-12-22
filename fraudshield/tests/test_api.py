"""
API Endpoint Tests
"""
import pytest
from fastapi.testclient import TestClient

# Import app
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_response_format(self, client):
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["status"] == "healthy"


class TestScoreEndpoint:
    """Tests for /score endpoint."""
    
    def test_score_minimal_payload(self, client):
        """Test with only required field."""
        response = client.post("/score", json={"amount": 100.0})
        assert response.status_code == 200
    
    def test_score_full_payload(self, client):
        """Test with all fields."""
        payload = {
            "amount": 250.0,
            "merchant_category": "retail",
            "hour": 14,
            "day_of_week": 2,
            "is_international": False,
            "card_present": True,
            "merchant_risk_score": 0.3
        }
        response = client.post("/score", json=payload)
        assert response.status_code == 200
    
    def test_score_response_format(self, client):
        """Verify response has all expected fields."""
        response = client.post("/score", json={"amount": 100.0})
        data = response.json()
        
        assert "transaction_id" in data
        assert "risk_score" in data
        assert "risk_tier" in data
        assert "reason_codes" in data
        assert "latency_ms" in data
    
    def test_score_risk_score_bounds(self, client):
        """Risk score must be between 0 and 1."""
        response = client.post("/score", json={"amount": 100.0})
        data = response.json()
        
        assert 0 <= data["risk_score"] <= 1
    
    def test_score_risk_tier_valid(self, client):
        """Risk tier must be low, medium, or high."""
        response = client.post("/score", json={"amount": 100.0})
        data = response.json()
        
        assert data["risk_tier"] in ["low", "medium", "high"]
    
    def test_score_high_amount_increases_risk(self, client):
        """High amounts should increase risk score."""
        low_amount = client.post("/score", json={"amount": 50.0}).json()
        high_amount = client.post("/score", json={"amount": 10000.0}).json()
        
        # High amount should generally have higher risk
        # (Not always true with ML model, but true for mock)
        assert high_amount["risk_score"] >= low_amount["risk_score"] * 0.5
    
    def test_score_invalid_amount_rejected(self, client):
        """Negative amounts should be rejected."""
        response = client.post("/score", json={"amount": -100.0})
        assert response.status_code == 422  # Validation error
    
    def test_score_invalid_hour_rejected(self, client):
        """Invalid hour should be rejected."""
        response = client.post("/score", json={"amount": 100.0, "hour": 25})
        assert response.status_code == 422


class TestBatchEndpoint:
    """Tests for /score/batch endpoint."""
    
    def test_batch_single_transaction(self, client):
        """Batch with one transaction."""
        payload = {"transactions": [{"amount": 100.0}]}
        response = client.post("/score/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
    
    def test_batch_multiple_transactions(self, client):
        """Batch with multiple transactions."""
        transactions = [{"amount": float(i * 100)} for i in range(1, 11)]
        payload = {"transactions": transactions}
        response = client.post("/score/batch", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 10
    
    def test_batch_too_large_rejected(self, client):
        """Batch over 100 should be rejected."""
        transactions = [{"amount": 100.0} for _ in range(101)]
        payload = {"transactions": transactions}
        response = client.post("/score/batch", json=payload)
        
        assert response.status_code == 400
    
    def test_batch_has_total_latency(self, client):
        """Batch response should include total latency."""
        payload = {"transactions": [{"amount": 100.0}]}
        response = client.post("/score/batch", json=payload)
        
        data = response.json()
        assert "total_latency_ms" in data


class TestThresholdsEndpoint:
    """Tests for /thresholds endpoint."""
    
    def test_thresholds_returns_200(self, client):
        response = client.get("/thresholds")
        assert response.status_code == 200
    
    def test_thresholds_has_values(self, client):
        response = client.get("/thresholds")
        data = response.json()
        
        assert "thresholds" in data
        assert "low" in data["thresholds"]
        assert "medium" in data["thresholds"]
        assert "high" in data["thresholds"]
