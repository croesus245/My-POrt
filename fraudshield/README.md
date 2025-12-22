# FraudShield

Real-time fraud detection system with delayed label reconciliation, drift monitoring, and CI-gated evaluation.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Croesus245/My-POrt.git
cd My-POrt/fraudshield

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model (generates synthetic data + trains XGBoost)
python scripts/train.py

# Run API
uvicorn src.api:app --reload

# Test it
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"amount": 150.0, "merchant_category": "retail", "hour": 14, "is_international": false}'
```

## Project Structure

```
fraudshield/
├── src/
│   ├── api.py              # FastAPI scoring endpoint
│   ├── model.py            # XGBoost model wrapper
│   ├── features.py         # Feature engineering pipeline
│   ├── drift.py            # PSI-based drift detection
│   └── config.py           # Configuration
├── tests/
│   ├── test_api.py         # API endpoint tests
│   ├── test_eval.py        # Slice-based evaluation
│   └── test_drift.py       # Drift detection tests
├── scripts/
│   ├── train.py            # Model training script
│   ├── evaluate.py         # Run full evaluation suite
│   └── benchmark.py        # Load test (locust)
├── models/                  # Trained model artifacts
├── data/                    # Synthetic training data
└── docs/                    # Documentation
```

## Key Features

### 1. Real-time Scoring API
- **Throughput:** ~1.2K TPS per instance
- **Latency:** p50 ≈ 12ms, p95 ≈ 45ms (dev load test)
- **Response:** Risk score (0-1), risk tier, reason codes

### 2. Delayed Label Handling
Fraud labels arrive 30+ days after transaction. We handle this via:
- Proxy labels (chargebacks, disputes) for fast feedback
- Periodic reconciliation with true labels
- Separate metrics for proxy vs. true label performance

### 3. Drift Detection
- PSI (Population Stability Index) on feature distributions
- Score distribution monitoring
- Automated alerts when PSI > 0.2

### 4. Slice-Based Evaluation
CI gates test model performance across:
- Transaction amounts (low/medium/high)
- Merchant categories
- Time of day
- Geographic regions

No slice may regress > 5% from baseline.

## API Endpoints

### POST /score
Score a single transaction.

```json
// Request
{
  "amount": 150.0,
  "merchant_category": "retail",
  "hour": 14,
  "day_of_week": 2,
  "is_international": false,
  "card_present": true,
  "merchant_risk_score": 0.3
}

// Response
{
  "transaction_id": "txn_abc123",
  "risk_score": 0.23,
  "risk_tier": "low",
  "reason_codes": ["normal_amount", "known_merchant_category"],
  "latency_ms": 12.4
}
```

### GET /health
Health check endpoint.

### GET /metrics
Prometheus metrics (request count, latency histogram, error rate).

## Running Tests

```bash
# Unit tests
pytest tests/ -v

# Evaluation suite (generates report)
python scripts/evaluate.py

# Load test (requires running API)
locust -f scripts/benchmark.py --host=http://localhost:8000
```

## Documentation

- [Evaluation Report](docs/eval_report.md) - Slice metrics & CI gates
- [Drift Runbook](docs/drift_runbook.md) - Response procedures
- [Cost Report](docs/cost_report.md) - Latency & infrastructure costs
- [Model Card](docs/model_card.md) - Model documentation
- [Postmortem](docs/postmortem.md) - Incident simulation

## License

MIT
