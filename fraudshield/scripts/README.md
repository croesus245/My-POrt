# FraudShield Scripts

Training, evaluation, and benchmarking scripts.

## Training

```bash
python scripts/train.py
```

Generates synthetic data and trains XGBoost model.

## Evaluation

```bash
python scripts/evaluate.py
```

Runs slice-based evaluation and outputs CI gate results.

## Benchmarking

```bash
# Quick benchmark (direct)
python scripts/benchmark.py http://localhost:8000 1000

# Full load test (Locust UI)
locust -f scripts/benchmark.py --host=http://localhost:8000
```
