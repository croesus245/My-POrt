# FraudShield Model Card

**Model Name:** FraudShield Transaction Scorer  
**Version:** xgb_v0.3.1  
**Type:** Binary Classification (XGBoost)  
**Last Updated:** 2025-12-15

---

## Model Details

### Overview

FraudShield is a real-time fraud detection model that scores payment transactions for fraud risk. It outputs a probability score (0-1) which is mapped to three decision buckets: Allow, Review, or Block.

### Architecture

- **Algorithm:** XGBoost (gradient boosted trees)
- **Trees:** 150
- **Max depth:** 6
- **Learning rate:** 0.1
- **Objective:** binary:logistic

### Intended Use

- **Primary use:** Real-time fraud scoring for payment transactions
- **Users:** Fraud operations teams, automated decisioning systems
- **Deployment:** REST API with < 50ms p95 latency requirement

### Out of Scope

- Account takeover detection (different model)
- Merchant fraud detection (B2B, different patterns)
- Credit risk assessment
- Identity verification

---

## Training Data

### Dataset Description

| Property | Value |
|----------|-------|
| Source | Synthetic transaction data |
| Size | 500,000 transactions |
| Time range | 2024-01-01 to 2024-10-31 |
| Fraud rate | 2.3% (after resampling) |
| Features | 47 input features |

### Data Generation

Training data is **synthetically generated** to simulate realistic fraud patterns without using real customer data. Generation process:

1. Base transaction patterns from published fraud research
2. Behavioral sequences generated via Markov chains
3. Fraud injection following known attack patterns
4. Temporal patterns (time-of-day, day-of-week) from aggregated statistics

### Label Definition

A transaction is labeled as fraud if:
- Chargeback filed within 90 days, OR
- Manual fraud team confirmation, OR
- Account takeover confirmed (linked transactions)

**Label delay:** 30-90 days typical (simulated in training)

### Data Splits

| Split | Period | Size | Fraud Rate |
|-------|--------|------|------------|
| Train | 2024-01 to 2024-08 | 400K | 2.3% |
| Validation | 2024-09 | 50K | 2.2% |
| Test | 2024-10 | 50K | 2.4% |

**Split method:** Temporal (not random) to simulate production conditions

---

## Features

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Transaction | 12 | amount, currency, channel |
| Behavioral | 15 | velocity, recency, frequency |
| Device | 6 | device_age, device_type, fingerprint_match |
| Location | 5 | geo_distance, country_risk |
| Merchant | 5 | merchant_risk, category, MCC |
| Temporal | 4 | hour_risk, day_of_week, time_since_last |

### Top Features by Importance

1. `velocity_user_1h` — Transaction count in last hour
2. `amount_zscore` — Amount relative to user's history
3. `time_since_last_txn` — Seconds since previous transaction
4. `merchant_risk_score` — Aggregated merchant risk
5. `device_age_days` — Days since device first seen

### Feature Engineering

- **Embeddings:** Merchant category embeddings (SVD, 8-dim)
- **Aggregations:** Rolling windows (1h, 24h, 7d, 30d)
- **Interactions:** Amount × velocity, device × location

---

## Evaluation Results

### Overall Performance

| Metric | Value | Target |
|--------|-------|--------|
| PR-AUC | 0.847 | ≥ 0.80 |
| ROC-AUC | 0.91 | — |
| Precision @ 0.5 | 0.82 | ≥ 0.75 |
| Recall @ 0.5 | 0.71 | ≥ 0.65 |
| ECE | 0.021 | < 0.05 |

### Slice Performance

| Slice | PR-AUC | Notes |
|-------|--------|-------|
| High-value (>$500) | 0.88 | Strong performance |
| New users (<30d) | 0.79 | Limited history |
| Mobile channel | 0.83 | — |
| Card-not-present | 0.86 | Higher base fraud rate |
| International | 0.81 | — |

### Decision Thresholds

| Decision | Score Range | Precision | Recall | Volume |
|----------|-------------|-----------|--------|--------|
| Block | ≥ 0.85 | 0.94 | 0.31 | 2.1% |
| Review | 0.45–0.85 | 0.67 | 0.52 | 8.3% |
| Allow | < 0.45 | — | — | 89.6% |

---

## Ethical Considerations

### Potential Biases

1. **Behavioral bias:** Users with limited transaction history receive higher risk scores
   - *Mitigation:* Separate scoring pathway for new users, higher review rate

2. **Geographic bias:** International transactions scored higher due to training distribution
   - *Mitigation:* Explicit country-risk feature, monitored slice metrics

3. **Merchant bias:** Certain merchant categories (travel, gambling) have higher baseline risk
   - *Mitigation:* Category-aware calibration, separate thresholds

### Fairness Monitoring

Slice metrics are tracked for:
- Transaction amount bands
- User tenure cohorts
- Geographic regions
- Device types
- Merchant categories

**Alert threshold:** Any slice dropping > 5% triggers investigation

### Privacy Considerations

- Model does not use PII directly (name, address, SSN)
- Device fingerprints are hashed
- User IDs are pseudonymized
- No demographic features (age, gender, race)

---

## Limitations

### Known Limitations

1. **Label delay:** 30-90 day delay means model trains on "stale" ground truth
2. **Synthetic data:** Training data is synthetic; real-world patterns may differ
3. **New attack patterns:** Model may not detect novel fraud schemes
4. **Cold start:** New users/merchants have limited signal
5. **Adversarial adaptation:** Fraudsters may adapt to model patterns

### Failure Modes

| Failure Mode | Likelihood | Impact | Mitigation |
|--------------|------------|--------|------------|
| Novel fraud pattern | Medium | High | Human review layer |
| Distribution shift | Medium | Medium | Drift monitoring |
| Feature pipeline failure | Low | High | Schema validation |
| Adversarial gaming | Medium | Medium | Regular retraining |

### When NOT to Use This Model

- As sole decision-maker for high-value transactions (> $10K)
- Without human review layer for edge cases
- For fraud types outside training distribution (ATO, merchant fraud)
- In regions with insufficient training data

---

## Deployment

### Infrastructure

- **Serving:** FastAPI on Kubernetes
- **Latency:** p50 ~15ms, p95 ~45ms
- **Throughput:** ~1.2K TPS per instance
- **Scaling:** Horizontal auto-scaling

### Monitoring

- Prediction score distribution (hourly)
- Feature drift (PSI, KS test)
- Slice performance (daily)
- Latency percentiles (continuous)

### Update Frequency

- **Automatic retraining:** On drift detection
- **Scheduled refresh:** Quarterly
- **Hotfix:** As needed for critical issues

---

## Maintenance

### Owners

- **Model:** ML Engineering Team
- **Features:** Data Engineering Team
- **Infrastructure:** Platform Team

### Documentation

- [Evaluation Report](eval_report.md)
- [Drift Runbook](drift_runbook.md)
- [Incident Postmortem](postmortem.md)
- [Cost Report](cost_report.md)

### Version History

| Version | Date | Changes |
|---------|------|---------|
| v0.3.1 | 2025-12 | Merchant embedding refresh, travel slice fix |
| v0.3.0 | 2025-09 | Added entity embeddings |
| v0.2.5 | 2025-06 | Improved new user handling |
| v0.2.0 | 2025-03 | Streaming features integration |
| v0.1.0 | 2024-12 | Initial release |

---

## Citation

If referencing this model:

```
FraudShield Transaction Scorer v0.3.1
Abdul-Sobur Ayinde, 2025
https://github.com/Croesus245/My-POrt/tree/main/fraudshield
```

---

*Model card format adapted from Mitchell et al., "Model Cards for Model Reporting" (2019)*
