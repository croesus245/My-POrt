# FraudShield Drift Monitoring Runbook

**Owner:** ML Engineering  
**Last Updated:** 2025-12-15  
**Alert Channel:** #ml-alerts (Slack)

---

## Overview

This runbook covers detection and response procedures for model drift in the FraudShield fraud detection system. Drift can manifest as:

1. **Prediction drift:** Model output distribution changes
2. **Feature drift:** Input feature distributions shift
3. **Label drift:** Underlying fraud rate changes
4. **Performance drift:** Model accuracy degrades on recent data

---

## Monitoring Dashboard

**Grafana:** `https://grafana.internal/d/fraudshield-drift`

Key panels:
- PSI (Population Stability Index) by feature
- Prediction score distribution (hourly)
- Base fraud rate (7-day rolling)
- Latency percentiles

---

## Alert Thresholds

| Metric | Method | Warning | Critical | Check Frequency |
|--------|--------|---------|----------|-----------------|
| Prediction score drift | PSI | > 0.1 | > 0.2 | Hourly |
| Feature drift | KS test | p < 0.01 on 2+ features | p < 0.001 on 3+ features | Hourly |
| Base rate drift | 7-day rolling | > 15% from baseline | > 25% from baseline | Daily |
| Null rate spike | Per-feature % | > 3% increase | > 5% increase | Hourly |
| Schema violation | Contract check | Any mismatch | — | Per request |
| Feature lag | Streaming delay | > 3 min | > 5 min | Continuous |
| Label freshness | Days since batch | > 30 days | > 45 days | Daily |
| Latency p95 | Service metrics | > 40ms | > 50ms | Continuous |

---

## Response Decision Tree

```
Alert Triggered
    │
    ├── Is it a schema violation?
    │   └── YES → Block pipeline, page on-call, see [Schema Response]
    │
    ├── Is it a latency alert?
    │   └── YES → Check load, scale workers, see [Latency Response]
    │
    ├── Is it feature drift (PSI/KS)?
    │   ├── Single feature → Investigate upstream data source
    │   └── Multiple features → Check for upstream pipeline changes
    │
    ├── Is it prediction drift only?
    │   └── YES → May indicate real distribution shift, monitor for 24h
    │
    ├── Is base rate drifting?
    │   ├── UP (more fraud) → Possible attack pattern, tighten thresholds
    │   └── DOWN (less fraud) → Verify label pipeline, check for seasonal effect
    │
    └── Is label pipeline stale?
        └── YES → Escalate to Data Engineering
```

---

## Response Procedures

### 1. Prediction Score Drift (PSI > 0.1)

**Severity:** Warning  
**Response time:** 4 hours

**Steps:**
1. Check feature drift dashboard—often root cause is upstream
2. Review recent deployments (model or feature pipeline)
3. Compare score distributions: `make drift-compare --days 7`
4. If PSI > 0.2, escalate to critical

**Commands:**
```bash
python -m src.monitoring.drift_analysis --metric psi --window 24h
python -m src.monitoring.score_distribution --compare baseline
```

---

### 2. Feature Drift (KS test failure)

**Severity:** Warning/Critical depending on count  
**Response time:** 2 hours

**Steps:**
1. Identify which features are drifting: check Grafana panel
2. Trace feature to upstream source
3. Check for:
   - Upstream schema changes
   - Data pipeline failures (nulls, delays)
   - Legitimate distribution shift (seasonality, new user cohort)
4. If legitimate shift → schedule recalibration
5. If data issue → escalate to Data Engineering

**Commands:**
```bash
python -m src.monitoring.drift_analysis --metric ks --features all
python -m src.monitoring.feature_nulls --window 24h
```

---

### 3. Base Rate Drift

**Severity:** Warning  
**Response time:** 24 hours (unless sudden spike)

**Steps:**
1. Verify label pipeline is functioning
2. Check for seasonal patterns (holiday fraud spikes)
3. If fraud rate UP significantly:
   - Review recent high-score transactions
   - Consider temporary threshold tightening
   - Alert fraud ops team
4. If fraud rate DOWN:
   - Verify labels are arriving (not pipeline failure)
   - Check for changes in transaction volume/mix

**Commands:**
```bash
python -m src.monitoring.base_rate --window 30d --plot
python -m src.monitoring.label_lag --check
```

---

### 4. Latency Spike (p95 > 50ms)

**Severity:** Critical  
**Response time:** 15 minutes

**Steps:**
1. Check current request volume (traffic spike?)
2. Check feature store latency (Redis)
3. Check model inference time
4. If load-related → scale horizontally
5. If not load-related → check for:
   - Feature computation bottleneck
   - Model size increase
   - Infrastructure issue

**Commands:**
```bash
kubectl get hpa fraudshield-api
kubectl top pods -l app=fraudshield
python -m src.monitoring.latency_breakdown --last 1h
```

---

### 5. Schema Violation

**Severity:** Critical  
**Response time:** Immediate

**Steps:**
1. Pipeline automatically blocks on schema mismatch
2. Identify which field violated contract
3. Check upstream producer for changes
4. Coordinate fix with data producer team
5. Do NOT bypass schema validation

**Commands:**
```bash
python -m src.data.validate_schema --sample recent
cat logs/schema_violations.log | tail -50
```

---

### 6. Label Pipeline Stale (> 45 days)

**Severity:** Critical  
**Response time:** 4 hours

**Steps:**
1. Check label ingestion job status
2. Verify chargeback data feed is active
3. Escalate to Data Engineering if pipeline broken
4. If intentional delay (vendor issue), document and monitor model closely

**Note:** Model can operate without fresh labels, but retraining will use stale ground truth.

---

## Escalation Matrix

| Situation | Primary | Secondary | Page? |
|-----------|---------|-----------|-------|
| PSI > 0.2 | ML Engineer on-call | ML Lead | No |
| KS fail on 3+ features | ML Engineer on-call | Data Eng | No |
| Schema violation | Data Eng on-call | ML Engineer | Yes |
| Latency p95 > 50ms | Platform on-call | ML Engineer | Yes |
| Base rate > 25% shift | ML Engineer on-call | Fraud Ops | No |
| Label pipeline down | Data Eng on-call | ML Lead | Yes |

---

## Retraining Trigger Criteria

Automatic retraining job is triggered when ANY of:

1. PSI > 0.15 sustained for 7 days
2. Base rate drift > 20% for 14 days
3. Manual trigger via `make retrain-trigger`
4. Scheduled quarterly refresh

**Retraining workflow:**
1. Pull latest labeled data (respecting 30-day label delay)
2. Train candidate model
3. Run full eval suite against current production model
4. If candidate passes all gates AND improves primary metric → auto-promote
5. If candidate fails any gate → alert ML team for review

---

## Rollback Procedure

If a newly deployed model degrades performance:

```bash
# List available model versions
make model-list

# Rollback to previous version
make model-rollback --version xgb_v0.3.0

# Verify rollback
make model-current
```

**Rollback criteria:**
- Any slice drops > 10% within 24h of deployment
- Latency increases > 50% from baseline
- Error rate exceeds 0.1%

---

## Appendix: PSI Interpretation

| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No significant shift | None |
| 0.1 – 0.2 | Moderate shift | Investigate |
| > 0.2 | Significant shift | Likely retrain needed |

---

## Appendix: Seasonal Patterns

Known seasonal effects to account for:

| Period | Expected Change | Notes |
|--------|-----------------|-------|
| Black Friday/Cyber Monday | +30-50% volume, +15% fraud rate | Pre-tighten thresholds |
| December holidays | +20% volume, variable fraud | Monitor closely |
| January | -20% volume, slight fraud decrease | Post-holiday normalization |
| Tax season (Apr) | +10% fraud rate | Refund fraud patterns |

---

*Runbook version 2.1 | Owner: ML Engineering*
