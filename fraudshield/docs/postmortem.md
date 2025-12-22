# FraudShield Incident Postmortem

**Incident:** Drift-Induced Performance Degradation  
**Date:** 2025-11-18  
**Duration:** 6 hours (detection to resolution)  
**Severity:** Medium  
**Status:** Resolved

---

## Executive Summary

A distribution shift in the `merchant_category` feature caused PR-AUC to drop from 0.84 to 0.61 on the "travel" merchant slice. The drift monitor detected the issue, automatic retraining was triggered, and the model was updated within 6 hours. No manual intervention was required for remediation.

**Impact:**
- ~340 additional fraudulent transactions in "review" bucket (would have been "block")
- Estimated $47K in potential fraud exposure during degradation window
- Zero false blocks on legitimate transactions

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 2025-11-18 02:00 | Upstream merchant data provider updates category taxonomy |
| 2025-11-18 06:15 | PSI alert fires: `merchant_category` PSI = 0.15 |
| 2025-11-18 06:20 | On-call acknowledges alert, begins investigation |
| 2025-11-18 06:45 | Root cause identified: merchant category distribution shift |
| 2025-11-18 07:00 | Feature KS test confirms: "travel" category volume +180% |
| 2025-11-18 07:15 | Slice analysis shows "travel" PR-AUC dropped to 0.61 |
| 2025-11-18 07:30 | Automatic retraining triggered (PSI threshold exceeded) |
| 2025-11-18 09:45 | Candidate model passes eval gates, promoted to production |
| 2025-11-18 10:00 | Post-deployment monitoring confirms recovery |
| 2025-11-18 12:00 | Incident closed |

---

## Root Cause Analysis

### What happened

The upstream merchant data provider (MerchantDB) updated their category taxonomy on 2025-11-18. This caused:

1. **Category redistribution:** Several subcategories were merged into "travel"
2. **Feature drift:** `merchant_category` distribution shifted significantly
3. **Embedding mismatch:** Pre-trained merchant embeddings no longer aligned with new categories
4. **Performance drop:** Model's learned patterns for "travel" became invalid

### Why it happened

1. **No schema contract with upstream:** We consumed merchant categories without version pinning
2. **Taxonomy change not communicated:** Provider did not notify downstream consumers
3. **Embedding staleness:** Merchant embeddings were trained on historical category distribution

### Why we didn't catch it sooner

We DID catch it—within 4 hours of the change propagating to our feature store. The drift monitoring system worked as designed:

- PSI alert fired at 0.15 (warning threshold: 0.1)
- Automatic investigation workflow initiated
- Retraining triggered without manual intervention

---

## Impact Assessment

### Quantified Impact

| Metric | Normal | During Incident | Delta |
|--------|--------|-----------------|-------|
| Travel slice PR-AUC | 0.84 | 0.61 | -27% |
| Travel slice volume | 2.1% | 5.8% | +176% |
| Misclassified fraud (travel) | ~12/day | ~85 (6h window) | +73 |
| Estimated exposure | — | $47,000 | — |

### What went well

1. ✅ Drift monitoring detected the issue automatically
2. ✅ Alert fired within 4 hours of upstream change
3. ✅ Automatic retraining pipeline worked end-to-end
4. ✅ New model passed all eval gates before promotion
5. ✅ Zero manual intervention required for remediation
6. ✅ No false positives (legitimate transactions blocked)

### What went poorly

1. ❌ 4-hour detection delay (PSI aggregation window)
2. ❌ No upstream change notification
3. ❌ Merchant embeddings not automatically refreshed

---

## Resolution

### Immediate actions taken

1. Automatic retraining with fresh data (including new category distribution)
2. Merchant embeddings regenerated with updated taxonomy
3. Model promoted after passing all eval gates

### Post-incident validation

```
Slice: travel merchants
Before incident: PR-AUC = 0.84
During incident: PR-AUC = 0.61
After fix:       PR-AUC = 0.86 ✅
```

---

## Lessons Learned

### What we learned

1. **Upstream dependencies are a risk:** External data providers can change schemas without notice
2. **Embeddings need refresh cycles:** Pre-trained embeddings can become stale
3. **Detection worked, but could be faster:** 4-hour aggregation window is too long for critical shifts

### Action items

| Action | Owner | Priority | Status |
|--------|-------|----------|--------|
| Implement schema versioning contract with MerchantDB | Data Eng | P1 | ✅ Done |
| Reduce PSI aggregation window to 1 hour | ML Eng | P1 | ✅ Done |
| Add embedding freshness check to drift monitoring | ML Eng | P2 | ✅ Done |
| Set up upstream change notifications | Data Eng | P2 | In Progress |
| Add "travel" as explicit high-risk slice in eval | ML Eng | P3 | ✅ Done |
| Quarterly embedding refresh automation | ML Eng | P3 | Planned |

---

## Prevention Measures Implemented

### 1. Schema Contract (Implemented)

```yaml
# data_contracts/merchant_category.yml
version: "2.1"
provider: merchantdb
fields:
  - name: category_code
    type: string
    allowed_values: [retail, travel, dining, ...]
    change_notification: required
```

### 2. Faster Drift Detection (Implemented)

```yaml
# monitoring/drift_config.yml
psi:
  aggregation_window: 1h  # was: 4h
  alert_threshold: 0.10
  critical_threshold: 0.15
```

### 3. Embedding Freshness Monitoring (Implemented)

```python
# Added to drift checks
def check_embedding_freshness():
    embedding_age = get_embedding_train_date()
    category_distribution = get_current_distribution()
    if distribution_divergence(embedding_age, category_distribution) > 0.2:
        trigger_embedding_refresh()
```

---

## Appendix: Detection Evidence

### PSI Alert (from monitoring logs)

```
2025-11-18 06:15:23 [ALERT] Feature drift detected
Feature: merchant_category
PSI: 0.153 (threshold: 0.10)
Baseline period: 2025-11-01 to 2025-11-17
Current period: 2025-11-18 00:00 to 06:00
Top shifted categories:
  - travel: 2.1% → 5.8% (+176%)
  - retail: 45.2% → 41.1% (-9%)
```

### Slice Performance Drop

```
2025-11-18 07:15:44 [ALERT] Slice performance degradation
Slice: merchant_category=travel
Baseline PR-AUC: 0.84
Current PR-AUC: 0.61
Delta: -27.4%
Action: Automatic retraining triggered
```

### Retraining Completion

```
2025-11-18 09:45:12 [INFO] Retraining complete
Candidate model: xgb_v0.3.1
Eval results:
  - Overall PR-AUC: 0.847 (baseline: 0.839) ✅
  - Travel slice PR-AUC: 0.86 (baseline: 0.61) ✅
  - All gates passed
Action: Promoting to production
```

---

## Sign-off

| Role | Name | Date |
|------|------|------|
| Incident Commander | ML Engineer On-Call | 2025-11-18 |
| Postmortem Author | Abdul-Sobur Ayinde | 2025-11-20 |
| Review | ML Lead | 2025-11-21 |

---

*This postmortem follows the blameless postmortem format. The goal is learning, not blame.*
