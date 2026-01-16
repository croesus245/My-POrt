# Eval Suites as CI Gates: A Practical Guide

**Author:** Abdul-Sobur Ayinde  
**Date:** 2025-11-05  
**Reading time:** 8 min

---

Your model passed code review. Tests are green. Time to merge?

Maybe not. I built this eval framework for my FraudShield project to catch regressions before they reach the "production" environment (it's synthetic data, but the principle matters).

Here's how I think about evaluation gates.

## The Problem

ML systems have two kinds of bugs:

1. **Code bugs:** The function throws an exception
2. **Model bugs:** The function returns the wrong answer

Traditional CI catches #1. It misses #2 entirely.

## The Solution: Eval Gates

Eval gates are automated checks that prevent merging if model quality degrades.

```yaml
# .github/workflows/eval-gate.yml
name: Evaluation Gate

on:
  pull_request:
    paths:
      - 'src/model/**'
      - 'src/features/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run evaluation
        run: python -m evaluation.run_suite --output results.json
      
      - name: Check gates
        run: python -m evaluation.check_gates --results results.json
```

## What to Gate On

### 1. Primary metric with threshold

The obvious one. Don't let AUC drop below some minimum.

```python
def check_primary_metric(results: dict) -> bool:
    auc = results['metrics']['auc']
    threshold = 0.80
    if auc < threshold:
        raise GateFailed(f"AUC {auc:.3f} below threshold {threshold}")
    return True
```

**Pitfall:** A single threshold can be gamed. Model improves one metric, degrades everything else.

### 2. Regression check against baseline

Compare to the current production model, not an arbitrary threshold.

```python
def check_regression(results: dict, baseline: dict) -> bool:
    for metric in ['auc', 'precision', 'recall']:
        current = results['metrics'][metric]
        baseline_val = baseline['metrics'][metric]
        regression = baseline_val - current
        
        if regression > 0.02:  # 2% max regression
            raise GateFailed(
                f"{metric} regressed {regression:.1%} from baseline"
            )
    return True
```

**Key insight:** Absolute thresholds are arbitrary. Regression limits are meaningful.

### 3. Slice-level requirements

Overall metrics hide slice disasters. Gate on every important slice.

```python
def check_slices(results: dict) -> bool:
    slice_threshold = 0.75
    
    for slice_name, slice_metrics in results['slices'].items():
        if slice_metrics['auc'] < slice_threshold:
            raise GateFailed(
                f"Slice '{slice_name}' AUC {slice_metrics['auc']:.3f} "
                f"below threshold {slice_threshold}"
            )
    return True
```

**Critical slices to always check:**
- New vs. returning users
- High-value vs. normal transactions
- By geographic region
- By device type
- By time of day

### 4. Calibration requirements

A model that ranks well but gives meaningless probabilities is dangerous.

```python
def check_calibration(results: dict) -> bool:
    ece = results['calibration']['expected_calibration_error']
    max_ece = 0.05
    
    if ece > max_ece:
        raise GateFailed(f"ECE {ece:.3f} exceeds maximum {max_ece}")
    return True
```

### 5. Latency requirements

The fastest model is useless if it's wrong. The most accurate model is useless if it's slow.

```python
def check_latency(results: dict) -> bool:
    p95 = results['latency']['p95_ms']
    max_p95 = 50
    
    if p95 > max_p95:
        raise GateFailed(f"p95 latency {p95}ms exceeds maximum {max_p95}ms")
    return True
```

## Implementation Details

### Fast feedback loop

Eval gates only work if they're fast. Target: < 10 minutes total.

Strategies:
- Use a representative sample, not full dataset
- Cache baseline results
- Parallelize slice evaluations
- Skip redundant computations

```python
# Use cached baseline
baseline = load_cached_baseline()  # Pre-computed, stored in S3

# Sample for speed
eval_data = full_data.sample(n=10000, random_state=42)

# Parallel slices
with ThreadPoolExecutor() as executor:
    slice_results = executor.map(evaluate_slice, slices)
```

### Clear failure messages

When a gate fails, the developer needs to know exactly what to fix.

Bad:
```
Error: Evaluation failed
```

Good:
```
❌ GATE FAILED: Slice 'new_users' regression

  Metric: AUC
  Baseline: 0.823
  Current:  0.761
  Regression: -7.5% (max allowed: 5%)

  Impact: ~12,000 users/day affected
  
  Debugging tips:
  - Check feature coverage for new users
  - Review recent changes to user_history features
  - Run: python -m debug.slice_analysis --slice new_users
```

### Escape hatches (with accountability)

Sometimes you need to bypass gates. Make it possible but visible.

```yaml
# In PR description
[eval-override] Accepting 3% regression on international slice
Reason: Known labeling issue, fix incoming in #456
Approved by: @ml-lead
```

```python
def check_override(pr_body: str) -> bool:
    if '[eval-override]' in pr_body:
        # Log for audit
        log_override(pr_number, pr_body, pr_author)
        return True
    return False
```

## Real Example: FraudShield

Here's the actual eval gate config from my fraud detection project:

```yaml
# evaluation/gates.yaml
gates:
  primary_metric:
    metric: pr_auc
    threshold: 0.80
    
  regression:
    max_regression: 0.05
    metrics: [pr_auc, precision, recall]
    
  slices:
    threshold: 0.75
    required:
      - high_value
      - new_users
      - card_not_present
      - international
      
  calibration:
    max_ece: 0.05
    
  latency:
    max_p95_ms: 50
    
  error_rate:
    max_percent: 0.1
```

And the output:

```
╔══════════════════════════════════════════════════════════════╗
║                    EVALUATION GATE RESULTS                    ║
╠══════════════════════════════════════════════════════════════╣
║ Primary Metric (PR-AUC)                                       ║
║   Current: 0.847  |  Threshold: 0.800  |  ✅ PASS            ║
╠══════════════════════════════════════════════════════════════╣
║ Regression Check                                              ║
║   Baseline: 0.839  |  Change: +0.95%  |  ✅ PASS             ║
╠══════════════════════════════════════════════════════════════╣
║ Slice Performance                                             ║
║   high_value:       0.88  |  ✅ PASS                         ║
║   new_users:        0.79  |  ✅ PASS                         ║
║   card_not_present: 0.86  |  ✅ PASS                         ║
║   international:    0.81  |  ✅ PASS                         ║
╠══════════════════════════════════════════════════════════════╣
║ Calibration (ECE)                                             ║
║   Current: 0.021  |  Maximum: 0.050  |  ✅ PASS              ║
╠══════════════════════════════════════════════════════════════╣
║ Latency (p95)                                                 ║
║   Current: 45ms  |  Maximum: 50ms  |  ✅ PASS                ║
╠══════════════════════════════════════════════════════════════╣
║                                                               ║
║   OVERALL: ✅ ALL GATES PASSED                                ║
║                                                               ║
╚══════════════════════════════════════════════════════════════╝
```

## Common Objections

### "It slows down development"

Yes, by 10 minutes per PR. It also prevents "fix production fire" weeks.

### "The thresholds are arbitrary"

Start with thresholds based on current production performance. Adjust based on experience.

### "It catches too many false positives"

Then your thresholds are wrong. Tune them. A gate that fails on every PR is useless.

### "We don't have time to build this"

You don't have time to not build this. Every hour invested in eval infrastructure saves days of incident response.

## What I'd Do Differently

1. **Start with one gate, not five.** Get the infrastructure working before adding complexity.

2. **Invest in fast evaluation.** A 2-hour eval suite won't be run. A 5-minute suite will be run on every PR.

3. **Make failures educational.** The gate should teach developers what went wrong, not just block them.

4. **Track gate metrics.** How often do gates fail? How often are overrides used? This tells you if gates are too strict or too loose.

---

*This is part 2 of my ML infrastructure series. See [Why Offline Metrics Lie](why-offline-metrics-lie.md) for the motivation.*
