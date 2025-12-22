# Why Offline Metrics Lie (And What to Do About It)

**Author:** Abdul-Sobur Ayinde  
**Date:** 2025-10-15  
**Reading time:** 6 min

---

Your model has 0.95 AUC on the test set. Ship it, right?

Not so fast. That test set was sampled from the same distribution as training. Your users live in a different world.

## The Gap

I've seen this pattern repeatedly:

| Metric | Offline | Production (Week 1) |
|--------|---------|---------------------|
| AUC | 0.95 | 0.89 |
| Precision | 0.88 | 0.71 |
| p95 Latency | 45ms | 180ms |

That's not a rounding error. That's the difference between a demo and a product.

## Why It Happens

### 1. Test sets are too clean

Your test set was created the same way as your training set. Same labeling process. Same time period. Same biases.

Real data has:
- Label noise (humans disagree)
- Distribution shift (the world changes)
- Edge cases you never imagined

### 2. Metrics don't capture what matters

AUC tells you about ranking. It says nothing about:
- Calibration (are your probabilities meaningful?)
- Latency under load
- Behavior on specific slices
- What happens when the model is wrong

### 3. Offline evaluation is static

You test once, get a number, move on.

Production is continuous. Your model sees thousands of requests per hour. Distribution drift happens. Adversaries adapt.

## What Actually Works

### 1. Slice your metrics

Overall AUC is hiding disasters. Always check:

```python
slices = ['new_users', 'high_value', 'mobile', 'international']
for slice_name in slices:
    slice_data = test_df[test_df.slice == slice_name]
    print(f"{slice_name}: AUC = {calculate_auc(model, slice_data):.3f}")
```

I've seen models with 0.92 overall AUC that dropped to 0.61 on the "new users" slice. That's a production incident waiting to happen.

### 2. Test calibration, not just discrimination

AUC measures ranking. ECE measures whether your probabilities are trustworthy.

```python
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10)
ece = np.mean(np.abs(prob_true - prob_pred))
```

A model that says "80% confident" should be right 80% of the time. If it's actually right 95% of the time, your downstream systems are making suboptimal decisions.

### 3. Build time-based test sets

Random splits are lying to you. Your training data is from the past. Your production data is from the future.

```python
# Bad: random split
train, test = train_test_split(df, test_size=0.2)

# Better: temporal split
cutoff = '2024-10-01'
train = df[df.date < cutoff]
test = df[df.date >= cutoff]
```

If your model can't generalize across time, it won't generalize to production.

### 4. Add regression tests in CI

Every PR should prove it doesn't break things:

```yaml
# .github/workflows/eval.yml
- name: Run evaluation
  run: |
    python evaluate.py --model ${{ github.sha }}
    python check_gates.py --min-auc 0.80 --max-regression 0.05
```

If you don't automate it, it won't happen.

### 5. Shadow mode before launch

Run your model in production without making decisions:

```python
def predict(request):
    # Existing model makes the decision
    decision = production_model.predict(request)
    
    # New model just logs
    shadow_prediction = candidate_model.predict(request)
    log_shadow_result(shadow_prediction, decision)
    
    return decision
```

After a week, you'll know if the candidate model agrees with reality.

## The Uncomfortable Truth

Offline metrics are necessary but not sufficient. They're a gate, not a guarantee.

The model that wins on your test set might lose in production. The model with the best AUC might have the worst calibration. The model that's fast on your laptop might be slow under load.

Build evaluation systems, not evaluation scripts.

---

## What I'd Do Differently

If I were starting over:

1. **Invest in slice infrastructure early.** You can't fix problems you can't see.

2. **Make temporal splits the default.** Random splits should require justification.

3. **Log everything.** You'll need it for debugging and retraining.

4. **Build dashboards before models.** If you can't measure production, you can't improve it.

The teams that win are the ones that close the feedback loop between production and development. Offline metrics are the beginning of that loop, not the end.

---

*Enjoyed this? Check out [Eval Suites as CI Gates](eval-suites-ci-gates.md) for the implementation details.*
