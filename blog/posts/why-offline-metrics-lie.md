# Why Offline Metrics Lie (And What to Do About It)

**Author:** Abdul-Sobur Ayinde  
**Date:** 2025-10-15  
**Reading time:** 6 min

---

Your model has 0.95 AUC on the test set. Ship it?

I did. On a personal project. Watched it fall apart when I tested on data from a month later.

This is what I learned.

## The Gap

On my fraud detection project, here's what I saw:

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

I've seen models with 0.92 overall AUC that dropped to 0.61 on specific slices. Once you start looking, you find these gaps everywhere.

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

## What I Learned

Offline metrics are necessary but not sufficient. They're a gate, not a guarantee.

I haven't deployed models at massive scale (yet). But even on personal projects with synthetic data, these patterns show up. The principles should transfer.

---

## What I'd Do Differently

Looking back at my projects:

1. **Start with temporal splits.** Random splits hide so much.

2. **Build slice infrastructure early.** You can't fix what you can't see.

3. **Log predictions.** You'll need them for debugging.

4. **Make dashboards before models.** Measurement first.

I'm still learning this stuff. These are notes to my future self as much as advice to anyone else.

---

*More on this: [Eval Suites as CI Gates](eval-suites-ci-gates.md)*
