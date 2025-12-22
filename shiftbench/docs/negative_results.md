# ShiftBench: What Didn't Work

**Author:** Abdul-Sobur Ayinde  
**Date:** 2025-11-20

---

## Overview

This document catalogs approaches that failed or underperformed during ShiftBench development. Recording negative results is essential for:

1. Avoiding wasted effort on known dead ends
2. Understanding why certain techniques fail
3. Providing honest context for what "worked"

---

## Failed Approaches

### 1. Automated Shift Detection via Feature Drift

**Hypothesis:** Use statistical tests (PSI, KS test) on penultimate layer features to automatically detect when a model encounters shifted data.

**What I tried:**
```python
# Extract features from penultimate layer
features_train = model.get_features(train_loader)
features_test = model.get_features(test_loader)

# Statistical tests
psi_scores = calculate_psi(features_train, features_test)
ks_results = ks_test_per_dimension(features_train, features_test)

# Alert if significant drift
if psi_scores.mean() > 0.1:
    print("Distribution shift detected!")
```

**Result:** ❌ Failed

**Why it failed:**
- High-dimensional features (2048-dim) made statistical tests unreliable
- PCA reduction lost the signal—shifts manifested in minor components
- PSI threshold was arbitrary; no principled way to set it
- False positive rate was ~30% on IID test data

**Lessons learned:**
- Feature-space drift detection works for tabular data, not raw CNN features
- Would need learned representations specifically for drift detection
- Confidence/uncertainty methods might work better

---

### 2. Domain Adaptation via Gradient Reversal

**Hypothesis:** Use domain-adversarial training to learn shift-invariant features.

**What I tried:**
- Added gradient reversal layer (GRL) between feature extractor and domain classifier
- Trained to classify disease while being unable to classify source domain
- Implementation based on DANN paper

**Result:** ❌ Worse than baseline

**Performance:**
| Method | Test-IID AUROC | Test-Hospital AUROC |
|--------|----------------|---------------------|
| Baseline (ERM) | 0.89 | 0.79 |
| DANN | 0.84 | 0.77 |

**Why it failed:**
- Domain-invariant features discarded diagnostically relevant information
- Color/texture differences between "hospitals" correlate with lesion appearance
- The shift I simulated isn't truly domain-independent

**Lessons learned:**
- Domain adaptation assumes shift is pure "nuisance"—not always true in medical imaging
- Would need to carefully define what should vs. shouldn't transfer
- Maybe useful if combined with explicit feature selection

---

### 3. Test-Time Augmentation (TTA) for Robustness

**Hypothesis:** Averaging predictions over augmented versions of test images improves robustness.

**What I tried:**
```python
def predict_with_tta(model, image, n_augments=10):
    preds = []
    for _ in range(n_augments):
        aug_image = random_augment(image)  # flip, rotate, color jitter
        preds.append(model(aug_image))
    return np.mean(preds)
```

**Result:** ⚠️ Marginal improvement, not worth the cost

**Performance:**
| Method | Test-IID | Test-Hospital | Inference Time |
|--------|----------|---------------|----------------|
| No TTA | 0.89 | 0.79 | 1x |
| TTA (10x) | 0.90 | 0.80 | 10x |

**Why it underperformed:**
- +1% AUROC for 10x inference cost is poor ROI
- Augmentations didn't address the actual shift (equipment differences)
- TTA helps with geometric invariance, not acquisition differences

**Lessons learned:**
- TTA is not a general robustness solution
- Need augmentations that match the expected shift type
- Consider using TTA only for high-stakes predictions

---

### 4. Calibration via Temperature Scaling (Post-hoc)

**Hypothesis:** Temperature scaling on IID validation set transfers to shifted test sets.

**What I tried:**
```python
# Find optimal temperature on val-IID
T = optimize_temperature(model, val_iid_loader)

# Apply to all predictions
calibrated_probs = softmax(logits / T)
```

**Result:** ❌ Failed on shifted data

**Calibration (ECE):**
| Split | Before T-scaling | After T-scaling |
|-------|------------------|-----------------|
| Test-IID | 0.08 | 0.02 |
| Test-Hospital | 0.15 | 0.18 |
| Test-Severity | 0.22 | 0.25 |

**Why it failed:**
- Temperature learned on IID data doesn't transfer
- Under shift, optimal T is different (and unknown)
- Actually made calibration worse on shifted data

**Lessons learned:**
- Calibration methods need shift-aware variants
- Can't assume calibration transfers to new distributions
- Need to report calibration per-shift, not just overall

---

### 5. Self-Training / Pseudo-Labels on Test Data

**Hypothesis:** Use confident predictions on unlabeled shifted data as pseudo-labels for adaptation.

**What I tried:**
```python
# Get confident predictions on shifted data
confident_mask = model.predict_proba(shifted_data) > 0.9
pseudo_labels = model.predict(shifted_data[confident_mask])

# Fine-tune on pseudo-labeled data
model.fit(shifted_data[confident_mask], pseudo_labels)
```

**Result:** ❌ Made things worse

**Performance:**
| Stage | Test-Hospital AUROC |
|-------|---------------------|
| Before self-training | 0.79 |
| After 1 epoch | 0.75 |
| After 3 epochs | 0.68 |

**Why it failed:**
- Confident predictions were systematically wrong on shifted data
- Self-training amplified the model's biases
- "Confident" ≠ "correct" under distribution shift

**Lessons learned:**
- Self-training requires calibrated confidence (which we don't have under shift)
- Need uncertainty estimation, not just confidence thresholds
- Maybe works with more sophisticated selection (MC dropout, ensembles)

---

### 6. Simple Ensemble of Different Architectures

**Hypothesis:** Ensemble of diverse architectures is more robust than single model.

**What I tried:**
- ResNet-50 + DenseNet-121 + EfficientNet-B0
- Average predictions

**Result:** ⚠️ Slight improvement, but not cost-effective

**Performance:**
| Model | Test-IID | Test-Hospital | Parameters |
|-------|----------|---------------|------------|
| ResNet-50 | 0.89 | 0.79 | 25M |
| DenseNet-121 | 0.88 | 0.78 | 8M |
| EfficientNet-B0 | 0.87 | 0.77 | 5M |
| Ensemble | 0.90 | 0.80 | 38M |

**Why it underperformed:**
- All models made correlated errors on shifted data
- Architectural diversity didn't translate to prediction diversity
- +1% for 3x compute is poor ROI

**Lessons learned:**
- Need diversity in training, not just architecture
- Consider ensembles trained on different data subsets
- Deep ensembles (same arch, different seeds) might be more efficient

---

## What Actually Worked

For contrast, here's what did improve robustness:

| Technique | Hospital AUROC | Why It Worked |
|-----------|----------------|---------------|
| Heavy augmentation (color jitter, blur) | 0.82 | Simulated equipment variation |
| Focal loss | 0.81 | Better calibration on imbalanced shifts |
| Smaller model (ResNet-18) | 0.80 | Less overfitting to training distribution |
| Mixup augmentation | 0.81 | Smoother decision boundaries |

---

## Meta-Lessons

### 1. Distribution shift is underspecified

"Robustness to shift" means nothing without specifying what kind of shift. Techniques that work for corruption robustness (blur, noise) don't work for dataset shift (equipment, demographics).

### 2. IID performance is a poor predictor

Methods that improved IID test performance often hurt shifted performance. Regularization that "underfits" IID data can be more robust.

### 3. Calibration and robustness are entangled

You can't fix calibration post-hoc if the model is wrong. Need to address robustness first, then calibrate.

### 4. Complexity rarely helps

The best robustness improvements came from simpler changes (augmentation, smaller models), not complex methods (domain adaptation, self-training).

---

## Future Directions

Based on these failures, promising directions to explore:

1. **Uncertainty-aware predictions** — Know when to abstain
2. **Shift-specific augmentation** — Match augmentation to expected deployment shift
3. **Multi-domain training** — Explicitly train on diverse acquisition conditions
4. **Conformal prediction** — Calibrated confidence intervals

---

## Reproducing Negative Results

All experiments are in `experiments/negative/`:

```bash
cd shiftbench
python -m experiments.negative.dann_training
python -m experiments.negative.self_training
python -m experiments.negative.feature_drift
```

Each script includes the exact hyperparameters and random seeds used.

---

*"Negative results are results." — Every frustrated researcher*
