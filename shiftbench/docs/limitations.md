# ShiftBench Limitations

**Author:** Abdul-Sobur Ayinde  
**Last Updated:** 2025-12-01

---

## Overview

This document honestly describes ShiftBench's limitations. Understanding these is crucial for proper interpretation of results and responsible use of the benchmark.

---

## Dataset Limitations

### 1. Single Imaging Modality

**Limitation:** ShiftBench only covers dermoscopy (skin imaging).

**Impact:** Results may not generalize to:
- Radiology (X-ray, CT, MRI)
- Pathology (histopathology slides)
- Ophthalmology (retinal imaging)
- Other medical imaging modalities

**Why this matters:** Different modalities have different shift characteristics. Equipment variation in MRI (scanner manufacturers, field strength) is very different from dermoscopy equipment variation.

### 2. Simulated Shifts

**Limitation:** The distribution shifts are simulated using metadata clustering, not actual deployment data.

**Impact:**
- "Hospital shift" is a proxy based on image appearance clustering, not actual multi-site data
- Real hospital-to-hospital shift includes factors we can't simulate (lighting, protocols, patient selection)
- Severity of simulated shifts may not match real deployment

**Honest assessment:** A model that performs well on ShiftBench test_hospital may still fail at actual new hospital deployment.

### 3. Limited Metadata

**Limitation:** 6-28% of images have missing metadata.

| Field | Missing % |
|-------|-----------|
| diagnosis | 0% |
| age | 6% |
| sex | 4% |
| anatomic_site | 11% |
| acquisition_device | 28% |

**Impact:**
- Demographic shift analysis is based on incomplete data
- Equipment-based clustering uses proxies (color histograms) due to missing device info
- Can't analyze all intersectional subgroups

### 4. Geographic and Demographic Bias

**Limitation:** ISIC source data is predominantly from North American and European institutions.

**Impact:**
- Underrepresentation of skin tones common in other regions
- Disease presentation differences across populations not captured
- Models evaluated on ShiftBench may fail on underrepresented groups

### 5. Label Quality

**Limitation:** Labels come from clinical diagnosis, which has inherent uncertainty.

**Impact:**
- Some "ground truth" labels may be incorrect
- Difficult cases are labeled by consensus, not definitive histopathology
- Ceiling on achievable performance is < 100%

---

## Evaluation Limitations

### 1. Binary Classification Focus

**Limitation:** Main benchmark uses binary (lesion vs. no lesion), simplifying the 8-class problem.

**Impact:**
- Doesn't capture fine-grained classification difficulty
- Some shifts may affect specific lesion types differently
- Real clinical need is often multi-class

**Mitigation:** Multi-class results available in supplementary experiments.

### 2. Single Threshold Evaluation

**Limitation:** Most metrics reported at default threshold (0.5).

**Impact:**
- Operating point may not match clinical needs
- Precision-recall tradeoffs not fully explored
- Calibration issues hidden by threshold choice

**Mitigation:** Full ROC/PR curves available in detailed results.

### 3. No Uncertainty Quantification

**Limitation:** Benchmark doesn't evaluate uncertainty estimates.

**Impact:**
- Can't assess whether models "know what they don't know"
- Abstention strategies not evaluated
- Clinically crucial for safe deployment

### 4. Static Evaluation

**Limitation:** One-time evaluation on fixed test sets.

**Impact:**
- Doesn't capture temporal drift (concept drift over time)
- No evaluation of online adaptation
- Real deployment is continuous, not one-shot

---

## Methodological Limitations

### 1. Train/Test Contamination Risk

**Limitation:** ISIC images may have duplicates or near-duplicates across years.

**Impact:**
- Temporal shift (train 2016-2018, test 2019) may have some leakage
- Models might memorize specific lesions that appear in multiple years

**Mitigation:** De-duplication based on image hashing was applied, but near-duplicates (same lesion, different photo) may remain.

### 2. Shift Severity Not Calibrated

**Limitation:** No objective measure of "how severe" each shift is.

**Impact:**
- Can't compare difficulty across shift types
- A model robust to "hospital shift" may face different magnitude shifts in practice
- Benchmarks don't come with deployment guarantees

### 3. Single Dataset Source

**Limitation:** All data from ISIC, a research consortium.

**Impact:**
- May not represent "typical" clinical images
- Images submitted to ISIC may be systematically different (more interesting cases, better quality)
- Selection bias in source data

---

## Scope Limitations

### 1. Not Clinical Validation

**Limitation:** ShiftBench is a research benchmark, not clinical validation.

**Impact:**
- Good ShiftBench performance ≠ clinical readiness
- No regulatory (FDA, CE) relevance
- Must not be used to justify clinical deployment

### 2. No Prospective Evaluation

**Limitation:** All evaluation is retrospective.

**Impact:**
- Doesn't capture prospective deployment challenges
- No real clinical workflow integration
- Patient outcomes not measured

### 3. English-Only Documentation

**Limitation:** All documentation and code in English.

**Impact:**
- Limited accessibility for non-English speakers
- May limit adoption in some regions

---

## Comparison to Related Benchmarks

| Benchmark | Advantages over ShiftBench | ShiftBench Advantages |
|-----------|---------------------------|----------------------|
| WILDS | Multiple domains, real shifts | Medical imaging focus |
| DomainBed | More adaptation methods | Realistic shift types |
| ImageNet-C | Comprehensive corruptions | Clinically relevant shifts |
| CheXpert | Larger, X-ray data | Explicit shift evaluation |

---

## What ShiftBench IS Good For

Despite limitations, ShiftBench is useful for:

1. **Comparing model robustness** — Relative ranking of methods under shift
2. **Identifying failure modes** — Which shift types hurt most
3. **Calibration research** — Studying calibration under shift
4. **Educational purposes** — Teaching distribution shift concepts
5. **Baseline establishment** — Standard reference for new methods

---

## What ShiftBench is NOT Good For

1. ❌ Predicting real-world deployment performance
2. ❌ Validating clinical safety
3. ❌ Comprehensive robustness certification
4. ❌ Production model selection (use real validation data)

---

## Recommendations for Users

### If using for research:

1. Report results on all shift types, not just favorable ones
2. Include confidence intervals (multiple seeds)
3. Acknowledge limitations in your paper
4. Consider additional benchmarks for comprehensive evaluation

### If using for model development:

1. Don't rely solely on ShiftBench for validation
2. Supplement with real deployment data when available
3. Test on actual target distribution before deployment
4. Use ShiftBench for development, not final validation

### If citing results:

1. Specify exact version and splits used
2. Report all metrics, not cherry-picked
3. Include standard deviations
4. Note these are benchmark results, not deployment performance

---

## Future Improvements

Planned improvements for future versions:

1. **Multi-site data:** Partner with hospitals for real multi-site images
2. **Additional modalities:** Expand to radiology, pathology
3. **Uncertainty evaluation:** Add metrics for uncertainty quality
4. **Temporal evaluation:** Rolling evaluation over time
5. **Fairness metrics:** Explicit subgroup performance reporting

---

## Feedback

If you discover additional limitations or issues, please:
- Open an issue on GitHub
- Contact via portfolio

Honest feedback improves the benchmark for everyone.

---

*"A benchmark that claims no limitations is hiding them."*
