# ShiftBench Dataset Datasheet

**Dataset Name:** ShiftBench  
**Version:** 1.0  
**Created:** 2025-11-01  
**Author:** Abdul-Sobur Ayinde

---

## Motivation

### Why was the dataset created?

ShiftBench was created to evaluate how well medical image classifiers maintain performance under realistic distribution shifts. Most existing benchmarks test models on IID (independent and identically distributed) data, which doesn't reflect real-world deployment where:

1. Imaging equipment varies across hospitals
2. Patient demographics shift over time
3. Disease prevalence changes seasonally
4. Image acquisition protocols differ

### What tasks does it support?

- Binary classification: lesion (any type) vs. no lesion
- Multi-class classification: specific lesion types
- Distribution shift robustness evaluation
- Calibration assessment under shift

### Who created the dataset?

Abdul-Sobur Ayinde, as a portfolio project demonstrating ML evaluation methodology.

---

## Composition

### Source Data

ShiftBench uses images from the **ISIC 2019 Challenge** dataset:

- **Source:** [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/)
- **Original size:** 25,331 dermoscopic images
- **Classes:** 8 diagnostic categories
- **License:** CC BY-NC 4.0

**Why ISIC 2019:** It's the largest publicly available dermoscopy dataset with diverse imaging conditions, making it suitable for simulating distribution shifts.

### ShiftBench Splits

| Split | N | Purpose | Shift Type |
|-------|---|---------|------------|
| Train | 18,000 | Model training | — |
| Val-IID | 2,000 | Hyperparameter tuning | None (IID) |
| Test-IID | 2,500 | Baseline performance | None (IID) |
| Test-Hospital | 1,200 | Equipment shift | Imaging device |
| Test-Demographic | 1,100 | Population shift | Age distribution |
| Test-Temporal | 800 | Time shift | Acquisition year |
| Test-Severity | 731 | Prevalence shift | Class balance |

**Total:** 25,331 images (full ISIC 2019)

### Shift Simulation Methodology

#### Hospital Shift (Equipment)
- Clustered images by color histogram features
- Used k-means to identify 5 "acquisition clusters"
- Test-Hospital contains only cluster 4 (underrepresented in training)
- Simulates deployment at a hospital with different imaging equipment

#### Demographic Shift (Age)
- Stratified by patient age metadata
- Training: uniform age distribution (20-80)
- Test-Demographic: skewed toward 60+ (simulates geriatric clinic)

#### Temporal Shift
- Used acquisition date metadata
- Training: 2016-2018 images
- Test-Temporal: 2019 images only
- Captures protocol changes over time

#### Severity Shift (Prevalence)
- Training: 2.3% melanoma prevalence (realistic screening)
- Test-Severity: 15% melanoma prevalence (simulates referral center)
- Tests calibration under prevalence shift

---

## Collection Process

### How was the data collected?

**Original ISIC collection:**
- Dermoscopic images from multiple institutions
- Submitted to ISIC archive by dermatologists
- Includes clinical metadata (age, sex, anatomic site, diagnosis)

**ShiftBench processing:**
1. Downloaded ISIC 2019 via official API
2. Filtered for images with complete metadata
3. Applied stratified splitting based on shift criteria
4. Generated split manifests (CSV files)

### Data integrity verification

```python
# Verification script (src/data/verify_splits.py)
- No image appears in multiple splits
- Class distribution matches design
- Shift criteria correctly applied
- All image files accessible
```

---

## Preprocessing

### Applied to all images

1. Resize to 224×224 (preserve aspect ratio, pad if needed)
2. Normalize to [0, 1]
3. No augmentation in test splits (training uses standard augmentations)

### Metadata extracted

| Field | Type | Coverage |
|-------|------|----------|
| diagnosis | categorical | 100% |
| age | integer | 94% |
| sex | categorical | 96% |
| anatomic_site | categorical | 89% |
| acquisition_device | categorical | 72% |

---

## Intended Use

### Primary use

Evaluate classifier robustness to distribution shift in medical imaging.

### Recommended evaluation protocol

```python
from shiftbench import load_benchmark

# Load all splits
benchmark = load_benchmark()

# Evaluate model
results = {}
for split_name, split_data in benchmark.items():
    preds = model.predict(split_data.images)
    results[split_name] = {
        'accuracy': accuracy(preds, split_data.labels),
        'auroc': auroc(preds, split_data.labels),
        'ece': calibration_error(preds, split_data.labels)
    }

# Report degradation
for split in ['hospital', 'demographic', 'temporal', 'severity']:
    degradation = results['test_iid']['auroc'] - results[f'test_{split}']['auroc']
    print(f"{split}: {degradation:+.1%} AUROC drop")
```

### Not intended for

- Clinical diagnosis (research only)
- Training production medical models
- Evaluating non-dermoscopy models

---

## Limitations

### Known issues

1. **Single domain:** Only dermoscopy; shifts may not generalize to other imaging modalities
2. **Simulated shifts:** Hospital/demographic shifts are proxies, not true deployment data
3. **Metadata completeness:** 6-28% missing metadata limits some analyses
4. **Geographic bias:** ISIC contributors are predominantly from North America/Europe

### What shifts are NOT covered

- Adversarial perturbations
- Severe image corruption
- Cross-modality transfer
- Novel disease classes (open-set)

---

## Ethical Considerations

### Privacy

- All images are de-identified in source dataset
- No personally identifiable information included
- Compliant with original ISIC data use agreement

### Fairness

- Age and sex metadata available for subgroup analysis
- Race/ethnicity not available (limitation of source data)
- Known bias: light skin tones overrepresented in dermoscopy datasets

### Clinical impact

- **This is a research benchmark, not a diagnostic tool**
- Results should not be used for clinical decision-making
- Models evaluated here require extensive validation before clinical use

---

## Distribution

### Access

```bash
# Download via script
python -m shiftbench.download --output ./data

# Or manual download
# 1. Register at https://challenge.isic-archive.com
# 2. Download ISIC 2019 Challenge dataset
# 3. Run split generation: python -m shiftbench.generate_splits
```

### License

- **ISIC source data:** CC BY-NC 4.0
- **ShiftBench splits/code:** MIT License

### Citation

```bibtex
@misc{shiftbench2025,
  author = {Abdul-Sobur Ayinde},
  title = {ShiftBench: Evaluating Medical Image Classifiers Under Distribution Shift},
  year = {2025},
  url = {https://github.com/Croesus245/My-POrt/tree/main/shiftbench}
}
```

---

## Maintenance

### Contact

- Issues: GitHub repository
- Email: [via portfolio contact page]

### Version history

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11 | Initial release |

### Known errata

None reported.

---

## Appendix: Split Generation Code

```python
# Simplified version of src/data/generate_splits.py

def generate_shiftbench_splits(isic_data):
    """Generate ShiftBench splits from ISIC 2019 data."""
    
    # IID splits (random stratified)
    train, test_iid = stratified_split(isic_data, test_size=0.2)
    train, val_iid = stratified_split(train, test_size=0.1)
    
    # Hospital shift (cluster-based)
    clusters = cluster_by_color_histogram(isic_data, n_clusters=5)
    test_hospital = isic_data[clusters == 4]  # Minority cluster
    
    # Demographic shift (age-stratified)
    test_demographic = isic_data[isic_data.age >= 60]
    
    # Temporal shift (year-based)
    test_temporal = isic_data[isic_data.year == 2019]
    
    # Severity shift (prevalence adjustment)
    test_severity = resample_to_prevalence(
        isic_data, target_melanoma_rate=0.15
    )
    
    return {
        'train': train,
        'val_iid': val_iid,
        'test_iid': test_iid,
        'test_hospital': test_hospital,
        'test_demographic': test_demographic,
        'test_temporal': test_temporal,
        'test_severity': test_severity
    }
```

---

*Datasheet format adapted from Gebru et al., "Datasheets for Datasets" (2021)*
