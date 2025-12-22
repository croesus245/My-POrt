# ShiftBench Results

## Benchmark Results (v1.0)

### Overview

ShiftBench evaluates medical image classifier robustness across 4 types of distribution shift.

**Model:** ResNet-50 pretrained on ImageNet, fine-tuned on ISIC 2019  
**Task:** Binary classification (lesion vs. no lesion)

### Main Results

| Split | AUROC | Accuracy | ECE | Drop from IID |
|-------|-------|----------|-----|---------------|
| Test-IID (baseline) | 0.89 | 0.85 | 0.04 | — |
| Test-Hospital | 0.79 | 0.76 | 0.12 | **-11%** |
| Test-Demographic | 0.84 | 0.80 | 0.08 | -6% |
| Test-Temporal | 0.82 | 0.78 | 0.09 | -8% |
| Test-Severity | 0.86 | 0.71 | 0.18 | -3% |

### Key Findings

1. **Hospital shift is most severe:** 11% AUROC drop from equipment variation
2. **Calibration degrades faster than accuracy:** ECE 3-4x worse under shift
3. **Severity shift hurts accuracy more than AUROC:** Class imbalance breaks threshold
4. **No single shift dominates:** Need to evaluate all types

### Robustness Techniques Comparison

| Technique | IID AUROC | Hospital AUROC | Delta |
|-----------|-----------|----------------|-------|
| Baseline (ERM) | 0.89 | 0.79 | -0.10 |
| Heavy augmentation | 0.88 | 0.82 | -0.06 |
| Mixup | 0.88 | 0.81 | -0.07 |
| Smaller model (ResNet-18) | 0.86 | 0.80 | -0.06 |
| Focal loss | 0.87 | 0.81 | -0.06 |

**Winner:** Heavy color augmentation — best robustness with minimal IID cost

### Full Results

See individual experiment files in this directory:
- `baseline_results.json` — Full metrics for all splits
- `augmentation_ablation.json` — Augmentation comparison
- `architecture_comparison.json` — Model architecture comparison
- `calibration_results.json` — Calibration method comparison

### Reproduction

```bash
cd shiftbench
python -m experiments.main_benchmark --output results/
```

See [reproduction_guide.md](../docs/reproduction_guide.md) for details.

---

*Results generated: 2025-12-01 | Model seed: 42*
