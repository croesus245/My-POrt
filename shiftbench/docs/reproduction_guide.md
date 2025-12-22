# ShiftBench Reproduction Guide

**Version:** 1.0  
**Last Updated:** 2025-12-01

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/Croesus245/My-POrt
cd My-POrt/shiftbench

# Setup environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Download data
python -m shiftbench.download --output ./data

# Reproduce main results
python -m experiments.main_benchmark --output results/
```

**Expected runtime:** ~4 hours on a single GPU (RTX 3080 or equivalent)

---

## Prerequisites

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1080 (8GB) | RTX 3080 (10GB) |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |

**Note:** Training can be done on CPU but will take 10-20x longer.

### Software

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Data Access

1. Create account at [ISIC Archive](https://challenge.isic-archive.com)
2. Accept data use agreement
3. Note your API credentials

---

## Step-by-Step Reproduction

### Step 1: Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Expected output:** `True` (if GPU available)

### Step 2: Download Source Data

```bash
# Automated download (requires ISIC credentials)
export ISIC_USERNAME="your_email"
export ISIC_PASSWORD="your_password"
python -m shiftbench.download --output ./data

# Alternative: manual download
# 1. Go to https://challenge.isic-archive.com/data/
# 2. Download "ISIC 2019: Training" dataset
# 3. Extract to ./data/isic2019/
```

**Expected:** ~25,000 images, ~12GB total

### Step 3: Generate Splits

```bash
python -m shiftbench.generate_splits \
    --data-dir ./data/isic2019 \
    --output-dir ./data/splits \
    --seed 42

# Verify splits
python -m shiftbench.verify_splits --splits-dir ./data/splits
```

**Expected output:**
```
Split verification:
  train: 18,000 images ✓
  val_iid: 2,000 images ✓
  test_iid: 2,500 images ✓
  test_hospital: 1,200 images ✓
  test_demographic: 1,100 images ✓
  test_temporal: 800 images ✓
  test_severity: 731 images ✓
No overlap between splits ✓
```

### Step 4: Train Baseline Model

```bash
python -m experiments.train_baseline \
    --data-dir ./data/splits \
    --output-dir ./models/baseline \
    --architecture resnet50 \
    --epochs 30 \
    --batch-size 32 \
    --lr 1e-4 \
    --seed 42
```

**Expected runtime:** ~2 hours on RTX 3080

**Expected output:**
```
Epoch 30/30:
  Train Loss: 0.234
  Val AUROC: 0.892
  Best model saved to ./models/baseline/best.pth
```

### Step 5: Evaluate on All Splits

```bash
python -m experiments.evaluate \
    --model ./models/baseline/best.pth \
    --data-dir ./data/splits \
    --output ./results/baseline_results.json
```

**Expected results (±0.02):**

| Split | AUROC | Accuracy | ECE |
|-------|-------|----------|-----|
| test_iid | 0.89 | 0.85 | 0.04 |
| test_hospital | 0.79 | 0.76 | 0.12 |
| test_demographic | 0.84 | 0.80 | 0.08 |
| test_temporal | 0.82 | 0.78 | 0.09 |
| test_severity | 0.86 | 0.71 | 0.18 |

### Step 6: Generate Report

```bash
python -m experiments.generate_report \
    --results ./results/baseline_results.json \
    --output ./results/README.md
```

---

## Reproducing Specific Experiments

### Augmentation Ablation

```bash
python -m experiments.augmentation_ablation \
    --data-dir ./data/splits \
    --output-dir ./results/augmentation
```

Tests: no augmentation, basic, heavy color jitter, mixup

### Model Architecture Comparison

```bash
python -m experiments.architecture_comparison \
    --data-dir ./data/splits \
    --output-dir ./results/architectures
```

Tests: ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0

### Calibration Methods

```bash
python -m experiments.calibration_comparison \
    --model ./models/baseline/best.pth \
    --data-dir ./data/splits \
    --output-dir ./results/calibration
```

Tests: uncalibrated, temperature scaling, isotonic regression

---

## Verifying Results

### Numerical Reproducibility

Due to GPU non-determinism, expect ±0.5% variation. To maximize reproducibility:

```python
import torch
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Note:** Setting `deterministic=True` may slow training by 10-20%.

### Statistical Significance

Run 5 seeds and report mean ± std:

```bash
for seed in 42 43 44 45 46; do
    python -m experiments.train_baseline --seed $seed --output-dir ./models/seed_$seed
done
python -m experiments.aggregate_results --seeds 42 43 44 45 46
```

### Hash Verification

```bash
# Verify data integrity
md5sum data/splits/*.csv

# Expected hashes
train.csv: a1b2c3d4e5f6...
test_iid.csv: b2c3d4e5f6a1...
...
```

---

## Common Issues

### Issue: CUDA out of memory

**Solution:** Reduce batch size
```bash
python -m experiments.train_baseline --batch-size 16
```

### Issue: Download fails

**Solution:** Use manual download from ISIC website, then run split generation.

### Issue: Results don't match

**Possible causes:**
1. Different random seed
2. Different PyTorch/CUDA version
3. Data version mismatch

**Check:**
```bash
python -m shiftbench.check_environment
```

### Issue: Slow training on CPU

**Solution:** Use Google Colab or rent cloud GPU

---

## Colab Notebook

For quick reproduction without local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Croesus245/My-POrt/blob/main/shiftbench/notebooks/reproduce_results.ipynb)

The notebook includes:
- Environment setup
- Data download
- Training
- Evaluation
- Visualization

---

## File Structure

```
shiftbench/
├── data/
│   ├── isic2019/           # Raw images
│   └── splits/             # Split CSV files
├── models/
│   └── baseline/           # Trained models
├── results/
│   ├── baseline_results.json
│   └── README.md           # Generated report
├── experiments/
│   ├── train_baseline.py
│   ├── evaluate.py
│   └── generate_report.py
├── shiftbench/
│   ├── download.py
│   ├── generate_splits.py
│   └── dataset.py
├── requirements.txt
└── README.md
```

---

## Configuration Reference

### Training Hyperparameters (Default)

```yaml
# config/default.yaml
model:
  architecture: resnet50
  pretrained: true
  num_classes: 2

training:
  epochs: 30
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  scheduler: cosine
  warmup_epochs: 5

augmentation:
  horizontal_flip: true
  vertical_flip: true
  rotation: 15
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1

evaluation:
  metrics: [auroc, accuracy, precision, recall, f1, ece]
  threshold: 0.5
```

---

## Citation

If you use ShiftBench in your research:

```bibtex
@misc{shiftbench2025,
  author = {Abdul-Sobur Ayinde},
  title = {ShiftBench: Evaluating Medical Image Classifiers Under Distribution Shift},
  year = {2025},
  url = {https://github.com/Croesus245/My-POrt/tree/main/shiftbench}
}
```

---

## Questions?

Open an issue on GitHub or contact via portfolio.

---

*Guide version 1.0 | Last verified: 2025-12-01*
