"""
FraudShield Evaluation Script

Runs slice-based evaluation and generates report.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from src.features import FeatureEngineer
from src.model import FraudModel
from src.config import DATA_DIR, AMOUNT_BINS, AMOUNT_LABELS


def evaluate_slice(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """Calculate metrics for a data slice."""
    y_pred = (y_proba >= threshold).astype(int)
    
    # Handle edge cases
    if len(y_true) == 0:
        return {"n_samples": 0, "fraud_rate": 0}
    
    metrics = {
        "n_samples": len(y_true),
        "fraud_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    # AUC metrics need both classes
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    
    return metrics


def run_evaluation():
    """Run full slice-based evaluation."""
    print("=" * 60)
    print("FraudShield Evaluation Suite")
    print("=" * 60)
    
    # Load data
    data_path = DATA_DIR / "transactions.csv"
    if not data_path.exists():
        print(f"❌ Data not found at {data_path}")
        print("   Run 'python scripts/train.py' first")
        return
    
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df):,} transactions")
    
    # Initialize model
    model = FraudModel()
    if not model.load():
        print("❌ Model not found. Run 'python scripts/train.py' first")
        return
    
    # Get predictions
    print("\nGenerating predictions...")
    fe = FeatureEngineer()
    transactions = df.to_dict("records")
    
    y_true = df["is_fraud"].values
    y_proba = np.array([model.predict(t)[0] for t in transactions])
    
    # Overall metrics
    print("\n" + "-" * 40)
    print("OVERALL METRICS")
    print("-" * 40)
    overall = evaluate_slice(y_true, y_proba)
    for k, v in overall.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Slice: Amount bins
    print("\n" + "-" * 40)
    print("SLICE: Transaction Amount")
    print("-" * 40)
    df["amount_bin"] = pd.cut(df["amount"], bins=AMOUNT_BINS, labels=AMOUNT_LABELS)
    
    amount_results = {}
    for label in AMOUNT_LABELS:
        mask = df["amount_bin"] == label
        if mask.sum() > 0:
            metrics = evaluate_slice(y_true[mask], y_proba[mask])
            amount_results[label] = metrics
            pr_auc = metrics.get("pr_auc")
            pr_auc_str = f"{pr_auc:.3f}" if pr_auc else "N/A"
            print(f"  {label:12} | n={metrics['n_samples']:6,} | PR-AUC={pr_auc_str}")
    
    # Slice: Merchant category
    print("\n" + "-" * 40)
    print("SLICE: Merchant Category")
    print("-" * 40)
    
    category_results = {}
    for category in df["merchant_category"].unique():
        mask = df["merchant_category"] == category
        if mask.sum() > 100:  # Only report categories with enough samples
            metrics = evaluate_slice(y_true[mask], y_proba[mask])
            category_results[category] = metrics
            pr_auc = metrics.get("pr_auc")
            pr_auc_str = f"{pr_auc:.3f}" if pr_auc else "N/A"
            print(f"  {category:15} | n={metrics['n_samples']:6,} | PR-AUC={pr_auc_str}")
    
    # Slice: Time of day
    print("\n" + "-" * 40)
    print("SLICE: Time of Day")
    print("-" * 40)
    
    time_bins = {
        "night (10pm-6am)": (df["hour"] >= 22) | (df["hour"] < 6),
        "morning (6am-12pm)": (df["hour"] >= 6) & (df["hour"] < 12),
        "afternoon (12pm-6pm)": (df["hour"] >= 12) & (df["hour"] < 18),
        "evening (6pm-10pm)": (df["hour"] >= 18) & (df["hour"] < 22),
    }
    
    time_results = {}
    for label, mask in time_bins.items():
        metrics = evaluate_slice(y_true[mask], y_proba[mask])
        time_results[label] = metrics
        pr_auc = metrics.get("pr_auc")
        pr_auc_str = f"{pr_auc:.3f}" if pr_auc else "N/A"
        print(f"  {label:20} | n={metrics['n_samples']:6,} | PR-AUC={pr_auc_str}")
    
    # Slice: Transaction type
    print("\n" + "-" * 40)
    print("SLICE: Transaction Type")
    print("-" * 40)
    
    type_slices = {
        "domestic": ~df["is_international"],
        "international": df["is_international"],
        "card_present": df["card_present"],
        "card_not_present": ~df["card_present"],
    }
    
    type_results = {}
    for label, mask in type_slices.items():
        metrics = evaluate_slice(y_true[mask], y_proba[mask])
        type_results[label] = metrics
        pr_auc = metrics.get("pr_auc")
        pr_auc_str = f"{pr_auc:.3f}" if pr_auc else "N/A"
        print(f"  {label:20} | n={metrics['n_samples']:6,} | PR-AUC={pr_auc_str}")
    
    # CI Gate Check
    print("\n" + "=" * 60)
    print("CI GATE CHECK")
    print("=" * 60)
    
    # Define minimum thresholds
    MIN_PR_AUC = 0.5
    MIN_RECALL = 0.6
    
    gates_passed = True
    
    # Check overall PR-AUC
    if overall.get("pr_auc", 0) < MIN_PR_AUC:
        print(f"❌ FAIL: Overall PR-AUC {overall['pr_auc']:.3f} < {MIN_PR_AUC}")
        gates_passed = False
    else:
        print(f"✅ PASS: Overall PR-AUC {overall['pr_auc']:.3f} >= {MIN_PR_AUC}")
    
    # Check recall
    if overall.get("recall", 0) < MIN_RECALL:
        print(f"❌ FAIL: Overall Recall {overall['recall']:.3f} < {MIN_RECALL}")
        gates_passed = False
    else:
        print(f"✅ PASS: Overall Recall {overall['recall']:.3f} >= {MIN_RECALL}")
    
    # Check no slice regresses more than 5%
    baseline_pr_auc = overall.get("pr_auc", 0)
    for slice_name, metrics in {**amount_results, **category_results}.items():
        slice_pr_auc = metrics.get("pr_auc")
        if slice_pr_auc and slice_pr_auc < baseline_pr_auc * 0.95:
            print(f"⚠️  WARNING: {slice_name} PR-AUC {slice_pr_auc:.3f} is >5% below baseline")
    
    print("\n" + "-" * 40)
    if gates_passed:
        print("✅ All CI gates PASSED")
    else:
        print("❌ CI gates FAILED - do not deploy")
    
    # Save results
    results = {
        "overall": overall,
        "amount_slices": amount_results,
        "category_slices": category_results,
        "time_slices": time_results,
        "type_slices": type_results,
        "ci_gates_passed": gates_passed
    }
    
    results_path = Path(__file__).parent.parent / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    run_evaluation()
