"""
Evaluate model performance.

Comprehensive evaluation including:
- Standard metrics
- Slice/disaggregated metrics
- Calibration analysis
- Threshold optimization
- Stress tests
"""

import argparse
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
from loguru import logger

from src.features.pipeline import FeaturePipeline
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import (
    compute_metrics,
    compute_slice_metrics,
    compute_calibration_metrics,
    compute_threshold_analysis,
)
from src.evaluation.report import generate_report, plot_evaluation_report
from src.evaluation.stress_tests import (
    label_noise_test,
    feature_perturbation_test,
    missing_feature_test,
    class_imbalance_test,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fraud detection model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="artifacts/models", help="Model directory")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports", help="Output directory")
    parser.add_argument("--run-stress-tests", action="store_true", help="Run stress tests")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # =========================================================================
    # Load Model and Data
    # =========================================================================
    logger.info("Loading model and data...")
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    
    # Load model
    trainer = ModelTrainer(config=config.get("model", {}))
    trainer.load(model_dir / "xgboost_model.json")
    
    # Load pipeline
    pipeline = FeaturePipeline.load(model_dir / "feature_pipeline.pkl")
    
    # Load test data
    ground_truth = pd.read_parquet(data_dir / "ground_truth.parquet")
    
    # Use last 20% as test (time-based split)
    split_date = ground_truth["timestamp"].quantile(0.8)
    test_df = ground_truth[ground_truth["timestamp"] > split_date].copy()
    
    logger.info(f"Test set: {len(test_df)} transactions")
    
    # Prepare features
    X_test = pipeline.transform(test_df)
    y_test = test_df["is_fraud"].values
    y_pred_proba = trainer.predict_proba(X_test)
    
    # =========================================================================
    # Standard Metrics
    # =========================================================================
    logger.info("Computing standard metrics...")
    
    metrics = compute_metrics(y_test, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("STANDARD METRICS")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"  {name:20s}: {value:.4f}")
    
    # =========================================================================
    # Slice Metrics
    # =========================================================================
    logger.info("Computing slice metrics...")
    
    slice_columns = config.get("evaluation", {}).get("slices", ["merchant_category"])
    slice_metrics = compute_slice_metrics(y_test, y_pred_proba, test_df, slice_columns)
    
    print("\n" + "=" * 60)
    print("SLICE METRICS")
    print("=" * 60)
    
    for slice_name, metrics in sorted(slice_metrics.items()):
        print(f"\n  {slice_name}:")
        print(f"    Support: {metrics['support']:,}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    F1:      {metrics['f1']:.4f}")
    
    # =========================================================================
    # Calibration Analysis
    # =========================================================================
    logger.info("Analyzing calibration...")
    
    calibration = compute_calibration_metrics(y_test, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("CALIBRATION")
    print("=" * 60)
    print(f"  ECE (Expected Calibration Error): {calibration['ece']:.4f}")
    print(f"  MCE (Max Calibration Error):      {calibration['mce']:.4f}")
    
    if calibration["ece"] > 0.1:
        print("  ⚠️  Model is poorly calibrated - consider Platt scaling")
    else:
        print("  ✓  Model is well calibrated")
    
    # =========================================================================
    # Threshold Analysis
    # =========================================================================
    logger.info("Analyzing thresholds...")
    
    thresholds = compute_threshold_analysis(y_test, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 44)
    
    for t in thresholds:
        print(f"  {t['threshold']:>10.2f} {t['precision']:>10.3f} {t['recall']:>10.3f} {t['f1']:>10.3f}")
    
    # =========================================================================
    # Stress Tests
    # =========================================================================
    if args.run_stress_tests:
        logger.info("Running stress tests...")
        
        print("\n" + "=" * 60)
        print("STRESS TESTS")
        print("=" * 60)
        
        # Label noise
        noise_result = label_noise_test(
            trainer, X_test, y_test,
            noise_rates=[0.01, 0.05, 0.1, 0.2],
        )
        print(f"\n  Label Noise Test: {'PASS' if noise_result.passed else 'FAIL'}")
        print(f"    Baseline F1: {noise_result.baseline_metric:.4f}")
        for rate, f1 in zip(noise_result.results["noise_rates"], noise_result.results["f1_scores"]):
            print(f"    {rate:.0%} noise: F1={f1:.4f}")
        
        # Feature perturbation
        perturb_result = feature_perturbation_test(
            trainer, X_test, y_test,
            noise_std=0.1,
        )
        print(f"\n  Feature Perturbation Test: {'PASS' if perturb_result.passed else 'FAIL'}")
        print(f"    Baseline F1: {perturb_result.baseline_metric:.4f}")
        print(f"    Perturbed F1: {perturb_result.results['perturbed_f1']:.4f}")
        
        # Missing features
        missing_result = missing_feature_test(
            trainer, X_test, y_test,
            missing_rate=0.1,
        )
        print(f"\n  Missing Feature Test: {'PASS' if missing_result.passed else 'FAIL'}")
        print(f"    Baseline F1: {missing_result.baseline_metric:.4f}")
        print(f"    With 10% missing: F1={missing_result.results['missing_f1']:.4f}")
    
    # =========================================================================
    # Generate Report
    # =========================================================================
    logger.info("Generating report...")
    
    report = generate_report(y_test, y_pred_proba, test_df, slice_columns)
    
    # Save report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    with open(report_path, "w") as f:
        yaml.dump({
            "timestamp": datetime.now().isoformat(),
            "test_size": len(test_df),
            "fraud_rate": float(y_test.mean()),
            "metrics": report.overall_metrics,
            "calibration": report.calibration,
            "slice_metrics": report.slice_metrics,
        }, f, default_flow_style=False)
    
    logger.info(f"Report saved to {report_path}")
    
    # Generate plots
    if args.plot:
        plot_path = output_dir / f"evaluation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_evaluation_report(report, save_path=plot_path)
        logger.info(f"Plots saved to {plot_path}")
    
    # =========================================================================
    # CI Regression Check
    # =========================================================================
    ci_gates = config.get("evaluation", {}).get("ci_gates", {})
    
    if ci_gates:
        print("\n" + "=" * 60)
        print("CI REGRESSION GATES")
        print("=" * 60)
        
        all_passed = True
        
        for metric_name, threshold in ci_gates.items():
            current_value = report.overall_metrics.get(metric_name, 0)
            passed = current_value >= threshold
            all_passed = all_passed and passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {metric_name}: {current_value:.4f} >= {threshold:.4f} [{status}]")
        
        print("\n" + "=" * 60)
        if all_passed:
            print("ALL CI GATES PASSED")
        else:
            print("CI GATES FAILED - Do not merge!")
            exit(1)
        print("=" * 60)


if __name__ == "__main__":
    main()
