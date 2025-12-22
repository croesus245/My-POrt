"""
Drift Detection for FraudShield

PSI-based monitoring for feature and score distribution shifts.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .config import PSI_WARNING_THRESHOLD, PSI_ALERT_THRESHOLD


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    feature_name: str
    psi_value: float
    status: str  # "ok", "warning", "alert"
    baseline_distribution: List[float]
    current_distribution: List[float]
    bin_edges: List[float]


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Calculate Population Stability Index (PSI) between two distributions.
    
    PSI measures how much a distribution has shifted from a baseline.
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Moderate change (warning)
    - PSI >= 0.2: Significant change (alert)
    
    Args:
        baseline: Baseline distribution samples
        current: Current distribution samples
        n_bins: Number of bins for histogram
        
    Returns:
        Tuple of (psi_value, baseline_percentages, current_percentages, bin_edges)
    """
    # Calculate bin edges from baseline
    bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Calculate histograms
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)
    
    # Convert to percentages (with smoothing to avoid division by zero)
    eps = 1e-6
    baseline_pct = (baseline_counts + eps) / (len(baseline) + eps * n_bins)
    current_pct = (current_counts + eps) / (len(current) + eps * n_bins)
    
    # Calculate PSI
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    
    return float(psi), baseline_pct.tolist(), current_pct.tolist(), bin_edges.tolist()


def get_drift_status(psi: float) -> str:
    """Map PSI value to drift status."""
    if psi < PSI_WARNING_THRESHOLD:
        return "ok"
    elif psi < PSI_ALERT_THRESHOLD:
        return "warning"
    else:
        return "alert"


class DriftDetector:
    """
    Monitor feature distributions for drift.
    
    Compares current production data against a baseline (training data)
    to detect distribution shifts that might indicate model degradation.
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize drift detector.
        
        Args:
            n_bins: Number of bins for PSI calculation
        """
        self.n_bins = n_bins
        self.baselines: Dict[str, np.ndarray] = {}
    
    def set_baseline(self, feature_name: str, values: np.ndarray):
        """
        Set baseline distribution for a feature.
        
        Args:
            feature_name: Name of the feature
            values: Baseline values (e.g., from training data)
        """
        self.baselines[feature_name] = np.array(values)
    
    def set_baselines_from_dict(self, baselines: Dict[str, np.ndarray]):
        """Set multiple baselines at once."""
        for name, values in baselines.items():
            self.set_baseline(name, values)
    
    def check_drift(
        self,
        feature_name: str,
        current_values: np.ndarray
    ) -> Optional[DriftResult]:
        """
        Check for drift in a single feature.
        
        Args:
            feature_name: Name of the feature to check
            current_values: Current production values
            
        Returns:
            DriftResult or None if no baseline exists
        """
        if feature_name not in self.baselines:
            return None
        
        baseline = self.baselines[feature_name]
        current = np.array(current_values)
        
        psi, baseline_dist, current_dist, bin_edges = calculate_psi(
            baseline, current, self.n_bins
        )
        
        return DriftResult(
            feature_name=feature_name,
            psi_value=psi,
            status=get_drift_status(psi),
            baseline_distribution=baseline_dist,
            current_distribution=current_dist,
            bin_edges=bin_edges
        )
    
    def check_all_features(
        self,
        current_data: Dict[str, np.ndarray]
    ) -> List[DriftResult]:
        """
        Check drift across all monitored features.
        
        Args:
            current_data: Dict mapping feature names to current values
            
        Returns:
            List of DriftResults for each feature
        """
        results = []
        for feature_name in self.baselines:
            if feature_name in current_data:
                result = self.check_drift(feature_name, current_data[feature_name])
                if result:
                    results.append(result)
        return results
    
    def get_alert_summary(
        self,
        current_data: Dict[str, np.ndarray]
    ) -> Dict[str, List[str]]:
        """
        Get summary of drift alerts.
        
        Args:
            current_data: Current feature values
            
        Returns:
            Dict with "warnings" and "alerts" lists
        """
        results = self.check_all_features(current_data)
        
        summary = {
            "ok": [],
            "warnings": [],
            "alerts": []
        }
        
        for result in results:
            if result.status == "warning":
                summary["warnings"].append(
                    f"{result.feature_name}: PSI={result.psi_value:.3f}"
                )
            elif result.status == "alert":
                summary["alerts"].append(
                    f"{result.feature_name}: PSI={result.psi_value:.3f}"
                )
            else:
                summary["ok"].append(result.feature_name)
        
        return summary


def generate_drift_report(
    detector: DriftDetector,
    current_data: Dict[str, np.ndarray]
) -> str:
    """
    Generate a markdown drift report.
    
    Args:
        detector: Configured DriftDetector
        current_data: Current production data
        
    Returns:
        Markdown formatted report
    """
    results = detector.check_all_features(current_data)
    summary = detector.get_alert_summary(current_data)
    
    lines = [
        "# Drift Detection Report",
        "",
        "## Summary",
        f"- **Features monitored:** {len(results)}",
        f"- **OK:** {len(summary['ok'])}",
        f"- **Warnings:** {len(summary['warnings'])}",
        f"- **Alerts:** {len(summary['alerts'])}",
        "",
    ]
    
    if summary["alerts"]:
        lines.extend([
            "## 🚨 Alerts (PSI >= 0.2)",
            "",
        ])
        for alert in summary["alerts"]:
            lines.append(f"- {alert}")
        lines.append("")
    
    if summary["warnings"]:
        lines.extend([
            "## ⚠️ Warnings (0.1 <= PSI < 0.2)",
            "",
        ])
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")
    
    lines.extend([
        "## Feature Details",
        "",
        "| Feature | PSI | Status |",
        "|---------|-----|--------|",
    ])
    
    for result in sorted(results, key=lambda x: -x.psi_value):
        status_icon = {"ok": "✅", "warning": "⚠️", "alert": "🚨"}[result.status]
        lines.append(
            f"| {result.feature_name} | {result.psi_value:.4f} | {status_icon} {result.status} |"
        )
    
    return "\n".join(lines)
