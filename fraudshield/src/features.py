"""
Feature Engineering Pipeline for FraudShield

Transforms raw transaction data into model features.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from .config import MERCHANT_CATEGORIES, AMOUNT_BINS, AMOUNT_LABELS


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    
    Handles:
    - Numerical feature scaling
    - Categorical encoding
    - Time-based features
    - Risk indicator derivation
    """
    
    def __init__(self):
        self.merchant_category_map = {cat: i for i, cat in enumerate(MERCHANT_CATEGORIES)}
        self.feature_names: List[str] = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names for model input."""
        self.feature_names = [
            "amount_log",
            "amount_bin",
            "merchant_category_encoded",
            "hour",
            "day_of_week",
            "is_weekend",
            "is_night",  # 10pm - 6am
            "is_international",
            "card_present",
            "merchant_risk_score",
            "hour_sin",
            "hour_cos",
        ]
    
    def transform_single(self, transaction: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single transaction into feature vector.
        
        Args:
            transaction: Raw transaction data
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Amount features
        amount = float(transaction.get("amount", 0))
        features.append(np.log1p(amount))  # amount_log
        features.append(self._get_amount_bin(amount))  # amount_bin
        
        # Merchant category
        category = transaction.get("merchant_category", "other")
        features.append(self.merchant_category_map.get(category, len(MERCHANT_CATEGORIES) - 1))
        
        # Time features
        hour = int(transaction.get("hour", 12))
        day_of_week = int(transaction.get("day_of_week", 0))
        
        features.append(hour)  # hour
        features.append(day_of_week)  # day_of_week
        features.append(1 if day_of_week >= 5 else 0)  # is_weekend
        features.append(1 if hour >= 22 or hour < 6 else 0)  # is_night
        
        # Transaction flags
        features.append(1 if transaction.get("is_international", False) else 0)
        features.append(1 if transaction.get("card_present", True) else 0)
        
        # Merchant risk
        features.append(float(transaction.get("merchant_risk_score", 0.5)))
        
        # Cyclical time encoding
        features.append(np.sin(2 * np.pi * hour / 24))  # hour_sin
        features.append(np.cos(2 * np.pi * hour / 24))  # hour_cos
        
        return np.array(features, dtype=np.float32)
    
    def transform_batch(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Transform a batch of transactions.
        
        Args:
            transactions: List of raw transaction data
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        return np.vstack([self.transform_single(t) for t in transactions])
    
    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a pandas DataFrame into feature matrix.
        
        Args:
            df: DataFrame with transaction columns
            
        Returns:
            Feature matrix
        """
        transactions = df.to_dict("records")
        return self.transform_batch(transactions)
    
    def _get_amount_bin(self, amount: float) -> int:
        """Map amount to bin index."""
        for i, threshold in enumerate(AMOUNT_BINS[1:]):
            if amount < threshold:
                return i
        return len(AMOUNT_BINS) - 2
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()


def generate_reason_codes(
    transaction: Dict[str, Any],
    risk_score: float
) -> List[str]:
    """
    Generate human-readable reason codes for a fraud score.
    
    Args:
        transaction: Raw transaction data
        risk_score: Model output score (0-1)
        
    Returns:
        List of reason codes explaining the score
    """
    reasons = []
    
    amount = float(transaction.get("amount", 0))
    hour = int(transaction.get("hour", 12))
    is_international = transaction.get("is_international", False)
    card_present = transaction.get("card_present", True)
    merchant_risk = float(transaction.get("merchant_risk_score", 0.5))
    
    # Amount-based reasons
    if amount > 5000:
        reasons.append("very_high_amount")
    elif amount > 1000:
        reasons.append("high_amount")
    elif amount < 50:
        reasons.append("normal_amount")
    
    # Time-based reasons
    if hour >= 22 or hour < 6:
        reasons.append("unusual_hour")
    
    # Transaction type reasons
    if is_international:
        reasons.append("international_transaction")
    
    if not card_present:
        reasons.append("card_not_present")
    
    # Merchant risk
    if merchant_risk > 0.7:
        reasons.append("high_risk_merchant")
    elif merchant_risk < 0.3:
        reasons.append("known_merchant_category")
    
    # Score-based summary
    if risk_score > 0.7:
        reasons.insert(0, "high_risk_pattern")
    elif risk_score < 0.3:
        if not reasons:
            reasons.append("normal_pattern")
    
    return reasons if reasons else ["standard_transaction"]
