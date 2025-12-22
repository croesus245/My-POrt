"""
FraudShield Model Wrapper

XGBoost-based fraud detection model with inference utilities.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import xgboost as xgb
    import joblib
except ImportError:
    xgb = None
    joblib = None

from .config import MODEL_FILE, FEATURE_NAMES_FILE, RISK_THRESHOLDS
from .features import FeatureEngineer, generate_reason_codes

logger = logging.getLogger(__name__)


class FraudModel:
    """
    XGBoost fraud detection model wrapper.
    
    Handles model loading, inference, and risk tier classification.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the fraud model.
        
        Args:
            model_path: Path to trained model file. If None, uses default.
        """
        self.model_path = model_path or MODEL_FILE
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_engineer = FeatureEngineer()
        self._loaded = False
    
    def load(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        if self._loaded:
            return True
        
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return False
        
        try:
            self.model = joblib.load(self.model_path)
            self._loaded = True
            logger.info(f"Model loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, transaction: Dict[str, Any]) -> Tuple[float, str, list]:
        """
        Predict fraud risk for a single transaction.
        
        Args:
            transaction: Transaction data dictionary
            
        Returns:
            Tuple of (risk_score, risk_tier, reason_codes)
        """
        if not self._loaded:
            # Return mock score if model not loaded (for demo purposes)
            return self._mock_predict(transaction)
        
        # Transform features
        features = self.feature_engineer.transform_single(transaction)
        features = features.reshape(1, -1)
        
        # Get probability
        risk_score = float(self.model.predict_proba(features)[0, 1])
        
        # Determine risk tier
        risk_tier = self._get_risk_tier(risk_score)
        
        # Generate reason codes
        reason_codes = generate_reason_codes(transaction, risk_score)
        
        return risk_score, risk_tier, reason_codes
    
    def predict_batch(self, transactions: list) -> list:
        """
        Predict fraud risk for a batch of transactions.
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            List of (risk_score, risk_tier, reason_codes) tuples
        """
        if not self._loaded:
            return [self._mock_predict(t) for t in transactions]
        
        # Transform features
        features = self.feature_engineer.transform_batch(transactions)
        
        # Get probabilities
        risk_scores = self.model.predict_proba(features)[:, 1]
        
        # Build results
        results = []
        for i, transaction in enumerate(transactions):
            score = float(risk_scores[i])
            tier = self._get_risk_tier(score)
            reasons = generate_reason_codes(transaction, score)
            results.append((score, tier, reasons))
        
        return results
    
    def _get_risk_tier(self, score: float) -> str:
        """Map score to risk tier."""
        if score < RISK_THRESHOLDS["low"]:
            return "low"
        elif score < RISK_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "high"
    
    def _mock_predict(self, transaction: Dict[str, Any]) -> Tuple[float, str, list]:
        """
        Generate mock prediction when model isn't loaded.
        Uses rule-based heuristics for demonstration.
        """
        score = 0.1
        
        # Amount-based risk
        amount = float(transaction.get("amount", 0))
        if amount > 5000:
            score += 0.3
        elif amount > 1000:
            score += 0.15
        
        # Time-based risk
        hour = int(transaction.get("hour", 12))
        if hour >= 22 or hour < 6:
            score += 0.1
        
        # Transaction type risk
        if transaction.get("is_international", False):
            score += 0.15
        if not transaction.get("card_present", True):
            score += 0.1
        
        # Merchant risk
        merchant_risk = float(transaction.get("merchant_risk_score", 0.5))
        score += merchant_risk * 0.2
        
        # Clamp score
        score = min(max(score, 0.0), 1.0)
        
        tier = self._get_risk_tier(score)
        reasons = generate_reason_codes(transaction, score)
        
        return score, tier, reasons


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_path: Optional[Path] = None
) -> xgb.XGBClassifier:
    """
    Train a fraud detection model.
    
    Args:
        X_train: Training features
        y_train: Training labels (0=legitimate, 1=fraud)
        output_path: Path to save trained model
        
    Returns:
        Trained XGBClassifier
    """
    # Class weight for imbalanced data (fraud is rare)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_path)
        logger.info(f"Model saved to {output_path}")
    
    return model
