"""
FraudShield Configuration
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Model settings
MODEL_FILE = MODELS_DIR / "fraud_model.joblib"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"

# Feature engineering
MERCHANT_CATEGORIES = [
    "retail", "grocery", "restaurant", "travel", "entertainment",
    "utilities", "healthcare", "gas_station", "online", "other"
]

AMOUNT_BINS = [0, 50, 200, 1000, 5000, float("inf")]
AMOUNT_LABELS = ["micro", "low", "medium", "high", "very_high"]

# Risk tiers
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.7,
    "high": 1.0
}

# Drift detection
PSI_WARNING_THRESHOLD = 0.1
PSI_ALERT_THRESHOLD = 0.2

# API settings
API_TIMEOUT_MS = 100
MAX_BATCH_SIZE = 100
