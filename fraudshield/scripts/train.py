"""
FraudShield Training Script

Generates synthetic fraud data and trains XGBoost model.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from src.features import FeatureEngineer
from src.model import train_model
from src.config import MODELS_DIR, DATA_DIR, MERCHANT_CATEGORIES


def generate_synthetic_data(n_samples: int = 50000, fraud_rate: float = 0.023) -> pd.DataFrame:
    """
    Generate synthetic fraud detection dataset.
    
    Mimics realistic patterns:
    - Most transactions are legitimate
    - Fraud more common at night, international, high amounts
    - Class imbalance (~2.3% fraud rate)
    """
    np.random.seed(42)
    
    n_fraud = int(n_samples * fraud_rate)
    n_legit = n_samples - n_fraud
    
    # Generate legitimate transactions
    legit_data = {
        "amount": np.random.lognormal(mean=4, sigma=1.2, size=n_legit).clip(1, 10000),
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, size=n_legit, 
            p=[0.25, 0.20, 0.15, 0.05, 0.10, 0.05, 0.05, 0.05, 0.08, 0.02]),
        "hour": np.random.choice(24, size=n_legit, 
            p=[0.02]*6 + [0.05]*4 + [0.07]*4 + [0.06]*4 + [0.04]*4 + [0.02]*2),  # More during day
        "day_of_week": np.random.choice(7, size=n_legit),
        "is_international": np.random.random(n_legit) < 0.05,
        "card_present": np.random.random(n_legit) < 0.7,
        "merchant_risk_score": np.random.beta(2, 5, n_legit),  # Skewed toward low risk
        "is_fraud": np.zeros(n_legit, dtype=int)
    }
    
    # Generate fraudulent transactions (different patterns)
    fraud_data = {
        "amount": np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud).clip(1, 50000),  # Higher amounts
        "merchant_category": np.random.choice(MERCHANT_CATEGORIES, size=n_fraud,
            p=[0.10, 0.05, 0.05, 0.15, 0.10, 0.02, 0.03, 0.05, 0.40, 0.05]),  # More online
        "hour": np.random.choice(24, size=n_fraud,
            p=[0.08]*6 + [0.03]*4 + [0.02]*4 + [0.03]*4 + [0.04]*4 + [0.06]*2),  # More at night
        "day_of_week": np.random.choice(7, size=n_fraud),
        "is_international": np.random.random(n_fraud) < 0.25,  # More international
        "card_present": np.random.random(n_fraud) < 0.3,  # Less card present
        "merchant_risk_score": np.random.beta(5, 2, n_fraud),  # Skewed toward high risk
        "is_fraud": np.ones(n_fraud, dtype=int)
    }
    
    # Combine
    df_legit = pd.DataFrame(legit_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    print("=" * 60)
    print("FraudShield Model Training")
    print("=" * 60)
    
    # Generate data
    print("\n[1/4] Generating synthetic data...")
    df = generate_synthetic_data(n_samples=50000, fraud_rate=0.023)
    
    # Save data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_path = DATA_DIR / "transactions.csv"
    df.to_csv(data_path, index=False)
    print(f"  - Saved {len(df):,} transactions to {data_path}")
    print(f"  - Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # Feature engineering
    print("\n[2/4] Engineering features...")
    fe = FeatureEngineer()
    X = fe.transform_dataframe(df)
    y = df["is_fraud"].values
    print(f"  - Feature matrix shape: {X.shape}")
    print(f"  - Features: {fe.get_feature_names()}")
    
    # Train/test split
    print("\n[3/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  - Train: {len(X_train):,} samples ({y_train.mean():.2%} fraud)")
    print(f"  - Test:  {len(X_test):,} samples ({y_test.mean():.2%} fraud)")
    
    # Train model
    print("\n[4/4] Training XGBoost model...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "fraud_model.joblib"
    model = train_model(X_train, y_train, output_path=model_path)
    print(f"  - Model saved to {model_path}")
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
    # Feature importance
    print("\nTop 5 Feature Importances:")
    importance = model.feature_importances_
    feature_names = fe.get_feature_names()
    for idx in np.argsort(importance)[::-1][:5]:
        print(f"  - {feature_names[idx]}: {importance[idx]:.4f}")
    
    print("\n✅ Training complete!")
    print(f"   Run the API with: uvicorn src.api:app --reload")


if __name__ == "__main__":
    main()
