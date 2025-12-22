"""
FraudShield API

FastAPI-based real-time fraud scoring endpoint.
"""
import time
import uuid
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model import FraudModel
from .config import RISK_THRESHOLDS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
model: Optional[FraudModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model = FraudModel()
    loaded = model.load()
    if loaded:
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not found - using mock predictions")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="FraudShield API",
    description="Real-time fraud detection scoring API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class TransactionRequest(BaseModel):
    """Single transaction scoring request."""
    amount: float = Field(..., ge=0, description="Transaction amount in dollars")
    merchant_category: str = Field(default="other", description="Merchant category")
    hour: int = Field(default=12, ge=0, le=23, description="Hour of transaction (0-23)")
    day_of_week: int = Field(default=0, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    is_international: bool = Field(default=False, description="International transaction flag")
    card_present: bool = Field(default=True, description="Card physically present")
    merchant_risk_score: float = Field(default=0.5, ge=0, le=1, description="Merchant risk score")


class ScoreResponse(BaseModel):
    """Fraud score response."""
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Fraud risk score (0-1)")
    risk_tier: str = Field(..., description="Risk tier: low, medium, high")
    reason_codes: List[str] = Field(default_factory=list, description="Explanation codes")
    latency_ms: float = Field(..., description="Processing latency in milliseconds")


class BatchRequest(BaseModel):
    """Batch scoring request."""
    transactions: List[TransactionRequest]


class BatchResponse(BaseModel):
    """Batch scoring response."""
    results: List[ScoreResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model._loaded if model else False,
        version="1.0.0"
    )


@app.post("/score", response_model=ScoreResponse)
async def score_transaction(request: TransactionRequest):
    """
    Score a single transaction for fraud risk.
    
    Returns risk score (0-1), risk tier, and reason codes.
    """
    start_time = time.perf_counter()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Convert to dict for model
    transaction = request.model_dump()
    
    # Get prediction
    risk_score, risk_tier, reason_codes = model.predict(transaction)
    
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return ScoreResponse(
        transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
        risk_score=round(risk_score, 4),
        risk_tier=risk_tier,
        reason_codes=reason_codes,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch(request: BatchRequest):
    """
    Score a batch of transactions.
    
    More efficient than calling /score multiple times.
    """
    start_time = time.perf_counter()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if len(request.transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100")
    
    transactions = [t.model_dump() for t in request.transactions]
    predictions = model.predict_batch(transactions)
    
    results = []
    for risk_score, risk_tier, reason_codes in predictions:
        results.append(ScoreResponse(
            transaction_id=f"txn_{uuid.uuid4().hex[:12]}",
            risk_score=round(risk_score, 4),
            risk_tier=risk_tier,
            reason_codes=reason_codes,
            latency_ms=0  # Individual latencies not tracked in batch
        ))
    
    total_latency_ms = (time.perf_counter() - start_time) * 1000
    
    return BatchResponse(
        results=results,
        total_latency_ms=round(total_latency_ms, 2)
    )


@app.get("/thresholds")
async def get_thresholds():
    """Get current risk tier thresholds."""
    return {
        "thresholds": RISK_THRESHOLDS,
        "description": {
            "low": "Score < 0.3 - Low risk, auto-approve",
            "medium": "0.3 <= Score < 0.7 - Review recommended",
            "high": "Score >= 0.7 - High risk, manual review required"
        }
    }


# Run with: uvicorn src.api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
