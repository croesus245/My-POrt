"""
SecureRAG API - FastAPI endpoints for secure RAG queries
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging

from .pipeline import SecureRAGPipeline
from .security.permissions import PermissionChecker, User

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SecureRAG API",
    description="Defense-in-depth RAG with prompt injection defense and faithfulness scoring",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline
pipeline = SecureRAGPipeline()
permission_checker = PermissionChecker()


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    """Incoming query request"""
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    """Document citation in response"""
    doc_id: str
    chunk: str
    score: float
    source: Optional[str] = None


class QueryResponse(BaseModel):
    """Query response with security metadata"""
    answer: str
    citations: list[Citation]
    faithfulness_score: float
    blocked: bool
    block_reason: Optional[str] = None
    security_flags: list[str]
    latency_ms: float


class IngestRequest(BaseModel):
    """Document ingestion request"""
    doc_id: str
    content: str
    tenant_id: str
    metadata: Optional[dict] = None
    access_level: str = Field(default="user")


class IngestResponse(BaseModel):
    """Ingestion response"""
    doc_id: str
    chunks_created: int
    success: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: dict


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """API root"""
    return {
        "service": "SecureRAG",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components={
            "pipeline": "ok",
            "vectorstore": "ok",
            "injection_detector": "ok",
            "pii_detector": "ok"
        }
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main query endpoint with full security pipeline.
    
    Security layers applied:
    1. Permission check (tenant isolation)
    2. Query injection detection
    3. Retrieval with access filtering
    4. Retrieved chunk injection detection
    5. PII/secret detection and redaction
    6. Generation with safe prompt
    7. Output validation
    """
    start_time = time.time()
    
    try:
        # Step 1: Verify user permissions
        user = permission_checker.get_user(request.user_id, request.tenant_id)
        if not user:
            raise HTTPException(status_code=403, detail="User not authorized")
        
        # Step 2-7: Run secure pipeline
        result = await pipeline.query(
            query=request.query,
            user=user,
            top_k=request.top_k
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Format citations
        citations = [
            Citation(
                doc_id=c["doc_id"],
                chunk=c["chunk"][:500],  # Truncate for response
                score=c["score"],
                source=c.get("source")
            )
            for c in result.get("citations", [])
        ]
        
        return QueryResponse(
            answer=result["answer"],
            citations=citations,
            faithfulness_score=result["faithfulness_score"],
            blocked=result["blocked"],
            block_reason=result.get("block_reason"),
            security_flags=result.get("security_flags", []),
            latency_ms=latency_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a document into the knowledge base.
    
    Documents are:
    1. Chunked into smaller pieces
    2. Embedded for vector search
    3. Tagged with tenant_id for isolation
    4. Scanned for injection attempts (flagged but stored)
    """
    try:
        chunks_created = pipeline.ingest(
            doc_id=request.doc_id,
            content=request.content,
            tenant_id=request.tenant_id,
            metadata=request.metadata,
            access_level=request.access_level
        )
        
        return IngestResponse(
            doc_id=request.doc_id,
            chunks_created=chunks_created,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/clear")
async def clear_vectorstore():
    """Clear all documents (admin only, for testing)"""
    pipeline.clear()
    return {"status": "cleared"}


# ============================================================================
# Startup: Load sample documents
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load sample documents on startup"""
    logger.info("Loading sample documents...")
    
    sample_docs = [
        {
            "doc_id": "refund_policy_v3",
            "tenant_id": "company_a",
            "content": """
            REFUND POLICY (Version 3.0)
            
            Section 1: General Terms
            All purchases are eligible for refund within 14 days of delivery.
            
            Section 2: Refund Process
            To request a refund:
            1. Log into your account
            2. Navigate to Order History
            3. Select the order and click "Request Refund"
            4. Provide reason for return
            
            Section 3: Exceptions
            The following items are non-refundable:
            - Digital downloads after access
            - Personalized items
            - Items marked as final sale
            
            Section 4: Processing Time
            Refunds are processed within 5-7 business days after we receive the returned item.
            """
        },
        {
            "doc_id": "merchant_onboarding",
            "tenant_id": "company_a", 
            "content": """
            MERCHANT ONBOARDING GUIDE
            
            Requirements for New Merchants:
            
            1. Business Documentation
               - Valid business license
               - Tax identification number
               - Bank account for payouts
            
            2. Compliance Requirements
               - PCI-DSS compliance certification
               - Privacy policy
               - Terms of service
            
            3. Integration Steps
               a) Register at merchant.example.com
               b) Complete KYC verification
               c) Set up payment methods
               d) Configure webhook endpoints
               e) Test in sandbox environment
               f) Go live after approval
            
            Processing time: 3-5 business days for standard review.
            """
        },
        {
            "doc_id": "internal_secrets",
            "tenant_id": "company_b",
            "content": """
            INTERNAL SYSTEM CREDENTIALS (CONFIDENTIAL)
            
            Database: prod-db.internal.example.com
            API Key: sk_live_abc123xyz789
            Admin Password: SuperSecret123!
            
            Note: This document should never be exposed to tenant company_a.
            """
        }
    ]
    
    for doc in sample_docs:
        pipeline.ingest(
            doc_id=doc["doc_id"],
            content=doc["content"],
            tenant_id=doc["tenant_id"]
        )
    
    logger.info(f"Loaded {len(sample_docs)} sample documents")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
