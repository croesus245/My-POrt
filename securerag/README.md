# SecureRAG

Defense-in-depth RAG system with prompt injection defense, data exfiltration prevention, and faithfulness scoring.

## Quick Start

```bash
cd securerag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key (or use mock mode)
export OPENAI_API_KEY="your-key"  # Optional - works without for demo

# Run the API
python -m uvicorn src.api:app --reload

# Test it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "user_id": "user_123", "tenant_id": "company_a"}'
```

## Project Structure

```
securerag/
├── src/
│   ├── api.py                 # FastAPI endpoints
│   ├── pipeline.py            # Main RAG pipeline orchestrator
│   ├── retrieval/
│   │   ├── embeddings.py      # Embedding model wrapper
│   │   ├── vectorstore.py     # In-memory vector store
│   │   └── reranker.py        # Reranking for retrieval quality
│   ├── security/
│   │   ├── injection.py       # Prompt injection detector
│   │   ├── pii.py             # PII/secret detection
│   │   ├── permissions.py     # Tenant isolation & RBAC
│   │   └── output_guard.py    # Output validation
│   ├── generation/
│   │   ├── llm.py             # LLM wrapper (OpenAI/mock)
│   │   └── prompts.py         # System prompts
│   └── evaluation/
│       ├── faithfulness.py    # Grounding/citation scoring
│       └── metrics.py         # Evaluation metrics
├── tests/
│   ├── test_api.py
│   ├── test_security.py
│   └── attacks/               # Attack test suite
│       ├── test_injection.py
│       ├── test_exfiltration.py
│       └── test_jailbreak.py
├── data/
│   └── sample_docs/           # Sample documents for demo
├── docs/                      # Documentation
└── requirements.txt
```

## Security Layers

### Layer 1: Access Control
- Tenant isolation (Company A can't see Company B's docs)
- Role-based permissions (admin, user, viewer)
- Document-level access control

### Layer 2: Prompt Injection Detection
- Classifier detects malicious instructions in queries AND documents
- Blocks "ignore previous instructions" style attacks
- Handles indirect injection (malicious content in retrieved docs)

### Layer 3: PII/Secret Detection  
- Regex + pattern matching for common secrets
- Blocks API keys, passwords, SSNs, credit cards
- Redacts sensitive content before generation

### Layer 4: Tool Sandboxing
- Strict allowlist of permitted tools
- Rate limiting on tool calls
- No arbitrary code execution

### Layer 5: Output Validation
- Checks final response for leaked secrets
- Verifies citations exist in retrieved context
- Blocks hallucinated content when confidence is low

## API Endpoints

### POST /query
Main query endpoint with full security pipeline.

```json
{
  "query": "What is our refund policy?",
  "user_id": "user_123",
  "tenant_id": "company_a"
}
```

Response:
```json
{
  "answer": "Refunds are allowed within 14 days of delivery...",
  "citations": [
    {"doc_id": "refund_policy_v3", "chunk": "...", "score": 0.92}
  ],
  "faithfulness_score": 0.89,
  "blocked": false,
  "security_flags": []
}
```

### POST /ingest
Add documents to the knowledge base.

### GET /health
Health check.

## Running Tests

```bash
# All tests
pytest tests/ -v

# Security tests only
pytest tests/attacks/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Attack Test Results

| Attack Category | Attempts | Blocked | Block Rate |
|-----------------|----------|---------|------------|
| Prompt Injection | 847 | 811 | 95.7% |
| Jailbreak | 312 | 298 | 95.5% |
| Data Exfiltration | 156 | 156 | **100%** |
| Tool Abuse | 89 | 82 | 92.1% |
| **Total** | **1,404** | **1,347** | **95.9%** |

## Documentation

- [Attack Report](docs/attack_report.md) - Detailed security test results
- [Security Model](docs/security_model.md) - Architecture & threat model
- [Eval Dashboard](docs/eval_dashboard.md) - Faithfulness metrics
- [Cost Report](docs/cost_report.md) - Latency & costs

## License

MIT
