# SecureRAG Evaluation Dashboard

**Last Updated:** 2025-12-15  
**System Version:** v0.4.2

---

## Quick Stats

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Answer Accuracy (w/ retrieval) | 84% | ≥ 80% | ✅ |
| Answer Accuracy (no retrieval) | 61% | — | Baseline |
| Retrieval Precision@5 | 0.72 | ≥ 0.70 | ✅ |
| Retrieval Recall@10 | 0.85 | ≥ 0.80 | ✅ |
| Security Block Rate | 95.9% | ≥ 95% | ✅ |
| Exfiltration Block Rate | 100% | 100% | ✅ |
| p95 Latency | 2.1s | < 3s | ✅ |
| Throughput | ~45 QPS | — | Info |

---

## Accuracy Breakdown by Query Type

| Query Type | N | Accuracy | Notes |
|------------|---|----------|-------|
| Factual lookup | 450 | 91% | Direct fact retrieval |
| Multi-hop reasoning | 180 | 76% | Requires connecting multiple docs |
| Summarization | 120 | 88% | Condense document content |
| Comparison | 95 | 79% | Compare across documents |
| Temporal | 65 | 72% | "What changed since..." |
| Unanswerable | 90 | 82% | Correctly says "I don't know" |

**Evaluation method:** Human-labeled ground truth on held-out test set

---

## Retrieval Quality

### Embedding Model: `text-embedding-3-small`

| Metric | Value |
|--------|-------|
| Precision@1 | 0.58 |
| Precision@5 | 0.72 |
| Precision@10 | 0.68 |
| Recall@5 | 0.71 |
| Recall@10 | 0.85 |
| MRR | 0.64 |
| NDCG@10 | 0.73 |

### Retrieval Failure Analysis

| Failure Mode | % of Errors | Example |
|--------------|-------------|---------|
| Semantic gap | 35% | Query uses different terminology than docs |
| Multi-doc answer | 28% | Answer spans multiple documents |
| Recency bias | 18% | Retrieved older doc, answer in newer |
| Embedding ambiguity | 12% | Query matches wrong domain |
| Chunking artifact | 7% | Answer split across chunks |

---

## Latency Breakdown

### End-to-End (p50 / p95 / p99)

| Stage | p50 | p95 | p99 |
|-------|-----|-----|-----|
| Input validation | 5ms | 12ms | 18ms |
| Embedding | 45ms | 62ms | 85ms |
| Vector search | 25ms | 38ms | 52ms |
| Permission filter | 3ms | 8ms | 15ms |
| LLM generation | 1.2s | 1.8s | 2.4s |
| Output validation | 35ms | 58ms | 95ms |
| **Total** | **1.3s** | **2.1s** | **2.8s** |

**Note:** LLM generation dominates. Using GPT-4-turbo with streaming.

### Throughput Under Load

| Concurrent Users | QPS | p95 Latency | Error Rate |
|------------------|-----|-------------|------------|
| 10 | 8 | 1.9s | 0% |
| 25 | 22 | 2.2s | 0% |
| 50 | 42 | 2.8s | 0.1% |
| 100 | 45 | 4.1s | 2.3% |
| 150 | 43 | 6.2s | 8.5% |

**Saturation point:** ~45 QPS with acceptable latency

---

## Security Metrics

### Attack Detection by Layer

| Layer | Attacks Blocked | % of Total Blocks |
|-------|-----------------|-------------------|
| Input Validator | 897 | 67% |
| Permission Filter | 156 | 12% |
| Tool Sandbox | 75 | 6% |
| Output Validator | 219 | 16% |

### False Positive Rate

| Metric | Value | Target |
|--------|-------|--------|
| Input validator FP | 0.3% | < 1% |
| Output validator FP | 1.2% | < 2% |
| Overall user friction | 1.5% | < 3% |

**Measurement:** 10,000 benign queries from production logs (anonymized)

---

## Cost Analysis

### Per-Query Cost Breakdown

| Component | Cost/Query | % of Total |
|-----------|------------|------------|
| Embedding (input) | $0.00002 | 1% |
| Vector DB query | $0.00001 | <1% |
| LLM (GPT-4-turbo) | $0.0045 | 97% |
| Output validation | $0.0001 | 2% |
| **Total** | **~$0.0046** | **100%** |

### Monthly Cost Projection

| Query Volume | LLM Cost | Infra | Total |
|--------------|----------|-------|-------|
| 100K/month | $460 | $150 | $610 |
| 500K/month | $2,300 | $300 | $2,600 |
| 1M/month | $4,600 | $500 | $5,100 |

---

## Quality Over Time

### Weekly Accuracy Trend (Last 8 Weeks)

```
Week 1:  ████████████████████░░░░░ 82%
Week 2:  █████████████████████░░░░ 83%
Week 3:  █████████████████████░░░░ 84%
Week 4:  ████████████████████░░░░░ 81%  ← Index refresh issue
Week 5:  █████████████████████░░░░ 84%
Week 6:  █████████████████████░░░░ 84%
Week 7:  ██████████████████████░░░ 85%
Week 8:  █████████████████████░░░░ 84%
```

### Drift Indicators

| Metric | Baseline | Current | Status |
|--------|----------|---------|--------|
| Query embedding drift (cosine) | — | 0.03 | ✅ Stable |
| Document distribution shift | — | 0.05 | ✅ Stable |
| Answer length variance | 142 tokens | 148 tokens | ✅ Normal |
| "I don't know" rate | 8.2% | 8.5% | ✅ Normal |

---

## A/B Test Results

### Chunking Strategy (Completed 2025-11)

| Variant | Accuracy | Latency | Winner |
|---------|----------|---------|--------|
| 512 token chunks | 82% | 2.0s | — |
| 256 token chunks + overlap | 84% | 2.1s | ✅ |
| Semantic chunking | 85% | 2.4s | — |

**Decision:** 256 token + 64 token overlap (balance of accuracy and latency)

### Reranking (In Progress)

| Variant | Accuracy | Latency | Cost |
|---------|----------|---------|------|
| No reranker (control) | 84% | 2.1s | $0.0046 |
| Cohere rerank | 86% | 2.4s | $0.0052 |
| Cross-encoder | 87% | 2.8s | $0.0048 |

**Status:** Evaluating cross-encoder for accuracy-sensitive use cases

---

## Failure Case Examples

### Retrieval Miss
**Query:** "What's our policy on remote work in Germany?"  
**Retrieved:** US remote work policy, UK policy  
**Correct doc:** Germany-specific addendum (different terminology)  
**Fix:** Added keyword expansion for country names

### Multi-hop Failure
**Query:** "Compare Q3 and Q4 revenue projections"  
**Issue:** Retrieved Q3 doc but missed Q4 doc  
**Fix:** Improved query decomposition for comparison queries

### Hallucination (Caught)
**Query:** "What's the deadline for the Phoenix project?"  
**LLM output:** "The deadline is March 15th" (not in any doc)  
**Output validator:** Flagged as unsupported claim  
**User response:** "I found relevant documents but couldn't verify a specific deadline."

---

## Monitoring Alerts

| Alert | Threshold | Current | Status |
|-------|-----------|---------|--------|
| Accuracy drop (24h) | < 75% | 84% | ✅ |
| Latency p95 spike | > 4s | 2.1s | ✅ |
| Error rate | > 1% | 0.2% | ✅ |
| Security block spike | > 5% of queries | 0.8% | ✅ |
| Cost/query spike | > $0.01 | $0.0046 | ✅ |

---

## Reproduce This Dashboard

```bash
cd securerag
make eval                    # Full evaluation suite
make eval-accuracy           # Accuracy metrics only
make eval-retrieval          # Retrieval quality
make eval-latency            # Latency benchmarks
make eval-security           # Security test suite
python -m src.eval.dashboard --output docs/eval_dashboard.md
```

---

*Dashboard generated by `src/eval/dashboard.py` | Data as of 2025-12-15*
