"""
Evaluation Metrics for SecureRAG

Metrics for:
- Retrieval quality
- Answer quality
- Security effectiveness
"""

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality"""
    precision_at_k: float  # Fraction of retrieved docs that are relevant
    recall_at_k: float     # Fraction of relevant docs that were retrieved
    mrr: float             # Mean Reciprocal Rank
    ndcg: float            # Normalized Discounted Cumulative Gain
    latency_ms: float      # Retrieval latency


@dataclass
class AnswerMetrics:
    """Metrics for answer quality"""
    faithfulness: float    # Grounding score (0-1)
    relevance: float       # Answer relevance to query (0-1)
    coherence: float       # Readability/coherence (0-1)
    citation_accuracy: float  # Citations are valid (0-1)


@dataclass
class SecurityMetrics:
    """Metrics for security effectiveness"""
    injection_block_rate: float      # % of injection attempts blocked
    exfiltration_block_rate: float   # % of exfil attempts blocked
    tenant_isolation_rate: float     # % of cross-tenant attempts blocked
    false_positive_rate: float       # % of benign requests incorrectly blocked


class MetricsCollector:
    """
    Collects and aggregates metrics during evaluation runs.
    """
    
    def __init__(self):
        self.retrieval_metrics: list[RetrievalMetrics] = []
        self.answer_metrics: list[AnswerMetrics] = []
        self.security_metrics: list[SecurityMetrics] = []
        self.latencies: list[float] = []
    
    def record_query(
        self,
        retrieval: Optional[RetrievalMetrics] = None,
        answer: Optional[AnswerMetrics] = None,
        latency_ms: Optional[float] = None
    ):
        """Record metrics for a single query"""
        if retrieval:
            self.retrieval_metrics.append(retrieval)
        if answer:
            self.answer_metrics.append(answer)
        if latency_ms:
            self.latencies.append(latency_ms)
    
    def record_security_test(self, metrics: SecurityMetrics):
        """Record security test results"""
        self.security_metrics.append(metrics)
    
    def aggregate(self) -> dict:
        """Aggregate all collected metrics"""
        result = {}
        
        if self.retrieval_metrics:
            result["retrieval"] = {
                "avg_precision": self._avg([m.precision_at_k for m in self.retrieval_metrics]),
                "avg_recall": self._avg([m.recall_at_k for m in self.retrieval_metrics]),
                "avg_mrr": self._avg([m.mrr for m in self.retrieval_metrics]),
                "avg_ndcg": self._avg([m.ndcg for m in self.retrieval_metrics]),
            }
        
        if self.answer_metrics:
            result["answer"] = {
                "avg_faithfulness": self._avg([m.faithfulness for m in self.answer_metrics]),
                "avg_relevance": self._avg([m.relevance for m in self.answer_metrics]),
                "avg_coherence": self._avg([m.coherence for m in self.answer_metrics]),
                "avg_citation_accuracy": self._avg([m.citation_accuracy for m in self.answer_metrics]),
            }
        
        if self.security_metrics:
            result["security"] = {
                "avg_injection_block": self._avg([m.injection_block_rate for m in self.security_metrics]),
                "avg_exfil_block": self._avg([m.exfiltration_block_rate for m in self.security_metrics]),
                "avg_isolation_rate": self._avg([m.tenant_isolation_rate for m in self.security_metrics]),
                "avg_false_positive": self._avg([m.false_positive_rate for m in self.security_metrics]),
            }
        
        if self.latencies:
            result["latency"] = {
                "avg_ms": self._avg(self.latencies),
                "p50_ms": self._percentile(self.latencies, 50),
                "p95_ms": self._percentile(self.latencies, 95),
                "p99_ms": self._percentile(self.latencies, 99),
            }
        
        return result
    
    def _avg(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
    
    def _percentile(self, values: list[float], p: int) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]


def compute_mrr(relevant_positions: list[int]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Args:
        relevant_positions: List of positions (1-indexed) where relevant docs appear
    """
    if not relevant_positions:
        return 0.0
    
    first_relevant = min(relevant_positions)
    return 1.0 / first_relevant


def compute_ndcg(relevance_scores: list[float], k: int = 10) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    
    Args:
        relevance_scores: Relevance score for each retrieved doc (in order)
        k: Number of results to consider
    """
    import math
    
    if not relevance_scores:
        return 0.0
    
    # DCG
    dcg = relevance_scores[0]
    for i, rel in enumerate(relevance_scores[1:k], start=2):
        dcg += rel / math.log2(i + 1)
    
    # Ideal DCG
    ideal_scores = sorted(relevance_scores, reverse=True)[:k]
    idcg = ideal_scores[0]
    for i, rel in enumerate(ideal_scores[1:], start=2):
        idcg += rel / math.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


class Timer:
    """Simple context manager for timing operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
    
    @property
    def elapsed_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0
