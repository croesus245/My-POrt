"""
Reranker for improving retrieval quality
"""

from typing import Optional


class Reranker:
    """
    Reranks retrieved documents for higher relevance.
    
    Uses cross-encoder when available, falls back to 
    simple keyword overlap scoring.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        
        self._load_model()
    
    def _load_model(self):
        """Try to load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name)
        except ImportError:
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int] = None
    ) -> list[dict]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: The search query
            documents: List of docs with 'text' field
            top_k: Optional limit on results
            
        Returns:
            Reranked documents with updated scores
        """
        if not documents:
            return []
        
        if self.model is not None:
            return self._rerank_with_model(query, documents, top_k)
        else:
            return self._rerank_simple(query, documents, top_k)
    
    def _rerank_with_model(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int]
    ) -> list[dict]:
        """Rerank using cross-encoder model"""
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs)
        
        # Update scores and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
            # Combine with original retrieval score
            doc["score"] = 0.3 * doc["score"] + 0.7 * float(score)
        
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        if top_k:
            return documents[:top_k]
        return documents
    
    def _rerank_simple(
        self,
        query: str,
        documents: list[dict],
        top_k: Optional[int]
    ) -> list[dict]:
        """Simple keyword-based reranking fallback"""
        query_words = set(query.lower().split())
        
        for doc in documents:
            doc_words = set(doc["text"].lower().split())
            
            # Jaccard similarity
            intersection = len(query_words & doc_words)
            union = len(query_words | doc_words)
            keyword_score = intersection / union if union > 0 else 0
            
            # Combine with embedding score
            doc["rerank_score"] = keyword_score
            doc["score"] = 0.5 * doc["score"] + 0.5 * keyword_score
        
        documents.sort(key=lambda x: x["score"], reverse=True)
        
        if top_k:
            return documents[:top_k]
        return documents
