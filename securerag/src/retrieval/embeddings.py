"""
Embedding model wrapper for semantic search
"""

import numpy as np
from typing import Optional
import hashlib


class EmbeddingModel:
    """
    Embedding model for converting text to vectors.
    
    Uses sentence-transformers when available, falls back to 
    simple TF-IDF style embeddings for demo/testing.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for MiniLM
        
        self._load_model()
    
    def _load_model(self):
        """Try to load sentence-transformers, fall back to simple embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            # Fallback to simple hash-based embeddings for demo
            self.model = None
            self.dimension = 128
    
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        if self.model is not None:
            return self.model.encode(text, convert_to_numpy=True)
        else:
            return self._simple_embed(text)
    
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts"""
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            return np.array([self._simple_embed(t) for t in texts])
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """
        Simple fallback embedding using character n-grams.
        Not as good as real embeddings but works for demo.
        """
        text = text.lower()
        
        # Generate n-gram based features
        ngrams = []
        for n in [2, 3, 4]:
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i+n])
        
        # Hash n-grams to fixed dimension
        vector = np.zeros(self.dimension)
        for ngram in ngrams:
            h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = h % self.dimension
            vector[idx] += 1
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))
