"""
In-memory vector store with metadata filtering
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from .embeddings import EmbeddingModel, cosine_similarity


@dataclass
class Document:
    """Stored document with embedding and metadata"""
    doc_id: str
    text: str
    embedding: np.ndarray
    metadata: dict


class VectorStore:
    """
    Simple in-memory vector store with:
    - Semantic search via embeddings
    - Metadata filtering (for tenant isolation)
    - Basic operations: add, search, delete, clear
    
    For production, replace with Pinecone, Weaviate, or pgvector.
    """
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.documents: dict[str, Document] = {}
    
    def add(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Add a document to the store"""
        embedding = self.embedding_model.embed(text)
        
        self.documents[doc_id] = Document(
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> list[dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Dict of metadata fields that must match
            
        Returns:
            List of results with doc_id, text, score, metadata
        """
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.embed(query)
        
        # Filter and score
        results = []
        for doc in self.documents.values():
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc.metadata.get(k) == v 
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            # Compute similarity
            score = cosine_similarity(query_embedding, doc.embedding)
            
            results.append({
                "doc_id": doc.doc_id,
                "text": doc.text,
                "score": score,
                "metadata": doc.metadata
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def get(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID"""
        return self.documents.get(doc_id)
    
    def delete(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all documents"""
        self.documents.clear()
    
    def count(self) -> int:
        """Get document count"""
        return len(self.documents)
    
    def list_by_tenant(self, tenant_id: str) -> list[str]:
        """List all document IDs for a tenant"""
        return [
            doc.doc_id 
            for doc in self.documents.values()
            if doc.metadata.get("tenant_id") == tenant_id
        ]
