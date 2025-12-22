"""
SecureRAG Pipeline - Orchestrates the full secure RAG flow
"""

import logging
from typing import Optional

from .retrieval.vectorstore import VectorStore
from .retrieval.reranker import Reranker
from .security.injection import InjectionDetector
from .security.pii import PIIDetector
from .security.permissions import User
from .security.output_guard import OutputGuard
from .generation.llm import LLMClient
from .generation.prompts import build_rag_prompt
from .evaluation.faithfulness import FaithfulnessScorer

logger = logging.getLogger(__name__)


class SecureRAGPipeline:
    """
    Main RAG pipeline with defense-in-depth security.
    
    Security layers:
    1. Query injection detection
    2. Tenant-isolated retrieval
    3. Retrieved chunk injection detection
    4. PII/secret detection and redaction
    5. Safe generation with strict prompt
    6. Output validation
    """
    
    def __init__(self):
        self.vectorstore = VectorStore()
        self.reranker = Reranker()
        self.injection_detector = InjectionDetector()
        self.pii_detector = PIIDetector()
        self.output_guard = OutputGuard()
        self.llm = LLMClient()
        self.faithfulness_scorer = FaithfulnessScorer()
        
    def ingest(
        self,
        doc_id: str,
        content: str,
        tenant_id: str,
        metadata: Optional[dict] = None,
        access_level: str = "user"
    ) -> int:
        """
        Ingest a document with security metadata.
        
        Returns number of chunks created.
        """
        # Check document for injection attempts (flag but still store)
        injection_result = self.injection_detector.detect(content)
        
        if injection_result["is_injection"]:
            logger.warning(
                f"Document {doc_id} contains potential injection: "
                f"{injection_result['reason']}"
            )
        
        # Build metadata
        doc_metadata = {
            "tenant_id": tenant_id,
            "access_level": access_level,
            "has_injection_flag": injection_result["is_injection"],
            **(metadata or {})
        }
        
        # Chunk and store
        chunks = self._chunk_document(content)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            self.vectorstore.add(
                doc_id=chunk_id,
                text=chunk,
                metadata={
                    **doc_metadata,
                    "parent_doc": doc_id,
                    "chunk_index": i
                }
            )
        
        return len(chunks)
    
    def _chunk_document(self, content: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """Simple chunking by character count with overlap"""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Try to break at sentence boundary
            if end < len(content):
                last_period = chunk.rfind('.')
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        return [c for c in chunks if c]  # Remove empty chunks
    
    async def query(
        self,
        query: str,
        user: User,
        top_k: int = 5
    ) -> dict:
        """
        Execute secure RAG query with all defense layers.
        """
        security_flags = []
        
        # =====================================================================
        # Layer 1: Query Injection Detection
        # =====================================================================
        query_injection = self.injection_detector.detect(query)
        
        if query_injection["is_injection"]:
            logger.warning(f"Blocked injection attempt from {user.user_id}: {query}")
            return {
                "answer": "I cannot process this request.",
                "citations": [],
                "faithfulness_score": 0.0,
                "blocked": True,
                "block_reason": "Query flagged as potential prompt injection",
                "security_flags": ["query_injection_blocked"]
            }
        
        # =====================================================================
        # Layer 2: Tenant-Isolated Retrieval
        # =====================================================================
        raw_results = self.vectorstore.search(
            query=query,
            top_k=top_k * 2,  # Retrieve more, filter later
            filter_metadata={"tenant_id": user.tenant_id}
        )
        
        if not raw_results:
            return {
                "answer": "I don't have information about that in the available documents.",
                "citations": [],
                "faithfulness_score": 1.0,  # Honest "I don't know"
                "blocked": False,
                "security_flags": []
            }
        
        # =====================================================================
        # Layer 3: Permission Filtering
        # =====================================================================
        permitted_results = [
            r for r in raw_results
            if self._user_can_access(user, r["metadata"])
        ]
        
        if not permitted_results:
            return {
                "answer": "You don't have access to documents about this topic.",
                "citations": [],
                "faithfulness_score": 1.0,
                "blocked": False,
                "security_flags": ["permission_filtered"]
            }
        
        # =====================================================================
        # Layer 4: Rerank for Quality
        # =====================================================================
        reranked = self.reranker.rerank(query, permitted_results)[:top_k]
        
        # =====================================================================
        # Layer 5: Chunk Injection Detection
        # =====================================================================
        safe_chunks = []
        for chunk in reranked:
            chunk_injection = self.injection_detector.detect(chunk["text"])
            
            if chunk_injection["is_injection"]:
                security_flags.append(f"chunk_injection_removed:{chunk['doc_id']}")
                logger.warning(f"Removed injected chunk: {chunk['doc_id']}")
            else:
                safe_chunks.append(chunk)
        
        if not safe_chunks:
            return {
                "answer": "The relevant documents appear to contain unsafe content. Please contact support.",
                "citations": [],
                "faithfulness_score": 0.0,
                "blocked": True,
                "block_reason": "All retrieved chunks contained injection attempts",
                "security_flags": security_flags
            }
        
        # =====================================================================
        # Layer 6: PII/Secret Detection
        # =====================================================================
        redacted_chunks = []
        for chunk in safe_chunks:
            pii_result = self.pii_detector.detect(chunk["text"])
            
            if pii_result["has_secrets"]:
                # Redact secrets
                redacted_text = self.pii_detector.redact(chunk["text"])
                chunk = {**chunk, "text": redacted_text}
                security_flags.append(f"pii_redacted:{chunk['doc_id']}")
            
            redacted_chunks.append(chunk)
        
        # =====================================================================
        # Layer 7: Safe Generation
        # =====================================================================
        context = "\n\n---\n\n".join([
            f"[Source: {c['doc_id']}]\n{c['text']}" 
            for c in redacted_chunks
        ])
        
        prompt = build_rag_prompt(query, context)
        answer = await self.llm.generate(prompt)
        
        # =====================================================================
        # Layer 8: Output Validation
        # =====================================================================
        output_check = self.output_guard.validate(answer, redacted_chunks)
        
        if output_check["blocked"]:
            return {
                "answer": "I cannot provide this response due to safety constraints.",
                "citations": [],
                "faithfulness_score": 0.0,
                "blocked": True,
                "block_reason": output_check["reason"],
                "security_flags": security_flags + ["output_blocked"]
            }
        
        if output_check["redacted"]:
            answer = output_check["clean_text"]
            security_flags.append("output_redacted")
        
        # =====================================================================
        # Layer 9: Faithfulness Scoring
        # =====================================================================
        faithfulness_score = self.faithfulness_scorer.score(
            answer=answer,
            context=context,
            query=query
        )
        
        # Build citations
        citations = [
            {
                "doc_id": c["metadata"].get("parent_doc", c["doc_id"]),
                "chunk": c["text"],
                "score": c["score"],
                "source": c["metadata"].get("source")
            }
            for c in redacted_chunks
        ]
        
        return {
            "answer": answer,
            "citations": citations,
            "faithfulness_score": faithfulness_score,
            "blocked": False,
            "security_flags": security_flags
        }
    
    def _user_can_access(self, user: User, metadata: dict) -> bool:
        """Check if user can access a document based on metadata"""
        # Tenant isolation (critical)
        if metadata.get("tenant_id") != user.tenant_id:
            return False
        
        # Access level check
        doc_level = metadata.get("access_level", "user")
        
        if doc_level == "admin" and user.role != "admin":
            return False
        
        return True
    
    def clear(self):
        """Clear all documents (for testing)"""
        self.vectorstore.clear()
