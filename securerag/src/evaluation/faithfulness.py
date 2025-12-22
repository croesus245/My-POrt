"""
Faithfulness Scoring - Check if answers are grounded in context

Methods:
1. Overlap-based scoring (simple, fast)
2. NLI-based scoring (accurate, requires model)
3. LLM-as-judge (most accurate, expensive)
"""

import re
from typing import Optional


class FaithfulnessScorer:
    """
    Scores how well an answer is grounded in the provided context.
    
    High faithfulness = answer claims are supported by context
    Low faithfulness = answer may contain hallucinations
    """
    
    def __init__(self, method: str = "overlap"):
        """
        Initialize scorer.
        
        Args:
            method: "overlap" (fast), "nli" (accurate), or "llm" (best)
        """
        self.method = method
        self.nli_model = None
        
        if method == "nli":
            self._load_nli_model()
    
    def _load_nli_model(self):
        """Load NLI model for entailment checking"""
        try:
            from transformers import pipeline
            self.nli_model = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli"
            )
        except ImportError:
            self.method = "overlap"  # Fallback
    
    def score(
        self,
        answer: str,
        context: str,
        query: Optional[str] = None
    ) -> float:
        """
        Score answer faithfulness to context.
        
        Args:
            answer: Generated answer
            context: Source context (concatenated chunks)
            query: Original query (optional, for relevance)
            
        Returns:
            Score from 0.0 (unfaithful) to 1.0 (faithful)
        """
        if not answer or not context:
            return 0.0
        
        if self.method == "nli" and self.nli_model:
            return self._score_nli(answer, context)
        else:
            return self._score_overlap(answer, context)
    
    def _score_overlap(self, answer: str, context: str) -> float:
        """
        Simple overlap-based faithfulness scoring.
        
        Checks what fraction of answer's key terms appear in context.
        """
        # Normalize text
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Extract key terms (nouns, verbs, numbers)
        answer_terms = self._extract_key_terms(answer_lower)
        context_terms = set(self._extract_key_terms(context_lower))
        
        if not answer_terms:
            return 1.0  # Empty answer is technically faithful
        
        # Count how many answer terms appear in context
        grounded_terms = sum(1 for t in answer_terms if t in context_terms)
        
        # Base overlap score
        overlap_score = grounded_terms / len(answer_terms)
        
        # Bonus for citing sources
        has_citation = "[source:" in answer_lower or "[source]" in answer_lower
        citation_bonus = 0.1 if has_citation else 0
        
        # Penalty for common hallucination patterns
        hallucination_penalty = self._detect_hallucination_patterns(answer)
        
        # Final score
        score = min(1.0, overlap_score + citation_bonus - hallucination_penalty)
        return max(0.0, score)
    
    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key terms from text"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'this', 'that', 'these', 'those', 'i',
            'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'whom', 'your', 'their', 'its', 'our', 'my'
        }
        
        return [w for w in words if w not in stop_words and len(w) > 2]
    
    def _detect_hallucination_patterns(self, answer: str) -> float:
        """Detect patterns that suggest hallucination"""
        penalty = 0.0
        answer_lower = answer.lower()
        
        # Specific numbers without citation often hallucinated
        if re.search(r'\b\d{4,}\b', answer) and '[source' not in answer_lower:
            penalty += 0.1
        
        # "According to my knowledge" suggests not from context
        knowledge_phrases = [
            "according to my knowledge",
            "based on what i know",
            "i believe",
            "i think",
            "generally speaking",
            "it is commonly known",
            "as far as i know"
        ]
        for phrase in knowledge_phrases:
            if phrase in answer_lower:
                penalty += 0.15
        
        # Very confident statements without citation
        confident_phrases = ["definitely", "certainly", "absolutely", "always", "never"]
        for phrase in confident_phrases:
            if phrase in answer_lower and '[source' not in answer_lower:
                penalty += 0.05
        
        return min(penalty, 0.5)  # Cap penalty
    
    def _score_nli(self, answer: str, context: str) -> float:
        """Score using NLI model (entailment checking)"""
        if not self.nli_model:
            return self._score_overlap(answer, context)
        
        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 1.0
        
        entailment_scores = []
        
        for sentence in sentences[:5]:  # Limit for performance
            # Check if context entails this sentence
            result = self.nli_model(
                f"{context[:1000]} [SEP] {sentence}",
                top_k=None
            )
            
            # Find entailment score
            for r in result:
                if r['label'] == 'ENTAILMENT':
                    entailment_scores.append(r['score'])
                    break
            else:
                entailment_scores.append(0.0)
        
        return sum(entailment_scores) / len(entailment_scores)


def compute_citation_accuracy(answer: str, context_chunks: list[dict]) -> dict:
    """
    Check if citations in answer actually appear in context.
    
    Returns:
        {
            "citations_found": list of cited doc_ids,
            "citations_valid": list of valid citations,
            "accuracy": float (valid / found)
        }
    """
    # Extract citations from answer
    citation_pattern = r'\[Source:\s*([^\]]+)\]'
    found_citations = re.findall(citation_pattern, answer, re.IGNORECASE)
    
    # Get actual doc_ids from context
    valid_doc_ids = {c.get("doc_id", "") for c in context_chunks}
    valid_doc_ids.update({c.get("parent_doc", "") for c in context_chunks if c.get("parent_doc")})
    
    # Check validity
    valid_citations = []
    for citation in found_citations:
        # Clean citation (remove section references)
        doc_id = citation.split(",")[0].strip()
        if doc_id in valid_doc_ids:
            valid_citations.append(citation)
    
    accuracy = len(valid_citations) / len(found_citations) if found_citations else 1.0
    
    return {
        "citations_found": found_citations,
        "citations_valid": valid_citations,
        "accuracy": accuracy
    }
