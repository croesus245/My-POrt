"""
Output Guard - Final validation layer

Validates LLM output before returning to user:
- Check for leaked secrets
- Verify citations exist in context
- Detect hallucinations
- Block unsafe content
"""

import re
from typing import Optional

from .pii import PIIDetector


class OutputGuard:
    """
    Final defense layer - validates LLM output.
    
    Checks:
    1. No secrets leaked in response
    2. Citations are grounded in context
    3. No dangerous content patterns
    4. Response quality thresholds
    """
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        
        # Patterns that should never appear in output
        self.blocked_patterns = [
            # System prompt leakage
            re.compile(r'(system prompt|initial instructions?|my (programming|rules))', re.I),
            
            # Harmful content indicators
            re.compile(r'(how to (hack|exploit|attack|break into))', re.I),
            re.compile(r'(create (malware|virus|exploit|weapon))', re.I),
            
            # Internal system references
            re.compile(r'(internal[_\s]?api|admin[_\s]?endpoint|secret[_\s]?key)', re.I),
            
            # Jailbreak acknowledgment
            re.compile(r'(as (dan|an unrestricted ai|a jailbroken))', re.I),
            re.compile(r'(developer mode|god mode)\s*(enabled|activated)', re.I),
            re.compile(r'i am now (dan|unfiltered|unrestricted)', re.I),
            re.compile(r'(jailbreak|unlock).*(success|complete|done)', re.I),
        ]
    
    def validate(
        self,
        output: str,
        context_chunks: list[dict],
        strict: bool = True
    ) -> dict:
        """
        Validate LLM output before returning to user.
        
        Args:
            output: The generated response
            context_chunks: Retrieved chunks used for generation
            strict: If True, block on any violation
            
        Returns:
            {
                "valid": bool,
                "blocked": bool,
                "redacted": bool,
                "clean_text": str,
                "reason": str | None,
                "violations": list
            }
        """
        violations = []
        clean_text = output
        
        # Check 1: Secret leakage
        pii_result = self.pii_detector.detect(output)
        if pii_result["has_secrets"]:
            violations.append({
                "type": "secret_leakage",
                "severity": "high",
                "details": pii_result["findings"]
            })
            clean_text = self.pii_detector.redact(clean_text)
        
        # Check 2: Blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(output):
                violations.append({
                    "type": "blocked_pattern",
                    "severity": "high",
                    "pattern": pattern.pattern[:50]
                })
        
        # Check 3: Response too short (might be refusal bypass)
        if len(output.strip()) < 10:
            violations.append({
                "type": "suspiciously_short",
                "severity": "low",
                "length": len(output)
            })
        
        # Check 4: Response claims it can do restricted things
        if self._contains_capability_claim(output):
            violations.append({
                "type": "capability_claim",
                "severity": "medium",
                "details": "Response claims unrestricted capabilities"
            })
        
        # Determine action
        high_severity = any(v["severity"] == "high" for v in violations)
        
        if strict and high_severity:
            return {
                "valid": False,
                "blocked": True,
                "redacted": False,
                "clean_text": "",
                "reason": "Response contained unsafe content",
                "violations": violations
            }
        elif pii_result["has_secrets"]:
            return {
                "valid": True,
                "blocked": False,
                "redacted": True,
                "clean_text": clean_text,
                "reason": "Secrets redacted from response",
                "violations": violations
            }
        else:
            return {
                "valid": True,
                "blocked": False,
                "redacted": False,
                "clean_text": output,
                "reason": None,
                "violations": violations
            }
    
    def _contains_capability_claim(self, text: str) -> bool:
        """Check if output claims unrestricted capabilities"""
        claims = [
            r"i can (now )?do anything",
            r"i (am|have been) (freed|unlocked|jailbroken)",
            r"no (restrictions?|limitations?|rules?) apply",
            r"i will (ignore|bypass) (my )?(rules|restrictions|safety)",
        ]
        
        for claim in claims:
            if re.search(claim, text, re.I):
                return True
        
        return False
    
    def check_citation_grounding(
        self,
        output: str,
        context_chunks: list[dict]
    ) -> dict:
        """
        Verify that claims in output are grounded in context.
        
        Simple heuristic: check if key phrases from output
        appear in the context.
        """
        # Extract key phrases from output (simplified)
        output_words = set(output.lower().split())
        
        # Build context vocabulary
        context_text = " ".join(c["text"] for c in context_chunks)
        context_words = set(context_text.lower().split())
        
        # Check overlap
        overlap = output_words & context_words
        coverage = len(overlap) / len(output_words) if output_words else 0
        
        return {
            "grounding_score": coverage,
            "is_grounded": coverage > 0.3,  # At least 30% word overlap
            "overlap_ratio": coverage
        }
