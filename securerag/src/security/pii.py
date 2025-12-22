"""
PII and Secret Detection

Detects and redacts sensitive information:
- Personal identifiable information (PII)
- API keys and credentials
- Financial data
"""

import re
from typing import Optional


class PIIDetector:
    """
    Detects PII and secrets in text.
    
    Categories:
    - API keys (OpenAI, AWS, Stripe, etc.)
    - Passwords and credentials
    - Personal info (SSN, phone, email)
    - Financial (credit cards, bank accounts)
    """
    
    def __init__(self):
        self.patterns = self._compile_patterns()
        
        # High-confidence patterns (definitely secrets)
        self.high_confidence = {
            "openai_key", "aws_key", "stripe_key", "github_token",
            "password_field", "private_key", "jwt", "connection_string",
            "ssn", "credit_card"
        }
    
    def _compile_patterns(self) -> dict:
        """Compile regex patterns for PII/secret detection"""
        return {
            # API Keys
            "openai_key": re.compile(r'sk-[a-zA-Z0-9_-]{20,}'),
            "aws_key": re.compile(r'AKIA[0-9A-Z]{16}'),
            "aws_secret": re.compile(r'[a-zA-Z0-9/+=]{40}'),
            "stripe_key": re.compile(r'(sk|pk)_(live|test)_[a-zA-Z0-9]{24,}'),
            "github_token": re.compile(r'(ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,}'),
            "generic_api_key": re.compile(
                r'(api[_-]?key|apikey|api[_-]?secret|secret[_-]?key)\s*[:=]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?',
                re.IGNORECASE
            ),
            
            # Passwords
            "password_field": re.compile(
                r'(password|passwd|pwd|pass)["\s]*[:=]["\s]*([^\s"\'<>]{4,})',
                re.IGNORECASE
            ),
            
            # Personal Info
            "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "phone_us": re.compile(r'\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            "phone_intl": re.compile(r'\+\d{1,3}[-.\s]?\d{6,14}'),
            "email": re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'),
            "nin_nigeria": re.compile(r'\b\d{11}\b'),  # Nigerian NIN
            "bvn_nigeria": re.compile(r'\b\d{11}\b'),  # Nigerian BVN
            
            # Financial
            "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "bank_account": re.compile(r'\b\d{8,17}\b'),
            "iban": re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b'),
            
            # IP Addresses (internal)
            "internal_ip": re.compile(
                r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|'
                r'172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|'
                r'192\.168\.\d{1,3}\.\d{1,3})\b'
            ),
            
            # Connection strings
            "connection_string": re.compile(
                r'(mongodb|mysql|postgres|redis|amqp)://[^\s<>"\']+',
                re.IGNORECASE
            ),
            
            # JWT tokens
            "jwt": re.compile(r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*'),
            
            # Private keys
            "private_key": re.compile(
                r'-----BEGIN (RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----',
                re.IGNORECASE
            )
        }
    
    def detect(self, text: str) -> dict:
        """
        Detect PII and secrets in text.
        
        Returns:
            {
                "has_secrets": bool,
                "has_pii": bool,
                "findings": [{"type": str, "value": str, "position": tuple}],
                "risk_level": "high" | "medium" | "low" | "none"
            }
        """
        findings = []
        
        for name, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                findings.append({
                    "type": name,
                    "value": match.group()[:20] + "..." if len(match.group()) > 20 else match.group(),
                    "position": (match.start(), match.end()),
                    "high_confidence": name in self.high_confidence
                })
        
        # Deduplicate overlapping findings
        findings = self._deduplicate_findings(findings)
        
        has_high_confidence = any(f["high_confidence"] for f in findings)
        
        if has_high_confidence:
            risk_level = "high"
        elif len(findings) > 3:
            risk_level = "medium"
        elif findings:
            risk_level = "low"
        else:
            risk_level = "none"
        
        pii_types = {"ssn", "phone_us", "phone_intl", "email", "credit_card", 
                     "nin_nigeria", "bvn_nigeria", "bank_account"}
        
        return {
            "has_secrets": has_high_confidence,
            "has_pii": any(f["type"] in pii_types for f in findings),
            "findings": findings,
            "risk_level": risk_level
        }
    
    def _deduplicate_findings(self, findings: list) -> list:
        """Remove overlapping findings, keeping highest confidence"""
        if not findings:
            return []
        
        # Sort by position, then by confidence
        findings.sort(key=lambda f: (f["position"][0], -f["high_confidence"]))
        
        deduped = [findings[0]]
        for f in findings[1:]:
            last = deduped[-1]
            # Check overlap
            if f["position"][0] >= last["position"][1]:
                deduped.append(f)
            elif f["high_confidence"] and not last["high_confidence"]:
                deduped[-1] = f
        
        return deduped
    
    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """
        Redact all detected PII and secrets from text.
        
        Returns text with sensitive content replaced.
        """
        result = self.detect(text)
        
        if not result["findings"]:
            return text
        
        # Sort by position descending to replace from end
        findings = sorted(result["findings"], key=lambda f: f["position"][0], reverse=True)
        
        redacted = text
        for finding in findings:
            start, end = finding["position"]
            redacted = redacted[:start] + replacement + redacted[end:]
        
        return redacted
    
    def mask(self, text: str) -> str:
        """
        Mask sensitive content (show partial).
        E.g., "sk-abc123xyz" -> "sk-***xyz"
        """
        result = self.detect(text)
        
        if not result["findings"]:
            return text
        
        findings = sorted(result["findings"], key=lambda f: f["position"][0], reverse=True)
        
        masked = text
        for finding in findings:
            start, end = finding["position"]
            original = text[start:end]
            
            if len(original) > 8:
                # Show first 3 and last 3 chars
                mask = original[:3] + "*" * (len(original) - 6) + original[-3:]
            else:
                mask = "*" * len(original)
            
            masked = masked[:start] + mask + masked[end:]
        
        return masked


# Test patterns for validation
SECRET_EXAMPLES = [
    "API Key: sk-abc123456789012345678901234567890",
    "Password: SuperSecret123!",
    "My SSN is 123-45-6789",
    "Call me at (555) 123-4567",
    "Credit card: 4111-1111-1111-1111",
    "AWS: AKIAIOSFODNN7EXAMPLE",
    "mongodb://user:pass@host:27017/db",
    "token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
]
