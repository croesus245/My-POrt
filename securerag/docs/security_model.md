# SecureRAG Security Model

**Version:** 1.2  
**Last Updated:** 2025-12-10  
**Classification:** Public

---

## Overview

SecureRAG implements a defense-in-depth security architecture for retrieval-augmented generation. This document describes the threat model, security controls, and trust boundaries.

---

## Threat Model

### Assets to Protect

1. **User documents** — Confidential content uploaded by users
2. **System prompts** — Instructions that shape LLM behavior
3. **User data** — Query history, preferences, PII
4. **Service availability** — Protection against DoS

### Threat Actors

| Actor | Capability | Motivation |
|-------|------------|------------|
| Malicious user | Crafted prompts, multi-turn attacks | Data theft, abuse |
| Compromised account | Legitimate credentials, API access | Lateral movement |
| External attacker | Network access, no credentials | Service disruption |

### Attack Vectors

1. **Prompt injection** — Manipulate LLM via user input
2. **Indirect injection** — Malicious content in retrieved documents
3. **Data exfiltration** — Extract documents, PII, or system prompts
4. **Privilege escalation** — Access other users' data
5. **Tool abuse** — Misuse integrated tools
6. **Denial of service** — Exhaust resources or rate limits

---

## Trust Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                        UNTRUSTED                                │
│  ┌──────────────┐                                               │
│  │  User Input  │                                               │
│  └──────┬───────┘                                               │
├─────────┼───────────────────────────────────────────────────────┤
│         ▼           TRUST BOUNDARY 1                            │
│  ┌──────────────┐                                               │
│  │Input Validator│ ← Pattern matching, ML classifier            │
│  └──────┬───────┘                                               │
├─────────┼───────────────────────────────────────────────────────┤
│         ▼           PARTIALLY TRUSTED                           │
│  ┌──────────────┐                                               │
│  │  Retrieved   │ ← May contain indirect injections             │
│  │  Documents   │                                               │
│  └──────┬───────┘                                               │
├─────────┼───────────────────────────────────────────────────────┤
│         ▼           TRUST BOUNDARY 2                            │
│  ┌──────────────┐                                               │
│  │Permission    │ ← Deterministic ACL check                     │
│  │Filter        │                                               │
│  └──────┬───────┘                                               │
├─────────┼───────────────────────────────────────────────────────┤
│         ▼           TRUSTED (but not infallible)                │
│  ┌──────────────┐                                               │
│  │  LLM Engine  │ ← GPT-4 with system prompt                    │
│  └──────┬───────┘                                               │
├─────────┼───────────────────────────────────────────────────────┤
│         ▼           TRUST BOUNDARY 3                            │
│  ┌──────────────┐                                               │
│  │Output        │ ← Final safety net                            │
│  │Validator     │                                               │
│  └──────┬───────┘                                               │
├─────────┼───────────────────────────────────────────────────────┤
│         ▼           TRUSTED OUTPUT                              │
│  ┌──────────────┐                                               │
│  │   Response   │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Controls

### 1. Input Validation Layer

**Purpose:** Block known-malicious patterns before LLM processing

**Implementation:**
```python
class InputValidator:
    def validate(self, user_input: str) -> ValidationResult:
        # Rule-based checks
        if self.regex_detector.detect_injection(user_input):
            return ValidationResult(blocked=True, reason="injection_pattern")
        
        # ML classifier
        injection_score = self.classifier.predict(user_input)
        if injection_score > 0.85:
            return ValidationResult(blocked=True, reason="ml_classifier")
        
        return ValidationResult(blocked=False)
```

**Patterns detected:**
- Instruction override ("ignore previous", "new instruction")
- Delimiter injection (fake XML/JSON/markdown boundaries)
- Role manipulation ("you are now", "pretend to be")
- Encoding obfuscation (base64, unicode escapes)

**Tuning:** Threshold set to minimize false positives (0.3% FP rate)

---

### 2. Permission Filter

**Purpose:** Enforce document-level access control

**Key principle:** No LLM involvement—pure deterministic check

**Implementation:**
```python
class PermissionFilter:
    def filter_documents(
        self, 
        user_id: str, 
        retrieved_docs: List[Document]
    ) -> List[Document]:
        allowed_docs = []
        for doc in retrieved_docs:
            if self.acl_service.user_can_access(user_id, doc.id):
                allowed_docs.append(doc)
            else:
                self.log_access_denied(user_id, doc.id)
        return allowed_docs
```

**Access control model:**
- Each document has an owner (user_id)
- Documents can be shared with explicit grants
- No implicit sharing, no group inheritance (simple model)
- All access checks logged for audit

**Why deterministic:** LLMs can be manipulated. Access control must not depend on LLM decisions.

---

### 3. Tool Sandbox

**Purpose:** Restrict capabilities of integrated tools

**Controls:**
| Control | Implementation |
|---------|----------------|
| Tool allowlist | Only pre-approved tools can be invoked |
| Parameter validation | Schema validation on all tool inputs |
| Rate limiting | Per-user, per-tool rate limits |
| Output sanitization | Tool outputs are sanitized before LLM sees them |
| Timeout | 10s max execution per tool call |

**Example tool definition:**
```python
@tool(
    name="search_documents",
    rate_limit="10/minute",
    timeout_seconds=10,
    allowed_params=["query", "limit"],
    max_limit=20
)
def search_documents(query: str, limit: int = 5):
    # Implementation with parameter validation
    ...
```

---

### 4. Output Validation

**Purpose:** Last line of defense—catch bypasses from earlier layers

**Checks performed:**
1. **PII detection** — Block responses containing SSN, credit cards, etc.
2. **Verbatim quote detection** — Flag large verbatim excerpts (> 50 tokens)
3. **System prompt leakage** — Detect if system prompt is being revealed
4. **Policy classification** — ML classifier for policy-violating content

**Implementation:**
```python
class OutputValidator:
    def validate(self, response: str, context: Context) -> ValidationResult:
        # PII check
        if self.pii_detector.contains_pii(response):
            return self.redact_pii(response)
        
        # Verbatim quote check
        for doc in context.retrieved_docs:
            if self.is_verbatim_quote(response, doc, threshold=50):
                return self.paraphrase_or_block(response)
        
        # Policy check
        if self.policy_classifier.is_violating(response):
            return ValidationResult(blocked=True, reason="policy_violation")
        
        return ValidationResult(blocked=False, response=response)
```

---

### 5. Tenant Isolation

**Purpose:** Prevent cross-tenant data access

**Implementation:**
- Separate vector store namespaces per tenant
- User ID included in all database queries
- Row-level security in PostgreSQL for metadata
- Audit logging of all cross-boundary attempts

```sql
-- Example RLS policy
CREATE POLICY user_documents ON documents
    FOR ALL
    USING (user_id = current_user_id());
```

---

## Security Assumptions

1. **LLM is not trusted:** We assume the LLM can be manipulated despite best efforts
2. **Retrieved documents may be adversarial:** Indirect injection is possible
3. **Users may be malicious:** All user input is untrusted
4. **Defense in depth:** No single layer is sufficient
5. **Fail secure:** When uncertain, block rather than allow

---

## Logging and Audit

### Events Logged

| Event | Data Captured | Retention |
|-------|---------------|-----------|
| Authentication | user_id, timestamp, IP, success/fail | 90 days |
| Document access | user_id, doc_id, action, allowed/denied | 1 year |
| Security blocks | user_id, input (truncated), block_reason | 1 year |
| Tool invocations | user_id, tool, params, result_status | 90 days |

### Alerting

| Condition | Alert |
|-----------|-------|
| > 10 blocks from single user in 1h | Potential attack, notify security |
| > 100 access denied events in 1h | Possible enumeration attack |
| Any system prompt leak detected | Critical, immediate review |

---

## Incident Response

See [Incident Response Playbook](incident_response.md) for detailed procedures.

**Quick reference:**
1. **Exfiltration detected:** Revoke user access, preserve logs, investigate
2. **Injection bypass:** Document payload, update validators, assess impact
3. **DoS attack:** Enable aggressive rate limiting, scale or block
4. **Account compromise:** Force password reset, revoke sessions, audit activity

---

## Compliance Considerations

| Requirement | How Addressed |
|-------------|---------------|
| Data minimization | Only retrieve necessary documents |
| Access control | Permission filter enforces need-to-know |
| Audit trail | Comprehensive logging of all access |
| Right to deletion | Document deletion propagates to vector store |
| Encryption at rest | Vector store and metadata encrypted |
| Encryption in transit | TLS 1.3 for all connections |

---

## Known Limitations

1. **LLM jailbreaks evolve:** New attacks may bypass current classifiers
2. **Indirect injection:** Difficult to fully prevent without content scanning
3. **Semantic attacks:** Some attacks are indistinguishable from legitimate queries
4. **Performance tradeoff:** More security checks = higher latency

---

## Security Contacts

- **Security issues:** security@example.com (placeholder)
- **Vulnerability disclosure:** See SECURITY.md in repository

---

*Document version 1.2 | Last security review: 2025-12-01*
