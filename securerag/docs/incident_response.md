# SecureRAG Incident Response Playbook

**Owner:** Security Team  
**Last Updated:** 2025-12-10  
**Review Cycle:** Quarterly

---

## Incident Classification

| Severity | Definition | Response Time | Examples |
|----------|------------|---------------|----------|
| **P1 - Critical** | Active data breach, service down | 15 minutes | Confirmed exfiltration, complete outage |
| **P2 - High** | Security bypass detected, partial outage | 1 hour | Injection bypass, DoS in progress |
| **P3 - Medium** | Attempted attack blocked, degraded service | 4 hours | Blocked exfiltration attempts, latency spike |
| **P4 - Low** | Minor security event, no impact | 24 hours | Failed auth attempts, policy edge cases |

---

## Incident Response Phases

### Phase 1: Detection & Triage (0-15 min)

**Goals:** Confirm incident, assess severity, notify stakeholders

**Checklist:**
- [ ] Alert received and acknowledged
- [ ] Initial severity assessment (P1-P4)
- [ ] Incident channel created (#incident-YYYY-MM-DD-brief)
- [ ] On-call responders notified
- [ ] Initial timeline started

**Key questions:**
1. Is this a real incident or false alarm?
2. What is the scope (users affected, data involved)?
3. Is the attack ongoing?
4. What evidence do we have?

---

### Phase 2: Containment (15-60 min)

**Goals:** Stop bleeding, preserve evidence, limit blast radius

**Actions by incident type:**

#### Data Exfiltration
```
1. Identify affected user accounts
2. Revoke sessions: POST /admin/revoke-sessions?user_id=<id>
3. Disable affected accounts if necessary
4. Preserve logs (do not delete)
5. Snapshot affected systems
```

#### Injection Bypass
```
1. Document the attack payload
2. Add payload to blocklist (immediate)
3. Review similar patterns in logs
4. Assess what was accessed/output
5. Consider temporary defensive mode
```

#### DoS Attack
```
1. Enable aggressive rate limiting
2. Block attacking IPs at edge
3. Scale up if legitimate traffic affected
4. Engage CDN/DDoS protection
5. Monitor for attack pattern changes
```

#### Account Compromise
```
1. Disable compromised account
2. Revoke all sessions and API keys
3. Notify account owner (if not attacker)
4. Audit account activity (last 30 days)
5. Check for lateral movement
```

---

### Phase 3: Eradication (1-4 hours)

**Goals:** Remove attacker access, fix vulnerability

**Checklist:**
- [ ] Vulnerability identified
- [ ] Fix developed and tested
- [ ] Fix deployed to production
- [ ] Verify fix effectiveness
- [ ] Check for persistence mechanisms

**For security control bypass:**
```python
# Example: Adding new pattern to input validator
INPUT_VALIDATOR_PATTERNS.append({
    "name": "incident_2025_12_payload",
    "pattern": r"<specific regex>",
    "severity": "high",
    "added_date": "2025-12-XX",
    "incident_ref": "INC-2025-XXX"
})
```

---

### Phase 4: Recovery (4-24 hours)

**Goals:** Restore normal operations, verify security

**Checklist:**
- [ ] All containment measures reviewed
- [ ] Affected accounts restored (if appropriate)
- [ ] Monitoring enhanced for recurrence
- [ ] User notifications sent (if required)
- [ ] External reporting (if required by compliance)

**User notification template (data breach):**
```
Subject: Security Notice - Action Required

We detected unauthorized access to your account on [DATE].

What happened: [Brief description]

What we did: [Actions taken]

What you should do:
1. Reset your password
2. Review your recent activity
3. Contact us if you notice anything unusual

We apologize for this incident and have taken steps to prevent recurrence.
```

---

### Phase 5: Post-Incident (24-72 hours)

**Goals:** Document lessons learned, improve defenses

**Postmortem template:**
```markdown
# Incident Postmortem: [Brief Title]

**Date:** YYYY-MM-DD
**Duration:** X hours
**Severity:** P1/P2/P3/P4
**Author:** [Name]

## Summary
[2-3 sentence description]

## Timeline
| Time | Event |
|------|-------|
| HH:MM | ... |

## Root Cause
[Detailed explanation]

## Impact
- Users affected: X
- Data exposed: Y
- Duration: Z hours

## What Went Well
- ...

## What Went Poorly
- ...

## Action Items
| Action | Owner | Due Date | Status |
|--------|-------|----------|--------|
| ... | ... | ... | ... |

## Lessons Learned
- ...
```

---

## Runbooks by Incident Type

### Runbook: Prompt Injection Bypass

**Trigger:** Output validator logs show blocked content that reveals injection success

**Steps:**
1. Pull logs for the session
   ```bash
   python -m src.admin.logs --session-id <id> --full
   ```

2. Extract attack payload
   ```bash
   python -m src.security.extract_payload --log-file <file>
   ```

3. Test payload against current defenses
   ```bash
   python -m src.security.test_payload --payload "<payload>"
   ```

4. Add to blocklist
   ```bash
   python -m src.security.add_blocklist --payload "<payload>" --incident <id>
   ```

5. Assess impact
   - What did the LLM output?
   - Did any sensitive data leak?
   - Was output validator effective?

6. Update input validator ML model (if pattern is novel)
   ```bash
   python -m src.security.retrain_classifier --add-sample <payload>
   ```

---

### Runbook: Suspected Exfiltration

**Trigger:** Unusual access patterns, high volume of document retrievals, blocked exfil attempts

**Steps:**
1. Identify suspicious user(s)
   ```bash
   python -m src.admin.anomaly_report --last 24h
   ```

2. Review their query history
   ```bash
   python -m src.admin.user_queries --user-id <id> --last 7d
   ```

3. Check what documents were retrieved
   ```bash
   python -m src.admin.user_retrievals --user-id <id> --last 7d
   ```

4. Look for exfiltration patterns:
   - Systematic document enumeration
   - Encoding in queries (base64, etc.)
   - Unusual query-to-retrieval ratios
   - Access outside normal hours

5. If confirmed malicious:
   - Revoke access immediately
   - Preserve all logs
   - Assess what was exposed
   - Determine if notification required

---

### Runbook: System Prompt Extraction

**Trigger:** Output contains fragments of system prompt

**Severity:** P2 (system prompt is not highly sensitive, but indicates bypass)

**Steps:**
1. Document the attack that extracted the prompt

2. Review system prompt content
   - Does it contain sensitive info? (Usually no)
   - Does exposure enable further attacks?

3. Update output validator
   ```python
   # Add system prompt fragments to blocklist
   OUTPUT_BLOCKLIST.append(system_prompt[:100])
   OUTPUT_BLOCKLIST.append(system_prompt[-100:])
   ```

4. Consider rotating system prompt structure (not content)
   - Add decoy instructions
   - Use different delimiters

5. Update input validator for the extraction technique

---

## Escalation Matrix

| Severity | Primary | Secondary | Executive |
|----------|---------|-----------|-----------|
| P1 | Security Lead | Engineering Lead | CTO |
| P2 | Security On-Call | Security Lead | — |
| P3 | Security On-Call | — | — |
| P4 | Async review | — | — |

**Contact methods:**
- Slack: #security-oncall
- PagerDuty: security-critical escalation policy
- Email: security@example.com (not for P1/P2)

---

## Communication Templates

### Internal Status Update
```
🔴 INCIDENT UPDATE - [TITLE]
Status: [Investigating/Contained/Resolved]
Severity: P[X]
Impact: [Brief description]
Current actions: [What we're doing now]
Next update: [Time]
```

### External Status Page
```
[Investigating/Identified/Monitoring/Resolved] - [Title]

We are currently investigating reports of [brief description].

Updates will be posted as they become available.

Posted: [Time] UTC
```

---

## Evidence Preservation

**What to preserve:**
- [ ] Application logs (API, security, audit)
- [ ] Database query logs
- [ ] Network flow logs
- [ ] System snapshots (if relevant)
- [ ] User session data

**How to preserve:**
```bash
# Snapshot logs to incident bucket
python -m src.admin.preserve_logs \
  --incident-id INC-YYYY-XXX \
  --start "YYYY-MM-DD HH:MM" \
  --end "YYYY-MM-DD HH:MM" \
  --user-id <affected_user>
```

**Chain of custody:**
1. Document who accessed evidence
2. Use read-only access after preservation
3. Hash all preserved files
4. Store in tamper-evident location

---

## Post-Incident Checklist

- [ ] Postmortem document created
- [ ] Action items assigned with due dates
- [ ] Security controls updated
- [ ] Monitoring/alerting improved
- [ ] Team debriefed
- [ ] Documentation updated
- [ ] Compliance notifications sent (if required)
- [ ] User notifications sent (if required)
- [ ] Incident closed in tracking system

---

*Playbook version 1.1 | Next review: 2026-03-01*
