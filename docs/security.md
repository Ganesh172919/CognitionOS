# Security & Safety

## Threat Model

### Attack Vectors

1. **Prompt Injection**
   - Attacker tries to override system instructions
   - Mitigation: Separate instructions from user data
   - Detection: Monitor for suspicious patterns

2. **Resource Exhaustion**
   - Attacker floods system with expensive requests
   - Mitigation: Rate limiting, budget caps
   - Detection: Usage anomaly detection

3. **Data Exfiltration**
   - Attacker tries to access other users' data
   - Mitigation: Row-level security, user isolation
   - Detection: Access pattern monitoring

4. **Tool Misuse**
   - Agent executes unauthorized operations
   - Mitigation: Permission system, sandboxing
   - Detection: Audit log analysis

5. **Model Jailbreaking**
   - Bypass safety guardrails in LLM
   - Mitigation: Output validation, safety classifiers
   - Detection: Pattern matching

## Defense-in-Depth

### Layer 1: Authentication & Authorization
- JWT with short expiration
- Role-based access control
- API key rotation
- MFA for admin accounts

### Layer 2: Input Validation
- Schema validation with Pydantic
- SQL injection prevention
- Path traversal protection
- Command injection protection

### Layer 3: Execution Isolation
- Docker containers for tools
- Network isolation by default
- Resource limits (CPU, memory, time)
- Read-only file systems

### Layer 4: Output Validation
- LLM output filtering
- Hallucination detection
- PII detection and redaction
- Malicious code detection

### Layer 5: Monitoring & Audit
- All actions logged
- Anomaly detection
- Real-time alerting
- Immutable audit trail

## Prompt Injection Defenses

### 1. Instruction/Data Separation
```python
# BAD: Mixing instructions and user data
prompt = f"You are a helpful assistant. User says: {user_input}"

# GOOD: Clear separation
system_prompt = "You are a helpful assistant."
user_message = user_input  # Sent separately
```

### 2. Input Sanitization
```python
def sanitize_input(text: str) -> str:
    # Remove potential instruction markers
    dangerous_patterns = [
        r"ignore previous instructions",
        r"system:",
        r"<\|im_start\|>",
        r"###",
    ]
    for pattern in dangerous_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text
```

### 3. Output Validation
```python
def validate_output(output: str) -> bool:
    # Check for signs of jailbreak
    if any(marker in output.lower() for marker in [
        "my previous instructions",
        "i apologize, but i cannot",
        "as an ai language model"
    ]):
        return False
    return True
```

## Tool Permission System

### Permission Levels

```python
PERMISSION_HIERARCHY = {
    "admin": [
        "code_execution",
        "network",
        "filesystem_read",
        "filesystem_write",
        "database_read",
        "database_write"
    ],
    "user": [
        "code_execution",
        "network",
        "filesystem_read",
        "database_read"
    ],
    "restricted": [
        "filesystem_read",
        "database_read"
    ]
}
```

### Permission Checks

```python
def check_permission(user_role: str, required_permission: str) -> bool:
    user_permissions = PERMISSION_HIERARCHY.get(user_role, [])
    return required_permission in user_permissions
```

## Memory Isolation

### Row-Level Security (PostgreSQL)

```sql
-- Enable RLS
ALTER TABLE memories ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only see their own memories
CREATE POLICY user_isolation ON memories
    FOR ALL
    TO app_user
    USING (user_id = current_setting('app.user_id')::uuid);

-- Set user context
SET app.user_id = '123e4567-e89b-12d3-a456-426614174000';

-- This automatically filters by user_id
SELECT * FROM memories;
```

### Application-Level Isolation

```python
async def get_memories(user_id: UUID) -> List[Memory]:
    # ALWAYS filter by user_id
    return await db.query(
        Memory
    ).filter(
        Memory.user_id == user_id
    ).all()
```

## Rate Limiting

### Per-User Limits

```python
RATE_LIMITS = {
    "requests_per_minute": 60,
    "requests_per_hour": 1000,
    "requests_per_day": 10000,
    "cost_per_day_usd": 10.0,
    "tokens_per_day": 100000
}
```

### Implementation

```python
class RateLimiter:
    def check_limit(self, user_id: UUID, metric: str) -> bool:
        current_usage = redis.get(f"{user_id}:{metric}")
        limit = RATE_LIMITS[metric]
        return current_usage < limit
```

## PII Protection

### Detection

```python
import re

PII_PATTERNS = {
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
    "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
}

def contains_pii(text: str) -> bool:
    for pattern in PII_PATTERNS.values():
        if re.search(pattern, text):
            return True
    return False
```

### Redaction

```python
def redact_pii(text: str) -> str:
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", text)
    return text
```

## Incident Response

### 1. Detection
- Automated alerts for suspicious activity
- Manual security reviews
- User reports

### 2. Containment
- Disable compromised accounts
- Rotate API keys
- Isolate affected services

### 3. Investigation
- Review audit logs
- Analyze attack patterns
- Identify affected data

### 4. Recovery
- Restore from backups
- Patch vulnerabilities
- Reset credentials

### 5. Post-Mortem
- Document incident
- Update security measures
- Notify affected users

## Compliance

### GDPR
- Right to access
- Right to deletion
- Data portability
- Consent management

### SOC 2
- Access controls
- Encryption at rest and in transit
- Audit logging
- Vulnerability management

### HIPAA (if handling health data)
- Data encryption
- Access audit trails
- Business associate agreements
- Breach notification

## Security Checklist

### Development
- [ ] Input validation on all endpoints
- [ ] Output encoding to prevent XSS
- [ ] Parameterized queries to prevent SQL injection
- [ ] CSRF protection
- [ ] Secure session management

### Deployment
- [ ] HTTPS/TLS for all communication
- [ ] Environment variables for secrets
- [ ] No hardcoded credentials
- [ ] Minimal container images
- [ ] Regular security updates

### Operations
- [ ] Monitoring and alerting
- [ ] Incident response plan
- [ ] Regular security audits
- [ ] Backup and disaster recovery
- [ ] Access reviews

## Responsible Disclosure

If you discover a security vulnerability:

1. **Do NOT** publicly disclose
2. Email: security@cognitionos.ai (example)
3. Include:
   - Vulnerability description
   - Steps to reproduce
   - Impact assessment
4. We will respond within 48 hours
5. We will credit security researchers
