# Security Hardening (Threat Model Extensions + Checklists)

This document extends `docs/security.md` with production-hardening checklists, multi-tenant isolation strategy, and agent/tool abuse considerations.

## Threat model extensions (prompt/tool systems)

| Threat | Example | Primary mitigations |
|---|---|---|
| Prompt injection | user content instructs tool abuse | instruction/data separation; allow-list tools; output validation |
| Tool abuse | agent calls filesystem/network beyond intent | permission gating; audit log; sandboxing |
| Multi-tenant leakage | tenant A reads tenant B memory | strong scoping + DB RLS + defense-in-depth checks |
| Credential theft | leaked API keys/JWTs | rotation, short TTLs, revocation lists, secrets hygiene |

## Practical hardening checklists

### AuthN/Z
- Use short-lived access tokens + refresh tokens with revocation (`user_sessions` in `database/migrations/007_auth_users.sql`).
- Enforce tenant-aware authorization checks on every route.
- Ensure API keys are **hashed** at rest (see `api_keys` in `database/migrations/009_multi_tenancy_billing.sql`).

### JWT pitfalls (documented)
- Validate `aud`, `iss`, `exp`, and algorithm.
- Prevent “none” algorithm attacks; pin algorithms explicitly.
- Rotate signing keys; publish a key-rotation runbook.

### DB row-level security (RLS) strategy
Document-only recommended posture:
- Add `tenant_id` columns to all tenant-scoped tables (migration `009_multi_tenancy_billing.sql` starts this).
- Enable Postgres RLS on tenant-scoped tables:
  - set `app.tenant_id` as a session variable
  - policies enforce `tenant_id = current_setting('app.tenant_id')::uuid`

### Supply-chain security
- Pin dependencies; run CVE scanning.
- Produce SBOM artifacts in CI (conceptually).
- Harden container builds (minimal base images, no dev tools in prod).

## Security tests we should have (documented scenarios)
- Cross-tenant memory search returns no results for other tenants.
- API key scope enforcement: forbidden scope fails with canonical error envelope.
- Prompt injection red-team suite for tool execution denial.
- Replay/idempotency: duplicate requests do not duplicate side effects.

## Advanced Engineering Notes

### Failure Scenarios: “Tenant isolation bypass”
Symptoms
- Reports of data from other tenants appearing in memory/workflow results.

Root cause
- Missing tenant_id filter in repository queries; no DB-level guardrails.

Mitigation
- Enforce tenant scoping at three layers:
  1) request auth context
  2) repository filters
  3) DB RLS policies

Observability signals (logs/metrics/traces)
- Audit logs show access to objects with mismatched tenant_id.
- Anomalous access patterns (tenant A accessing tenant B identifiers).

### Complexity & Performance
- Authorization checks should be **O(1)** per request with cached policy evaluation; avoid per-request “full permission scans”.
- Hashing/verifying tokens is typically **O(n)** in token length; keep token sizes bounded.

## Research Extensions

- Formalize permission policies and property-test them (deny-by-default invariants).
- Differential fuzzing for prompt injection defenses (same intent, adversarial phrasing).

## System Design Deep Dive

- See `../system_design_deep_dive.md` for where security boundaries exist (API → workers → DB).

## Future Evolution Strategy

- Near-term: tenant scoping consistency + canonical auth audit logs.
- Mid-term: RLS enforcement + outbox-based audit events.
- Long-term: multi-region key management + tenant-isolated encryption.

