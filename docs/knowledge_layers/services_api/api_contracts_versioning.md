# API Contracts & Versioning (`/api/v3/*`)

This document describes a versioning and compatibility strategy aligned with the current routing approach (many routers declare `prefix="/api/v3/..."` in `services/api/src/routes/*`).

## Versioning strategy

### URL-path versioning (current)
- Contract version is encoded in the path: `/api/v3/...`
- New breaking changes should land in `/api/v4/...` (or as a new service) while keeping v3 stable.

Trade-off table
| Strategy | Pros | Cons |
|---|---|---|
| Path versioning (`/api/v3`) | explicit, cache-friendly | route duplication; clients must upgrade URLs |
| Header versioning | fewer routes | harder to debug; proxy/cache complexity |

### Compatibility rules (documented)
- **Additive changes** (new optional fields) are allowed within v3.
- **Breaking changes** (rename/remove fields, semantic changes) require a new major version (`v4`).
- **Enum expansion** is allowed if clients treat unknown values safely.

### Deprecation policy (documented)
- Mark endpoints/fields as deprecated in OpenAPI descriptions.
- Keep deprecated behavior for a fixed window (e.g., 90 days) before removing in next major version.
- Publish a migration guide in `docs/` when cutting a new major version.

## Pagination and filtering contracts

Documented defaults:
- Prefer cursor-based pagination for large tables (stable under inserts) over offset pagination.
- If offset pagination is used, define max page size and warn about inconsistent ordering under concurrent writes.

Complexity note:
- Offset pagination can be **O(offset + limit)** on the server if it requires scanning; cursor pagination is closer to **O(limit)** with a supporting index.

## Idempotency recommendations (POST/execute-like endpoints)

For endpoints that trigger side effects (execution, tool invocation, memory store):
- Accept `Idempotency-Key` header (client-generated).
- Persist key at the boundary where side effects are recorded.

Anchor:
- `database/migrations/008_execution_persistence.sql` defines `step_execution_attempts.idempotency_key` (unique).

## Error model spec (canonical envelope)

Canonical error envelope should match `services/api/src/error_handlers.py`:
```json
{
  "error": {
    "id": "err_123abc...",
    "type": "ValidationError",
    "message": "Request validation failed",
    "details": { "errors": [] },
    "timestamp": "2026-02-27T00:00:00Z",
    "path": "/api/v3/..."
  },
  "status": 422
}
```

Client guidance:
- Always log `error.id` and the response `X-Request-ID` for correlation.
- Treat `status` + `error.type` as the stable machine interface; `message` is for humans.

## Advanced Engineering Notes

### Failure Scenarios: “Breaking change inside v3”
Symptoms
- Existing clients start failing after server deployment.
- Increased 4xx/5xx on endpoints that previously worked.

Root cause
- Schema or semantic change shipped without version bump (v3 contract broken).

Mitigation
- Enforce contract review checklist for any request/response schema change.
- Add contract tests (golden responses) for key endpoints.

Observability signals (logs/metrics/traces)
- Spike in request validation errors (422) after deployment.
- Logs show new error types/fields not expected by clients.

### Complexity & Performance
- Contract evolution complexity grows with number of clients (**O(C)**). Versioning reduces coordination cost by decoupling client upgrades.
- Error envelope size should be bounded; avoid dumping full tracebacks in production responses.

## Research Extensions

- Generate SDKs from OpenAPI and run compatibility diff checks between versions.
- Add an automated “breaking change detector” for Pydantic schemas.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for where contracts interact with orchestration (execute endpoints) and persistence semantics.

## Future Evolution Strategy

- Near-term: standardize error envelope + idempotency headers across all v3 routers.
- Mid-term: add explicit “retryability” classification to errors (client-safe retries).

