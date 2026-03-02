# Memory Hierarchy + Vector Search (Tiering, Indexes, Concurrency)

Anchors:
- Domain entities/services: `core/domain/memory_hierarchy/entities.py`, `core/domain/memory_hierarchy/services.py`
- Use cases: `core/application/memory_hierarchy/use_cases.py`
- Persistence (pgvector): `infrastructure/persistence/memory_hierarchy_models.py`, `infrastructure/persistence/memory_hierarchy_repository.py`
- Schema: `database/migrations/003_phase3_extended_operation.sql` (L1/L2/L3 tables + ivfflat indexes)

## Memory tiers (L1/L2/L3)

```text
          +----------------------+
          |  L1 Working Memory   |  (hot, recent, high-churn)
          |  table: working_memory
          +----------+-----------+
                     |
         promote (compress + cluster)
                     v
          +----------------------+
          |  L2 Episodic Memory  |  (summaries, clusters)
          |  table: episodic_memory
          +----------+-----------+
                     |
         promote (extract durable knowledge)
                     v
          +----------------------+
          |  L3 Long-term Memory |  (durable, archive-able)
          |  table: longterm_memory
          +----------------------+
```

Tier transitions are managed by domain services such as `MemoryTierManager` (`core/domain/memory_hierarchy/services.py`).

## Embeddings: dimensionality, drift, versioning

Observed defaults in adapters:
- `Vector(1536)` in `infrastructure/persistence/memory_hierarchy_models.py`
- `MemoryEmbedding.dimensions` inferred from vector length in domain (`core/domain/memory_hierarchy/entities.py`)

Production-aware recommendations (document-only):
- Store `embedding_model` + `embedding_version` in metadata.
- Plan for model migration:
  - dual-write (old+new embeddings) during transition, or
  - background re-embed with backfill progress tracking.

## Vector index choices (IVFFlat vs HNSW)

| Index | Typical trade-off | When it wins | Notes |
|---|---|---|---|
| IVFFlat | fast build, good for large datasets; approximate | bulk-ingest + high-volume search | requires `lists` tuning; needs ANALYZE and good clustering |
| HNSW | high recall at low latency; heavier build | interactive search with tight latency budgets | migration `005_phase5_v4_evolution.sql` uses HNSW for semantic cache table |

In current schema:
- L1/L2/L3 memory tables use IVFFlat indexes (migration `003_phase3_extended_operation.sql`).
- Semantic cache explores HNSW (migration `005_phase5_v4_evolution.sql`).

## Concurrency + isolation notes (async API + pgvector)

Key considerations:
- **Read-your-writes (RYW)**: if a request writes memory then immediately searches, you may or may not see it depending on transaction boundaries and commit timing.
- **Write amplification**: frequent L1 writes + embedding updates can create vacuum pressure and index bloat.
- **Hot partitions**: a single agent/workflow_execution_id can concentrate writes; indexes on `(agent_id, created_at)` help.

Document-only patterns:
- Use explicit transaction scopes for multi-step operations (store + promote).
- Consider “eventual” promotion: promotions can lag slightly behind writes.

## Complexity & Performance

### Writes
- Insert into memory table is typically **O(log N)** for btree indexes plus vector index maintenance (index-dependent).
- If you update embeddings post-hoc, each update triggers index maintenance cost (amplified at high write rates).

### Vector search
- Distance computation is **O(d)** per candidate (d = embedding dimensions, often 1536).
- ANN index complexity is implementation-dependent; treat it as ~**O(log N)** (HNSW-like) or **O(N/lists)** (IVFFlat scanning within lists) as a rough mental model.
- Merging results across tiers is **O(k log k)** if you heap-merge top-k from each tier.

## Advanced Engineering Notes

### Failure Scenarios: “Stale embeddings after model migration”
Symptoms
- Search relevance degrades immediately after changing embedding model/version.
- Similarity thresholds no longer correlate with relevance.

Root cause
- Mixing embeddings from different models/dimensions without version gating.

Mitigation
- Version embeddings; filter search to a single active version.
- Backfill embeddings and switch traffic only after backfill completes.

Observability signals (logs/metrics/traces)
- Increased “low similarity” results or “no results” rate.
- Drift in embedding norm distributions by model/version.

### Failure Scenarios: “Vector index bloat / latency regression”
Symptoms
- p95 search latency climbs over time; CPU usage increases during search.

Root cause
- High churn tables with vector indexes accumulate bloat; vacuum/analyze not keeping up.

Mitigation
- Monitor bloat; schedule vacuum/analyze appropriately.
- Consider partitioning by agent/tenant or time if cardinality is high.

Observability signals (logs/metrics/traces)
- DB metrics: increased buffer reads, higher CPU in `pg_stat_statements`, slower `cosine_distance` queries.
- Traces: DB spans dominate request latency in memory search endpoints.

### Failure Scenarios: “Namespace / tenant leakage”
Symptoms
- One tenant/agent retrieves memories that belong to another.

Root cause
- Missing scoping filters (agent_id/tenant_id) in queries; insufficient DB-level isolation.

Mitigation
- Enforce scoping at the repository layer and (ideally) with DB RLS policies.
- Add audit logs for cross-tenant accesses.

Observability signals (logs/metrics/traces)
- Logs with mismatched `(tenant_id, agent_id)` pairs.
- Security alerts on anomalous access patterns.

## Research Extensions

- Evaluate tier promotion policies (importance thresholds, access-count triggers) with offline replay.
- Retrieval scoring research: hybrid (bm25 + vector) and learned re-rankers per tier.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for how memory writes/reads compose with workflow execution.

## Future Evolution Strategy

- Near-term: define embedding versioning and migration playbooks.
- Mid-term: add tier-aware caching and backpressure for embedding generation.
- Long-term: multi-region memory consistency model (RYW vs eventual) and conflict resolution strategy.

