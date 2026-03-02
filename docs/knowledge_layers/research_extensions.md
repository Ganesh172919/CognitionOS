# Research Extensions (Experiments & Hypotheses)

This document lists research-grade extensions that can be explored without committing to them in production. Each item should be treated as a hypothesis with measurable outcomes.

## Experiments (table)

| Experiment | Hypothesis | Method | Metrics | Code anchors |
|---|---|---|---|---|
| Retrieval evaluation harness | Better evaluation reduces regressions in memory relevance | offline dataset + replay queries + judge model | nDCG@k, MRR, p95 latency, cost/query | `core/application/memory_hierarchy/use_cases.py`, `infrastructure/persistence/memory_hierarchy_repository.py` |
| CRDT-style memory merges | Multi-writer memory updates can converge without conflicts | define CRDT for tags/metadata + vector versioning | conflict rate, convergence time, storage overhead | `core/domain/memory_hierarchy/entities.py` |
| Causal tracing for agent decisions | Trace-level causality improves debugging of “why did agent do X” | propagate decision IDs into spans + memory references | time-to-root-cause, # unknown failures | `infrastructure/observability/*`, `services/api/src/middleware/request_id.py` |
| Deterministic replay validator | Replay catches nondeterminism early | compare `step_execution_attempts` outputs hashes | divergence %, false positive rate | `database/migrations/008_execution_persistence.sql` |

## Advanced Engineering Notes

### Failure Scenarios: “Offline/online mismatch”
Symptoms
- Offline evaluation improves while production relevance degrades (or vice versa).

Root cause
- Training/evaluation data not representative; production query distribution shifts.

Mitigation
- Sample real production queries (privacy-safe) and build stratified eval sets.
- Track embedding model/version and segment metrics by version.

Observability signals (logs/metrics/traces)
- Increased “no results” rate in memory search endpoints.
- Drift in embedding distributions (mean norm, cosine similarity histograms).

### Complexity & Performance
- A naïve evaluation harness that compares all pairs is **O(N²)**; avoid it by sampling and using ANN indexes for candidate generation.
- Replay validation is typically **O(A)** where A is number of attempts; hashing adds **O(S)** per payload size.

## Research Extensions

- Add a “retrieval scoreboard” to CI using fixed seeds and pinned embedding models.
- Explore approximate nearest neighbor quality vs cost by sweeping IVFFlat/HNSW parameters.

## System Design Deep Dive

- Use `system_design_deep_dive.md` to ensure experiments reflect actual production flows (where caching, retries, and timeouts matter).

## Future Evolution Strategy

- Graduating a research extension into production should produce a small PR that updates `future_evolution_strategy.md` with a concrete adoption plan and rollback strategy.

