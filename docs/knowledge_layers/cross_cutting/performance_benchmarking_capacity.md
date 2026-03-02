# Performance, Benchmarking, and Capacity Planning

This document is “how to measure” rather than “how to optimize”. It focuses on reproducible benchmarking and a capacity planning template.

## Benchmarking methodology

| Benchmark type | What it answers | Examples in CognitionOS |
|---|---|---|
| Micro-benchmark | “Is algorithm X fast?” | DAG validation (`Workflow.get_execution_order`), readiness computation, vector search merge |
| Macro-benchmark | “Does the system meet SLOs?” | `/api/v3/memory/search` under load; workflow execution throughput |
| Soak test | “Does it degrade over time?” | vector index bloat; Redis eviction; queue backlogs |

### Avoiding noisy measurements (documented)
- Pin CPU cores where possible; run on quiet machines.
- Warm caches; report both cold and warm behavior.
- Use fixed seeds for generated workflows and embeddings.
- Report distributions (p50/p95/p99), not just mean.

## Load testing approach (document-only)

Traffic shaping recommendations:
- Mix endpoints by production proportions (read-heavy vs write-heavy).
- Include realistic payload sizes (embeddings, step graphs).
- Model retries and client timeouts to expose retry storms.

Saturation signals:
- DB connection pool saturation
- Queue depth increases (RabbitMQ)
- Worker concurrency maxed (Celery)
- Event loop lag (async API)

## Complexity “hotspot map”

Where accidental **O(N²)** appears:
- Topo-sort using list `pop(0)` (documented in `core/workflow_dag_execution.md`).
- Recomputing readiness across all steps on every minor update.
- JSONB-heavy queries without selective indexes.
- High-cardinality metrics labels causing monitoring overhead growth.

## Capacity planning template (fill-in)

| Parameter | Assumption | Notes |
|---|---:|---|
| Target p99 latency | ____ ms | per endpoint |
| Max workflows/min | ____ | |
| Avg steps/workflow | ____ | affects scheduling cost |
| Memory writes/sec | ____ | affects vacuum/index bloat |
| Memory searches/sec | ____ | affects vector index load |
| Embedding dims | 1536 | from pgvector models |
| DB connections | ____ | pool size + max connections |
| Headroom | 30–50% | for bursts and failover |

## Advanced Engineering Notes

### Failure Scenarios: “Benchmarks mislead due to cache effects”
Symptoms
- Optimizations regress production even though benchmarks improved.

Root cause
- Benchmarks ran with warm caches or unrealistic payloads; production has cold paths and varied data.

Mitigation
- Always run paired cold/warm benchmarks; include representative data distributions.
- Use production-like DB sizes for vector index benchmarks.

Observability signals (logs/metrics/traces)
- Production p99 latency diverges from benchmark expectations.
- Traces show new hot spans not present in benchmark runs.

### Complexity & Performance
- Benchmark cost itself is **O(R)** in runs × **O(W)** in workload size; keep workloads representative but bounded.
- Capacity planning is only as good as the worst-case tail behavior (p99) and failure modes (retry storms).

## Research Extensions

- Build a retrieval regression suite with pinned embeddings and deterministic seeds.
- Add “profiling mode” endpoints to capture flamegraphs under controlled load.

## System Design Deep Dive

- See `../system_design_deep_dive.md` for where to place measurement boundaries (API vs worker vs DB).

## Future Evolution Strategy

- Near-term: define SLOs per endpoint; add load test scenarios to CI (nightly).
- Mid-term: automated capacity reports from Prometheus time series.

