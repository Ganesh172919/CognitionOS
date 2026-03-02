# Workflow DAG Execution (Algorithms, Determinism, Failure Modes)

This document is anchored to:
- DAG/topological ordering: `core/domain/workflow/entities.py` (`Workflow.get_execution_order`)
- Validation + template references: `core/domain/workflow/services.py` (`WorkflowValidator`)
- Step readiness: `core/domain/workflow/services.py` (`WorkflowExecutionOrchestrator`)
- Use cases: `core/application/workflow/use_cases.py`

## Algorithmic deep dive (topological sort + cycle detection)

### Current implementation (Kahn’s algorithm)
`Workflow.get_execution_order()` builds:
- `step_map`: StepId → step
- `in_degree`: StepId → integer
- `adjacency`: StepId → list[StepId]

Then it:
1. Seeds a queue with nodes where `in_degree == 0`
2. Pops from the queue, appends to result
3. Decrements neighbors’ in-degrees; enqueue when they hit 0
4. If `len(result) != len(steps)`: cycle detected → error

Example DAG
```text
  A      B
   \    /
    \  /
     C
     |
     D

Topo order (one valid): A, B, C, D
Parallel “level sets”: {A,B} -> {C} -> {D}
```

### Stable ordering + determinism (tie-breaking)
Topological sorting is not unique. For deterministic replay and consistent UX:
- Define a stable tie-break rule when multiple nodes have `in_degree == 0` (e.g., sort by `StepId.value`).
- Persist (or derive) an **execution plan ID** so replays compute the same ordering.

Document-only recommendation:
```text
ready = all steps with in_degree==0
while ready:
  choose next = min(ready, key=StepId.value)   # stable tie-break
  pop next, relax edges, add newly-ready nodes
```

### Critical path + parallelism estimation (planning)
To estimate “how parallel can this workflow be?”:
- Compute a longest path in the DAG when edges represent “must happen before”.
- For unit-weight steps, you can compute levels during topo traversal:
  - `level[node] = 1 + max(level[dep] for dep in depends_on)` else 0
- Peak parallelism can be approximated by counting nodes per level.

## Determinism notes (replay semantics and idempotency)

**What determinism should mean (documented)**
- Given the same workflow definition + inputs, the system should produce the same step scheduling decisions.
- External calls (tools, HTTP) are nondeterministic; capture their inputs/outputs at the boundary.

**Idempotency keys**
- Migration `database/migrations/008_execution_persistence.sql` introduces `step_execution_attempts.idempotency_key`.
- Documented pattern: derive key from immutable execution identifiers.

Pseudocode:
```text
idempotency_key = sha256("exec:{execution_id}:step:{step_id}:attempt:{attempt_number}")
```

## Complexity & Performance

### DAG validation
- Building adjacency/in-degree is **O(V + E)** (V steps, E dependencies).
- Kahn’s algorithm is **O(V + E)** in theory.

**Important note about the current queue**
- If the queue is a Python list and uses `pop(0)`, each pop is **O(V)** due to shifting.
- Worst-case topo traversal becomes **O(V² + E)** for large V.
- Document-only optimization: use `collections.deque` for amortized **O(1)** pops.

### Step readiness computation
`WorkflowExecutionOrchestrator.get_ready_steps()`:
- Builds a status map: **O(V)**
- Checks dependencies: worst-case **O(E)** across all steps
- Total: **O(V + E)** per readiness computation

## Benchmarking (DAG validation + scheduling)

Micro-benchmark guidance (document-only):
- Generate DAGs with controlled shapes:
  - chain (critical path length V)
  - wide fan-out (high parallelism)
  - random sparse/dense
- Measure:
  - p50/p95/p99 validation latency
  - allocations (`tracemalloc`) and peak RSS
  - stability (variance across runs)

Suggested metrics to record:
- `validation_time_ms` (p50/p99)
- `execution_order_length`
- `queue_ops_count`
- `alloc_bytes_total`

## Advanced Engineering Notes

### Failure Scenarios: “Circular dependencies”
Symptoms
- Workflow creation fails with “circular dependencies”.
- Execution never starts (no ready steps).

Root cause
- Cycle in `depends_on` relations; topo-sort cannot visit all nodes.

Mitigation
- Validate on create (`WorkflowValidator.validate_dag`).
- Provide a cycle “witness” in error messages (documented enhancement).

Observability signals (logs/metrics/traces)
- Error logs from the create-workflow route with invalid DAG details.
- Metrics: spikes in 4xx validation failures on `/api/v3/workflows`.

### Failure Scenarios: “Template reference bugs”
Symptoms
- Workflow validation passes DAG but execution fails due to missing step outputs.
- Intermittent failures when steps reference outputs “from the future”.

Root cause
- Invalid `${{ steps.X... }}` references; or references to steps that appear later in execution order.

Mitigation
- Use `WorkflowValidator.validate_template_references()` before activation.
- Document constraints: template references must only target steps that are earlier in the topo order.

Observability signals (logs/metrics/traces)
- Errors containing “references non-existent step” or “comes after it in execution order”.
- Trace spans showing step execution started with missing input fields.

### Failure Scenarios: “Stuck steps / no progress”
Symptoms
- Execution remains in “running” but no steps are scheduled.

Root cause
- Dependency status never transitions to completed (lost updates, worker crash, missing event).

Mitigation
- Add a “driver loop” that periodically recomputes readiness and detects lack of progress.
- Persist leases/locks to avoid two drivers scheduling the same step.

Observability signals (logs/metrics/traces)
- Metrics: `workflows_in_progress` steady, but `workflow_steps_executed_total` flat.
- Logs: repeated “no ready steps” warnings for same execution_id.

## Research Extensions

- Property tests: generate random DAGs; assert topo order validity and determinism under stable tie-break.
- Replay semantics: compare original outputs to replay outputs, marking nondeterminism flags (migration 008 fields).

## System Design Deep Dive

- See `../system_design_deep_dive.md` for how API/use-case/repo boundaries interact with scheduling semantics.

## Future Evolution Strategy

- Near-term: define stable ordering + idempotency keys for side effects.
- Mid-term: durable orchestration loop + distributed locks (`execution_locks`).
- Long-term: adopt a Temporal-like engine if replay/durability requirements dominate (see `../future_evolution_strategy.md`).

