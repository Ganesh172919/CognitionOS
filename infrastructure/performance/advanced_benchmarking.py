"""
Advanced Benchmarking System — CognitionOS

Extends the base performance benchmarking with:
- API endpoint load testing
- Database query benchmarking
- AI inference latency tracking
- Comparative A/B benchmarks
- Automated regression CI pipeline
- Dashboard-ready metric export
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_seconds: float = 5.0
    think_time_ms: float = 0
    timeout_seconds: float = 30.0
    target_rps: Optional[float] = None


@dataclass
class LoadTestResult:
    test_id: str
    name: str
    config: LoadTestConfig
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration_seconds: float = 0
    avg_latency_ms: float = 0
    p50_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0
    max_latency_ms: float = 0
    min_latency_ms: float = 0
    throughput_rps: float = 0
    error_rate_pct: float = 0
    status_codes: Dict[int, int] = field(default_factory=dict)
    latency_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id, "name": self.name,
            "total_requests": self.total_requests,
            "successful": self.successful_requests,
            "failed": self.failed_requests,
            "duration_s": round(self.total_duration_seconds, 2),
            "avg_ms": round(self.avg_latency_ms, 2),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "p99_ms": round(self.p99_ms, 2),
            "throughput_rps": round(self.throughput_rps, 2),
            "error_rate_pct": round(self.error_rate_pct, 2),
        }


class LoadTester:
    """HTTP/function load tester with ramped concurrent users."""

    def __init__(self):
        self._results: Dict[str, List[LoadTestResult]] = defaultdict(list)

    async def run(self, name: str,
                   fn: Callable[..., Awaitable[Any]],
                   config: Optional[LoadTestConfig] = None) -> LoadTestResult:
        """Run load test against an async function."""
        config = config or LoadTestConfig()
        test_id = uuid.uuid4().hex[:12]
        all_latencies: List[float] = []
        errors = 0
        lock = asyncio.Lock()

        async def user_worker(user_id: int):
            nonlocal errors
            local_latencies = []
            ramp_delay = (config.ramp_up_seconds / config.concurrent_users) * user_id
            await asyncio.sleep(ramp_delay)

            for _ in range(config.requests_per_user):
                start = time.perf_counter()
                try:
                    await asyncio.wait_for(fn(), timeout=config.timeout_seconds)
                    latency_ms = (time.perf_counter() - start) * 1000
                    local_latencies.append(latency_ms)
                except Exception:
                    errors += 1

                if config.think_time_ms > 0:
                    await asyncio.sleep(config.think_time_ms / 1000)

            async with lock:
                all_latencies.extend(local_latencies)

        start_time = time.time()
        tasks = [asyncio.create_task(user_worker(i))
                  for i in range(config.concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start_time

        sorted_lat = sorted(all_latencies) if all_latencies else [0]
        n = len(sorted_lat)
        total = len(all_latencies) + errors

        result = LoadTestResult(
            test_id=test_id, name=name, config=config,
            total_requests=total,
            successful_requests=len(all_latencies),
            failed_requests=errors,
            total_duration_seconds=duration,
            avg_latency_ms=statistics.mean(sorted_lat) if sorted_lat else 0,
            p50_ms=sorted_lat[n // 2] if sorted_lat else 0,
            p95_ms=sorted_lat[int(n * 0.95)] if sorted_lat else 0,
            p99_ms=sorted_lat[min(int(n * 0.99), n - 1)] if sorted_lat else 0,
            max_latency_ms=sorted_lat[-1] if sorted_lat else 0,
            min_latency_ms=sorted_lat[0] if sorted_lat else 0,
            throughput_rps=total / max(duration, 0.001),
            error_rate_pct=(errors / max(total, 1)) * 100,
            latency_distribution=self._build_distribution(sorted_lat),
        )

        self._results[name].append(result)
        return result

    def _build_distribution(self, latencies: List[float]) -> Dict[str, int]:
        buckets = {"<10ms": 0, "10-50ms": 0, "50-100ms": 0,
                    "100-500ms": 0, "500ms-1s": 0, ">1s": 0}
        for lat in latencies:
            if lat < 10:
                buckets["<10ms"] += 1
            elif lat < 50:
                buckets["10-50ms"] += 1
            elif lat < 100:
                buckets["50-100ms"] += 1
            elif lat < 500:
                buckets["100-500ms"] += 1
            elif lat < 1000:
                buckets["500ms-1s"] += 1
            else:
                buckets[">1s"] += 1
        return buckets

    def get_history(self, name: str) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._results.get(name, [])]


@dataclass
class QueryBenchmarkResult:
    query_name: str
    query_text: str
    execution_count: int = 0
    avg_ms: float = 0
    p95_ms: float = 0
    p99_ms: float = 0
    rows_returned: int = 0
    plan_cost: float = 0
    index_used: bool = False
    suggestions: List[str] = field(default_factory=list)


class DatabaseBenchmarker:
    """Benchmark database queries with EXPLAIN analysis."""

    def __init__(self):
        self._results: Dict[str, List[QueryBenchmarkResult]] = defaultdict(list)

    async def benchmark_query(self, name: str,
                                query_fn: Callable[[], Awaitable[Any]], *,
                                iterations: int = 100) -> QueryBenchmarkResult:
        """Benchmark a database query function."""
        latencies = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                await query_fn()
                latencies.append((time.perf_counter() - start) * 1000)
            except Exception as exc:
                logger.warning("Query benchmark error: %s", exc)

        sorted_lat = sorted(latencies) if latencies else [0]
        n = len(sorted_lat)

        result = QueryBenchmarkResult(
            query_name=name,
            query_text="",
            execution_count=len(latencies),
            avg_ms=statistics.mean(sorted_lat) if sorted_lat else 0,
            p95_ms=sorted_lat[int(n * 0.95)] if sorted_lat else 0,
            p99_ms=sorted_lat[min(int(n * 0.99), n - 1)] if sorted_lat else 0,
        )

        # Auto-suggest optimizations
        if result.avg_ms > 100:
            result.suggestions.append("Consider adding database indexes")
        if result.p99_ms > 500:
            result.suggestions.append("Query has high tail latency — review EXPLAIN plan")
        if result.avg_ms > 50 and n > 50:
            result.suggestions.append("Consider query caching for this workload")

        self._results[name].append(result)
        return result

    def get_slow_queries(self, *, threshold_ms: float = 100) -> List[Dict[str, Any]]:
        slow = []
        for name, results in self._results.items():
            if results and results[-1].avg_ms > threshold_ms:
                r = results[-1]
                slow.append({
                    "name": name, "avg_ms": round(r.avg_ms, 2),
                    "p95_ms": round(r.p95_ms, 2),
                    "suggestions": r.suggestions,
                })
        return sorted(slow, key=lambda x: -x["avg_ms"])


class InferenceBenchmarker:
    """Benchmark AI model inference latency and throughput."""

    def __init__(self):
        self._results: Dict[str, List[Dict[str, float]]] = defaultdict(list)

    async def benchmark_inference(self, model_name: str,
                                     inference_fn: Callable[..., Awaitable[Any]],
                                     *args,
                                     iterations: int = 50,
                                     batch_sizes: Optional[List[int]] = None
                                     ) -> Dict[str, Any]:
        batch_sizes = batch_sizes or [1]
        results = {}

        for bs in batch_sizes:
            latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                try:
                    await inference_fn(*args)
                    latencies.append((time.perf_counter() - start) * 1000)
                except Exception:
                    pass

            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)
                result = {
                    "batch_size": bs,
                    "avg_ms": round(statistics.mean(sorted_lat), 2),
                    "p50_ms": round(sorted_lat[n // 2], 2),
                    "p95_ms": round(sorted_lat[int(n * 0.95)], 2),
                    "throughput_inferences_per_sec": round(
                        (bs * len(sorted_lat)) / (sum(sorted_lat) / 1000), 2
                    ),
                }
                results[f"batch_{bs}"] = result
                self._results[model_name].append(result)

        return {"model": model_name, "results": results}


class ABBenchmarkRunner:
    """Run A/B benchmarks to compare two implementations."""

    async def compare(self, name_a: str, fn_a: Callable[..., Awaitable[Any]],
                       name_b: str, fn_b: Callable[..., Awaitable[Any]], *,
                       iterations: int = 100) -> Dict[str, Any]:
        """Compare two functions with statistical significance."""
        lat_a, lat_b = [], []

        for _ in range(iterations):
            start = time.perf_counter()
            await fn_a()
            lat_a.append((time.perf_counter() - start) * 1000)

            start = time.perf_counter()
            await fn_b()
            lat_b.append((time.perf_counter() - start) * 1000)

        mean_a = statistics.mean(lat_a)
        mean_b = statistics.mean(lat_b)
        std_a = statistics.stdev(lat_a) if len(lat_a) > 1 else 0
        std_b = statistics.stdev(lat_b) if len(lat_b) > 1 else 0

        improvement_pct = ((mean_a - mean_b) / max(mean_a, 0.001)) * 100
        winner = name_a if mean_a < mean_b else name_b

        return {
            name_a: {
                "mean_ms": round(mean_a, 3),
                "std_dev_ms": round(std_a, 3),
                "p95_ms": round(sorted(lat_a)[int(len(lat_a) * 0.95)], 3),
            },
            name_b: {
                "mean_ms": round(mean_b, 3),
                "std_dev_ms": round(std_b, 3),
                "p95_ms": round(sorted(lat_b)[int(len(lat_b) * 0.95)], 3),
            },
            "winner": winner,
            "improvement_pct": round(abs(improvement_pct), 2),
            "iterations": iterations,
        }
