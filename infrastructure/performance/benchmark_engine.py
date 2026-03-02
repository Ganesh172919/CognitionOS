"""
Performance Benchmarking System — CognitionOS

Micro-benchmarks with percentile analysis, memory profiling,
regression detection, and JSON report generation.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import statistics
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


def _percentile(data: List[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = (pct / 100.0) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    w = idx - lo
    return s[lo] * (1 - w) + s[hi] * w


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    total_time_s: float
    avg_ms: float
    min_ms: float
    max_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    ops_per_sec: float
    mem_peak_mb: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in {
            "name": self.name, "iterations": self.iterations,
            "total_time_s": self.total_time_s, "avg_ms": self.avg_ms,
            "min_ms": self.min_ms, "max_ms": self.max_ms,
            "median_ms": self.median_ms, "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms, "std_dev_ms": self.std_dev_ms,
            "ops_per_sec": self.ops_per_sec, "mem_peak_mb": self.mem_peak_mb,
            "timestamp": self.timestamp,
        }.items()}


@dataclass
class ComparisonResult:
    baseline: BenchmarkResult
    candidate: BenchmarkResult
    speedup: float
    latency_pct: float
    is_regression: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {"baseline": self.baseline.name, "candidate": self.candidate.name,
                "speedup": round(self.speedup, 3), "latency_pct": round(self.latency_pct, 2),
                "is_regression": self.is_regression, "summary": self.summary}


class BenchmarkEngine:
    def __init__(self, *, iters: int = 100, warmup: int = 5,
                 mem_profile: bool = True, results_dir: str = ".benchmarks",
                 regression_threshold: float = 10.0) -> None:
        self._iters = iters
        self._warmup = warmup
        self._mem = mem_profile
        self._dir = Path(results_dir)
        self._threshold = regression_threshold
        self._registered: Dict[str, Callable] = {}
        self._history: Dict[str, List[BenchmarkResult]] = defaultdict(list)

    def register(self, name: str | None = None):
        def dec(fn):
            self._registered[name or fn.__qualname__] = fn
            return fn
        return dec

    def _build(self, name: str, iters: int, total: float,
               times: List[float], mem: float) -> BenchmarkResult:
        return BenchmarkResult(
            name=name, iterations=iters, total_time_s=total,
            avg_ms=statistics.mean(times), min_ms=min(times), max_ms=max(times),
            median_ms=statistics.median(times), p95_ms=_percentile(times, 95),
            p99_ms=_percentile(times, 99),
            std_dev_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            ops_per_sec=iters / total if total > 0 else 0, mem_peak_mb=mem)

    def run_sync(self, func: Callable, *a, name: str | None = None,
                 iterations: int | None = None, **kw) -> BenchmarkResult:
        n = name or func.__qualname__
        it = iterations or self._iters
        for _ in range(self._warmup):
            func(*a, **kw)
        gc.collect()
        if self._mem:
            tracemalloc.start()
        times = []
        t0 = time.perf_counter()
        for _ in range(it):
            s = time.perf_counter()
            func(*a, **kw)
            times.append((time.perf_counter() - s) * 1000)
        total = time.perf_counter() - t0
        mem = 0.0
        if self._mem:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem = peak / (1024 * 1024)
        r = self._build(n, it, total, times, mem)
        self._history[n].append(r)
        return r

    async def run_async(self, func: Callable[..., Awaitable], *a,
                        name: str | None = None, iterations: int | None = None,
                        **kw) -> BenchmarkResult:
        n = name or func.__qualname__
        it = iterations or self._iters
        for _ in range(self._warmup):
            await func(*a, **kw)
        gc.collect()
        if self._mem:
            tracemalloc.start()
        times = []
        t0 = time.perf_counter()
        for _ in range(it):
            s = time.perf_counter()
            await func(*a, **kw)
            times.append((time.perf_counter() - s) * 1000)
        total = time.perf_counter() - t0
        mem = 0.0
        if self._mem:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem = peak / (1024 * 1024)
        r = self._build(n, it, total, times, mem)
        self._history[n].append(r)
        return r

    def compare(self, baseline: BenchmarkResult, candidate: BenchmarkResult) -> ComparisonResult:
        sp = baseline.avg_ms / candidate.avg_ms if candidate.avg_ms else float("inf")
        pct = ((baseline.avg_ms - candidate.avg_ms) / baseline.avg_ms * 100) if baseline.avg_ms else 0
        reg = pct < -self._threshold
        if reg:
            s = f"REGRESSION: {abs(pct):.1f}% slower"
        elif pct > 0:
            s = f"IMPROVEMENT: {pct:.1f}% faster"
        else:
            s = "NEUTRAL"
        return ComparisonResult(baseline, candidate, sp, pct, reg, s)

    def detect_regression(self, name: str) -> ComparisonResult | None:
        h = self._history.get(name, [])
        return self.compare(h[-2], h[-1]) if len(h) >= 2 else None

    def save_report(self, fname: str | None = None) -> str:
        self._dir.mkdir(parents=True, exist_ok=True)
        fn = fname or f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        p = self._dir / fn
        report = {"generated": datetime.now(timezone.utc).isoformat(),
                  "benchmarks": {n: [r.to_dict() for r in rs] for n, rs in self._history.items()}}
        p.write_text(json.dumps(report, indent=2, default=str))
        return str(p)

    def get_history(self, name: str) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._history.get(name, [])]

_engine: BenchmarkEngine | None = None

def get_benchmark_engine() -> BenchmarkEngine:
    global _engine
    if not _engine:
        _engine = BenchmarkEngine()
    return _engine
