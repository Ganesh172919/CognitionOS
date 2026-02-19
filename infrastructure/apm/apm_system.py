"""
Application Performance Monitoring (APM)

Comprehensive performance monitoring with distributed tracing and profiling.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import time


@dataclass
class PerformanceMetric:
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str]


@dataclass
class TransactionTrace:
    trace_id: str
    name: str
    duration_ms: float
    started_at: datetime
    ended_at: datetime
    status: str
    spans: List[Dict]
    metadata: Dict


class APMSystem:
    """
    Application Performance Monitoring system.
    
    Features:
    - Transaction tracing
    - Performance metrics
    - Error tracking
    - Resource monitoring
    - Custom instrumentation
    - Alert generation
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.traces: List[TransactionTrace] = []
        self.active_transactions: Dict[str, Dict] = {}
    
    async def start_transaction(
        self,
        name: str,
        trace_id: str
    ) -> str:
        """Start monitoring a transaction"""
        self.active_transactions[trace_id] = {
            "name": name,
            "started_at": datetime.utcnow(),
            "spans": []
        }
        return trace_id
    
    async def end_transaction(
        self,
        trace_id: str,
        status: str = "success"
    ) -> TransactionTrace:
        """End transaction and record trace"""
        if trace_id not in self.active_transactions:
            raise ValueError(f"Transaction {trace_id} not found")
        
        txn = self.active_transactions[trace_id]
        ended_at = datetime.utcnow()
        duration = (ended_at - txn["started_at"]).total_seconds() * 1000
        
        trace = TransactionTrace(
            trace_id=trace_id,
            name=txn["name"],
            duration_ms=duration,
            started_at=txn["started_at"],
            ended_at=ended_at,
            status=status,
            spans=txn["spans"],
            metadata={}
        )
        
        self.traces.append(trace)
        del self.active_transactions[trace_id]
        
        return trace
    
    async def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "count",
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            tags=tags or {}
        )
        self.metrics.append(metric)
    
    async def get_performance_summary(
        self,
        time_window: int = 3600
    ) -> Dict:
        """Get performance summary for time window"""
        cutoff = datetime.utcnow() - timedelta(seconds=time_window)
        
        recent_traces = [
            t for t in self.traces
            if t.started_at >= cutoff
        ]
        
        if not recent_traces:
            return {
                "total_transactions": 0,
                "avg_duration_ms": 0,
                "p50_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "error_rate": 0
            }
        
        durations = sorted([t.duration_ms for t in recent_traces])
        errors = sum(1 for t in recent_traces if t.status != "success")
        
        return {
            "total_transactions": len(recent_traces),
            "avg_duration_ms": sum(durations) / len(durations),
            "p50_ms": durations[int(len(durations) * 0.5)],
            "p95_ms": durations[int(len(durations) * 0.95)],
            "p99_ms": durations[int(len(durations) * 0.99)],
            "error_rate": (errors / len(recent_traces)) * 100
        }
