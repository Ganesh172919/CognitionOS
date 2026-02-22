"""
Distributed Background Job Queue System

Enterprise job processing with:
- Priority-based job scheduling
- Distributed worker coordination
- Job retry with exponential backoff
- Dead letter queue handling
- Job dependencies and workflows
- Real-time job monitoring
- Worker health tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import uuid
import json
import time
from collections import deque
import asyncio


class JobStatus(Enum):
    """Job execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    DEAD_LETTER = "dead_letter"


class JobPriority(Enum):
    """Job priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BULK = 4


class WorkerStatus(Enum):
    """Worker health status"""
    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class Job:
    """Background job definition"""
    job_id: str
    job_type: str
    payload: Dict[str, Any]
    priority: JobPriority
    status: JobStatus
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    depends_on: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    result: Optional[Any] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Worker:
    """Background worker"""
    worker_id: str
    worker_type: str
    status: WorkerStatus
    current_job_id: Optional[str] = None
    jobs_processed: int = 0
    jobs_failed: int = 0
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    capabilities: Set[str] = field(default_factory=set)


@dataclass
class JobStatistics:
    """Job processing statistics"""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    retried_jobs: int = 0
    avg_execution_time: float = 0.0
    jobs_per_minute: float = 0.0
    active_workers: int = 0


class DistributedJobQueue:
    """
    Distributed background job queue system.

    Features:
    - Priority-based scheduling
    - Job dependencies
    - Retry with backoff
    - Dead letter queue
    - Worker coordination
    - Real-time monitoring
    """

    def __init__(
        self,
        max_queue_size: int = 10000,
        default_timeout: int = 300,
        heartbeat_interval: int = 30
    ):
        self.max_queue_size = max_queue_size
        self.default_timeout = default_timeout
        self.heartbeat_interval = heartbeat_interval

        # Job storage
        self.jobs: Dict[str, Job] = {}
        self.pending_jobs: deque = deque()
        self.scheduled_jobs: List[Job] = []
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}
        self.dead_letter_queue: Dict[str, Job] = {}

        # Workers
        self.workers: Dict[str, Worker] = {}
        self.job_handlers: Dict[str, Callable] = {}

        # Statistics
        self.stats = JobStatistics()

        # Dependencies
        self.dependency_graph: Dict[str, Set[str]] = {}

    def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        depends_on: Optional[List[str]] = None,
        max_retries: int = 3,
        timeout_seconds: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> str:
        """
        Enqueue a new job.

        Args:
            job_type: Type of job to execute
            payload: Job data
            priority: Job priority level
            scheduled_at: Schedule for future execution
            depends_on: List of job IDs this depends on
            max_retries: Maximum retry attempts
            timeout_seconds: Execution timeout
            tags: Tags for filtering

        Returns:
            Job ID
        """
        if len(self.jobs) >= self.max_queue_size:
            raise ValueError("Queue is full")

        job_id = str(uuid.uuid4())

        job = Job(
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            priority=priority,
            status=JobStatus.SCHEDULED if scheduled_at else JobStatus.PENDING,
            created_at=datetime.utcnow(),
            scheduled_at=scheduled_at,
            depends_on=depends_on or [],
            max_retries=max_retries,
            timeout_seconds=timeout_seconds or self.default_timeout,
            tags=tags or set()
        )

        self.jobs[job_id] = job
        self.stats.total_jobs += 1

        if scheduled_at:
            self.scheduled_jobs.append(job)
            self.scheduled_jobs.sort(key=lambda j: j.scheduled_at)
        else:
            if depends_on:
                # Track dependency
                self.dependency_graph[job_id] = set(depends_on)
            else:
                self._add_to_pending_queue(job)

        return job_id

    def register_handler(
        self,
        job_type: str,
        handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Register job handler function"""
        self.job_handlers[job_type] = handler

    def register_worker(
        self,
        worker_type: str,
        capabilities: Optional[Set[str]] = None
    ) -> str:
        """
        Register a new worker.

        Args:
            worker_type: Type of worker
            capabilities: Job types this worker can handle

        Returns:
            Worker ID
        """
        worker_id = f"worker_{uuid.uuid4().hex[:8]}"

        worker = Worker(
            worker_id=worker_id,
            worker_type=worker_type,
            status=WorkerStatus.IDLE,
            capabilities=capabilities or set()
        )

        self.workers[worker_id] = worker
        self.stats.active_workers += 1

        return worker_id

    def dequeue(
        self,
        worker_id: str
    ) -> Optional[Job]:
        """
        Dequeue next job for worker.

        Args:
            worker_id: Worker requesting job

        Returns:
            Job to execute or None
        """
        worker = self.workers.get(worker_id)
        if not worker or worker.status != WorkerStatus.IDLE:
            return None

        # Process scheduled jobs
        self._process_scheduled_jobs()

        # Find suitable job
        job = self._find_job_for_worker(worker)
        if not job:
            return None

        # Assign job to worker
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.worker_id = worker_id

        worker.status = WorkerStatus.BUSY
        worker.current_job_id = job.job_id
        worker.last_heartbeat = datetime.utcnow()

        self.running_jobs[job.job_id] = job

        return job

    def complete_job(
        self,
        job_id: str,
        result: Any,
        worker_id: str
    ) -> bool:
        """
        Mark job as completed.

        Args:
            job_id: Job ID
            result: Job result
            worker_id: Worker that executed job

        Returns:
            True if successful
        """
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False

        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.result = result

        # Update worker
        worker = self.workers.get(worker_id)
        if worker:
            worker.status = WorkerStatus.IDLE
            worker.current_job_id = None
            worker.jobs_processed += 1

        # Move to completed
        if job_id in self.running_jobs:
            del self.running_jobs[job_id]
        self.completed_jobs[job_id] = job

        self.stats.completed_jobs += 1

        # Update avg execution time
        execution_time = (job.completed_at - job.started_at).total_seconds()
        if self.stats.completed_jobs == 1:
            self.stats.avg_execution_time = execution_time
        else:
            self.stats.avg_execution_time = (
                self.stats.avg_execution_time * 0.9 +
                execution_time * 0.1
            )

        # Process dependent jobs
        self._process_dependencies(job_id)

        return True

    def fail_job(
        self,
        job_id: str,
        error: str,
        worker_id: str,
        retry: bool = True
    ) -> bool:
        """
        Mark job as failed and optionally retry.

        Args:
            job_id: Job ID
            error: Error message
            worker_id: Worker that failed
            retry: Whether to retry

        Returns:
            True if handled
        """
        job = self.jobs.get(job_id)
        if not job:
            return False

        job.error = error

        # Update worker
        worker = self.workers.get(worker_id)
        if worker:
            worker.status = WorkerStatus.IDLE
            worker.current_job_id = None
            worker.jobs_failed += 1

        # Retry logic
        if retry and job.retry_count < job.max_retries:
            job.retry_count += 1
            job.status = JobStatus.RETRYING

            # Exponential backoff
            delay_seconds = min(300, 2 ** job.retry_count)
            job.scheduled_at = datetime.utcnow() + timedelta(seconds=delay_seconds)

            self.scheduled_jobs.append(job)
            self.scheduled_jobs.sort(key=lambda j: j.scheduled_at)

            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

            self.stats.retried_jobs += 1
            return True

        # Move to dead letter queue
        job.status = JobStatus.DEAD_LETTER
        self.dead_letter_queue[job_id] = job

        if job_id in self.running_jobs:
            del self.running_jobs[job_id]

        self.stats.failed_jobs += 1
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or scheduled job"""
        job = self.jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.RUNNING]:
            return False  # Cannot cancel running job

        job.status = JobStatus.CANCELLED

        # Remove from queues
        if job in self.scheduled_jobs:
            self.scheduled_jobs.remove(job)

        return True

    def worker_heartbeat(
        self,
        worker_id: str
    ) -> bool:
        """Update worker heartbeat"""
        worker = self.workers.get(worker_id)
        if not worker:
            return False

        worker.last_heartbeat = datetime.utcnow()
        return True

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status"""
        job = self.jobs.get(job_id)
        if not job:
            return None

        return {
            "job_id": job.job_id,
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority.value,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retry_count": job.retry_count,
            "worker_id": job.worker_id,
            "error": job.error,
            "result": job.result
        }

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            "total_jobs": self.stats.total_jobs,
            "pending_jobs": len(self.pending_jobs),
            "scheduled_jobs": len(self.scheduled_jobs),
            "running_jobs": len(self.running_jobs),
            "completed_jobs": self.stats.completed_jobs,
            "failed_jobs": self.stats.failed_jobs,
            "dead_letter_jobs": len(self.dead_letter_queue),
            "active_workers": len([w for w in self.workers.values() if w.status != WorkerStatus.STOPPED]),
            "avg_execution_time": self.stats.avg_execution_time,
            "success_rate": (
                self.stats.completed_jobs / self.stats.total_jobs * 100
                if self.stats.total_jobs > 0 else 0
            )
        }

    def get_worker_statistics(self) -> Dict[str, Any]:
        """Get worker statistics"""
        worker_stats = []

        for worker in self.workers.values():
            worker_stats.append({
                "worker_id": worker.worker_id,
                "worker_type": worker.worker_type,
                "status": worker.status.value,
                "jobs_processed": worker.jobs_processed,
                "jobs_failed": worker.jobs_failed,
                "uptime_seconds": (datetime.utcnow() - worker.started_at).total_seconds(),
                "last_heartbeat": worker.last_heartbeat.isoformat()
            })

        return {
            "workers": worker_stats,
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.workers.values() if w.status == WorkerStatus.BUSY]),
            "idle_workers": len([w for w in self.workers.values() if w.status == WorkerStatus.IDLE])
        }

    def retry_dead_letter_jobs(self) -> int:
        """Retry all jobs in dead letter queue"""
        retried = 0

        for job_id in list(self.dead_letter_queue.keys()):
            job = self.dead_letter_queue[job_id]
            job.status = JobStatus.PENDING
            job.retry_count = 0
            job.error = None

            self._add_to_pending_queue(job)
            del self.dead_letter_queue[job_id]
            retried += 1

        return retried

    # Private helper methods

    def _add_to_pending_queue(self, job: Job) -> None:
        """Add job to pending queue with priority"""
        # Insert based on priority
        inserted = False
        for i, pending_job in enumerate(self.pending_jobs):
            if job.priority.value < pending_job.priority.value:
                self.pending_jobs.insert(i, job)
                inserted = True
                break

        if not inserted:
            self.pending_jobs.append(job)

    def _find_job_for_worker(self, worker: Worker) -> Optional[Job]:
        """Find suitable job for worker"""
        if not self.pending_jobs:
            return None

        # Find job matching worker capabilities
        for _ in range(len(self.pending_jobs)):
            job = self.pending_jobs.popleft()

            if not worker.capabilities or job.job_type in worker.capabilities:
                return job

            # Re-add to queue if not suitable
            self.pending_jobs.append(job)

        return None

    def _process_scheduled_jobs(self) -> None:
        """Move scheduled jobs to pending"""
        now = datetime.utcnow()

        while self.scheduled_jobs and self.scheduled_jobs[0].scheduled_at <= now:
            job = self.scheduled_jobs.pop(0)
            job.status = JobStatus.PENDING
            self._add_to_pending_queue(job)

    def _process_dependencies(self, completed_job_id: str) -> None:
        """Process jobs dependent on completed job"""
        jobs_to_unblock = []

        for job_id, dependencies in list(self.dependency_graph.items()):
            if completed_job_id in dependencies:
                dependencies.remove(completed_job_id)

                if not dependencies:
                    # All dependencies satisfied
                    jobs_to_unblock.append(job_id)
                    del self.dependency_graph[job_id]

        for job_id in jobs_to_unblock:
            job = self.jobs.get(job_id)
            if job:
                job.status = JobStatus.PENDING
                self._add_to_pending_queue(job)

    def check_stale_jobs(self) -> List[str]:
        """Check for stale jobs (running too long)"""
        stale_jobs = []
        now = datetime.utcnow()

        for job_id, job in self.running_jobs.items():
            if job.started_at:
                runtime = (now - job.started_at).total_seconds()
                if runtime > job.timeout_seconds:
                    stale_jobs.append(job_id)

        return stale_jobs

    def check_worker_health(self) -> List[str]:
        """Check for unhealthy workers"""
        unhealthy = []
        now = datetime.utcnow()

        for worker_id, worker in self.workers.items():
            time_since_heartbeat = (now - worker.last_heartbeat).total_seconds()
            if time_since_heartbeat > self.heartbeat_interval * 3:
                unhealthy.append(worker_id)
                worker.status = WorkerStatus.ERROR

        return unhealthy
