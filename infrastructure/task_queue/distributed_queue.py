"""
Distributed Task Queue - Async Job Processing System

Provides background job processing with priority queues, retry logic,
scheduled tasks, job chaining, and comprehensive monitoring.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from heapq import heappush, heappop
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SCHEDULED = "scheduled"


class JobPriority(Enum):
    LOW = 0
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class JobResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    retry_count: int = 0


@dataclass
class Job:
    """Represents a unit of work to be executed asynchronously."""

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    queue_name: str = "default"
    task_name: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None

    # Retry config
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    retry_backoff_factor: float = 2.0
    retry_max_delay_seconds: float = 300.0

    # Timeout
    timeout_seconds: float = 300.0

    # Scheduling
    scheduled_at: Optional[datetime] = None
    cron_expression: Optional[str] = None
    repeat_count: int = 0
    max_repeats: int = 0

    # Chaining
    parent_job_id: Optional[str] = None
    chain_on_success: Optional[str] = None
    chain_on_failure: Optional[str] = None

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[JobResult] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Progress tracking
    progress: float = 0.0
    progress_message: str = ""

    def __lt__(self, other: "Job") -> bool:
        return self.priority.value > other.priority.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "queue_name": self.queue_name,
            "task_name": self.task_name,
            "status": self.status.value,
            "priority": self.priority.name,
            "tenant_id": self.tenant_id,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": {
                "success": self.result.success,
                "error": self.result.error,
                "duration_ms": self.result.duration_ms,
            } if self.result else None,
        }


class TaskExecutor(ABC):
    """Base class for background task executors."""

    @abstractmethod
    async def execute(self, job: Job) -> JobResult: ...

    def get_name(self) -> str:
        return self.__class__.__name__


class ProgressCallback:
    """Callback for reporting job progress."""

    def __init__(self, job: Job):
        self._job = job

    async def update(self, progress: float, message: str = "") -> None:
        self._job.progress = min(max(progress, 0.0), 100.0)
        self._job.progress_message = message


class WorkerPool:
    """Manages a pool of workers for processing jobs."""

    def __init__(
        self,
        concurrency: int = 10,
        name: str = "default",
    ):
        self.name = name
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
        self._active_jobs: Dict[str, Job] = {}
        self._completed_count = 0
        self._failed_count = 0

    @property
    def active_count(self) -> int:
        return len(self._active_jobs)

    @property
    def available_slots(self) -> int:
        return self.concurrency - self.active_count

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "concurrency": self.concurrency,
            "active": self.active_count,
            "available": self.available_slots,
            "completed": self._completed_count,
            "failed": self._failed_count,
        }


class JobScheduler:
    """Handles scheduled and recurring jobs."""

    def __init__(self):
        self._scheduled: List[Tuple[datetime, Job]] = []
        self._recurring: Dict[str, Tuple[Job, float]] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def schedule(self, job: Job, run_at: datetime) -> None:
        job.scheduled_at = run_at
        job.status = JobStatus.SCHEDULED
        heappush(self._scheduled, (run_at, job))

    async def schedule_recurring(
        self, job: Job, interval_seconds: float, max_runs: int = 0
    ) -> None:
        job.max_repeats = max_runs
        self._recurring[job.job_id] = (job, interval_seconds)

    async def start(self, enqueue_fn: Callable) -> None:
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop(enqueue_fn))

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()

    async def _scheduler_loop(self, enqueue_fn: Callable) -> None:
        while self._running:
            now = datetime.utcnow()

            # Check scheduled jobs
            while self._scheduled and self._scheduled[0][0] <= now:
                _, job = heappop(self._scheduled)
                job.status = JobStatus.PENDING
                await enqueue_fn(job)

            # Check recurring jobs
            for job_id, (job, interval) in list(self._recurring.items()):
                if job.max_repeats > 0 and job.repeat_count >= job.max_repeats:
                    del self._recurring[job_id]
                    continue

                if not job.started_at or (
                    now - job.started_at
                ).total_seconds() >= interval:
                    new_job = Job(
                        queue_name=job.queue_name,
                        task_name=job.task_name,
                        payload=job.payload,
                        priority=job.priority,
                        tenant_id=job.tenant_id,
                        parent_job_id=job.job_id,
                    )
                    job.repeat_count += 1
                    job.started_at = now
                    await enqueue_fn(new_job)

            await asyncio.sleep(1.0)


class DistributedTaskQueue:
    """
    Production-grade distributed task queue.

    Features:
    - Priority-based job scheduling
    - Configurable retry with exponential backoff
    - Job chaining (success/failure callbacks)
    - Scheduled and recurring jobs
    - Progress tracking
    - Job dependency resolution
    - Worker pool management
    - Dead-letter queue for permanently failed jobs
    - Tenant-aware job isolation
    - Comprehensive metrics
    """

    def __init__(
        self,
        default_concurrency: int = 10,
        max_queue_size: int = 100_000,
        enable_scheduler: bool = True,
    ):
        self._executors: Dict[str, TaskExecutor] = {}
        self._fn_executors: Dict[str, Callable] = {}
        self._queues: Dict[str, List[Job]] = defaultdict(list)
        self._pools: Dict[str, WorkerPool] = {}
        self._jobs: Dict[str, Job] = {}
        self._completed_jobs: Dict[str, Job] = {}
        self._dead_letter: List[Job] = []
        self._max_queue_size = max_queue_size
        self._default_concurrency = default_concurrency
        self._running = False
        self._loop_tasks: List[asyncio.Task] = []

        # Scheduler
        self._scheduler = JobScheduler() if enable_scheduler else None

        # Metrics
        self._metrics = {
            "enqueued": 0,
            "completed": 0,
            "failed": 0,
            "retried": 0,
            "cancelled": 0,
            "timed_out": 0,
            "total_duration_ms": 0.0,
        }

    # -- Registration -------------------------------------------------------

    def register_executor(self, task_name: str, executor: TaskExecutor) -> None:
        self._executors[task_name] = executor
        logger.info("Registered executor %s for task %s", executor.get_name(), task_name)

    def register_handler(self, task_name: str):
        """Decorator to register a function as a task handler."""

        def decorator(fn: Callable):
            self._fn_executors[task_name] = fn
            logger.info("Registered handler %s for task %s", fn.__name__, task_name)
            return fn

        return decorator

    def create_pool(
        self, queue_name: str, concurrency: Optional[int] = None
    ) -> WorkerPool:
        pool = WorkerPool(
            concurrency=concurrency or self._default_concurrency,
            name=queue_name,
        )
        self._pools[queue_name] = pool
        return pool

    # -- Enqueue ------------------------------------------------------------

    async def enqueue(self, job: Job) -> str:
        if len(self._queues[job.queue_name]) >= self._max_queue_size:
            raise RuntimeError(f"Queue {job.queue_name} is full")

        job.status = JobStatus.QUEUED
        self._jobs[job.job_id] = job
        heappush(self._queues[job.queue_name], job)
        self._metrics["enqueued"] += 1

        logger.debug(
            "Enqueued job %s [%s] on queue %s",
            job.job_id,
            job.task_name,
            job.queue_name,
        )
        return job.job_id

    async def enqueue_many(self, jobs: List[Job]) -> List[str]:
        ids = []
        for job in jobs:
            jid = await self.enqueue(job)
            ids.append(jid)
        return ids

    async def schedule_job(self, job: Job, run_at: datetime) -> str:
        if self._scheduler:
            self._jobs[job.job_id] = job
            await self._scheduler.schedule(job, run_at)
            return job.job_id
        raise RuntimeError("Scheduler is disabled")

    async def schedule_recurring(
        self, job: Job, interval_seconds: float, max_runs: int = 0
    ) -> str:
        if self._scheduler:
            self._jobs[job.job_id] = job
            await self._scheduler.schedule_recurring(job, interval_seconds, max_runs)
            return job.job_id
        raise RuntimeError("Scheduler is disabled")

    # -- Job management -----------------------------------------------------

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id) or self._completed_jobs.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.QUEUED, JobStatus.SCHEDULED):
            job.status = JobStatus.CANCELLED
            self._metrics["cancelled"] += 1
            return True
        return False

    def get_queue_stats(self) -> Dict[str, Any]:
        return {
            "queues": {
                name: {
                    "pending": len(q),
                    "pool": self._pools.get(name, WorkerPool(name=name)).get_stats(),
                }
                for name, q in self._queues.items()
            },
            "total_jobs": len(self._jobs),
            "completed_jobs": len(self._completed_jobs),
            "dead_letter_size": len(self._dead_letter),
            "metrics": self._metrics.copy(),
        }

    # -- Processing ---------------------------------------------------------

    async def start(self) -> None:
        self._running = True

        # Create default pool if not exists
        if "default" not in self._pools:
            self.create_pool("default")

        # Start scheduler
        if self._scheduler:
            await self._scheduler.start(self.enqueue)

        # Start worker loops for known queues
        for queue_name in list(self._queues.keys()) + list(self._pools.keys()):
            if queue_name not in [t.get_name() for t in self._loop_tasks if hasattr(t, "get_name")]:
                task = asyncio.create_task(self._process_queue(queue_name))
                self._loop_tasks.append(task)

        logger.info("Task queue started with %d pools", len(self._pools))

    async def stop(self, graceful: bool = True, timeout: float = 30.0) -> None:
        self._running = False

        if self._scheduler:
            await self._scheduler.stop()

        if graceful:
            # Wait for active jobs to complete
            deadline = time.monotonic() + timeout
            while any(
                pool.active_count > 0 for pool in self._pools.values()
            ) and time.monotonic() < deadline:
                await asyncio.sleep(0.5)

        for task in self._loop_tasks:
            task.cancel()
        self._loop_tasks.clear()

        logger.info("Task queue stopped")

    async def _process_queue(self, queue_name: str) -> None:
        pool = self._pools.get(queue_name)
        if not pool:
            pool = self.create_pool(queue_name)

        while self._running:
            queue = self._queues.get(queue_name, [])
            if not queue or pool.available_slots <= 0:
                await asyncio.sleep(0.1)
                continue

            job = heappop(queue)
            if job.status == JobStatus.CANCELLED:
                continue

            # Check dependencies
            if job.depends_on:
                deps_met = all(
                    self._completed_jobs.get(dep_id, Job()).status == JobStatus.COMPLETED
                    for dep_id in job.depends_on
                )
                if not deps_met:
                    heappush(queue, job)
                    await asyncio.sleep(0.5)
                    continue

            async with pool._semaphore:
                pool._active_jobs[job.job_id] = job
                try:
                    await self._execute_job(job, pool)
                finally:
                    pool._active_jobs.pop(job.job_id, None)

    async def _execute_job(self, job: Job, pool: WorkerPool) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        start = time.monotonic()

        try:
            # Find executor
            executor = self._executors.get(job.task_name)
            fn = self._fn_executors.get(job.task_name)

            if executor:
                result = await asyncio.wait_for(
                    executor.execute(job), timeout=job.timeout_seconds
                )
            elif fn:
                result = await asyncio.wait_for(
                    fn(job), timeout=job.timeout_seconds
                )
            else:
                raise RuntimeError(f"No executor registered for task: {job.task_name}")

            elapsed_ms = (time.monotonic() - start) * 1000
            result.duration_ms = elapsed_ms

            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            pool._completed_count += 1
            self._metrics["completed"] += 1
            self._metrics["total_duration_ms"] += elapsed_ms

            # Move to completed storage
            self._completed_jobs[job.job_id] = self._jobs.pop(job.job_id, job)

            # Chain on success
            if result.success and job.chain_on_success:
                chain_job = Job(
                    task_name=job.chain_on_success,
                    queue_name=job.queue_name,
                    payload={**job.payload, "parent_result": result.data},
                    parent_job_id=job.job_id,
                    tenant_id=job.tenant_id,
                    correlation_id=job.correlation_id,
                )
                await self.enqueue(chain_job)

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            job.status = JobStatus.TIMEOUT
            job.result = JobResult(
                success=False, error="Job timed out", duration_ms=elapsed_ms
            )
            self._metrics["timed_out"] += 1
            await self._handle_failure(job, pool)

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            job.result = JobResult(
                success=False, error=str(exc), duration_ms=elapsed_ms
            )
            await self._handle_failure(job, pool)

    async def _handle_failure(self, job: Job, pool: WorkerPool) -> None:
        job.retry_count += 1

        if job.retry_count <= job.max_retries:
            delay = min(
                job.retry_delay_seconds * (job.retry_backoff_factor ** (job.retry_count - 1)),
                job.retry_max_delay_seconds,
            )
            job.status = JobStatus.RETRYING
            self._metrics["retried"] += 1
            logger.warning(
                "Retrying job %s (%d/%d) in %.1fs",
                job.job_id, job.retry_count, job.max_retries, delay,
            )
            await asyncio.sleep(delay)
            job.status = JobStatus.QUEUED
            heappush(self._queues[job.queue_name], job)
        else:
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            pool._failed_count += 1
            self._metrics["failed"] += 1
            self._dead_letter.append(job)
            self._completed_jobs[job.job_id] = self._jobs.pop(job.job_id, job)

            # Chain on failure
            if job.chain_on_failure:
                chain_job = Job(
                    task_name=job.chain_on_failure,
                    queue_name=job.queue_name,
                    payload={**job.payload, "parent_error": job.result.error if job.result else None},
                    parent_job_id=job.job_id,
                    tenant_id=job.tenant_id,
                    correlation_id=job.correlation_id,
                )
                await self.enqueue(chain_job)

            logger.error("Job %s permanently failed after %d retries", job.job_id, job.max_retries)
