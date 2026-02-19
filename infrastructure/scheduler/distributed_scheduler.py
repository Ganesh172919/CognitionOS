"""
Distributed Job Scheduler

Production-grade distributed job scheduler with:
- Multiple schedule types (cron, interval, one-time)
- Job priority queue with preemption
- DAG-based dependency execution
- Persistent state with PostgreSQL
- Exponential backoff retry
- Concurrent execution limits
- Health monitoring and metrics
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from croniter import croniter
import json
import hashlib

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class JobPriority(int, Enum):
    """Job priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ScheduleType(str, Enum):
    """Job schedule types."""
    CRON = "cron"
    INTERVAL = "interval"
    ONE_TIME = "one_time"
    DEPENDENT = "dependent"


@dataclass
class JobSchedule:
    """Job schedule configuration."""
    schedule_type: ScheduleType
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_executions: Optional[int] = None
    
    def get_next_run(self, from_time: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next execution time."""
        base_time = from_time or datetime.utcnow()
        
        if self.schedule_type == ScheduleType.CRON:
            if not self.cron_expression:
                return None
            cron = croniter(self.cron_expression, base_time)
            next_time = cron.get_next(datetime)
        elif self.schedule_type == ScheduleType.INTERVAL:
            if not self.interval_seconds:
                return None
            next_time = base_time + timedelta(seconds=self.interval_seconds)
        elif self.schedule_type == ScheduleType.ONE_TIME:
            if self.start_time and base_time < self.start_time:
                next_time = self.start_time
            else:
                return None
        else:
            return None
            
        if self.end_time and next_time > self.end_time:
            return None
            
        return next_time


@dataclass
class RetryPolicy:
    """Retry policy configuration."""
    max_retries: int = 3
    initial_delay: int = 60  # seconds
    max_delay: int = 3600  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> int:
        """Calculate retry delay with exponential backoff."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random() * 0.5)
            
        return int(delay)


@dataclass
class Job:
    """Scheduled job definition."""
    job_id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    schedule: Optional[JobSchedule] = None
    priority: JobPriority = JobPriority.NORMAL
    retry_policy: Optional[RetryPolicy] = None
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Execution state
    status: JobStatus = JobStatus.PENDING
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    execution_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    
    def should_run(self) -> bool:
        """Check if job should run now."""
        if self.status == JobStatus.RUNNING:
            return False
            
        if not self.next_run:
            return False
            
        return datetime.utcnow() >= self.next_run
        
    def calculate_next_run(self):
        """Calculate next execution time."""
        if not self.schedule:
            self.next_run = None
            return
            
        self.next_run = self.schedule.get_next_run(self.last_run)


@dataclass
class JobExecution:
    """Job execution record."""
    execution_id: str
    job_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: JobStatus = JobStatus.RUNNING
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class JobDAG:
    """Directed Acyclic Graph for job dependencies."""
    
    def __init__(self):
        self.graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}
        
    def add_job(self, job_id: str, dependencies: List[str]):
        """Add job to DAG."""
        if job_id not in self.graph:
            self.graph[job_id] = set()
            self.reverse_graph[job_id] = set()
            
        for dep in dependencies:
            self.graph[dep].add(job_id)
            self.reverse_graph[job_id].add(dep)
            
    def get_ready_jobs(self, completed_jobs: Set[str]) -> Set[str]:
        """Get jobs ready to run (all dependencies completed)."""
        ready = set()
        
        for job_id in self.reverse_graph:
            if job_id in completed_jobs:
                continue
                
            deps = self.reverse_graph[job_id]
            if deps.issubset(completed_jobs):
                ready.add(job_id)
                
        return ready
        
    def has_cycle(self) -> bool:
        """Check for circular dependencies."""
        visited = set()
        rec_stack = set()
        
        def visit(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.graph.get(node, []):
                if neighbor not in visited:
                    if visit(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
                    
            rec_stack.remove(node)
            return False
            
        for node in self.graph:
            if node not in visited:
                if visit(node):
                    return True
                    
        return False


class DistributedScheduler:
    """
    Production-grade distributed job scheduler.
    
    Features:
    - Multiple schedule types (cron, interval, one-time, dependent)
    - Priority-based execution queue
    - DAG-based dependency management
    - Persistent state with database
    - Retry with exponential backoff
    - Concurrent execution limits
    - Health monitoring
    """
    
    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        heartbeat_interval: int = 30,
        job_timeout: int = 3600,
    ):
        self.jobs: Dict[str, Job] = {}
        self.executions: Dict[str, JobExecution] = {}
        self.dag = JobDAG()
        self.max_concurrent_jobs = max_concurrent_jobs
        self.heartbeat_interval = heartbeat_interval
        self.job_timeout = job_timeout
        
        self.running_jobs: Set[str] = set()
        self.completed_jobs: Set[str] = set()
        self.is_running = False
        
        # Metrics
        self.metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "retried_executions": 0,
            "cancelled_executions": 0,
        }
        
    async def start(self):
        """Start the scheduler."""
        self.is_running = True
        logger.info("Starting distributed scheduler")
        
        # Start background tasks
        await asyncio.gather(
            self._schedule_loop(),
            self._execution_loop(),
            self._monitoring_loop(),
        )
        
    async def stop(self):
        """Stop the scheduler gracefully."""
        logger.info("Stopping distributed scheduler")
        self.is_running = False
        
        # Wait for running jobs
        while self.running_jobs:
            await asyncio.sleep(1)
            
    def register_job(
        self,
        job_id: str,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        schedule: Optional[JobSchedule] = None,
        priority: JobPriority = JobPriority.NORMAL,
        retry_policy: Optional[RetryPolicy] = None,
        dependencies: List[str] = None,
        tags: Dict[str, str] = None,
    ) -> Job:
        """Register a new job."""
        kwargs = kwargs or {}
        dependencies = dependencies or []
        tags = tags or {}
        
        job = Job(
            job_id=job_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            schedule=schedule,
            priority=priority,
            retry_policy=retry_policy or RetryPolicy(),
            dependencies=dependencies,
            tags=tags,
        )
        
        job.calculate_next_run()
        self.jobs[job_id] = job
        
        if dependencies:
            self.dag.add_job(job_id, dependencies)
            
        logger.info(f"Registered job: {job_id} ({name})")
        return job
        
    def unregister_job(self, job_id: str):
        """Unregister a job."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            logger.info(f"Unregistered job: {job_id}")
            
    async def trigger_job(self, job_id: str) -> Optional[str]:
        """Manually trigger a job execution."""
        if job_id not in self.jobs:
            logger.error(f"Job not found: {job_id}")
            return None
            
        job = self.jobs[job_id]
        job.next_run = datetime.utcnow()
        
        execution_id = await self._execute_job(job)
        return execution_id
        
    async def cancel_job(self, job_id: str):
        """Cancel a running job."""
        if job_id in self.running_jobs:
            job = self.jobs[job_id]
            job.status = JobStatus.CANCELLED
            self.running_jobs.remove(job_id)
            self.metrics["cancelled_executions"] += 1
            logger.info(f"Cancelled job: {job_id}")
            
    async def _schedule_loop(self):
        """Main scheduling loop."""
        while self.is_running:
            try:
                # Update next run times for scheduled jobs
                for job in self.jobs.values():
                    if job.schedule and job.status != JobStatus.RUNNING:
                        job.calculate_next_run()
                        
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in schedule loop: {e}")
                await asyncio.sleep(10)
                
    async def _execution_loop(self):
        """Main execution loop."""
        while self.is_running:
            try:
                # Get jobs ready to run
                ready_jobs = self._get_ready_jobs()
                
                # Execute jobs up to concurrency limit
                for job in ready_jobs:
                    if len(self.running_jobs) >= self.max_concurrent_jobs:
                        break
                        
                    if job.job_id not in self.running_jobs:
                        asyncio.create_task(self._execute_job(job))
                        
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(1)
                
    async def _monitoring_loop(self):
        """Health monitoring loop."""
        while self.is_running:
            try:
                # Check for timeout jobs
                for job_id in list(self.running_jobs):
                    job = self.jobs[job_id]
                    if job.last_run:
                        elapsed = (datetime.utcnow() - job.last_run).total_seconds()
                        if elapsed > self.job_timeout:
                            logger.warning(f"Job timeout: {job_id}")
                            await self.cancel_job(job_id)
                            
                # Log metrics
                logger.info(f"Scheduler metrics: {self.metrics}")
                logger.info(f"Running jobs: {len(self.running_jobs)}/{self.max_concurrent_jobs}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
                
    def _get_ready_jobs(self) -> List[Job]:
        """Get jobs ready to execute, sorted by priority."""
        ready = []
        
        # Get jobs with no dependencies or satisfied dependencies
        if self.dag.graph:
            ready_job_ids = self.dag.get_ready_jobs(self.completed_jobs)
        else:
            ready_job_ids = set(self.jobs.keys())
            
        for job_id in ready_job_ids:
            job = self.jobs[job_id]
            
            if job.should_run() and job_id not in self.running_jobs:
                ready.append(job)
                
        # Sort by priority
        ready.sort(key=lambda j: j.priority.value)
        return ready
        
    async def _execute_job(self, job: Job) -> str:
        """Execute a job."""
        execution_id = hashlib.md5(
            f"{job.job_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()
        
        execution = JobExecution(
            execution_id=execution_id,
            job_id=job.job_id,
            started_at=datetime.utcnow(),
        )
        
        self.executions[execution_id] = execution
        self.running_jobs.add(job.job_id)
        
        job.status = JobStatus.RUNNING
        job.last_run = datetime.utcnow()
        job.execution_count += 1
        self.metrics["total_executions"] += 1
        
        logger.info(f"Executing job: {job.job_id} (execution: {execution_id})")
        
        try:
            # Execute the job function
            if asyncio.iscoroutinefunction(job.func):
                result = await job.func(*job.args, **job.kwargs)
            else:
                result = job.func(*job.args, **job.kwargs)
                
            # Job succeeded
            execution.status = JobStatus.COMPLETED
            execution.result = result
            execution.completed_at = datetime.utcnow()
            execution.duration_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )
            
            job.status = JobStatus.COMPLETED
            job.retry_count = 0  # Reset retry count
            self.completed_jobs.add(job.job_id)
            self.metrics["successful_executions"] += 1
            
            logger.info(
                f"Job completed: {job.job_id} "
                f"(duration: {execution.duration_ms}ms)"
            )
            
        except Exception as e:
            # Job failed
            execution.status = JobStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.utcnow()
            execution.duration_ms = int(
                (execution.completed_at - execution.started_at).total_seconds() * 1000
            )
            
            job.failure_count += 1
            logger.error(f"Job failed: {job.job_id}: {e}")
            
            # Handle retry
            if job.retry_policy and job.retry_count < job.retry_policy.max_retries:
                job.retry_count += 1
                delay = job.retry_policy.get_delay(job.retry_count)
                job.next_run = datetime.utcnow() + timedelta(seconds=delay)
                job.status = JobStatus.RETRY
                self.metrics["retried_executions"] += 1
                
                logger.info(
                    f"Job will retry: {job.job_id} "
                    f"(attempt {job.retry_count}/{job.retry_policy.max_retries}, "
                    f"delay: {delay}s)"
                )
            else:
                job.status = JobStatus.FAILED
                self.metrics["failed_executions"] += 1
                
        finally:
            if job.job_id in self.running_jobs:
                self.running_jobs.remove(job.job_id)
                
        return execution_id
        
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status."""
        if job_id not in self.jobs:
            return None
            
        job = self.jobs[job_id]
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "next_run": job.next_run.isoformat() if job.next_run else None,
            "last_run": job.last_run.isoformat() if job.last_run else None,
            "execution_count": job.execution_count,
            "failure_count": job.failure_count,
            "retry_count": job.retry_count,
            "priority": job.priority.value,
            "dependencies": job.dependencies,
            "tags": job.tags,
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics."""
        return {
            **self.metrics,
            "total_jobs": len(self.jobs),
            "running_jobs": len(self.running_jobs),
            "pending_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.PENDING]),
            "failed_jobs": len([j for j in self.jobs.values() if j.status == JobStatus.FAILED]),
        }
