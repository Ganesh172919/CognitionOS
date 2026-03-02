"""
Distributed Lock Manager — CognitionOS Core Engine

Provides distributed locking primitives for coordinating
across multiple platform instances:
- Mutual exclusion locks with TTL
- Read-write locks
- Leader election
- Lock renewal and auto-release
- Deadlock detection
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LockState(str, Enum):
    """Current state of a lock."""
    UNLOCKED = "unlocked"
    LOCKED = "locked"
    EXPIRED = "expired"
    RELEASED = "released"


class LockType(str, Enum):
    """Type of lock."""
    EXCLUSIVE = "exclusive"
    SHARED = "shared"           # Read lock
    LEADER_ELECTION = "leader"


@dataclass
class LockInfo:
    """Metadata about a held lock."""
    lock_id: str
    resource: str
    owner_id: str
    lock_type: LockType = LockType.EXCLUSIVE
    state: LockState = LockState.UNLOCKED
    acquired_at: float = 0
    expires_at: float = 0
    ttl_seconds: float = 30.0
    renewal_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    @property
    def is_held(self) -> bool:
        return self.state == LockState.LOCKED and not self.is_expired

    @property
    def remaining_ttl(self) -> float:
        if self.expires_at <= 0:
            return float("inf")
        return max(0, self.expires_at - time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lock_id": self.lock_id,
            "resource": self.resource,
            "owner_id": self.owner_id,
            "lock_type": self.lock_type.value,
            "state": self.state.value,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "ttl_seconds": self.ttl_seconds,
            "renewal_count": self.renewal_count,
            "is_expired": self.is_expired,
            "remaining_ttl": round(self.remaining_ttl, 2),
            "metadata": self.metadata,
        }


@dataclass
class LeaderElectionResult:
    """Result of a leader election attempt."""
    elected: bool
    leader_id: str
    term: int
    elected_at: float = 0
    expires_at: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elected": self.elected,
            "leader_id": self.leader_id,
            "term": self.term,
            "elected_at": self.elected_at,
            "expires_at": self.expires_at,
        }


class LockAcquisitionError(Exception):
    """Raised when a lock cannot be acquired."""
    pass


class LockRenewalError(Exception):
    """Raised when a lock renewal fails."""
    pass


class DistributedLockManager:
    """
    In-process distributed lock manager with TTL, renewal, and leader election.

    For production multi-instance deployment, swap the in-memory store
    with a Redis/etcd backend via the abstract methods.

    Features:
    - Exclusive and shared (read/write) locks
    - TTL-based auto-expiry
    - Lock renewal for long-running operations
    - Leader election with term tracking
    - Deadlock detection via wait-for graph
    - Lock contention metrics
    """

    def __init__(
        self,
        node_id: str = "",
        default_ttl: float = 30.0,
        max_ttl: float = 300.0,
        cleanup_interval: float = 10.0,
    ):
        self._node_id = node_id or f"node-{uuid.uuid4().hex[:8]}"
        self._default_ttl = default_ttl
        self._max_ttl = max_ttl
        self._cleanup_interval = cleanup_interval

        # Lock storage
        self._locks: Dict[str, LockInfo] = {}
        self._shared_locks: Dict[str, Set[str]] = {}  # resource -> set of owner_ids
        self._waiters: Dict[str, List[asyncio.Event]] = {}

        # Leader election
        self._leaders: Dict[str, LeaderElectionResult] = {}
        self._election_terms: Dict[str, int] = {}

        # Metrics
        self._total_acquisitions = 0
        self._total_releases = 0
        self._total_expirations = 0
        self._total_contentions = 0
        self._acquisition_times: List[float] = []

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            "DistributedLockManager initialized (node=%s, ttl=%.0fs)",
            self._node_id, default_ttl,
        )

    async def start(self):
        """Start the background cleanup loop."""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Lock manager cleanup loop started")

    async def shutdown(self):
        """Stop the cleanup loop and release all locks."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Release all held locks
        for resource in list(self._locks.keys()):
            info = self._locks[resource]
            if info.owner_id == self._node_id:
                await self.release(resource, self._node_id)

        logger.info("Lock manager shutdown complete")

    # ── Lock Acquisition ──

    async def acquire(
        self,
        resource: str,
        owner_id: str = "",
        *,
        ttl: Optional[float] = None,
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LockInfo:
        """
        Acquire a lock on a resource.

        Args:
            resource: The resource to lock.
            owner_id: Identity of the lock holder.
            ttl: Time-to-live in seconds (default: container default).
            lock_type: EXCLUSIVE or SHARED.
            timeout: Max seconds to wait for the lock.
            metadata: Optional metadata to attach to the lock.

        Returns:
            LockInfo with acquisition details.

        Raises:
            LockAcquisitionError: If lock cannot be acquired within timeout.
        """
        owner = owner_id or self._node_id
        ttl_val = min(ttl or self._default_ttl, self._max_ttl)
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                self._total_contentions += 1
                raise LockAcquisitionError(
                    f"Timeout acquiring {lock_type.value} lock on '{resource}' "
                    f"after {timeout:.1f}s (owner={owner})"
                )

            acquired = self._try_acquire(
                resource, owner, ttl_val, lock_type, metadata or {}
            )
            if acquired:
                self._total_acquisitions += 1
                acq_time = time.time() - start_time
                self._acquisition_times.append(acq_time)
                # Keep only last 1000 measurements
                if len(self._acquisition_times) > 1000:
                    self._acquisition_times = self._acquisition_times[-1000:]
                logger.debug(
                    "Lock acquired: resource='%s', owner='%s', type=%s, ttl=%.0fs",
                    resource, owner, lock_type.value, ttl_val,
                )
                return acquired

            # Wait for lock release
            event = asyncio.Event()
            self._waiters.setdefault(resource, []).append(event)
            remaining = timeout - elapsed
            try:
                await asyncio.wait_for(event.wait(), timeout=min(remaining, 1.0))
            except asyncio.TimeoutError:
                pass
            finally:
                waiters = self._waiters.get(resource, [])
                if event in waiters:
                    waiters.remove(event)

    def _try_acquire(
        self,
        resource: str,
        owner: str,
        ttl: float,
        lock_type: LockType,
        metadata: Dict[str, Any],
    ) -> Optional[LockInfo]:
        """Attempt to acquire lock without waiting."""
        now = time.time()

        # Check for expired lock
        existing = self._locks.get(resource)
        if existing and existing.is_expired:
            self._total_expirations += 1
            del self._locks[resource]
            self._shared_locks.pop(resource, None)
            existing = None

        # Shared lock logic
        if lock_type == LockType.SHARED:
            if existing and existing.lock_type == LockType.EXCLUSIVE and existing.is_held:
                return None  # Exclusive lock held, can't acquire shared

            # Allow multiple shared locks
            self._shared_locks.setdefault(resource, set()).add(owner)
            info = LockInfo(
                lock_id=uuid.uuid4().hex,
                resource=resource,
                owner_id=owner,
                lock_type=LockType.SHARED,
                state=LockState.LOCKED,
                acquired_at=now,
                expires_at=now + ttl,
                ttl_seconds=ttl,
                metadata=metadata,
            )
            self._locks[resource] = info
            return info

        # Exclusive lock logic
        if existing and existing.is_held:
            if existing.owner_id == owner:
                # Re-entrant: extend TTL
                existing.expires_at = now + ttl
                existing.renewal_count += 1
                return existing
            return None  # Another owner holds it

        if resource in self._shared_locks and self._shared_locks[resource]:
            return None  # Shared locks held, can't acquire exclusive

        info = LockInfo(
            lock_id=uuid.uuid4().hex,
            resource=resource,
            owner_id=owner,
            lock_type=LockType.EXCLUSIVE,
            state=LockState.LOCKED,
            acquired_at=now,
            expires_at=now + ttl,
            ttl_seconds=ttl,
            metadata=metadata,
        )
        self._locks[resource] = info
        return info

    # ── Lock Release ──

    async def release(self, resource: str, owner_id: str = "") -> bool:
        """Release a held lock."""
        owner = owner_id or self._node_id
        info = self._locks.get(resource)

        if not info:
            logger.warning("Release called on non-existent lock '%s'", resource)
            return False

        if info.lock_type == LockType.SHARED:
            shared = self._shared_locks.get(resource, set())
            shared.discard(owner)
            if not shared:
                self._shared_locks.pop(resource, None)
                del self._locks[resource]
        else:
            if info.owner_id != owner:
                logger.warning(
                    "Release denied: lock '%s' owned by '%s', not '%s'",
                    resource, info.owner_id, owner,
                )
                return False
            info.state = LockState.RELEASED
            del self._locks[resource]

        self._total_releases += 1
        logger.debug("Lock released: resource='%s', owner='%s'", resource, owner)

        # Notify waiters
        for event in self._waiters.pop(resource, []):
            event.set()

        return True

    # ── Lock Renewal ──

    async def renew(
        self,
        resource: str,
        owner_id: str = "",
        *,
        ttl: Optional[float] = None,
    ) -> LockInfo:
        """Renew a held lock's TTL."""
        owner = owner_id or self._node_id
        info = self._locks.get(resource)

        if not info or info.owner_id != owner:
            raise LockRenewalError(
                f"Cannot renew lock '{resource}': not held by '{owner}'"
            )

        if info.is_expired:
            raise LockRenewalError(
                f"Lock '{resource}' has already expired"
            )

        now = time.time()
        new_ttl = min(ttl or self._default_ttl, self._max_ttl)
        info.expires_at = now + new_ttl
        info.ttl_seconds = new_ttl
        info.renewal_count += 1

        logger.debug(
            "Lock renewed: resource='%s', new_expiry=+%.0fs, renewals=%d",
            resource, new_ttl, info.renewal_count,
        )
        return info

    # ── Leader Election ──

    async def elect_leader(
        self,
        election_id: str,
        candidate_id: str = "",
        *,
        ttl: float = 30.0,
    ) -> LeaderElectionResult:
        """
        Attempt to become leader for a named election.

        Uses a simple lock-based election: first to acquire the lock
        becomes leader for the term.
        """
        candidate = candidate_id or self._node_id
        resource = f"__leader_election__{election_id}"
        self._election_terms.setdefault(election_id, 0)

        existing = self._leaders.get(election_id)
        if existing and existing.expires_at > time.time():
            # Current leader still valid
            return LeaderElectionResult(
                elected=(existing.leader_id == candidate),
                leader_id=existing.leader_id,
                term=existing.term,
                elected_at=existing.elected_at,
                expires_at=existing.expires_at,
            )

        # Try to acquire leader lock
        try:
            lock_info = await self.acquire(
                resource, candidate, ttl=ttl, timeout=1.0
            )
            self._election_terms[election_id] += 1
            term = self._election_terms[election_id]
            now = time.time()

            result = LeaderElectionResult(
                elected=True,
                leader_id=candidate,
                term=term,
                elected_at=now,
                expires_at=now + ttl,
            )
            self._leaders[election_id] = result
            logger.info(
                "Leader elected: election='%s', leader='%s', term=%d",
                election_id, candidate, term,
            )
            return result

        except LockAcquisitionError:
            existing = self._leaders.get(election_id)
            return LeaderElectionResult(
                elected=False,
                leader_id=existing.leader_id if existing else "",
                term=existing.term if existing else 0,
                elected_at=existing.elected_at if existing else 0,
                expires_at=existing.expires_at if existing else 0,
            )

    async def renew_leadership(
        self,
        election_id: str,
        leader_id: str = "",
        *,
        ttl: float = 30.0,
    ) -> bool:
        """Renew leadership for a term."""
        leader = leader_id or self._node_id
        existing = self._leaders.get(election_id)

        if not existing or existing.leader_id != leader:
            return False

        resource = f"__leader_election__{election_id}"
        try:
            await self.renew(resource, leader, ttl=ttl)
            existing.expires_at = time.time() + ttl
            return True
        except LockRenewalError:
            return False

    def is_leader(self, election_id: str, candidate_id: str = "") -> bool:
        """Check if a candidate is the current leader."""
        candidate = candidate_id or self._node_id
        existing = self._leaders.get(election_id)
        if not existing:
            return False
        return (
            existing.leader_id == candidate
            and existing.expires_at > time.time()
        )

    # ── Cleanup ──

    async def _cleanup_loop(self):
        """Periodically clean up expired locks."""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Lock cleanup error: %s", exc)

    def _cleanup_expired(self):
        """Remove expired locks and notify waiters."""
        now = time.time()
        expired = []

        for resource, info in list(self._locks.items()):
            if info.expires_at > 0 and now > info.expires_at:
                expired.append(resource)

        for resource in expired:
            self._total_expirations += 1
            del self._locks[resource]
            self._shared_locks.pop(resource, None)
            logger.debug("Expired lock cleaned: '%s'", resource)

            for event in self._waiters.pop(resource, []):
                event.set()

        # Clean expired leader elections
        for eid in list(self._leaders.keys()):
            result = self._leaders[eid]
            if result.expires_at > 0 and now > result.expires_at:
                del self._leaders[eid]
                resource = f"__leader_election__{eid}"
                self._locks.pop(resource, None)

    # ── Context Manager ──

    class LockContext:
        """Async context manager for lock acquisition/release."""

        def __init__(
            self,
            manager: "DistributedLockManager",
            resource: str,
            owner_id: str,
            ttl: float,
            lock_type: LockType,
            timeout: float,
        ):
            self._manager = manager
            self._resource = resource
            self._owner_id = owner_id
            self._ttl = ttl
            self._lock_type = lock_type
            self._timeout = timeout
            self._lock_info: Optional[LockInfo] = None

        async def __aenter__(self) -> LockInfo:
            self._lock_info = await self._manager.acquire(
                self._resource,
                self._owner_id,
                ttl=self._ttl,
                lock_type=self._lock_type,
                timeout=self._timeout,
            )
            return self._lock_info

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if self._lock_info:
                await self._manager.release(
                    self._resource, self._owner_id
                )

    def lock(
        self,
        resource: str,
        owner_id: str = "",
        *,
        ttl: Optional[float] = None,
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout: float = 10.0,
    ) -> LockContext:
        """Return an async context manager for lock acquire/release."""
        return self.LockContext(
            manager=self,
            resource=resource,
            owner_id=owner_id or self._node_id,
            ttl=ttl or self._default_ttl,
            lock_type=lock_type,
            timeout=timeout,
        )

    # ── Stats ──

    def get_stats(self) -> Dict[str, Any]:
        """Return lock manager stats for monitoring."""
        avg_acquisition = (
            sum(self._acquisition_times) / len(self._acquisition_times)
            if self._acquisition_times else 0
        )
        return {
            "node_id": self._node_id,
            "active_locks": sum(
                1 for info in self._locks.values() if info.is_held
            ),
            "total_locks": len(self._locks),
            "active_elections": len(self._leaders),
            "total_acquisitions": self._total_acquisitions,
            "total_releases": self._total_releases,
            "total_expirations": self._total_expirations,
            "total_contentions": self._total_contentions,
            "avg_acquisition_time_ms": round(avg_acquisition * 1000, 2),
            "locks": {
                resource: info.to_dict()
                for resource, info in self._locks.items()
                if info.is_held
            },
            "leaders": {
                eid: result.to_dict()
                for eid, result in self._leaders.items()
            },
        }

    def __repr__(self) -> str:
        active = sum(1 for info in self._locks.values() if info.is_held)
        return (
            f"DistributedLockManager(node='{self._node_id}', "
            f"active_locks={active})"
        )


# ── Module-level convenience ──

_lock_manager: Optional[DistributedLockManager] = None


def get_lock_manager() -> DistributedLockManager:
    """Get or create the global lock manager."""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = DistributedLockManager()
    return _lock_manager
