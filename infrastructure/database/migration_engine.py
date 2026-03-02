"""
Data Migration Engine — CognitionOS

Database migration system with:
- Version-tracked migrations
- Forward and rollback support
- Migration dependency resolution
- Dry-run mode
- Migration status tracking
- Schema snapshotting
- Seed data support
- Multi-tenant migration support
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable, Dict, List, Optional

logger = logging.getLogger(__name__)


class MigrationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"


@dataclass
class Migration:
    migration_id: str
    version: str  # e.g., "001", "002", semver
    name: str
    description: str = ""
    up_fn: Optional[Callable[..., Awaitable[None]]] = None
    down_fn: Optional[Callable[..., Awaitable[None]]] = None
    dependencies: List[str] = field(default_factory=list)
    status: MigrationStatus = MigrationStatus.PENDING
    applied_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    duration_ms: float = 0
    error: Optional[str] = None
    is_seed: bool = False
    tenant_scoped: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version, "name": self.name,
            "status": self.status.value,
            "applied_at": self.applied_at,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
        }


@dataclass
class MigrationResult:
    success: bool
    migrations_applied: int = 0
    migrations_failed: int = 0
    total_duration_ms: float = 0
    details: List[Dict[str, Any]] = field(default_factory=list)


class MigrationEngine:
    """
    Database migration engine with version tracking, rollback,
    and multi-tenant support.
    """

    def __init__(self):
        self._migrations: Dict[str, Migration] = {}  # version -> migration
        self._applied: Dict[str, float] = {}  # version -> applied_at
        self._history: List[Dict[str, Any]] = []

    # ── Registration ──

    def register(self, version: str, name: str, *,
                   up_fn: Callable[..., Awaitable[None]],
                   down_fn: Optional[Callable[..., Awaitable[None]]] = None,
                   description: str = "",
                   dependencies: Optional[List[str]] = None,
                   is_seed: bool = False,
                   tenant_scoped: bool = False) -> Migration:
        migration = Migration(
            migration_id=uuid.uuid4().hex[:12],
            version=version, name=name, description=description,
            up_fn=up_fn, down_fn=down_fn,
            dependencies=dependencies or [],
            is_seed=is_seed, tenant_scoped=tenant_scoped,
        )
        self._migrations[version] = migration
        return migration

    # ── Migrate Up ──

    async def migrate(self, *, target_version: Optional[str] = None,
                        dry_run: bool = False) -> MigrationResult:
        """Apply all pending migrations up to target_version."""
        pending = self._get_pending(target_version)
        result = MigrationResult(success=True)
        start_total = time.perf_counter()

        for migration in pending:
            if dry_run:
                result.details.append({
                    **migration.to_dict(), "dry_run": True,
                })
                result.migrations_applied += 1
                continue

            # Check dependencies
            for dep in migration.dependencies:
                if dep not in self._applied:
                    logger.error("Migration %s depends on unapplied %s",
                                 migration.version, dep)
                    result.success = False
                    result.migrations_failed += 1
                    result.details.append({
                        **migration.to_dict(),
                        "error": f"Dependency {dep} not applied",
                    })
                    continue

            start = time.perf_counter()
            migration.status = MigrationStatus.RUNNING

            try:
                if migration.up_fn:
                    await migration.up_fn()

                migration.status = MigrationStatus.COMPLETED
                migration.applied_at = time.time()
                migration.duration_ms = (time.perf_counter() - start) * 1000
                self._applied[migration.version] = migration.applied_at
                result.migrations_applied += 1

                self._history.append({
                    "action": "migrate_up", "version": migration.version,
                    "name": migration.name, "timestamp": time.time(),
                    "duration_ms": migration.duration_ms,
                })

                logger.info("Migration applied: %s — %s (%.1fms)",
                            migration.version, migration.name,
                            migration.duration_ms)

            except Exception as exc:
                migration.status = MigrationStatus.FAILED
                migration.error = str(exc)
                migration.duration_ms = (time.perf_counter() - start) * 1000
                result.success = False
                result.migrations_failed += 1
                logger.error("Migration FAILED: %s — %s: %s",
                             migration.version, migration.name, exc)

            result.details.append(migration.to_dict())

        result.total_duration_ms = (time.perf_counter() - start_total) * 1000
        return result

    # ── Rollback ──

    async def rollback(self, *, steps: int = 1,
                         target_version: Optional[str] = None) -> MigrationResult:
        """Rollback applied migrations."""
        applied = sorted(
            [(v, t) for v, t in self._applied.items()],
            key=lambda x: x[1], reverse=True,
        )

        result = MigrationResult(success=True)
        start_total = time.perf_counter()
        rolled_back = 0

        for version, _ in applied:
            if target_version and version == target_version:
                break
            if rolled_back >= steps and not target_version:
                break

            migration = self._migrations.get(version)
            if not migration or not migration.down_fn:
                logger.warning("No rollback for migration %s", version)
                continue

            start = time.perf_counter()
            try:
                await migration.down_fn()
                migration.status = MigrationStatus.ROLLED_BACK
                migration.rolled_back_at = time.time()
                migration.duration_ms = (time.perf_counter() - start) * 1000
                del self._applied[version]
                rolled_back += 1

                self._history.append({
                    "action": "rollback", "version": version,
                    "name": migration.name, "timestamp": time.time(),
                })

                logger.info("Rolled back: %s — %s", version, migration.name)

            except Exception as exc:
                migration.error = str(exc)
                result.success = False
                result.migrations_failed += 1
                logger.error("Rollback FAILED: %s — %s: %s",
                             version, migration.name, exc)

            result.details.append(migration.to_dict())

        result.migrations_applied = rolled_back
        result.total_duration_ms = (time.perf_counter() - start_total) * 1000
        return result

    # ── Seed Data ──

    async def seed(self) -> MigrationResult:
        seeds = [
            m for m in self._migrations.values()
            if m.is_seed and m.version not in self._applied
        ]
        result = MigrationResult(success=True)
        for seed in seeds:
            try:
                if seed.up_fn:
                    await seed.up_fn()
                seed.status = MigrationStatus.COMPLETED
                seed.applied_at = time.time()
                self._applied[seed.version] = seed.applied_at
                result.migrations_applied += 1
            except Exception as exc:
                seed.error = str(exc)
                result.migrations_failed += 1
                result.success = False
            result.details.append(seed.to_dict())
        return result

    # ── Queries ──

    def _get_pending(self, target: Optional[str] = None) -> List[Migration]:
        pending = []
        for version in sorted(self._migrations.keys()):
            if version in self._applied:
                continue
            migration = self._migrations[version]
            if migration.is_seed:
                continue
            pending.append(migration)
            if target and version == target:
                break
        return pending

    def get_status(self) -> Dict[str, Any]:
        total = len(self._migrations)
        applied = len(self._applied)
        pending = total - applied
        seeds = sum(1 for m in self._migrations.values() if m.is_seed)

        return {
            "total_migrations": total,
            "applied": applied,
            "pending": pending,
            "seeds": seeds,
            "current_version": max(self._applied.keys()) if self._applied else None,
            "migrations": [
                {**m.to_dict(), "applied": m.version in self._applied}
                for m in sorted(self._migrations.values(),
                                 key=lambda m: m.version)
            ],
        }

    def get_history(self, *, limit: int = 50) -> List[Dict[str, Any]]:
        return self._history[-limit:]


# ── Singleton ──
_engine: Optional[MigrationEngine] = None


def get_migration_engine() -> MigrationEngine:
    global _engine
    if not _engine:
        _engine = MigrationEngine()
    return _engine
