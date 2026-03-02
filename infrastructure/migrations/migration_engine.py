"""
Data Migration Engine — CognitionOS

Schema migration and data transformation:
- Version-tracked migrations
- Rollback support
- Data transformation pipelines
- Migration dependency resolution
- Dry-run mode
- Migration history
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import uuid4

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
    version: str
    name: str
    description: str = ""
    up_handler: Optional[Callable[..., Awaitable[Any]]] = None
    down_handler: Optional[Callable[..., Awaitable[Any]]] = None
    dependencies: List[str] = field(default_factory=list)
    is_destructive: bool = False
    estimated_duration_seconds: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id, "version": self.version,
            "name": self.name, "description": self.description,
            "is_destructive": self.is_destructive,
            "dependencies": self.dependencies}


@dataclass
class MigrationResult:
    migration_id: str
    version: str
    status: MigrationStatus
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0
    error: Optional[str] = None
    rows_affected: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id, "version": self.version,
            "status": self.status.value, "started_at": self.started_at,
            "completed_at": self.completed_at, "duration_ms": round(self.duration_ms, 2),
            "error": self.error, "rows_affected": self.rows_affected}


class DataMigrationEngine:
    """Version-tracked schema and data migrations with rollback."""

    def __init__(self) -> None:
        self._migrations: Dict[str, Migration] = {}  # version -> migration
        self._history: List[MigrationResult] = []
        self._applied: set[str] = set()

    def register(self, migration: Migration) -> None:
        self._migrations[migration.version] = migration

    def register_handler(self, version: str, name: str, *,
                          up: Callable[..., Awaitable[Any]],
                          down: Callable[..., Awaitable[Any]] | None = None,
                          dependencies: List[str] | None = None,
                          description: str = "") -> Migration:
        m = Migration(
            migration_id=str(uuid4()), version=version, name=name,
            description=description, up_handler=up, down_handler=down,
            dependencies=dependencies or [])
        self.register(m)
        return m

    def get_pending(self) -> List[Migration]:
        return [m for v, m in sorted(self._migrations.items())
                if v not in self._applied]

    def get_applied(self) -> List[str]:
        return sorted(self._applied)

    async def migrate_up(self, *, target_version: str | None = None,
                          dry_run: bool = False) -> List[MigrationResult]:
        pending = self.get_pending()
        if target_version:
            pending = [m for m in pending if m.version <= target_version]

        results = []
        for migration in pending:
            # Check dependencies
            for dep in migration.dependencies:
                if dep not in self._applied:
                    result = MigrationResult(
                        migration_id=migration.migration_id,
                        version=migration.version,
                        status=MigrationStatus.SKIPPED,
                        error=f"Dependency not met: {dep}")
                    results.append(result)
                    continue

            if dry_run:
                result = MigrationResult(
                    migration_id=migration.migration_id,
                    version=migration.version,
                    status=MigrationStatus.PENDING)
                results.append(result)
                continue

            result = await self._run_migration(migration, "up")
            results.append(result)
            if result.status == MigrationStatus.COMPLETED:
                self._applied.add(migration.version)
            else:
                break  # Stop on failure

        return results

    async def migrate_down(self, target_version: str) -> List[MigrationResult]:
        applied = sorted(self._applied, reverse=True)
        results = []

        for version in applied:
            if version <= target_version:
                break
            migration = self._migrations.get(version)
            if not migration or not migration.down_handler:
                continue

            result = await self._run_migration(migration, "down")
            results.append(result)
            if result.status == MigrationStatus.COMPLETED:
                self._applied.discard(version)
                result.status = MigrationStatus.ROLLED_BACK
            else:
                break

        return results

    async def _run_migration(self, migration: Migration, direction: str) -> MigrationResult:
        import time
        handler = migration.up_handler if direction == "up" else migration.down_handler
        result = MigrationResult(
            migration_id=migration.migration_id,
            version=migration.version,
            status=MigrationStatus.RUNNING,
            started_at=datetime.now(timezone.utc).isoformat())

        start = time.monotonic()
        try:
            if handler:
                ret = await handler()
                if isinstance(ret, int):
                    result.rows_affected = ret
            result.status = MigrationStatus.COMPLETED
            logger.info("Migration %s %s completed: %s", direction, migration.version, migration.name)
        except Exception as e:
            result.status = MigrationStatus.FAILED
            result.error = str(e)
            logger.error("Migration %s %s failed: %s — %s",
                          direction, migration.version, migration.name, e)

        result.duration_ms = (time.monotonic() - start) * 1000
        result.completed_at = datetime.now(timezone.utc).isoformat()
        self._history.append(result)
        return result

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._history[-limit:]]

    def get_status(self) -> Dict[str, Any]:
        return {
            "total_migrations": len(self._migrations),
            "applied": len(self._applied),
            "pending": len(self.get_pending()),
            "history_count": len(self._history)}


_engine: DataMigrationEngine | None = None

def get_migration_engine() -> DataMigrationEngine:
    global _engine
    if not _engine:
        _engine = DataMigrationEngine()
    return _engine
