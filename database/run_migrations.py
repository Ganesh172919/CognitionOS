#!/usr/bin/env python3
"""
Database migration runner for CognitionOS

Usage:
    python run_migrations.py up    # Apply all pending migrations
    python run_migrations.py down  # Rollback last migration
    python run_migrations.py init  # Initialize database (create tables)
"""

import sys
import os
from pathlib import Path
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://cognition:cognition@localhost:5432/cognitionos"
)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_engine():
    """Create database engine"""
    return create_engine(DATABASE_URL, echo=True)


def create_migrations_table(engine):
    """Create migrations tracking table if it doesn't exist"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        conn.commit()
        logger.info("Migrations table ready")


def get_applied_migrations(engine):
    """Get list of applied migrations"""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT version FROM schema_migrations ORDER BY version"
        ))
        return {row[0] for row in result}


def get_pending_migrations(applied):
    """Get list of pending migration files"""
    migrations = []
    for file in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = file.stem
        if version not in applied:
            migrations.append((version, file))
    return migrations


def apply_migration(engine, version, filepath):
    """Apply a single migration"""
    logger.info(f"Applying migration: {version}")

    with open(filepath) as f:
        sql = f.read()

    with engine.connect() as conn:
        # Execute migration
        conn.execute(text(sql))
        # Record migration
        conn.execute(
            text("INSERT INTO schema_migrations (version) VALUES (:version)"),
            {"version": version}
        )
        conn.commit()

    logger.info(f"Migration {version} applied successfully")


def rollback_migration(engine):
    """Rollback the last applied migration"""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        ))
        row = result.fetchone()

        if not row:
            logger.warning("No migrations to rollback")
            return

        version = row[0]
        logger.warning(f"Rolling back migration: {version}")

        # Note: This requires a down migration file
        down_file = MIGRATIONS_DIR / f"{version}_down.sql"
        if not down_file.exists():
            logger.error(f"Rollback file not found: {down_file}")
            logger.warning("Manual rollback required")
            return

        with open(down_file) as f:
            sql = f.read()

        conn.execute(text(sql))
        conn.execute(
            text("DELETE FROM schema_migrations WHERE version = :version"),
            {"version": version}
        )
        conn.commit()
        logger.info(f"Migration {version} rolled back")


def migrate_up():
    """Apply all pending migrations"""
    engine = get_engine()
    create_migrations_table(engine)

    applied = get_applied_migrations(engine)
    pending = get_pending_migrations(applied)

    if not pending:
        logger.info("No pending migrations")
        return

    logger.info(f"Found {len(pending)} pending migrations")
    for version, filepath in pending:
        apply_migration(engine, version, filepath)

    logger.info("All migrations applied successfully")


def migrate_down():
    """Rollback the last migration"""
    engine = get_engine()
    rollback_migration(engine)


def init_db():
    """Initialize database with all migrations"""
    logger.info("Initializing database...")
    migrate_up()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    try:
        if command == "up":
            migrate_up()
        elif command == "down":
            migrate_down()
        elif command == "init":
            init_db()
        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    main()
