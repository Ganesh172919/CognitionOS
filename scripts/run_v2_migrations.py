#!/usr/bin/env python3
"""
V2 Database Migration Runner

Applies V2 migrations to the CognitionOS database.
"""

import os
import asyncio
from pathlib import Path

# Add parent directory to path for imports

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy import text
from shared.libs.config import load_config, BaseConfig
from shared.libs.logger import setup_logger


logger = setup_logger(__name__)


async def run_migration_file(session: AsyncSession, filepath: Path):
    """
    Run a single migration file.

    Args:
        session: Database session
        filepath: Path to SQL migration file
    """
    logger.info(f"Running migration: {filepath.name}")

    try:
        # Read SQL file
        with open(filepath, 'r') as f:
            sql = f.read()

        # Execute migration
        await session.execute(text(sql))
        await session.commit()

        logger.info(f"✓ Migration {filepath.name} completed successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Migration {filepath.name} failed: {e}")
        await session.rollback()
        raise


async def run_v2_migrations():
    """Run all V2 migrations in order."""
    config = load_config(BaseConfig)

    # Create database engine
    logger.info("Connecting to database...")
    engine = create_async_engine(config.database_url, echo=True)

    # Get migration files
    migrations_dir = Path(__file__).parent.parent / 'database' / 'migrations' / 'v2'

    migration_files = sorted([
        f for f in migrations_dir.glob('*.sql')
        if f.name.startswith('00')  # Only numbered migrations
    ])

    if not migration_files:
        logger.warning("No V2 migrations found")
        return

    logger.info(f"Found {len(migration_files)} V2 migrations")

    async with engine.begin() as conn:
        session = AsyncSession(bind=conn, expire_on_commit=False)

        for migration_file in migration_files:
            await run_migration_file(session, migration_file)

    logger.info("✓ All V2 migrations completed successfully!")

    await engine.dispose()


async def check_migration_status():
    """Check which migrations have been applied."""
    config = load_config(BaseConfig)
    engine = create_async_engine(config.database_url)

    async with engine.begin() as conn:
        session = AsyncSession(bind=conn)

        # Check if V2 tables exist
        tables_to_check = [
            'workflows',
            'workflow_executions',
            'workflow_execution_steps',
            'agent_metrics',
            'agent_performance_history',
            'memory_namespaces',
            'memory_lifecycle_policies',
            'quality_gate_policies',
            'quality_gate_results'
        ]

        logger.info("\nV2 Migration Status:")
        logger.info("=" * 60)

        for table in tables_to_check:
            result = await session.execute(text(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = '{table}'
                );
            """))
            exists = result.scalar()
            status = "✓ Exists" if exists else "✗ Missing"
            logger.info(f"{table:40s} {status}")

    await engine.dispose()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='V2 Database Migration Runner')
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check migration status without applying'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply V2 migrations'
    )

    args = parser.parse_args()

    if args.check:
        asyncio.run(check_migration_status())
    elif args.apply:
        asyncio.run(run_v2_migrations())
    else:
        print("Usage:")
        print("  python run_v2_migrations.py --check   # Check migration status")
        print("  python run_v2_migrations.py --apply   # Apply migrations")
