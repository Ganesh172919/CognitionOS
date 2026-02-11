"""
Persistence Infrastructure - Base Components

Provides database connection, session management, and base repository classes.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base

# SQLAlchemy declarative base
Base = declarative_base()


class DatabaseConfig:
    """Database configuration"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "cognition_os",
        user: str = "postgres",
        password: str = "postgres",
        pool_size: int = 20,
        max_overflow: int = 10,
        echo: bool = False
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo

    def get_connection_string(self) -> str:
        """Get PostgreSQL async connection string"""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class DatabaseSession:
    """
    Database session manager.

    Provides async session factory and lifecycle management.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine: AsyncEngine = create_async_engine(
            config.get_connection_string(),
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            echo=config.echo,
            pool_pre_ping=True  # Verify connections before using
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session (async context manager).

        Usage:
            async with db.get_session() as session:
                # Use session
                pass
        """
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        """Close database engine"""
        await self.engine.dispose()


# Global database session instance (to be initialized)
_db_session: DatabaseSession | None = None


def init_database(config: DatabaseConfig) -> DatabaseSession:
    """
    Initialize database session.

    Args:
        config: Database configuration

    Returns:
        DatabaseSession instance
    """
    global _db_session
    _db_session = DatabaseSession(config)
    return _db_session


def get_database() -> DatabaseSession:
    """
    Get global database session.

    Returns:
        DatabaseSession instance

    Raises:
        RuntimeError: If database not initialized
    """
    if _db_session is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_session
