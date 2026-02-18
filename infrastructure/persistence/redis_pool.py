"""
Redis Connection Pool Manager

Provides singleton Redis connection pool for efficient resource management.
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

from core.config import get_config

logger = logging.getLogger(__name__)


class RedisPoolManager:
    """
    Singleton Redis connection pool manager.
    
    Provides efficient connection pooling with:
    - Single instance across application
    - Connection health checking
    - Automatic reconnection
    - Resource cleanup
    """
    
    _instance: Optional['RedisPoolManager'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __init__(self):
        """Initialize pool manager (use get_instance instead)."""
        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._initialized = False
    
    @classmethod
    async def get_instance(cls) -> 'RedisPoolManager':
        """
        Get or create the singleton instance.
        
        Returns:
            RedisPoolManager instance
        """
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = RedisPoolManager()
                    await cls._instance._initialize()
        
        return cls._instance
    
    async def _initialize(self) -> None:
        """Initialize Redis connection pool."""
        if self._initialized:
            return
        
        try:
            config = get_config()
            
            # Create connection pool
            self._pool = ConnectionPool.from_url(
                config.redis.url,
                max_connections=config.redis.max_connections,
                decode_responses=True,
                encoding="utf-8",
                socket_connect_timeout=5,
                socket_keepalive=True,
                health_check_interval=30,
            )
            
            # Create Redis client
            self._client = aioredis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            
            self._initialized = True
            logger.info(
                f"Redis connection pool initialized: "
                f"max_connections={config.redis.max_connections}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis pool: {e}", exc_info=True)
            raise
    
    async def get_client(self) -> aioredis.Redis:
        """
        Get Redis client from pool.
        
        Returns:
            Redis client instance
            
        Raises:
            RuntimeError: If pool not initialized
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("Redis pool not initialized")
        
        return self._client
    
    @asynccontextmanager
    async def get_connection(self):
        """
        Get Redis connection from pool with context manager.
        
        Usage:
            async with redis_pool.get_connection() as conn:
                await conn.set('key', 'value')
        
        Yields:
            Redis connection
        """
        client = await self.get_client()
        try:
            yield client
        finally:
            # Connection automatically returned to pool
            pass
    
    async def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized or self._client is None:
                return False
            
            response = await self._client.ping()
            return response is True
            
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.close()
            logger.info("Redis connection closed")
        
        if self._pool:
            await self._pool.disconnect()
            logger.info("Redis connection pool closed")
        
        self._initialized = False
        self._client = None
        self._pool = None
    
    @classmethod
    async def reset(cls) -> None:
        """Reset singleton instance (mainly for testing)."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None


# Convenience function for getting Redis client
async def get_redis_client() -> aioredis.Redis:
    """
    Get Redis client from singleton pool.
    
    Returns:
        Redis client instance
        
    Example:
        redis = await get_redis_client()
        await redis.set('key', 'value')
    """
    pool_manager = await RedisPoolManager.get_instance()
    return await pool_manager.get_client()


async def check_redis_health() -> bool:
    """
    Check Redis connection health.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        pool_manager = await RedisPoolManager.get_instance()
        return await pool_manager.health_check()
    except Exception:
        return False
