"""
System Health Checks

Comprehensive health checks for all system dependencies.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any

import aio_pika
import redis.asyncio as redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service: str
    status: HealthStatus
    latency_ms: float
    details: Dict[str, Any]
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class RedisHealthCheck:
    """Health check for Redis"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
    
    async def check(self) -> HealthCheckResult:
        """
        Perform Redis health check.
        
        Returns:
            HealthCheckResult with Redis status
        """
        start_time = time.time()
        
        try:
            # Connect to Redis
            client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test PING command
            ping_start = time.time()
            await client.ping()
            ping_latency = (time.time() - ping_start) * 1000
            
            # Get server info
            info = await client.info()
            memory_used_mb = int(info.get('used_memory', 0)) / (1024 * 1024)
            connected_clients = info.get('connected_clients', 0)
            uptime_seconds = info.get('uptime_in_seconds', 0)
            
            # Test SET/GET
            test_key = f"health_check_{int(time.time())}"
            await client.set(test_key, "test", ex=10)
            value = await client.get(test_key)
            await client.delete(test_key)
            
            # Close connection
            await client.close()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine status based on latency and memory
            if latency_ms > 1000 or memory_used_mb > 1000:  # > 1s or > 1GB
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                service="redis",
                status=status,
                latency_ms=round(latency_ms, 2),
                details={
                    "ping_latency_ms": round(ping_latency, 2),
                    "memory_used_mb": round(memory_used_mb, 2),
                    "connected_clients": connected_clients,
                    "uptime_seconds": uptime_seconds,
                    "read_write_test": "passed" if value == "test" else "failed",
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="redis",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                details={},
                error=str(e)
            )


class RabbitMQHealthCheck:
    """Health check for RabbitMQ"""
    
    def __init__(self, rabbitmq_url: str):
        self.rabbitmq_url = rabbitmq_url
    
    async def check(self) -> HealthCheckResult:
        """
        Perform RabbitMQ health check.
        
        Returns:
            HealthCheckResult with RabbitMQ status
        """
        start_time = time.time()
        
        try:
            # Connect to RabbitMQ
            connection = await aio_pika.connect_robust(self.rabbitmq_url)
            
            # Create channel
            channel = await connection.channel()
            
            # Declare a test queue
            test_queue_name = f"health_check_{int(time.time())}"
            queue = await channel.declare_queue(test_queue_name, auto_delete=True)
            
            # Publish and consume a test message
            test_message = f"health_check_{int(time.time())}"
            await channel.default_exchange.publish(
                aio_pika.Message(body=test_message.encode()),
                routing_key=test_queue_name,
            )
            
            # Consume message
            message = await queue.get(timeout=5.0)
            if message:
                await message.ack()
                message_test_passed = message.body.decode() == test_message
            else:
                message_test_passed = False
            
            # Get queue info
            queue_info = await queue.declare(passive=True)
            message_count = queue_info.message_count
            
            # Close connection
            await queue.delete()
            await connection.close()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine status
            if latency_ms > 1000:  # > 1s
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                service="rabbitmq",
                status=status,
                latency_ms=round(latency_ms, 2),
                details={
                    "connection_test": "passed",
                    "message_test": "passed" if message_test_passed else "failed",
                    "test_queue_messages": message_count,
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="rabbitmq",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                details={},
                error=str(e)
            )


class DatabaseHealthCheck:
    """Health check for PostgreSQL database"""
    
    async def check(self, session: AsyncSession) -> HealthCheckResult:
        """
        Perform database health check.
        
        Args:
            session: Database session
            
        Returns:
            HealthCheckResult with database status
        """
        start_time = time.time()
        
        try:
            # Test simple query
            query_start = time.time()
            result = await session.execute(text("SELECT 1 as alive"))
            row = result.scalar_one()
            query_latency = (time.time() - query_start) * 1000
            
            # Get connection pool status
            pool = session.bind.pool
            pool_size = pool.size()
            pool_overflow = pool.overflow()
            pool_checked_in = pool.checkedin()
            pool_checked_out = pool.checkedout()
            
            # Get database version
            version_result = await session.execute(text("SELECT version()"))
            db_version = version_result.scalar_one()
            
            # Get active connections count
            connections_result = await session.execute(
                text("SELECT count(*) FROM pg_stat_activity WHERE state = 'active'")
            )
            active_connections = connections_result.scalar_one()
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Determine status based on latency and pool utilization
            pool_utilization = (pool_checked_out / pool_size) if pool_size > 0 else 0
            
            if latency_ms > 500 or pool_utilization > 0.9:  # > 500ms or > 90% pool used
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                service="database",
                status=status,
                latency_ms=round(latency_ms, 2),
                details={
                    "query_latency_ms": round(query_latency, 2),
                    "query_test": "passed" if row == 1 else "failed",
                    "pool_size": pool_size,
                    "pool_checked_out": pool_checked_out,
                    "pool_checked_in": pool_checked_in,
                    "pool_utilization_percent": round(pool_utilization * 100, 1),
                    "active_connections": active_connections,
                    "database_version": db_version.split(',')[0] if db_version else "unknown",
                }
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="database",
                status=HealthStatus.UNHEALTHY,
                latency_ms=round(latency_ms, 2),
                details={},
                error=str(e)
            )


class SystemHealthAggregator:
    """Aggregate health checks for all system components"""
    
    def __init__(
        self,
        redis_url: str,
        rabbitmq_url: str,
    ):
        self.redis_check = RedisHealthCheck(redis_url)
        self.rabbitmq_check = RabbitMQHealthCheck(rabbitmq_url)
        self.db_check = DatabaseHealthCheck()
    
    async def check_all(self, db_session: AsyncSession) -> Dict[str, Any]:
        """
        Perform all health checks concurrently.
        
        Args:
            db_session: Database session
            
        Returns:
            Aggregated health check results
        """
        start_time = time.time()
        
        # Run all checks concurrently
        results = await asyncio.gather(
            self.redis_check.check(),
            self.rabbitmq_check.check(),
            self.db_check.check(db_session),
            return_exceptions=True,
        )
        
        total_latency = (time.time() - start_time) * 1000
        
        # Process results
        checks = {}
        all_healthy = True
        any_unhealthy = False
        
        for result in results:
            if isinstance(result, Exception):
                # Handle exceptions from gather
                all_healthy = False
                any_unhealthy = True
                continue
            
            checks[result.service] = {
                "status": result.status.value,
                "latency_ms": result.latency_ms,
                "details": result.details,
                "error": result.error,
                "timestamp": result.timestamp.isoformat(),
            }
            
            if result.status == HealthStatus.UNHEALTHY:
                any_unhealthy = True
                all_healthy = False
            elif result.status == HealthStatus.DEGRADED:
                all_healthy = False
        
        # Determine overall status
        if any_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif not all_healthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "total_latency_ms": round(total_latency, 2),
            "checks": checks,
            "summary": {
                "total_checks": len(checks),
                "healthy": sum(1 for c in checks.values() if c["status"] == "healthy"),
                "degraded": sum(1 for c in checks.values() if c["status"] == "degraded"),
                "unhealthy": sum(1 for c in checks.values() if c["status"] == "unhealthy"),
            }
        }
