"""
Cache Invalidation - Tenant and subscription invalidation.

Invalidates L1 and L2 cache when tenant/subscription data changes.
"""

import logging
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)

_l1_cache: Optional["L1Cache"] = None
_l2_cache: Optional["L2RedisCache"] = None


def _get_l1() -> "L1Cache":
    global _l1_cache
    if _l1_cache is None:
        from infrastructure.caching.l1_cache import L1Cache
        _l1_cache = L1Cache(max_size=1000, default_ttl=300)
    return _l1_cache


def _get_l2():
    global _l2_cache
    if _l2_cache is None:
        try:
            from infrastructure.caching.l2_redis import L2RedisCache
            _l2_cache = L2RedisCache(namespace="cognitionos", default_ttl=600)
        except ImportError:
            _l2_cache = None
    return _l2_cache


async def invalidate_tenant(tenant_id: UUID) -> None:
    """
    Invalidate all cache entries for a tenant.
    Call when tenant or subscription is updated.
    """
    l1 = _get_l1()
    count_l1 = await l1.delete_tenant(tenant_id, prefix="tenant")
    count_l1 += await l1.delete_tenant(tenant_id, prefix="subscription")
    if count_l1:
        logger.debug("L1 invalidated %d keys for tenant %s", count_l1, tenant_id)
    l2 = _get_l2()
    if l2:
        count_l2 = await l2.delete_tenant_keys(tenant_id, prefix="tenant")
        count_l2 += await l2.delete_tenant_keys(tenant_id, prefix="subscription")
        if count_l2:
            logger.debug("L2 invalidated %d keys for tenant %s", count_l2, tenant_id)


async def invalidate_subscription(tenant_id: UUID) -> None:
    """
    Invalidate subscription cache for a tenant.
    Call when subscription tier, status, or period changes.
    """
    l1 = _get_l1()
    key = l1.tenant_subscription_key(tenant_id)
    await l1.delete(key)
    logger.debug("Invalidated subscription cache for tenant %s", tenant_id)
    l2 = _get_l2()
    if l2:
        await l2.delete_tenant_keys(tenant_id, prefix="subscription")
