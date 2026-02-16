"""
Multi-Layer LLM Caching System for CognitionOS V4
Phase 5.2: Performance Dominance

Implements a 4-layer caching hierarchy:
- L1: Redis (hot cache, ~1ms latency)
- L2: PostgreSQL (warm cache, ~10ms latency)
- L3: Semantic cache with pgvector (~100ms latency)
- L4: LLM API fallback (~1800ms latency)

Target: 90% cache hit rate → 10x performance improvement
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import asyncio

from infrastructure.observability import get_logger
from infrastructure.llm.provider import LLMRequest, LLMResponse, LLMProvider


logger = get_logger(__name__)


class CacheLayer(str, Enum):
    """Cache layer enumeration"""
    L1_REDIS = "l1_redis"
    L2_DATABASE = "l2_database"
    L3_SEMANTIC = "l3_semantic"
    L4_LLM_API = "l4_llm_api"


@dataclass
class CacheEntry:
    """Cached LLM response entry"""
    cache_key: str
    request_hash: str
    messages: List[Dict[str, str]]
    model: str
    response_content: str
    provider: str
    usage: Dict[str, int]
    latency_ms: int
    cost_usd: Optional[float]
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl_seconds: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheHitResult:
    """Result of cache lookup"""
    hit: bool
    layer: Optional[CacheLayer]
    response: Optional[LLMResponse]
    latency_ms: int


class CacheKeyGenerator:
    """Generate deterministic cache keys for LLM requests"""
    
    @staticmethod
    def generate_exact_key(request: LLMRequest) -> str:
        """
        Generate exact cache key based on request parameters.
        
        Args:
            request: LLM request
            
        Returns:
            SHA256 hash of canonical request representation
        """
        # Create canonical representation
        canonical = {
            "messages": request.messages,
            "model": request.model,
            "temperature": round(request.temperature, 2),
            "max_tokens": request.max_tokens,
            "top_p": round(request.top_p, 2),
        }
        
        # Sort for determinism
        canonical_str = json.dumps(canonical, sort_keys=True)
        
        # Generate hash
        hash_obj = hashlib.sha256(canonical_str.encode('utf-8'))
        return f"llm:exact:{hash_obj.hexdigest()}"
    
    @staticmethod
    def generate_semantic_key(messages: List[Dict[str, str]]) -> str:
        """
        Generate semantic key for approximate matching.
        
        Args:
            messages: Message history
            
        Returns:
            SHA256 hash of message content only
        """
        # Extract just the content for semantic matching
        content = " ".join([msg.get("content", "") for msg in messages])
        hash_obj = hashlib.sha256(content.encode('utf-8'))
        return f"llm:semantic:{hash_obj.hexdigest()}"


class L1RedisCache:
    """
    L1 Cache: Redis (hot cache, ~1ms latency)
    
    Stores exact matches for very recent requests.
    Short TTL (5 minutes) for fastest possible access.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_ttl = 300  # 5 minutes
        
    async def get(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from L1 cache"""
        try:
            start_time = time.time()
            
            data = await self.redis.get(cache_key)
            if not data:
                return None
            
            # Deserialize
            entry_dict = json.loads(data)
            entry_dict['created_at'] = datetime.fromisoformat(entry_dict['created_at'])
            entry_dict['accessed_at'] = datetime.fromisoformat(entry_dict['accessed_at'])
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"L1 cache hit", extra={
                "cache_key": cache_key,
                "latency_ms": latency_ms
            })
            
            return CacheEntry(**entry_dict)
            
        except Exception as e:
            logger.error(f"L1 cache error: {str(e)}")
            return None
    
    async def set(self, entry: CacheEntry):
        """Store entry in L1 cache"""
        try:
            # Serialize
            entry_dict = asdict(entry)
            entry_dict['created_at'] = entry.created_at.isoformat()
            entry_dict['accessed_at'] = entry.accessed_at.isoformat()
            
            data = json.dumps(entry_dict)
            
            # Store with TTL
            await self.redis.setex(
                entry.cache_key,
                entry.ttl_seconds or self.default_ttl,
                data
            )
            
            logger.debug(f"L1 cache set", extra={"cache_key": entry.cache_key})
            
        except Exception as e:
            logger.error(f"L1 cache set error: {str(e)}")
    
    async def invalidate(self, cache_key: str):
        """Invalidate specific cache entry"""
        try:
            await self.redis.delete(cache_key)
            logger.info(f"L1 cache invalidated", extra={"cache_key": cache_key})
        except Exception as e:
            logger.error(f"L1 cache invalidation error: {str(e)}")
    
    async def clear_all(self):
        """Clear all LLM cache entries"""
        try:
            # Find all LLM cache keys
            keys = await self.redis.keys("llm:*")
            if keys:
                await self.redis.delete(*keys)
                logger.info(f"L1 cache cleared", extra={"keys_deleted": len(keys)})
        except Exception as e:
            logger.error(f"L1 cache clear error: {str(e)}")


class L2DatabaseCache:
    """
    L2 Cache: PostgreSQL (warm cache, ~10ms latency)
    
    Stores exact matches for recent requests.
    Medium TTL (1 hour) for persistence across Redis restarts.
    """
    
    def __init__(self, db_session):
        self.db = db_session
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from L2 cache"""
        try:
            start_time = time.time()
            
            # Query database (assumes llm_cache table exists)
            query = """
                SELECT * FROM llm_cache
                WHERE cache_key = $1
                AND (created_at + (ttl_seconds || ' seconds')::interval) > NOW()
            """
            
            result = await self.db.fetchrow(query, cache_key)
            
            if not result:
                return None
            
            # Update access count and time
            await self.db.execute("""
                UPDATE llm_cache
                SET accessed_at = NOW(), access_count = access_count + 1
                WHERE cache_key = $1
            """, cache_key)
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"L2 cache hit", extra={
                "cache_key": cache_key,
                "latency_ms": latency_ms
            })
            
            return CacheEntry(**dict(result))
            
        except Exception as e:
            logger.error(f"L2 cache error: {str(e)}")
            return None
    
    async def set(self, entry: CacheEntry):
        """Store entry in L2 cache"""
        try:
            query = """
                INSERT INTO llm_cache (
                    cache_key, request_hash, messages, model,
                    response_content, provider, usage, latency_ms,
                    cost_usd, created_at, accessed_at, access_count,
                    ttl_seconds, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (cache_key) DO UPDATE SET
                    accessed_at = EXCLUDED.accessed_at,
                    access_count = llm_cache.access_count + 1
            """
            
            await self.db.execute(
                query,
                entry.cache_key,
                entry.request_hash,
                json.dumps(entry.messages),
                entry.model,
                entry.response_content,
                entry.provider,
                json.dumps(entry.usage),
                entry.latency_ms,
                entry.cost_usd,
                entry.created_at,
                entry.accessed_at,
                entry.access_count,
                entry.ttl_seconds or self.default_ttl,
                json.dumps(entry.metadata) if entry.metadata else None
            )
            
            logger.debug(f"L2 cache set", extra={"cache_key": entry.cache_key})
            
        except Exception as e:
            logger.error(f"L2 cache set error: {str(e)}")


class L3SemanticCache:
    """
    L3 Cache: Semantic similarity with pgvector (~100ms latency)
    
    Finds semantically similar requests even if not exact match.
    Similarity threshold: 0.92+ (configurable)
    Longer TTL (24 hours) for stable, high-quality responses.
    """
    
    def __init__(self, db_session, embedding_service, similarity_threshold: float = 0.92):
        self.db = db_session
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.default_ttl = 86400  # 24 hours
    
    async def get(self, request: LLMRequest) -> Optional[Tuple[CacheEntry, float]]:
        """
        Get entry from L3 cache using semantic similarity.
        
        Returns:
            Tuple of (CacheEntry, similarity_score) if found, None otherwise
        """
        try:
            start_time = time.time()
            
            # Generate embedding for request
            content = " ".join([msg.get("content", "") for msg in request.messages])
            embedding = await self.embedding_service.embed(content)
            
            # Find similar entries using vector search
            query = """
                SELECT *, 
                    1 - (embedding <=> $1::vector) as similarity
                FROM llm_semantic_cache
                WHERE model = $2
                AND (created_at + (ttl_seconds || ' seconds')::interval) > NOW()
                AND 1 - (embedding <=> $1::vector) >= $3
                ORDER BY similarity DESC
                LIMIT 1
            """
            
            result = await self.db.fetchrow(
                query,
                embedding,
                request.model,
                self.similarity_threshold
            )
            
            if not result:
                return None
            
            similarity = result['similarity']
            
            # Update access count
            await self.db.execute("""
                UPDATE llm_semantic_cache
                SET accessed_at = NOW(), access_count = access_count + 1
                WHERE cache_key = $1
            """, result['cache_key'])
            
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(f"L3 cache hit", extra={
                "cache_key": result['cache_key'],
                "similarity": similarity,
                "latency_ms": latency_ms
            })
            
            entry = CacheEntry(**{k: result[k] for k in result.keys() if k != 'embedding' and k != 'similarity'})
            return (entry, similarity)
            
        except Exception as e:
            logger.error(f"L3 cache error: {str(e)}")
            return None
    
    async def set(self, entry: CacheEntry, embedding: List[float]):
        """Store entry in L3 semantic cache"""
        try:
            query = """
                INSERT INTO llm_semantic_cache (
                    cache_key, request_hash, messages, model,
                    response_content, provider, usage, latency_ms,
                    cost_usd, created_at, accessed_at, access_count,
                    ttl_seconds, embedding, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::vector, $15)
                ON CONFLICT (cache_key) DO UPDATE SET
                    accessed_at = EXCLUDED.accessed_at,
                    access_count = llm_semantic_cache.access_count + 1
            """
            
            await self.db.execute(
                query,
                entry.cache_key,
                entry.request_hash,
                json.dumps(entry.messages),
                entry.model,
                entry.response_content,
                entry.provider,
                json.dumps(entry.usage),
                entry.latency_ms,
                entry.cost_usd,
                entry.created_at,
                entry.accessed_at,
                entry.access_count,
                entry.ttl_seconds or self.default_ttl,
                embedding,
                json.dumps(entry.metadata) if entry.metadata else None
            )
            
            logger.debug(f"L3 cache set", extra={"cache_key": entry.cache_key})
            
        except Exception as e:
            logger.error(f"L3 cache set error: {str(e)}")


class MultiLayerLLMCache:
    """
    Multi-layer LLM cache coordinator.
    
    Manages cache hierarchy and fallback logic:
    L1 (Redis) → L2 (DB) → L3 (Semantic) → L4 (LLM API)
    """
    
    def __init__(
        self,
        l1_cache: L1RedisCache,
        l2_cache: L2DatabaseCache,
        l3_cache: L3SemanticCache,
        llm_provider,
        enable_l3: bool = True
    ):
        self.l1 = l1_cache
        self.l2 = l2_cache
        self.l3 = l3_cache
        self.llm = llm_provider
        self.enable_l3 = enable_l3
        
        # Metrics
        self.hits_l1 = 0
        self.hits_l2 = 0
        self.hits_l3 = 0
        self.hits_l4 = 0
        self.total_requests = 0
    
    async def generate(self, request: LLMRequest) -> Tuple[LLMResponse, CacheLayer]:
        """
        Generate LLM response with multi-layer caching.
        
        Returns:
            Tuple of (LLMResponse, CacheLayer indicating which layer served the response)
        """
        self.total_requests += 1
        start_time = time.time()
        
        # Generate cache keys
        exact_key = CacheKeyGenerator.generate_exact_key(request)
        
        # L1: Try Redis (exact match)
        entry = await self.l1.get(exact_key)
        if entry:
            self.hits_l1 += 1
            response = self._entry_to_response(entry)
            await self._track_cache_hit(CacheLayer.L1_REDIS, time.time() - start_time)
            return (response, CacheLayer.L1_REDIS)
        
        # L2: Try Database (exact match)
        entry = await self.l2.get(exact_key)
        if entry:
            self.hits_l2 += 1
            # Promote to L1
            await self.l1.set(entry)
            response = self._entry_to_response(entry)
            await self._track_cache_hit(CacheLayer.L2_DATABASE, time.time() - start_time)
            return (response, CacheLayer.L2_DATABASE)
        
        # L3: Try Semantic cache (approximate match)
        if self.enable_l3:
            result = await self.l3.get(request)
            if result:
                entry, similarity = result
                self.hits_l3 += 1
                # Promote to L1 and L2
                await self.l1.set(entry)
                await self.l2.set(entry)
                response = self._entry_to_response(entry)
                await self._track_cache_hit(CacheLayer.L3_SEMANTIC, time.time() - start_time)
                logger.info(f"Semantic cache hit", extra={"similarity": similarity})
                return (response, CacheLayer.L3_SEMANTIC)
        
        # L4: Call LLM API (cache miss)
        self.hits_l4 += 1
        response = await self.llm.generate(request)
        
        # Store in all cache layers
        await self._store_in_cache(exact_key, request, response)
        
        await self._track_cache_hit(CacheLayer.L4_LLM_API, time.time() - start_time)
        return (response, CacheLayer.L4_LLM_API)
    
    async def _store_in_cache(self, cache_key: str, request: LLMRequest, response: LLMResponse):
        """Store response in all cache layers"""
        now = datetime.utcnow()
        
        # Create cache entry
        entry = CacheEntry(
            cache_key=cache_key,
            request_hash=CacheKeyGenerator.generate_semantic_key(request.messages),
            messages=request.messages,
            model=request.model,
            response_content=response.content,
            provider=response.provider.value,
            usage=response.usage,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            created_at=now,
            accessed_at=now,
            access_count=1,
            ttl_seconds=300,  # 5 minutes for L1
            metadata=response.metadata
        )
        
        # Store in L1 and L2
        await asyncio.gather(
            self.l1.set(entry),
            self.l2.set(entry)
        )
        
        # Store in L3 if enabled
        if self.enable_l3:
            content = " ".join([msg.get("content", "") for msg in request.messages])
            # Note: This requires embedding service - placeholder for now
            # embedding = await self.l3.embedding_service.embed(content)
            # await self.l3.set(entry, embedding)
            pass
    
    def _entry_to_response(self, entry: CacheEntry) -> LLMResponse:
        """Convert cache entry to LLM response"""
        return LLMResponse(
            content=entry.response_content,
            model=entry.model,
            provider=LLMProvider(entry.provider),
            usage=entry.usage,
            latency_ms=entry.latency_ms,
            cost_usd=entry.cost_usd,
            metadata=entry.metadata
        )
    
    async def _track_cache_hit(self, layer: CacheLayer, latency: float):
        """Track cache hit metrics"""
        latency_ms = int(latency * 1000)
        logger.info(f"Cache result", extra={
            "layer": layer.value,
            "latency_ms": latency_ms,
            "cache_hit_rate": self.get_cache_hit_rate()
        })
    
    def get_cache_hit_rate(self) -> Dict[str, float]:
        """Get cache hit rates by layer"""
        if self.total_requests == 0:
            return {"l1": 0.0, "l2": 0.0, "l3": 0.0, "l4": 0.0, "total": 0.0}
        
        total = self.total_requests
        return {
            "l1": self.hits_l1 / total,
            "l2": self.hits_l2 / total,
            "l3": self.hits_l3 / total,
            "l4": self.hits_l4 / total,
            "total": (self.hits_l1 + self.hits_l2 + self.hits_l3) / total
        }
    
    async def invalidate(self, cache_key: str):
        """Invalidate cache entry across all layers"""
        await asyncio.gather(
            self.l1.invalidate(cache_key),
            # L2 and L3 invalidation would require DB queries
        )
    
    async def clear_all(self):
        """Clear all cache layers"""
        await self.l1.clear_all()
        # L2 and L3 clearing would require DB operations
