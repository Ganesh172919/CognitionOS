"""
Agent Memory System — CognitionOS

Persistent, queryable memory for AI agents:
- Short-term (conversation) memory
- Long-term (semantic) memory with embeddings
- Episodic memory (task execution history)
- Working memory (current context)
- Memory consolidation and pruning
- Relevance search
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    WORKING = "working"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    memory_id: str
    content: str
    memory_type: MemoryType
    agent_id: str = ""
    tenant_id: str = ""
    importance: float = 0.5  # 0-1
    embedding: List[float] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    decay_factor: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id, "content": self.content[:200],
            "type": self.memory_type.value, "importance": self.importance,
            "tags": self.tags, "access_count": self.access_count,
            "created_at": self.created_at}


@dataclass
class MemorySearchResult:
    entry: MemoryEntry
    relevance_score: float
    recency_score: float
    combined_score: float


class AgentMemorySystem:
    """Multi-layer memory for AI agents with relevance search."""

    def __init__(self, *, max_short_term: int = 50,
                 max_long_term: int = 10000,
                 decay_rate: float = 0.01) -> None:
        self._memories: Dict[str, Dict[str, MemoryEntry]] = defaultdict(dict)
        self._max_short_term = max_short_term
        self._max_long_term = max_long_term
        self._decay_rate = decay_rate
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"mem_{self._id_counter:08d}"

    # ---- store ----
    def store(self, content: str, memory_type: MemoryType, *,
              agent_id: str = "", tenant_id: str = "",
              importance: float = 0.5, tags: List[str] | None = None,
              metadata: Dict[str, Any] | None = None) -> MemoryEntry:
        mid = self._next_id()
        entry = MemoryEntry(
            memory_id=mid, content=content, memory_type=memory_type,
            agent_id=agent_id, tenant_id=tenant_id,
            importance=importance, tags=tags or [],
            metadata=metadata or {},
            embedding=self._compute_embedding(content))

        key = f"{agent_id}:{tenant_id}"
        self._memories[key][mid] = entry
        self._enforce_limits(key)
        return entry

    def store_episodic(self, agent_id: str, task_id: str, *,
                        requirement: str, result: str,
                        success: bool, duration_ms: float) -> MemoryEntry:
        content = f"Task {task_id}: {requirement[:200]} → {'SUCCESS' if success else 'FAILURE'}: {result[:200]}"
        return self.store(
            content, MemoryType.EPISODIC, agent_id=agent_id,
            importance=0.7 if success else 0.9,
            tags=["task", task_id],
            metadata={"task_id": task_id, "success": success,
                      "duration_ms": duration_ms})

    # ---- search ----
    def search(self, query: str, *, agent_id: str = "", tenant_id: str = "",
               memory_type: MemoryType | None = None,
               top_k: int = 10, min_score: float = 0.1) -> List[MemorySearchResult]:
        key = f"{agent_id}:{tenant_id}"
        memories = self._memories.get(key, {})
        if not memories:
            return []

        query_embedding = self._compute_embedding(query)
        query_words = set(query.lower().split())
        now = datetime.now(timezone.utc)
        results = []

        for entry in memories.values():
            if memory_type and entry.memory_type != memory_type:
                continue

            # Relevance via cosine similarity
            relevance = self._cosine_similarity(query_embedding, entry.embedding)

            # Keyword overlap boost
            content_words = set(entry.content.lower().split())
            overlap = len(query_words & content_words)
            keyword_boost = min(0.3, overlap * 0.05)
            relevance = min(1.0, relevance + keyword_boost)

            # Recency score
            age_hours = 1
            try:
                created = datetime.fromisoformat(entry.created_at)
                age_hours = max(1, (now - created).total_seconds() / 3600)
            except (ValueError, TypeError):
                pass
            recency = 1.0 / (1.0 + math.log(age_hours))

            # Importance weighting
            combined = (relevance * 0.5 + recency * 0.2 +
                        entry.importance * 0.2 + entry.decay_factor * 0.1)

            if combined >= min_score:
                results.append(MemorySearchResult(
                    entry=entry, relevance_score=relevance,
                    recency_score=recency, combined_score=combined))

        results.sort(key=lambda r: -r.combined_score)

        # Update access on returned results
        for r in results[:top_k]:
            r.entry.access_count += 1
            r.entry.last_accessed = now.isoformat()

        return results[:top_k]

    def get_working_memory(self, agent_id: str, tenant_id: str = "") -> List[Dict[str, Any]]:
        key = f"{agent_id}:{tenant_id}"
        memories = self._memories.get(key, {})
        working = [e for e in memories.values() if e.memory_type == MemoryType.WORKING]
        working.sort(key=lambda e: e.created_at, reverse=True)
        return [e.to_dict() for e in working[:20]]

    def get_recent(self, agent_id: str, *, limit: int = 10,
                   memory_type: MemoryType | None = None,
                   tenant_id: str = "") -> List[Dict[str, Any]]:
        key = f"{agent_id}:{tenant_id}"
        memories = list(self._memories.get(key, {}).values())
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return [m.to_dict() for m in memories[:limit]]

    # ---- consolidation ----
    def consolidate(self, agent_id: str, tenant_id: str = "") -> int:
        """Move important short-term memories to long-term, prune stale entries."""
        key = f"{agent_id}:{tenant_id}"
        memories = self._memories.get(key, {})
        promoted = 0

        for entry in list(memories.values()):
            # Promote important short-term to long-term
            if (entry.memory_type == MemoryType.SHORT_TERM and
                    entry.importance >= 0.7 and entry.access_count >= 2):
                entry.memory_type = MemoryType.LONG_TERM
                promoted += 1

            # Apply decay
            try:
                created = datetime.fromisoformat(entry.created_at)
                age_days = (datetime.now(timezone.utc) - created).days
                entry.decay_factor = max(0.1, 1.0 - self._decay_rate * age_days)
            except (ValueError, TypeError):
                pass

        return promoted

    def prune(self, agent_id: str, *, max_age_days: int = 90,
              min_importance: float = 0.2, tenant_id: str = "") -> int:
        key = f"{agent_id}:{tenant_id}"
        memories = self._memories.get(key, {})
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        to_remove = []

        for mid, entry in memories.items():
            if (entry.created_at < cutoff and entry.importance < min_importance
                    and entry.memory_type != MemoryType.PROCEDURAL):
                to_remove.append(mid)

        for mid in to_remove:
            del memories[mid]
        return len(to_remove)

    # ---- helpers ----
    def _compute_embedding(self, text: str) -> List[float]:
        """Simple hash-based pseudo-embedding for demo. Replace with real embeddings."""
        h = hashlib.sha256(text.lower().encode()).digest()
        return [b / 255.0 for b in h[:32]]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return max(0, dot / (mag_a * mag_b))

    def _enforce_limits(self, key: str) -> None:
        memories = self._memories.get(key, {})
        short = [(m.memory_id, m) for m in memories.values()
                 if m.memory_type == MemoryType.SHORT_TERM]
        if len(short) > self._max_short_term:
            short.sort(key=lambda x: x[1].created_at)
            for mid, _ in short[:len(short) - self._max_short_term]:
                del memories[mid]

    # ---- stats ----
    def get_stats(self, agent_id: str = "", tenant_id: str = "") -> Dict[str, Any]:
        key = f"{agent_id}:{tenant_id}"
        memories = self._memories.get(key, {})
        by_type: Dict[str, int] = defaultdict(int)
        for m in memories.values():
            by_type[m.memory_type.value] += 1
        return {"total": len(memories), "by_type": dict(by_type),
                "agent_id": agent_id}


_system: AgentMemorySystem | None = None

def get_agent_memory() -> AgentMemorySystem:
    global _system
    if not _system:
        _system = AgentMemorySystem()
    return _system
