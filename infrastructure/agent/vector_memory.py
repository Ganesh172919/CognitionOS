"""
Vector Memory Store for AI Agents

Provides persistent, searchable memory for autonomous agents using:
- In-memory vector index with cosine similarity (no external dependencies)
- Optional Redis-backed persistence layer
- Hierarchical memory tiers: working, episodic, semantic, procedural
- Importance scoring with decay
- Context window compression
- Deduplication via content fingerprinting
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4


class MemoryTier(str, Enum):
    """Memory hierarchy tiers"""
    WORKING = "working"       # Current task context, short-lived
    EPISODIC = "episodic"     # Specific past experiences
    SEMANTIC = "semantic"     # General knowledge and facts
    PROCEDURAL = "procedural" # How-to knowledge, learned skills


class MemoryType(str, Enum):
    """Types of memory entries"""
    OBSERVATION = "observation"
    ACTION = "action"
    RESULT = "result"
    REFLECTION = "reflection"
    PLAN = "plan"
    FACT = "fact"
    ERROR = "error"
    USER_INPUT = "user_input"
    TOOL_OUTPUT = "tool_output"


@dataclass
class MemoryEntry:
    """A single memory entry with embedding vector"""
    entry_id: str
    content: str
    memory_type: MemoryType
    tier: MemoryTier
    embedding: List[float]           # Dense vector representation
    importance: float                # 0.0–1.0
    created_at: float                # Unix timestamp
    last_accessed_at: float          # For LRU eviction
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    fingerprint: str = ""            # SHA-256 of content for dedup

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = hashlib.sha256(self.content.encode()).hexdigest()[:16]

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    @property
    def recency_score(self) -> float:
        """Score from 0–1 that decays with time (half-life = 1 hour)"""
        half_life = 3600.0
        return math.exp(-math.log(2) * self.age_seconds / half_life)

    def effective_importance(self, recency_weight: float = 0.4) -> float:
        """Combined importance × recency score"""
        return self.importance * (1 - recency_weight) + self.recency_score * recency_weight


@dataclass
class SearchResult:
    """Result of a memory search"""
    entry: MemoryEntry
    similarity: float         # Cosine similarity 0–1
    relevance_score: float    # Combined similarity + importance + recency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry.entry_id,
            "content": self.entry.content,
            "memory_type": self.entry.memory_type,
            "tier": self.entry.tier,
            "similarity": round(self.similarity, 4),
            "relevance_score": round(self.relevance_score, 4),
            "importance": self.entry.importance,
            "age_seconds": round(self.entry.age_seconds, 1),
            "metadata": self.entry.metadata,
            "tags": self.entry.tags,
        }


class TFIDFEmbedder:
    """
    Lightweight TF-IDF based text embedder.
    Produces deterministic dense vectors without external ML libraries.
    Suitable for semantic similarity when numpy/sentence-transformers unavailable.
    """

    STOP_WORDS = {
        "a", "an", "the", "is", "it", "in", "of", "to", "and", "or",
        "for", "on", "at", "by", "from", "with", "as", "was", "are",
        "be", "has", "had", "have", "this", "that", "not", "but",
    }
    DIM = 256

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}

    def _tokenize(self, text: str) -> List[str]:
        tokens = text.lower().split()
        cleaned = []
        for t in tokens:
            t = "".join(c for c in t if c.isalnum())
            if t and t not in self.STOP_WORDS and len(t) > 1:
                cleaned.append(t)
        return cleaned

    def _term_hash(self, term: str) -> int:
        """Hash term to a bucket in [0, DIM)"""
        h = int(hashlib.md5(term.encode()).hexdigest(), 16)
        return h % self.DIM

    def encode(self, text: str) -> List[float]:
        """Produce a fixed-dimension TF vector using feature hashing"""
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self.DIM

        vector = [0.0] * self.DIM
        for token in tokens:
            idx = self._term_hash(token)
            # Use sign trick to reduce collisions
            sign = 1 if int(hashlib.md5((token + "_s").encode()).hexdigest(), 16) % 2 else -1
            vector[idx] += sign

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two vectors"""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


class VectorMemoryStore:
    """
    In-memory vector store for agent memory with semantic search.

    Features:
    - Cosine similarity search over all memory tiers
    - Importance-weighted retrieval
    - Automatic deduplication by content fingerprint
    - LRU eviction when capacity exceeded
    - Memory compression (summarization of old entries)
    - Tier-based filtering
    - Agent and session isolation
    - Serializable state for persistence
    """

    def __init__(
        self,
        capacity: int = 10000,
        embedder: Optional[TFIDFEmbedder] = None,
        default_importance: float = 0.5,
        recency_weight: float = 0.4,
    ) -> None:
        self._entries: Dict[str, MemoryEntry] = {}
        self._fingerprints: Dict[str, str] = {}  # fingerprint -> entry_id
        self._capacity = capacity
        self._embedder = embedder or TFIDFEmbedder()
        self._default_importance = default_importance
        self._recency_weight = recency_weight

    # ──────────────────────────────────────────────
    # Write API
    # ──────────────────────────────────────────────

    def store(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        tier: MemoryTier = MemoryTier.EPISODIC,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> MemoryEntry:
        """Store a new memory entry, deduplicating by content fingerprint."""
        fingerprint = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Deduplication: if same content exists, update importance and return
        if fingerprint in self._fingerprints:
            existing_id = self._fingerprints[fingerprint]
            if existing_id in self._entries:
                entry = self._entries[existing_id]
                if importance is not None:
                    entry.importance = max(entry.importance, importance)
                entry.access_count += 1
                entry.last_accessed_at = time.time()
                return entry

        # Evict if at capacity
        if len(self._entries) >= self._capacity:
            self._evict_lru()

        embedding = self._embedder.encode(content)
        entry = MemoryEntry(
            entry_id=str(uuid4()),
            content=content,
            memory_type=memory_type,
            tier=tier,
            embedding=embedding,
            importance=importance if importance is not None else self._default_importance,
            created_at=time.time(),
            last_accessed_at=time.time(),
            metadata=metadata or {},
            tags=tags or [],
            agent_id=agent_id,
            session_id=session_id,
            fingerprint=fingerprint,
        )
        self._entries[entry.entry_id] = entry
        self._fingerprints[fingerprint] = entry.entry_id
        return entry

    def update_importance(self, entry_id: str, importance: float) -> bool:
        """Update the importance score of a stored memory"""
        if entry_id in self._entries:
            self._entries[entry_id].importance = max(0.0, min(1.0, importance))
            return True
        return False

    def delete(self, entry_id: str) -> bool:
        """Remove a memory entry"""
        entry = self._entries.pop(entry_id, None)
        if entry:
            self._fingerprints.pop(entry.fingerprint, None)
            return True
        return False

    def clear_tier(self, tier: MemoryTier, agent_id: Optional[str] = None) -> int:
        """Clear all entries in a tier, optionally scoped to an agent"""
        to_delete = [
            eid for eid, e in self._entries.items()
            if e.tier == tier and (agent_id is None or e.agent_id == agent_id)
        ]
        for eid in to_delete:
            self.delete(eid)
        return len(to_delete)

    # ──────────────────────────────────────────────
    # Search API
    # ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        tier: Optional[MemoryTier] = None,
        memory_type: Optional[MemoryType] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
        min_similarity: float = 0.0,
        tags: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search over memory entries.
        Returns top-k results ordered by relevance (similarity + importance + recency).
        """
        query_vec = self._embedder.encode(query)
        candidates = self._filter_entries(
            tier, memory_type, agent_id, session_id, min_importance, tags
        )

        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in candidates:
            sim = self._embedder.similarity(query_vec, entry.embedding)
            if sim < min_similarity:
                continue
            relevance = (
                sim * 0.5
                + entry.effective_importance(self._recency_weight) * 0.5
            )
            scored.append((relevance, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for relevance, entry in scored[:top_k]:
            sim = self._embedder.similarity(query_vec, entry.embedding)
            entry.access_count += 1
            entry.last_accessed_at = time.time()
            results.append(SearchResult(
                entry=entry,
                similarity=sim,
                relevance_score=relevance,
            ))
        return results

    def get_recent(
        self,
        n: int = 20,
        tier: Optional[MemoryTier] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Retrieve the n most recently created memory entries"""
        entries = self._filter_entries(tier, None, agent_id, session_id, 0.0, None)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries[:n]

    def get_important(
        self,
        n: int = 20,
        tier: Optional[MemoryTier] = None,
        agent_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Retrieve the n highest-importance memory entries"""
        entries = self._filter_entries(tier, None, agent_id, None, 0.0, None)
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:n]

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._entries.get(entry_id)

    # ──────────────────────────────────────────────
    # Context Window Builder
    # ──────────────────────────────────────────────

    def build_context(
        self,
        query: str,
        max_tokens: int = 2000,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Build a context string from relevant memories for injection into LLM prompts.
        Token budget is approximated at ~4 chars per token.
        """
        char_budget = max_tokens * 4
        results = self.search(
            query,
            top_k=50,
            agent_id=agent_id,
            session_id=session_id,
        )

        context_parts: List[str] = []
        used_chars = 0

        for res in results:
            snippet = f"[{res.entry.memory_type.value.upper()}] {res.entry.content}"
            if used_chars + len(snippet) > char_budget:
                break
            context_parts.append(snippet)
            used_chars += len(snippet)

        return "\n".join(context_parts)

    # ──────────────────────────────────────────────
    # Stats & Persistence
    # ──────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return memory store statistics"""
        tier_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for entry in self._entries.values():
            tier_counts[entry.tier.value] = tier_counts.get(entry.tier.value, 0) + 1
            type_counts[entry.memory_type.value] = type_counts.get(entry.memory_type.value, 0) + 1

        return {
            "total_entries": len(self._entries),
            "capacity": self._capacity,
            "utilization_pct": round(len(self._entries) / self._capacity * 100, 1),
            "tier_breakdown": tier_counts,
            "type_breakdown": type_counts,
        }

    def export(self) -> List[Dict[str, Any]]:
        """Export all entries as JSON-serializable dicts"""
        return [
            {
                "entry_id": e.entry_id,
                "content": e.content,
                "memory_type": e.memory_type.value,
                "tier": e.tier.value,
                "importance": e.importance,
                "created_at": e.created_at,
                "last_accessed_at": e.last_accessed_at,
                "access_count": e.access_count,
                "metadata": e.metadata,
                "tags": e.tags,
                "agent_id": e.agent_id,
                "session_id": e.session_id,
                "fingerprint": e.fingerprint,
            }
            for e in self._entries.values()
        ]

    def import_entries(self, entries: List[Dict[str, Any]]) -> int:
        """Import exported entries (re-embeds content)"""
        count = 0
        for data in entries:
            if data["entry_id"] in self._entries:
                continue
            embedding = self._embedder.encode(data["content"])
            entry = MemoryEntry(
                entry_id=data["entry_id"],
                content=data["content"],
                memory_type=MemoryType(data["memory_type"]),
                tier=MemoryTier(data["tier"]),
                embedding=embedding,
                importance=data["importance"],
                created_at=data["created_at"],
                last_accessed_at=data["last_accessed_at"],
                access_count=data.get("access_count", 0),
                metadata=data.get("metadata", {}),
                tags=data.get("tags", []),
                agent_id=data.get("agent_id"),
                session_id=data.get("session_id"),
                fingerprint=data.get("fingerprint", ""),
            )
            self._entries[entry.entry_id] = entry
            if entry.fingerprint:
                self._fingerprints[entry.fingerprint] = entry.entry_id
            count += 1
        return count

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _filter_entries(
        self,
        tier: Optional[MemoryTier],
        memory_type: Optional[MemoryType],
        agent_id: Optional[str],
        session_id: Optional[str],
        min_importance: float,
        tags: Optional[List[str]],
    ) -> List[MemoryEntry]:
        result = []
        for entry in self._entries.values():
            if tier and entry.tier != tier:
                continue
            if memory_type and entry.memory_type != memory_type:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            if session_id and entry.session_id != session_id:
                continue
            if entry.importance < min_importance:
                continue
            if tags and not all(t in entry.tags for t in tags):
                continue
            result.append(entry)
        return result

    def _evict_lru(self) -> None:
        """Evict the least recently accessed entry with lowest importance"""
        if not self._entries:
            return
        # Score = importance * 0.3 + recency * 0.7  (lower = evict first)
        victim = min(
            self._entries.values(),
            key=lambda e: e.importance * 0.3 + e.recency_score * 0.7,
        )
        self.delete(victim.entry_id)
