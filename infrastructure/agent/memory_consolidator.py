"""
Agent Memory Consolidator

Background memory management system that:
- Promotes important working memories to long-term storage
- Deduplicates overlapping memory entries
- Clusters related memories for coherent recall
- Decays low-importance memories over time
- Generates semantic summaries of memory clusters
- Compresses episodic memories into procedural knowledge
- Enforces capacity limits per tier per agent
- Tracks consolidation statistics for observability
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .vector_memory import (
    MemoryEntry,
    MemoryTier,
    MemoryType,
    TFIDFEmbedder,
    VectorMemoryStore,
)


class ConsolidationStrategy(str, Enum):
    """Strategy for consolidating memory clusters"""
    KEEP_HIGHEST_IMPORTANCE = "keep_highest_importance"
    MERGE_INTO_SUMMARY = "merge_into_summary"
    PROMOTE_TO_SEMANTIC = "promote_to_semantic"
    DECAY_AND_DROP = "decay_and_drop"


@dataclass
class ConsolidationRule:
    """Rule governing how memories are consolidated"""
    source_tier: MemoryTier
    strategy: ConsolidationStrategy
    min_access_count: int = 3           # Must be accessed this many times
    min_importance: float = 0.6         # Minimum importance to promote
    max_age_hours: float = 24.0         # Max age before decay applies
    similarity_threshold: float = 0.85  # Cluster merging threshold
    capacity_limit: int = 500           # Max entries per tier per agent
    decay_rate: float = 0.05            # Importance reduction per hour


@dataclass
class MemoryCluster:
    """Group of related memory entries"""
    cluster_id: str
    centroid: List[float]           # Average embedding vector
    member_ids: List[str]
    dominant_type: MemoryType
    avg_importance: float
    summary: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.member_ids)


@dataclass
class ConsolidationStats:
    """Statistics from a consolidation run"""
    run_id: str
    agent_id: str
    started_at: float
    completed_at: float
    entries_examined: int = 0
    entries_promoted: int = 0
    entries_merged: int = 0
    entries_decayed: int = 0
    entries_dropped: int = 0
    clusters_formed: int = 0
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "entries_examined": self.entries_examined,
            "entries_promoted": self.entries_promoted,
            "entries_merged": self.entries_merged,
            "entries_decayed": self.entries_decayed,
            "entries_dropped": self.entries_dropped,
            "clusters_formed": self.clusters_formed,
            "duration_ms": round(self.duration_ms, 2),
        }


class MemoryConsolidator:
    """
    Consolidates agent memories by promoting, merging, and decaying entries.

    Consolidation pipeline:
    1. Decay: reduce importance of old, rarely-accessed memories
    2. Cluster: group similar working memories
    3. Promote: move high-importance episodic memories to semantic tier
    4. Merge: combine duplicate/near-duplicate entries
    5. Evict: remove lowest-value entries when capacity exceeded

    Usage::

        store = VectorMemoryStore()
        consolidator = MemoryConsolidator(store)
        stats = consolidator.consolidate("agent-1")
    """

    DEFAULT_RULES: List[ConsolidationRule] = [
        ConsolidationRule(
            source_tier=MemoryTier.WORKING,
            strategy=ConsolidationStrategy.PROMOTE_TO_SEMANTIC,
            min_access_count=2,
            min_importance=0.7,
            max_age_hours=1.0,
            capacity_limit=200,
            decay_rate=0.1,
        ),
        ConsolidationRule(
            source_tier=MemoryTier.EPISODIC,
            strategy=ConsolidationStrategy.MERGE_INTO_SUMMARY,
            min_access_count=3,
            min_importance=0.5,
            max_age_hours=48.0,
            similarity_threshold=0.80,
            capacity_limit=1000,
            decay_rate=0.02,
        ),
        ConsolidationRule(
            source_tier=MemoryTier.SEMANTIC,
            strategy=ConsolidationStrategy.KEEP_HIGHEST_IMPORTANCE,
            min_importance=0.3,
            max_age_hours=720.0,  # 30 days
            capacity_limit=2000,
            decay_rate=0.005,
        ),
    ]

    def __init__(
        self,
        memory_store: VectorMemoryStore,
        rules: Optional[List[ConsolidationRule]] = None,
    ) -> None:
        self._store = memory_store
        self._rules = rules or self.DEFAULT_RULES
        self._embedder = TFIDFEmbedder()
        self._consolidation_history: List[ConsolidationStats] = []

    def consolidate(
        self,
        agent_id: str,
        session_id: Optional[str] = None,
    ) -> ConsolidationStats:
        """
        Run a full consolidation cycle for an agent.
        Returns statistics about what was done.
        """
        import uuid
        run_id = str(uuid.uuid4())
        started_at = time.time()
        stats = ConsolidationStats(
            run_id=run_id,
            agent_id=agent_id,
            started_at=started_at,
            completed_at=0.0,
        )

        for rule in self._rules:
            entries = self._store.get_recent(
                n=10000,
                tier=rule.source_tier,
                agent_id=agent_id,
                session_id=session_id,
            )
            stats.entries_examined += len(entries)

            # Step 1: Decay old entries
            decayed, dropped = self._apply_decay(entries, rule)
            stats.entries_decayed += decayed
            stats.entries_dropped += dropped

            # Refresh entries after decay
            entries = self._store.get_recent(
                n=10000,
                tier=rule.source_tier,
                agent_id=agent_id,
                session_id=session_id,
            )

            # Step 2: Handle strategy
            if rule.strategy == ConsolidationStrategy.PROMOTE_TO_SEMANTIC:
                promoted = self._promote_entries(entries, rule, agent_id)
                stats.entries_promoted += promoted

            elif rule.strategy == ConsolidationStrategy.MERGE_INTO_SUMMARY:
                clusters = self._cluster_entries(entries, rule.similarity_threshold)
                stats.clusters_formed += len(clusters)
                merged = self._merge_clusters(clusters, rule, agent_id)
                stats.entries_merged += merged

            elif rule.strategy == ConsolidationStrategy.KEEP_HIGHEST_IMPORTANCE:
                dropped = self._enforce_capacity(entries, rule)
                stats.entries_dropped += dropped

        stats.completed_at = time.time()
        stats.duration_ms = (stats.completed_at - started_at) * 1000
        self._consolidation_history.append(stats)
        if len(self._consolidation_history) > 100:
            self._consolidation_history = self._consolidation_history[-100:]
        return stats

    def get_consolidation_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        history = self._consolidation_history
        if agent_id:
            history = [s for s in history if s.agent_id == agent_id]
        return [s.to_dict() for s in history[-limit:]]

    def cluster_entries(
        self,
        entries: List[MemoryEntry],
        threshold: float = 0.8,
    ) -> List[MemoryCluster]:
        """Public API: cluster a list of memory entries"""
        return self._cluster_entries(entries, threshold)

    # ──────────────────────────────────────────────
    # Pipeline steps
    # ──────────────────────────────────────────────

    def _apply_decay(
        self,
        entries: List[MemoryEntry],
        rule: ConsolidationRule,
    ) -> Tuple[int, int]:
        """Apply importance decay and remove entries below threshold"""
        decayed = 0
        dropped = 0
        now = time.time()

        for entry in entries:
            age_hours = (now - entry.created_at) / 3600
            if age_hours < 1.0:
                continue
            decay_amount = rule.decay_rate * age_hours
            new_importance = max(0.0, entry.importance - decay_amount)

            if new_importance < 0.05:
                self._store.delete(entry.entry_id)
                dropped += 1
            elif abs(new_importance - entry.importance) > 0.01:
                self._store.update_importance(entry.entry_id, new_importance)
                decayed += 1

        return decayed, dropped

    def _promote_entries(
        self,
        entries: List[MemoryEntry],
        rule: ConsolidationRule,
        agent_id: str,
    ) -> int:
        """Promote high-value working memories to semantic tier"""
        promoted = 0
        now = time.time()

        for entry in entries:
            age_hours = (now - entry.created_at) / 3600
            if (
                entry.importance >= rule.min_importance
                and entry.access_count >= rule.min_access_count
                and age_hours >= 0.1  # At least 6 minutes old
            ):
                # Store as semantic memory
                self._store.store(
                    content=entry.content,
                    memory_type=MemoryType.FACT,
                    tier=MemoryTier.SEMANTIC,
                    importance=min(1.0, entry.importance * 1.1),  # Slight boost
                    metadata={
                        **entry.metadata,
                        "promoted_from": MemoryTier.WORKING.value,
                        "promoted_at": now,
                        "original_entry_id": entry.entry_id,
                    },
                    tags=entry.tags,
                    agent_id=agent_id,
                )
                # Remove from working memory
                self._store.delete(entry.entry_id)
                promoted += 1

        return promoted

    def _cluster_entries(
        self,
        entries: List[MemoryEntry],
        threshold: float,
    ) -> List[MemoryCluster]:
        """Group similar entries using greedy agglomerative clustering"""
        if not entries:
            return []

        import uuid

        clusters: List[MemoryCluster] = []
        assigned: set[str] = set()

        for entry in entries:
            if entry.entry_id in assigned:
                continue

            # Find similar unassigned entries
            members = [entry.entry_id]
            assigned.add(entry.entry_id)

            for other in entries:
                if other.entry_id in assigned:
                    continue
                sim = self._embedder.similarity(entry.embedding, other.embedding)
                if sim >= threshold:
                    members.append(other.entry_id)
                    assigned.add(other.entry_id)

            # Compute centroid
            member_entries = [e for e in entries if e.entry_id in members]
            centroid = self._compute_centroid(member_entries)
            avg_imp = sum(e.importance for e in member_entries) / len(member_entries)
            types = [e.memory_type for e in member_entries]
            dominant_type = max(set(types), key=types.count)

            clusters.append(MemoryCluster(
                cluster_id=str(uuid.uuid4()),
                centroid=centroid,
                member_ids=members,
                dominant_type=dominant_type,
                avg_importance=avg_imp,
            ))

        return clusters

    def _merge_clusters(
        self,
        clusters: List[MemoryCluster],
        rule: ConsolidationRule,
        agent_id: str,
    ) -> int:
        """Merge clusters with >1 member into summary entries"""
        merged = 0
        for cluster in clusters:
            if cluster.size <= 1:
                continue

            # Get member entries
            member_entries = [
                e for e in (self._store.get_by_id(mid) for mid in cluster.member_ids)
                if e is not None
            ]
            if not member_entries:
                continue

            # Create a merged summary
            contents = [e.content for e in member_entries]
            summary = self._summarize_contents(contents)

            # Keep the best entry, update with summary
            best = max(member_entries, key=lambda e: e.importance)
            self._store.store(
                content=summary,
                memory_type=cluster.dominant_type,
                tier=MemoryTier.EPISODIC,
                importance=min(1.0, cluster.avg_importance * 1.05),
                metadata={
                    "merged_count": cluster.size,
                    "original_ids": cluster.member_ids,
                    "cluster_id": cluster.cluster_id,
                },
                agent_id=agent_id,
            )

            # Remove originals
            for entry in member_entries:
                self._store.delete(entry.entry_id)

            merged += cluster.size

        return merged

    def _enforce_capacity(
        self,
        entries: List[MemoryEntry],
        rule: ConsolidationRule,
    ) -> int:
        """Drop lowest-value entries when over capacity"""
        if len(entries) <= rule.capacity_limit:
            return 0

        # Sort by effective importance (descending), drop extras
        sorted_entries = sorted(
            entries,
            key=lambda e: e.effective_importance(),
            reverse=True,
        )
        to_drop = sorted_entries[rule.capacity_limit:]
        for entry in to_drop:
            self._store.delete(entry.entry_id)
        return len(to_drop)

    # ──────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _compute_centroid(entries: List[MemoryEntry]) -> List[float]:
        if not entries:
            return []
        dim = len(entries[0].embedding)
        centroid = [0.0] * dim
        for entry in entries:
            for i, v in enumerate(entry.embedding):
                centroid[i] += v
        return [v / len(entries) for v in centroid]

    @staticmethod
    def _summarize_contents(contents: List[str]) -> str:
        """Create a simple extractive summary by picking the longest content"""
        if not contents:
            return ""
        if len(contents) == 1:
            return contents[0]
        # Pick the most representative (median length) content as base
        sorted_by_len = sorted(contents, key=len)
        median_idx = len(sorted_by_len) // 2
        base = sorted_by_len[median_idx]
        # Append unique key terms from other entries
        base_tokens = set(base.lower().split())
        extras: List[str] = []
        for content in contents:
            if content == base:
                continue
            unique = [t for t in content.split() if t.lower() not in base_tokens and len(t) > 4]
            if unique:
                extras.extend(unique[:3])
        if extras:
            return f"{base} [related: {', '.join(extras[:5])}]"
        return base
