"""
Context Management System with Memory Hierarchy

Advanced context management for autonomous agents with:
- Multi-tier memory (working, short-term, long-term)
- Intelligent context compression
- Relevance scoring
- Memory consolidation
- Context window optimization
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, field
import hashlib
import json

logger = logging.getLogger(__name__)


class MemoryTier(str, Enum):
    """Memory hierarchy tiers."""
    WORKING = "working"  # Active context, immediately accessible
    SHORT_TERM = "short_term"  # Recent context, cached
    LONG_TERM = "long_term"  # Persistent context, vector indexed


class MemoryImportance(str, Enum):
    """Memory importance levels."""
    CRITICAL = "critical"  # Core facts, never forget
    HIGH = "high"  # Important context
    MEDIUM = "medium"  # Useful context
    LOW = "low"  # Optional context
    TRIVIAL = "trivial"  # Can be discarded


@dataclass
class MemoryEntry:
    """A single memory entry."""
    entry_id: str
    content: str
    tier: MemoryTier
    importance: MemoryImportance
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[List[float]] = None
    
    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1
    
    def calculate_decay_factor(self) -> float:
        """Calculate memory decay based on age and access."""
        age_hours = (datetime.utcnow() - self.last_accessed).total_seconds() / 3600
        
        # Decay formula: importance * (1 / (1 + age))
        importance_weight = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TRIVIAL: 0.2,
        }
        
        base_importance = importance_weight[self.importance]
        decay = 1.0 / (1.0 + age_hours / 24)  # Decay over days
        
        # Access frequency boosts retention
        access_boost = min(1.0, self.access_count / 10)
        
        return base_importance * decay * (1.0 + access_boost)


@dataclass
class ContextWindow:
    """Context window for agent execution."""
    max_tokens: int = 8000
    current_tokens: int = 0
    entries: List[MemoryEntry] = field(default_factory=list)
    
    def can_fit(self, entry: MemoryEntry, tokens_per_char: float = 0.25) -> bool:
        """Check if entry can fit in context window."""
        estimated_tokens = len(entry.content) * tokens_per_char
        return (self.current_tokens + estimated_tokens) <= self.max_tokens
    
    def add_entry(self, entry: MemoryEntry, tokens_per_char: float = 0.25):
        """Add entry to context window."""
        estimated_tokens = len(entry.content) * tokens_per_char
        self.entries.append(entry)
        self.current_tokens += int(estimated_tokens)
    
    def to_text(self) -> str:
        """Convert context window to text."""
        sections = []
        
        # Group by importance
        by_importance = {}
        for entry in self.entries:
            if entry.importance not in by_importance:
                by_importance[entry.importance] = []
            by_importance[entry.importance].append(entry)
        
        # Format with hierarchy
        for importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH, 
                          MemoryImportance.MEDIUM, MemoryImportance.LOW]:
            if importance in by_importance:
                sections.append(f"\n[{importance.upper()}]")
                for entry in by_importance[importance]:
                    sections.append(f"- {entry.content}")
        
        return "\n".join(sections)


class ContextManager:
    """
    Advanced context management system with memory hierarchy.
    
    Features:
    - Multi-tier memory architecture
    - Intelligent context window optimization
    - Automatic memory consolidation
    - Relevance-based retrieval
    - Memory decay and pruning
    """
    
    def __init__(
        self,
        max_working_memory: int = 10,
        max_short_term_memory: int = 100,
        memory_service: Any = None,
        embedding_service: Any = None,
    ):
        """
        Initialize context manager.
        
        Args:
            max_working_memory: Maximum entries in working memory
            max_short_term_memory: Maximum entries in short-term memory
            memory_service: Long-term memory persistence service
            embedding_service: Service for generating embeddings
        """
        self.max_working_memory = max_working_memory
        self.max_short_term_memory = max_short_term_memory
        self.memory_service = memory_service
        self.embedding_service = embedding_service
        
        # Memory stores
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.short_term_memory: Dict[str, MemoryEntry] = {}
        
        logger.info("Context manager initialized")
    
    async def add_memory(
        self,
        content: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add new memory entry.
        
        Automatically places in appropriate tier based on importance.
        
        Args:
            content: Memory content
            importance: Importance level
            metadata: Optional metadata
            
        Returns:
            Entry ID
        """
        entry_id = self._generate_entry_id(content)
        
        # Check for duplicate
        if entry_id in self.working_memory or entry_id in self.short_term_memory:
            logger.debug(f"Duplicate memory entry: {entry_id}")
            return entry_id
        
        # Generate embeddings for long-term storage
        embeddings = None
        if self.embedding_service and importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
            embeddings = await self.embedding_service.generate(content)
        
        # Create entry
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            tier=MemoryTier.WORKING,
            importance=importance,
            created_at=datetime.utcnow(),
            last_accessed=datetime.utcnow(),
            metadata=metadata or {},
            embeddings=embeddings,
        )
        
        # Add to working memory
        self.working_memory[entry_id] = entry
        
        # Trigger consolidation if needed
        if len(self.working_memory) > self.max_working_memory:
            await self._consolidate_working_memory()
        
        logger.info(f"Added memory entry: {entry_id} (importance={importance})")
        
        return entry_id
    
    async def retrieve_relevant_context(
        self,
        query: str,
        max_entries: int = 10,
        min_relevance: float = 0.5,
    ) -> List[MemoryEntry]:
        """
        Retrieve most relevant context for a query.
        
        Searches across all memory tiers and ranks by relevance.
        
        Args:
            query: Query text
            max_entries: Maximum entries to return
            min_relevance: Minimum relevance threshold
            
        Returns:
            List of relevant memory entries
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        candidates = []
        
        # Search working memory (always included)
        for entry in self.working_memory.values():
            entry.update_access()
            relevance = self._calculate_relevance(query, entry)
            if relevance >= min_relevance:
                entry.relevance_score = relevance
                candidates.append(entry)
        
        # Search short-term memory
        for entry in self.short_term_memory.values():
            entry.update_access()
            relevance = self._calculate_relevance(query, entry)
            if relevance >= min_relevance:
                entry.relevance_score = relevance
                candidates.append(entry)
        
        # Search long-term memory (vector search)
        if self.memory_service:
            long_term_entries = await self.memory_service.retrieve(
                query=query,
                k=max_entries,
            )
            
            for entry_data in long_term_entries:
                entry = self._memory_data_to_entry(entry_data, MemoryTier.LONG_TERM)
                relevance = entry_data.get("relevance", 0.0)
                if relevance >= min_relevance:
                    entry.relevance_score = relevance
                    candidates.append(entry)
        
        # Sort by relevance and importance
        candidates.sort(
            key=lambda e: (e.relevance_score * 0.7 + self._get_importance_weight(e.importance) * 0.3),
            reverse=True
        )
        
        # Return top entries
        result = candidates[:max_entries]
        
        logger.info(f"Retrieved {len(result)} relevant entries")
        
        return result
    
    async def build_context_window(
        self,
        query: str,
        max_tokens: int = 8000,
    ) -> ContextWindow:
        """
        Build optimized context window for query.
        
        Intelligently selects and orders context to fit within token limit.
        
        Args:
            query: Query to build context for
            max_tokens: Maximum tokens allowed
            
        Returns:
            Optimized context window
        """
        window = ContextWindow(max_tokens=max_tokens)
        
        # Retrieve relevant entries
        relevant_entries = await self.retrieve_relevant_context(
            query=query,
            max_entries=50,  # Get more candidates than we'll use
        )
        
        # Add entries in priority order until we hit limit
        for entry in relevant_entries:
            if window.can_fit(entry):
                window.add_entry(entry)
            else:
                # Try to compress less important entries
                if entry.importance not in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
                    compressed = await self._compress_entry(entry)
                    if window.can_fit(compressed):
                        window.add_entry(compressed)
        
        logger.info(f"Built context window: {window.current_tokens}/{max_tokens} tokens, "
                   f"{len(window.entries)} entries")
        
        return window
    
    async def consolidate_memory(self):
        """
        Consolidate memory across tiers.
        
        Moves entries between tiers based on access patterns and importance.
        """
        logger.info("Starting memory consolidation")
        
        # Consolidate working memory
        await self._consolidate_working_memory()
        
        # Consolidate short-term memory
        await self._consolidate_short_term_memory()
        
        # Prune old low-importance entries
        await self._prune_memory()
        
        logger.info("Memory consolidation complete")
    
    async def _consolidate_working_memory(self):
        """Move entries from working to short-term memory."""
        if len(self.working_memory) <= self.max_working_memory:
            return
        
        # Get entries sorted by decay factor
        entries = list(self.working_memory.values())
        entries.sort(key=lambda e: e.calculate_decay_factor())
        
        # Keep most recent/important in working memory
        to_keep = entries[-self.max_working_memory:]
        to_move = entries[:-self.max_working_memory]
        
        # Move to short-term
        for entry in to_move:
            entry.tier = MemoryTier.SHORT_TERM
            self.short_term_memory[entry.entry_id] = entry
            del self.working_memory[entry.entry_id]
        
        logger.info(f"Moved {len(to_move)} entries from working to short-term memory")
    
    async def _consolidate_short_term_memory(self):
        """Move entries from short-term to long-term storage."""
        if len(self.short_term_memory) <= self.max_short_term_memory:
            return
        
        # Get entries sorted by decay factor
        entries = list(self.short_term_memory.values())
        entries.sort(key=lambda e: e.calculate_decay_factor())
        
        # Keep most recent/important in short-term
        to_keep = entries[-self.max_short_term_memory:]
        to_move = entries[:-self.max_short_term_memory]
        
        # Move to long-term storage
        if self.memory_service:
            for entry in to_move:
                if entry.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
                    await self._store_in_long_term(entry)
                
                # Remove from short-term
                del self.short_term_memory[entry.entry_id]
        
        logger.info(f"Moved {len(to_move)} entries from short-term to long-term memory")
    
    async def _store_in_long_term(self, entry: MemoryEntry):
        """Store entry in long-term memory service."""
        if not self.memory_service:
            return
        
        await self.memory_service.store(
            user_id=entry.metadata.get("user_id", "system"),
            content=entry.content,
            memory_type="contextual",
            importance=entry.importance.value,
            metadata={
                **entry.metadata,
                "access_count": entry.access_count,
                "created_at": entry.created_at.isoformat(),
            },
            embeddings=entry.embeddings,
        )
    
    async def _prune_memory(self):
        """Remove old low-importance entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        # Prune short-term memory
        to_remove = []
        for entry_id, entry in self.short_term_memory.items():
            if (entry.last_accessed < cutoff_date and 
                entry.importance in [MemoryImportance.LOW, MemoryImportance.TRIVIAL]):
                to_remove.append(entry_id)
        
        for entry_id in to_remove:
            del self.short_term_memory[entry_id]
        
        if to_remove:
            logger.info(f"Pruned {len(to_remove)} old entries from short-term memory")
    
    def _calculate_relevance(self, query: str, entry: MemoryEntry) -> float:
        """
        Calculate relevance score between query and entry.
        
        Uses simple text matching. Production would use embeddings.
        """
        query_lower = query.lower()
        content_lower = entry.content.lower()
        
        # Simple keyword matching
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        relevance = overlap / len(query_words)
        
        # Boost for exact matches
        if query_lower in content_lower:
            relevance += 0.3
        
        return min(1.0, relevance)
    
    async def _compress_entry(self, entry: MemoryEntry) -> MemoryEntry:
        """Compress entry content to save tokens."""
        # Simple compression - take first sentences
        sentences = entry.content.split('. ')
        compressed_content = '. '.join(sentences[:2]) + '...'
        
        return MemoryEntry(
            entry_id=entry.entry_id,
            content=compressed_content,
            tier=entry.tier,
            importance=entry.importance,
            created_at=entry.created_at,
            last_accessed=entry.last_accessed,
            access_count=entry.access_count,
            relevance_score=entry.relevance_score,
            metadata={**entry.metadata, "compressed": True},
        )
    
    def _generate_entry_id(self, content: str) -> str:
        """Generate unique entry ID from content."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_importance_weight(self, importance: MemoryImportance) -> float:
        """Get numerical weight for importance level."""
        weights = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TRIVIAL: 0.2,
        }
        return weights.get(importance, 0.5)
    
    def _memory_data_to_entry(
        self,
        data: Dict[str, Any],
        tier: MemoryTier,
    ) -> MemoryEntry:
        """Convert memory service data to MemoryEntry."""
        return MemoryEntry(
            entry_id=data.get("id", str(uuid4())),
            content=data.get("content", ""),
            tier=tier,
            importance=MemoryImportance(data.get("importance", "medium")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            last_accessed=datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "working_memory": {
                "count": len(self.working_memory),
                "max": self.max_working_memory,
                "utilization": len(self.working_memory) / self.max_working_memory,
            },
            "short_term_memory": {
                "count": len(self.short_term_memory),
                "max": self.max_short_term_memory,
                "utilization": len(self.short_term_memory) / self.max_short_term_memory,
            },
            "total_entries": len(self.working_memory) + len(self.short_term_memory),
        }
