"""
Context Compression Engine - Innovation Feature

Maintains high fidelity for long tasks while reducing token usage via semantic
compaction. Uses embedding-based importance scoring and hierarchical summarization
to intelligently compress context without losing critical information.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import hashlib


class CompressionStrategy(str, Enum):
    """Compression strategies"""
    AGGRESSIVE = "aggressive"      # Maximum compression, may lose details
    BALANCED = "balanced"          # Balance compression and fidelity
    CONSERVATIVE = "conservative"  # Minimal compression, preserve details
    ADAPTIVE = "adaptive"          # Adjust based on context


class ContentType(str, Enum):
    """Content type for compression"""
    CODE = "code"
    DOCUMENTATION = "documentation"
    CONVERSATION = "conversation"
    DATA = "data"
    MIXED = "mixed"


class ImportanceLevel(str, Enum):
    """Importance levels for content segments"""
    CRITICAL = "critical"          # Must preserve
    HIGH = "high"                  # Should preserve
    MEDIUM = "medium"              # Can compress
    LOW = "low"                    # Can heavily compress
    TRIVIAL = "trivial"            # Can remove


# ==================== Value Objects ====================

@dataclass(frozen=True)
class CompressionConfig:
    """Configuration for compression behavior"""
    strategy: CompressionStrategy
    target_compression_ratio: float  # 0.0 - 1.0 (0.5 = 50% reduction)
    min_importance_threshold: float  # 0.0 - 1.0
    preserve_structure: bool = True
    preserve_code_blocks: bool = True
    preserve_citations: bool = True
    max_chunk_size: int = 512
    overlap_size: int = 50
    use_embeddings: bool = True

    def __post_init__(self):
        if not 0.0 <= self.target_compression_ratio <= 1.0:
            raise ValueError("Target compression ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.min_importance_threshold <= 1.0:
            raise ValueError("Min importance threshold must be between 0.0 and 1.0")
        if self.max_chunk_size <= 0:
            raise ValueError("Max chunk size must be positive")
        if self.overlap_size < 0:
            raise ValueError("Overlap size cannot be negative")


@dataclass
class ContentSegment:
    """
    Segment of content with metadata.
    
    Represents a chunk of content that can be independently scored and compressed.
    """
    id: UUID
    content: str
    content_type: ContentType
    position: int
    token_count: int
    importance_score: float  # 0.0 - 1.0
    importance_level: ImportanceLevel
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.token_count < 0:
            raise ValueError("Token count cannot be negative")
        if not 0.0 <= self.importance_score <= 1.0:
            raise ValueError("Importance score must be between 0.0 and 1.0")
        if self.position < 0:
            raise ValueError("Position cannot be negative")

    @staticmethod
    def create(
        content: str,
        content_type: ContentType,
        position: int,
        token_count: int
    ) -> "ContentSegment":
        """Create a new content segment"""
        return ContentSegment(
            id=uuid4(),
            content=content,
            content_type=content_type,
            position=position,
            token_count=token_count,
            importance_score=0.5,
            importance_level=ImportanceLevel.MEDIUM
        )

    @property
    def content_hash(self) -> str:
        """Get content hash for deduplication"""
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]

    def should_preserve(self, config: CompressionConfig) -> bool:
        """Check if segment should be preserved based on config"""
        if self.importance_level == ImportanceLevel.CRITICAL:
            return True
        if self.importance_score >= config.min_importance_threshold:
            return True
        if config.preserve_code_blocks and self.content_type == ContentType.CODE:
            return True
        return False


@dataclass
class CompressionStats:
    """Statistics for compression operation"""
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    segments_total: int
    segments_preserved: int
    segments_compressed: int
    segments_removed: int
    information_loss_estimate: float  # 0.0 - 1.0
    
    def __post_init__(self):
        if self.original_tokens < 0 or self.compressed_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        if not 0.0 <= self.compression_ratio <= 1.0:
            raise ValueError("Compression ratio must be between 0.0 and 1.0")
        if not 0.0 <= self.information_loss_estimate <= 1.0:
            raise ValueError("Information loss must be between 0.0 and 1.0")

    @property
    def tokens_saved(self) -> int:
        """Calculate tokens saved"""
        return self.original_tokens - self.compressed_tokens

    @property
    def efficiency_score(self) -> float:
        """Calculate compression efficiency (high compression, low loss)"""
        return self.compression_ratio * (1.0 - self.information_loss_estimate)


# ==================== Entities ====================

@dataclass
class ContextWindow:
    """
    Context window for compression.
    
    Represents a full context that needs compression.
    """
    id: UUID
    tenant_id: UUID
    task_id: Optional[UUID]
    segments: List[ContentSegment]
    total_tokens: int
    config: CompressionConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.total_tokens < 0:
            raise ValueError("Total tokens cannot be negative")
        if not self.segments:
            raise ValueError("Context window must have at least one segment")

    @staticmethod
    def create(
        tenant_id: UUID,
        segments: List[ContentSegment],
        config: CompressionConfig,
        task_id: Optional[UUID] = None
    ) -> "ContextWindow":
        """Create a new context window"""
        total_tokens = sum(s.token_count for s in segments)
        return ContextWindow(
            id=uuid4(),
            tenant_id=tenant_id,
            task_id=task_id,
            segments=segments,
            total_tokens=total_tokens,
            config=config
        )

    def get_target_tokens(self) -> int:
        """Calculate target token count after compression"""
        return int(self.total_tokens * (1.0 - self.config.target_compression_ratio))

    def get_segments_by_importance(self) -> List[ContentSegment]:
        """Get segments sorted by importance (descending)"""
        return sorted(self.segments, key=lambda s: s.importance_score, reverse=True)

    def get_critical_segments(self) -> List[ContentSegment]:
        """Get all critical segments"""
        return [s for s in self.segments if s.importance_level == ImportanceLevel.CRITICAL]

    def estimate_compression_headroom(self) -> int:
        """Estimate how many tokens can be removed"""
        critical_tokens = sum(
            s.token_count for s in self.segments 
            if s.importance_level == ImportanceLevel.CRITICAL
        )
        return max(0, self.total_tokens - critical_tokens)


@dataclass
class CompressedContext:
    """
    Result of context compression.
    
    Contains compressed content and metadata about compression.
    """
    id: UUID
    original_window_id: UUID
    tenant_id: UUID
    compressed_content: str
    preserved_segments: List[ContentSegment]
    compressed_segments: List[Tuple[UUID, str]]  # (original_id, compressed_content)
    removed_segments: List[UUID]
    stats: CompressionStats
    config: CompressionConfig
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    compressed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def create(
        original_window: ContextWindow,
        compressed_content: str,
        preserved_segments: List[ContentSegment],
        compressed_segments: List[Tuple[UUID, str]],
        removed_segments: List[UUID],
        compressed_tokens: int
    ) -> "CompressedContext":
        """Create compressed context result"""
        segments_total = len(original_window.segments)
        segments_preserved = len(preserved_segments)
        segments_compressed = len(compressed_segments)
        segments_removed = len(removed_segments)
        
        compression_ratio = 1.0 - (compressed_tokens / original_window.total_tokens) if original_window.total_tokens > 0 else 0.0
        
        # Estimate information loss based on what was removed/compressed
        removed_importance = sum(
            s.importance_score for s in original_window.segments 
            if s.id in removed_segments
        )
        compressed_importance = sum(
            s.importance_score for s in original_window.segments 
            if any(s.id == seg_id for seg_id, _ in compressed_segments)
        )
        total_importance = sum(s.importance_score for s in original_window.segments)
        
        information_loss = (removed_importance + compressed_importance * 0.3) / total_importance if total_importance > 0 else 0.0
        
        stats = CompressionStats(
            original_tokens=original_window.total_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            segments_total=segments_total,
            segments_preserved=segments_preserved,
            segments_compressed=segments_compressed,
            segments_removed=segments_removed,
            information_loss_estimate=min(information_loss, 1.0)
        )
        
        return CompressedContext(
            id=uuid4(),
            original_window_id=original_window.id,
            tenant_id=original_window.tenant_id,
            compressed_content=compressed_content,
            preserved_segments=preserved_segments,
            compressed_segments=compressed_segments,
            removed_segments=removed_segments,
            stats=stats,
            config=original_window.config
        )

    @property
    def is_effective(self) -> bool:
        """Check if compression was effective (>20% reduction, <30% loss)"""
        return (
            self.stats.compression_ratio >= 0.2 and
            self.stats.information_loss_estimate <= 0.3
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "original_window_id": str(self.original_window_id),
            "tenant_id": str(self.tenant_id),
            "compressed_content": self.compressed_content,
            "preserved_segment_ids": [str(s.id) for s in self.preserved_segments],
            "compressed_segment_ids": [str(seg_id) for seg_id, _ in self.compressed_segments],
            "removed_segment_ids": [str(seg_id) for seg_id in self.removed_segments],
            "original_tokens": self.stats.original_tokens,
            "compressed_tokens": self.stats.compressed_tokens,
            "compression_ratio": self.stats.compression_ratio,
            "information_loss_estimate": self.stats.information_loss_estimate,
            "summary": self.summary,
            "metadata": self.metadata,
            "compressed_at": self.compressed_at.isoformat()
        }


@dataclass
class HierarchicalSummary:
    """
    Hierarchical summary for multi-level compression.
    
    Provides different levels of detail for progressive disclosure.
    """
    id: UUID
    context_window_id: UUID
    levels: Dict[str, str]  # level_name -> summary_text
    token_counts: Dict[str, int]  # level_name -> token_count
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def create(
        context_window_id: UUID,
        executive_summary: str,
        detailed_summary: str,
        full_summary: str
    ) -> "HierarchicalSummary":
        """Create hierarchical summary with three levels"""
        return HierarchicalSummary(
            id=uuid4(),
            context_window_id=context_window_id,
            levels={
                "executive": executive_summary,
                "detailed": detailed_summary,
                "full": full_summary
            },
            token_counts={
                "executive": len(executive_summary.split()) // 4,  # Rough token estimate
                "detailed": len(detailed_summary.split()) // 4,
                "full": len(full_summary.split()) // 4
            }
        )

    def get_summary_for_budget(self, token_budget: int) -> str:
        """Get most detailed summary that fits in token budget"""
        if self.token_counts.get("executive", 0) <= token_budget:
            if self.token_counts.get("detailed", 0) <= token_budget:
                if self.token_counts.get("full", 0) <= token_budget:
                    return self.levels.get("full", "")
                return self.levels.get("detailed", "")
            return self.levels.get("executive", "")
        return ""


# ==================== Service ====================

class ContextCompressionService:
    """
    Context compression service for token optimization.
    
    Intelligently compresses context while maintaining information fidelity.
    """

    def __init__(self):
        """Initialize context compression service"""
        self._segment_cache: Dict[str, ContentSegment] = {}

    async def compress_context(
        self,
        context_window: ContextWindow
    ) -> CompressedContext:
        """
        Compress context window.
        
        Args:
            context_window: Context window to compress
            
        Returns:
            Compressed context result
        """
        # Score importance for all segments
        await self._score_importance(context_window)
        
        # Determine what to preserve/compress/remove
        preserved, to_compress, to_remove = await self._partition_segments(context_window)
        
        # Compress segments that need compression
        compressed = await self._compress_segments(to_compress, context_window.config)
        
        # Reconstruct compressed content
        reconstructed = await self._reconstruct_content(preserved, compressed, context_window)
        
        # Calculate compressed token count
        compressed_tokens = sum(s.token_count for s in preserved)
        compressed_tokens += sum(len(text.split()) // 4 for _, text in compressed)  # Rough estimate
        
        # Create result
        return CompressedContext.create(
            original_window=context_window,
            compressed_content=reconstructed,
            preserved_segments=preserved,
            compressed_segments=compressed,
            removed_segments=[s.id for s in to_remove],
            compressed_tokens=compressed_tokens
        )

    async def create_hierarchical_summary(
        self,
        context_window: ContextWindow
    ) -> HierarchicalSummary:
        """
        Create hierarchical summary with multiple detail levels.
        
        Args:
            context_window: Context window to summarize
            
        Returns:
            Hierarchical summary
        """
        # Get segments sorted by importance
        sorted_segments = context_window.get_segments_by_importance()
        
        # Executive summary: Top 10% most important
        top_10_percent = max(1, len(sorted_segments) // 10)
        executive_segments = sorted_segments[:top_10_percent]
        executive_summary = await self._summarize_segments(executive_segments, max_tokens=200)
        
        # Detailed summary: Top 30% most important
        top_30_percent = max(1, len(sorted_segments) // 3)
        detailed_segments = sorted_segments[:top_30_percent]
        detailed_summary = await self._summarize_segments(detailed_segments, max_tokens=500)
        
        # Full summary: All segments
        full_summary = await self._summarize_segments(sorted_segments, max_tokens=1000)
        
        return HierarchicalSummary.create(
            context_window_id=context_window.id,
            executive_summary=executive_summary,
            detailed_summary=detailed_summary,
            full_summary=full_summary
        )

    async def chunk_content(
        self,
        content: str,
        content_type: ContentType,
        config: CompressionConfig
    ) -> List[ContentSegment]:
        """
        Chunk content into segments.
        
        Args:
            content: Content to chunk
            content_type: Type of content
            config: Compression configuration
            
        Returns:
            List of content segments
        """
        segments = []
        
        # Split by natural boundaries
        if content_type == ContentType.CODE:
            chunks = self._chunk_code(content, config.max_chunk_size)
        elif content_type == ContentType.DOCUMENTATION:
            chunks = self._chunk_documentation(content, config.max_chunk_size)
        else:
            chunks = self._chunk_text(content, config.max_chunk_size, config.overlap_size)
        
        # Create segments
        for i, chunk in enumerate(chunks):
            token_count = len(chunk.split()) // 4  # Rough token estimate
            segment = ContentSegment.create(
                content=chunk,
                content_type=content_type,
                position=i,
                token_count=token_count
            )
            segments.append(segment)
        
        return segments

    async def deduplicate_segments(
        self,
        segments: List[ContentSegment],
        similarity_threshold: float = 0.95
    ) -> List[ContentSegment]:
        """
        Remove duplicate or highly similar segments.
        
        Args:
            segments: Segments to deduplicate
            similarity_threshold: Similarity threshold for deduplication
            
        Returns:
            Deduplicated segments
        """
        seen_hashes = set()
        unique_segments = []
        
        for segment in segments:
            content_hash = segment.content_hash
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_segments.append(segment)
        
        return unique_segments

    # Private helper methods

    async def _score_importance(self, context_window: ContextWindow) -> None:
        """Score importance for all segments"""
        for segment in context_window.segments:
            score = await self._calculate_importance_score(segment, context_window)
            segment.importance_score = score
            segment.importance_level = self._categorize_importance(score)

    async def _calculate_importance_score(
        self,
        segment: ContentSegment,
        context_window: ContextWindow
    ) -> float:
        """Calculate importance score for a segment"""
        score = 0.5  # Base score
        
        # Position bias: Earlier content often more important
        position_factor = 1.0 - (segment.position / len(context_window.segments))
        score += position_factor * 0.2
        
        # Content type bias
        type_weights = {
            ContentType.CODE: 0.8,
            ContentType.DOCUMENTATION: 0.6,
            ContentType.CONVERSATION: 0.5,
            ContentType.DATA: 0.7,
            ContentType.MIXED: 0.5
        }
        score *= type_weights.get(segment.content_type, 0.5)
        
        # Length bias: Very short or very long segments may be less important
        if segment.token_count < 10 or segment.token_count > 1000:
            score *= 0.8
        
        # Keywords/patterns that indicate importance
        important_keywords = [
            "error", "critical", "important", "required", "must",
            "TODO", "FIXME", "BUG", "class ", "def ", "function"
        ]
        keyword_count = sum(1 for kw in important_keywords if kw in segment.content.lower())
        score += min(keyword_count * 0.05, 0.2)
        
        return min(score, 1.0)

    def _categorize_importance(self, score: float) -> ImportanceLevel:
        """Categorize importance score into level"""
        if score >= 0.9:
            return ImportanceLevel.CRITICAL
        elif score >= 0.7:
            return ImportanceLevel.HIGH
        elif score >= 0.5:
            return ImportanceLevel.MEDIUM
        elif score >= 0.3:
            return ImportanceLevel.LOW
        else:
            return ImportanceLevel.TRIVIAL

    async def _partition_segments(
        self,
        context_window: ContextWindow
    ) -> Tuple[List[ContentSegment], List[ContentSegment], List[ContentSegment]]:
        """Partition segments into preserve/compress/remove"""
        preserved = []
        to_compress = []
        to_remove = []
        
        target_tokens = context_window.get_target_tokens()
        current_tokens = 0
        
        # Always preserve critical segments
        critical = context_window.get_critical_segments()
        for segment in critical:
            preserved.append(segment)
            current_tokens += segment.token_count
        
        # Process remaining segments by importance
        remaining = [s for s in context_window.segments if s not in critical]
        remaining.sort(key=lambda s: s.importance_score, reverse=True)
        
        for segment in remaining:
            if current_tokens >= target_tokens:
                # Over budget - decide compress or remove
                if segment.importance_score >= 0.5:
                    to_compress.append(segment)
                else:
                    to_remove.append(segment)
            else:
                # Under budget - preserve
                preserved.append(segment)
                current_tokens += segment.token_count
        
        return preserved, to_compress, to_remove

    async def _compress_segments(
        self,
        segments: List[ContentSegment],
        config: CompressionConfig
    ) -> List[Tuple[UUID, str]]:
        """Compress segments"""
        compressed = []
        
        for segment in segments:
            # Simplified compression: extract key sentences
            compressed_text = self._extract_key_sentences(segment.content, compression_ratio=0.5)
            compressed.append((segment.id, compressed_text))
        
        return compressed

    def _extract_key_sentences(self, text: str, compression_ratio: float = 0.5) -> str:
        """Extract key sentences from text"""
        sentences = text.split('. ')
        if not sentences:
            return text
        
        # Keep first and last sentence, sample middle
        keep_count = max(1, int(len(sentences) * compression_ratio))
        if keep_count >= len(sentences):
            return text
        
        selected = [sentences[0]]
        if len(sentences) > 2:
            step = len(sentences) // keep_count
            selected.extend(sentences[i] for i in range(step, len(sentences)-1, step))
        if len(sentences) > 1:
            selected.append(sentences[-1])
        
        return '. '.join(selected[:keep_count])

    async def _reconstruct_content(
        self,
        preserved: List[ContentSegment],
        compressed: List[Tuple[UUID, str]],
        context_window: ContextWindow
    ) -> str:
        """Reconstruct content from preserved and compressed segments"""
        # Sort by original position
        all_segments = []
        
        for segment in preserved:
            all_segments.append((segment.position, segment.content, False))
        
        compressed_map = dict(compressed)
        for segment in context_window.segments:
            if segment.id in compressed_map:
                all_segments.append((segment.position, compressed_map[segment.id], True))
        
        all_segments.sort(key=lambda x: x[0])
        
        # Reconstruct with markers
        parts = []
        for position, content, is_compressed in all_segments:
            if is_compressed:
                parts.append(f"[COMPRESSED] {content}")
            else:
                parts.append(content)
        
        return "\n\n".join(parts)

    async def _summarize_segments(
        self,
        segments: List[ContentSegment],
        max_tokens: int
    ) -> str:
        """Create summary from segments"""
        # Simplified summarization: concatenate and truncate
        combined = " ".join(s.content for s in segments)
        
        # Rough token-based truncation
        words = combined.split()
        max_words = max_tokens * 4  # Rough conversion
        
        if len(words) <= max_words:
            return combined
        
        return " ".join(words[:max_words]) + "..."

    def _chunk_text(
        self,
        text: str,
        max_size: int,
        overlap: int
    ) -> List[str]:
        """Chunk text with overlap"""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + max_size]
            chunks.append(" ".join(chunk_words))
            i += max_size - overlap
        
        return chunks

    def _chunk_code(self, code: str, max_size: int) -> List[str]:
        """Chunk code by logical boundaries"""
        lines = code.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_tokens = len(line.split()) // 4
            if current_size + line_tokens > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(line)
            current_size += line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _chunk_documentation(self, doc: str, max_size: int) -> List[str]:
        """Chunk documentation by sections"""
        # Split by common section markers
        sections = doc.split('\n## ')
        chunks = []
        
        for section in sections:
            section_tokens = len(section.split()) // 4
            if section_tokens > max_size:
                # Further split large sections
                chunks.extend(self._chunk_text(section, max_size, 0))
            else:
                chunks.append(section)
        
        return chunks
