"""
Context Manager — CognitionOS AI Intelligence Layer

Smart context window management for AI agent conversations:
- Token budget tracking and enforcement
- Message prioritization with relevance scoring
- Old message summarization and compression
- Sliding attention window
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SUMMARY = "summary"


class MessagePriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    EPHEMERAL = "ephemeral"


@dataclass
class ContextMessage:
    message_id: str
    role: MessageRole
    content: str
    priority: MessagePriority = MessagePriority.MEDIUM
    token_count: int = 0
    timestamp: float = field(default_factory=time.time)
    turn_number: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_summary: bool = False
    original_message_ids: List[str] = field(default_factory=list)
    relevance_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role.value,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "priority": self.priority.value,
            "token_count": self.token_count,
            "turn_number": self.turn_number,
            "is_summary": self.is_summary,
            "relevance_score": round(self.relevance_score, 2),
        }


@dataclass
class CompressionResult:
    original_tokens: int
    compressed_tokens: int
    messages_removed: int
    messages_summarized: int
    compression_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "messages_removed": self.messages_removed,
            "messages_summarized": self.messages_summarized,
            "compression_ratio": round(self.compression_ratio, 3),
            "savings_pct": round((1 - self.compression_ratio) * 100, 1),
        }


class ContextManager:
    """
    Smart context window manager for AI agent conversations.

    Manages conversation history within a token budget by prioritizing
    messages, compressing old context, and maintaining a sliding window.
    """

    CHARS_PER_TOKEN = 4

    def __init__(self, *, token_budget: int = 8192, attention_window: int = 10,
                 compression_threshold: float = 0.85, min_messages_to_keep: int = 4,
                 system_prompt: str = ""):
        self._token_budget = token_budget
        self._attention_window = attention_window
        self._compression_threshold = compression_threshold
        self._min_messages_to_keep = min_messages_to_keep
        self._messages: List[ContextMessage] = []
        self._turn_count = 0
        self._total_tokens_processed = 0
        self._compressions_performed = 0
        self._messages_evicted = 0

        if system_prompt:
            self.add_message(role=MessageRole.SYSTEM, content=system_prompt,
                             priority=MessagePriority.CRITICAL)

        logger.info("ContextManager initialized (budget=%d, window=%d)",
                     token_budget, attention_window)

    def add_message(self, role: MessageRole, content: str, *,
                    priority: Optional[MessagePriority] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> ContextMessage:
        if role == MessageRole.USER:
            self._turn_count += 1

        if priority is None:
            priority = {
                MessageRole.SYSTEM: MessagePriority.CRITICAL,
                MessageRole.USER: MessagePriority.HIGH,
                MessageRole.ASSISTANT: MessagePriority.MEDIUM,
                MessageRole.TOOL: MessagePriority.LOW,
                MessageRole.SUMMARY: MessagePriority.MEDIUM,
            }.get(role, MessagePriority.MEDIUM)

        token_count = self._estimate_tokens(content)
        msg_id = hashlib.md5(f"{role.value}{content[:50]}{time.time()}".encode()).hexdigest()[:12]

        message = ContextMessage(
            message_id=msg_id, role=role, content=content,
            priority=priority, token_count=token_count,
            turn_number=self._turn_count, metadata=metadata or {},
        )
        self._messages.append(message)
        self._total_tokens_processed += token_count

        if self.usage_ratio > self._compression_threshold:
            self.compress()

        return message

    def get_context_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": msg.role.value if msg.role != MessageRole.SUMMARY else "system",
             "content": msg.content}
            for msg in self._messages
        ]

    @property
    def total_tokens(self) -> int:
        return sum(m.token_count for m in self._messages)

    @property
    def remaining_budget(self) -> int:
        return max(0, self._token_budget - self.total_tokens)

    @property
    def usage_ratio(self) -> float:
        return self.total_tokens / self._token_budget if self._token_budget > 0 else 0

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def compress(self) -> CompressionResult:
        """Compress context by evicting ephemeral and summarizing old messages."""
        original_tokens = self.total_tokens

        critical, window, evictable = [], [], []
        for i, msg in enumerate(self._messages):
            if msg.priority == MessagePriority.CRITICAL:
                critical.append(msg)
            elif i >= len(self._messages) - self._attention_window:
                window.append(msg)
            else:
                evictable.append(msg)

        messages_removed = 0
        messages_summarized = 0
        summary_parts, summarized_ids = [], []

        remaining = []
        for msg in evictable:
            if msg.priority == MessagePriority.EPHEMERAL:
                messages_removed += 1
                self._messages_evicted += 1
            elif msg.priority in (MessagePriority.LOW, MessagePriority.MEDIUM):
                preview = msg.content[:150].replace("\n", " ")
                summary_parts.append(f"[Turn {msg.turn_number}, {msg.role.value}]: {preview}")
                summarized_ids.append(msg.message_id)
                messages_summarized += 1
                self._messages_evicted += 1
            else:
                remaining.append(msg)

        new_messages = critical[:]
        if summary_parts:
            summary_text = "## Previous Context Summary\n" + "\n".join(summary_parts[-20:])
            new_messages.append(ContextMessage(
                message_id=hashlib.md5(summary_text[:50].encode()).hexdigest()[:12],
                role=MessageRole.SUMMARY, content=summary_text,
                priority=MessagePriority.MEDIUM,
                token_count=self._estimate_tokens(summary_text),
                is_summary=True, original_message_ids=summarized_ids,
            ))
        new_messages.extend(remaining)
        new_messages.extend(window)
        self._messages = new_messages

        compressed_tokens = self.total_tokens
        ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
        self._compressions_performed += 1

        logger.info("Compressed: %d→%d tokens, %d removed, %d summarized",
                     original_tokens, compressed_tokens, messages_removed, messages_summarized)

        return CompressionResult(
            original_tokens=original_tokens, compressed_tokens=compressed_tokens,
            messages_removed=messages_removed, messages_summarized=messages_summarized,
            compression_ratio=ratio,
        )

    def update_relevance(self, query: str):
        """Update relevance scores based on query similarity."""
        query_tokens = set(query.lower().split())
        for msg in self._messages:
            if msg.priority == MessagePriority.CRITICAL:
                msg.relevance_score = 1.0
                continue
            content_tokens = set(msg.content.lower().split())
            if not content_tokens:
                msg.relevance_score = 0.1
                continue
            intersection = query_tokens & content_tokens
            union = query_tokens | content_tokens
            similarity = len(intersection) / len(union) if union else 0
            recency = max(0.1, 1.0 - (self._turn_count - msg.turn_number) / max(self._turn_count, 1))
            msg.relevance_score = similarity * 0.6 + recency * 0.4

    def can_fit(self, content: str) -> bool:
        return self._estimate_tokens(content) <= self.remaining_budget

    def set_budget(self, new_budget: int):
        self._token_budget = new_budget
        if self.usage_ratio > self._compression_threshold:
            self.compress()

    def get_stats(self) -> Dict[str, Any]:
        role_dist: Dict[str, int] = {}
        for msg in self._messages:
            role_dist[msg.role.value] = role_dist.get(msg.role.value, 0) + 1
        return {
            "total_messages": len(self._messages),
            "total_tokens": self.total_tokens,
            "token_budget": self._token_budget,
            "remaining_budget": self.remaining_budget,
            "usage_ratio": round(self.usage_ratio, 3),
            "turn_count": self._turn_count,
            "compressions_performed": self._compressions_performed,
            "messages_evicted_total": self._messages_evicted,
            "role_distribution": role_dist,
        }

    def __repr__(self) -> str:
        return (f"ContextManager(messages={len(self._messages)}, "
                f"tokens={self.total_tokens}/{self._token_budget})")
