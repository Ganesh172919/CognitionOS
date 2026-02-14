"""
Checkpoint Domain - Events

Domain events for checkpoint/resume operations.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass(frozen=True)
class CheckpointEvent:
    """Base checkpoint domain event"""
    checkpoint_id: UUID
    workflow_execution_id: UUID
    occurred_at: datetime
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class CheckpointCreated(CheckpointEvent):
    """Event raised when a checkpoint is created"""
    checkpoint_number: int
    completion_percentage: float
    budget_consumed: float

    @classmethod
    def create(
        cls,
        checkpoint_id: UUID,
        workflow_execution_id: UUID,
        checkpoint_number: int,
        completion_percentage: float,
        budget_consumed: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointCreated":
        """Factory method to create event"""
        return cls(
            checkpoint_id=checkpoint_id,
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            completion_percentage=completion_percentage,
            budget_consumed=budget_consumed,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class CheckpointRestored(CheckpointEvent):
    """Event raised when a checkpoint is restored"""
    checkpoint_number: int
    recovery_reason: str

    @classmethod
    def create(
        cls,
        checkpoint_id: UUID,
        workflow_execution_id: UUID,
        checkpoint_number: int,
        recovery_reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointRestored":
        """Factory method to create event"""
        return cls(
            checkpoint_id=checkpoint_id,
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            recovery_reason=recovery_reason,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class CheckpointDeleted(CheckpointEvent):
    """Event raised when a checkpoint is deleted"""
    checkpoint_number: int
    deletion_reason: str

    @classmethod
    def create(
        cls,
        checkpoint_id: UUID,
        workflow_execution_id: UUID,
        checkpoint_number: int,
        deletion_reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointDeleted":
        """Factory method to create event"""
        return cls(
            checkpoint_id=checkpoint_id,
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            deletion_reason=deletion_reason,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class CheckpointCompressionCompleted(CheckpointEvent):
    """Event raised when checkpoint compression completes"""
    checkpoint_number: int
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float

    @classmethod
    def create(
        cls,
        checkpoint_id: UUID,
        workflow_execution_id: UUID,
        checkpoint_number: int,
        original_size_bytes: int,
        compressed_size_bytes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointCompressionCompleted":
        """Factory method to create event"""
        compression_ratio = (
            compressed_size_bytes / original_size_bytes
            if original_size_bytes > 0
            else 0.0
        )
        return cls(
            checkpoint_id=checkpoint_id,
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            original_size_bytes=original_size_bytes,
            compressed_size_bytes=compressed_size_bytes,
            compression_ratio=compression_ratio,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class CheckpointFailed(CheckpointEvent):
    """Event raised when checkpoint creation or restoration fails"""
    checkpoint_number: int
    error_message: str
    error_type: str

    @classmethod
    def create(
        cls,
        checkpoint_id: UUID,
        workflow_execution_id: UUID,
        checkpoint_number: int,
        error_message: str,
        error_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointFailed":
        """Factory method to create event"""
        return cls(
            checkpoint_id=checkpoint_id,
            workflow_execution_id=workflow_execution_id,
            checkpoint_number=checkpoint_number,
            error_message=error_message,
            error_type=error_type,
            occurred_at=datetime.utcnow(),
            metadata=metadata or {},
        )
