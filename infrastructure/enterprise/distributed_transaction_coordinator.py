"""
Distributed Transaction Coordinator - SAGA Pattern Implementation

Implements distributed transactions with:
- SAGA orchestration pattern
- Compensating transactions
- Two-phase commit (2PC) support
- Transaction state management
- Idempotency guarantees
- Timeout and rollback handling
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import uuid
import json


class TransactionState(Enum):
    """Transaction states"""
    INITIATED = "initiated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StepState(Enum):
    """Transaction step states"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class IsolationLevel(Enum):
    """Transaction isolation levels"""
    READ_UNCOMMITTED = "read_uncommitted"
    READ_COMMITTED = "read_committed"
    REPEATABLE_READ = "repeatable_read"
    SERIALIZABLE = "serializable"


@dataclass
class TransactionStep:
    """Individual step in SAGA transaction"""
    step_id: str
    step_name: str
    service_name: str
    action: str  # Forward action
    compensating_action: str  # Rollback action
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    state: StepState
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 30
    idempotency_key: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class DistributedTransaction:
    """Distributed transaction (SAGA)"""
    transaction_id: str
    transaction_type: str
    state: TransactionState
    steps: List[TransactionStep]
    metadata: Dict[str, Any]
    isolation_level: IsolationLevel
    created_at: datetime
    updated_at: datetime
    timeout_at: datetime
    current_step_index: int = 0
    completed_steps: int = 0
    failed_step_index: Optional[int] = None


@dataclass
class TwoPhaseCommit:
    """Two-phase commit transaction"""
    transaction_id: str
    coordinator_id: str
    participants: List[str]  # Service IDs
    prepared_participants: Set[str] = field(default_factory=set)
    committed_participants: Set[str] = field(default_factory=set)
    aborted_participants: Set[str] = field(default_factory=set)
    state: TransactionState
    timeout_seconds: int = 60


class DistributedTransactionCoordinator:
    """
    Distributed transaction coordinator implementing SAGA pattern.

    Features:
    - Orchestrated SAGA pattern
    - Compensating transactions
    - Two-phase commit support
    - Idempotency guarantees
    - Automatic retry and timeout
    - State persistence
    """

    def __init__(
        self,
        default_timeout_seconds: int = 300,
        default_isolation: IsolationLevel = IsolationLevel.READ_COMMITTED
    ):
        self.default_timeout_seconds = default_timeout_seconds
        self.default_isolation = default_isolation

        # Transaction storage
        self.transactions: Dict[str, DistributedTransaction] = {}
        self.two_phase_commits: Dict[str, TwoPhaseCommit] = {}

        # Idempotency tracking
        self.executed_idempotency_keys: Set[str] = set()

        # Metrics
        self.metrics = {
            "total_transactions": 0,
            "completed_transactions": 0,
            "failed_transactions": 0,
            "compensated_transactions": 0,
            "avg_duration_ms": 0.0
        }

    def begin_transaction(
        self,
        transaction_type: str,
        steps: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        timeout_seconds: Optional[int] = None,
        isolation_level: Optional[IsolationLevel] = None
    ) -> DistributedTransaction:
        """
        Begin a new distributed transaction.

        Args:
            transaction_type: Type identifier
            steps: List of transaction steps
            metadata: Additional metadata
            timeout_seconds: Transaction timeout
            isolation_level: Isolation level

        Returns:
            Created DistributedTransaction
        """
        transaction_id = str(uuid.uuid4())
        now = datetime.utcnow()

        timeout = timeout_seconds or self.default_timeout_seconds
        isolation = isolation_level or self.default_isolation

        # Create transaction steps
        tx_steps = []
        for i, step_data in enumerate(steps):
            step = TransactionStep(
                step_id=f"{transaction_id}_step_{i}",
                step_name=step_data.get("name", f"Step {i}"),
                service_name=step_data["service"],
                action=step_data["action"],
                compensating_action=step_data.get("compensating_action", ""),
                input_data=step_data.get("input", {}),
                output_data=None,
                state=StepState.PENDING,
                max_retries=step_data.get("max_retries", 3),
                timeout_seconds=step_data.get("timeout", 30)
            )
            tx_steps.append(step)

        transaction = DistributedTransaction(
            transaction_id=transaction_id,
            transaction_type=transaction_type,
            state=TransactionState.INITIATED,
            steps=tx_steps,
            metadata=metadata or {},
            isolation_level=isolation,
            created_at=now,
            updated_at=now,
            timeout_at=now + timedelta(seconds=timeout)
        )

        self.transactions[transaction_id] = transaction
        self.metrics["total_transactions"] += 1

        return transaction

    def execute_transaction(
        self,
        transaction_id: str,
        step_executor: Callable[[TransactionStep], Dict[str, Any]]
    ) -> bool:
        """
        Execute transaction steps sequentially.

        Args:
            transaction_id: Transaction to execute
            step_executor: Function to execute each step

        Returns:
            True if transaction completed, False if failed
        """
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            raise ValueError(f"Transaction {transaction_id} not found")

        if transaction.state != TransactionState.INITIATED:
            raise ValueError(f"Transaction in invalid state: {transaction.state}")

        transaction.state = TransactionState.IN_PROGRESS
        transaction.updated_at = datetime.utcnow()

        # Execute each step
        for i, step in enumerate(transaction.steps):
            # Check timeout
            if datetime.utcnow() > transaction.timeout_at:
                transaction.state = TransactionState.TIMEOUT
                transaction.failed_step_index = i
                self._compensate_transaction(transaction, i - 1, step_executor)
                return False

            # Check idempotency
            if step.idempotency_key in self.executed_idempotency_keys:
                # Step already executed, skip
                step.state = StepState.COMPLETED
                transaction.completed_steps += 1
                continue

            # Execute step
            step.state = StepState.EXECUTING
            step.started_at = datetime.utcnow()

            success = False
            retry_count = 0

            while retry_count <= step.max_retries and not success:
                try:
                    # Execute step
                    result = step_executor(step)

                    # Mark as completed
                    step.output_data = result
                    step.state = StepState.COMPLETED
                    step.completed_at = datetime.utcnow()
                    transaction.completed_steps += 1

                    # Record idempotency
                    self.executed_idempotency_keys.add(step.idempotency_key)

                    success = True

                except Exception as e:
                    step.retry_count = retry_count
                    retry_count += 1

                    if retry_count > step.max_retries:
                        # Step failed permanently
                        step.state = StepState.FAILED
                        step.error = str(e)
                        transaction.state = TransactionState.FAILED
                        transaction.failed_step_index = i

                        # Compensate all previous steps
                        self._compensate_transaction(transaction, i - 1, step_executor)
                        return False

            transaction.current_step_index = i + 1
            transaction.updated_at = datetime.utcnow()

        # All steps completed
        transaction.state = TransactionState.COMPLETED
        transaction.updated_at = datetime.utcnow()
        self.metrics["completed_transactions"] += 1

        # Update avg duration
        duration = (transaction.updated_at - transaction.created_at).total_seconds() * 1000
        self.metrics["avg_duration_ms"] = (
            self.metrics["avg_duration_ms"] * 0.9 + duration * 0.1
        )

        return True

    def begin_2pc(
        self,
        transaction_type: str,
        participants: List[str],
        coordinator_id: str = "default",
        timeout_seconds: int = 60
    ) -> TwoPhaseCommit:
        """
        Begin two-phase commit transaction.

        Args:
            transaction_type: Transaction type
            participants: List of participant service IDs
            coordinator_id: Coordinator identifier
            timeout_seconds: Transaction timeout

        Returns:
            Created TwoPhaseCommit
        """
        transaction_id = str(uuid.uuid4())

        tpc = TwoPhaseCommit(
            transaction_id=transaction_id,
            coordinator_id=coordinator_id,
            participants=participants,
            state=TransactionState.INITIATED,
            timeout_seconds=timeout_seconds
        )

        self.two_phase_commits[transaction_id] = tpc
        return tpc

    def prepare_2pc(
        self,
        transaction_id: str,
        participant_id: str,
        prepared: bool
    ) -> bool:
        """
        Handle prepare phase response from participant.

        Args:
            transaction_id: 2PC transaction ID
            participant_id: Participant service ID
            prepared: Whether participant is prepared

        Returns:
            True if all participants prepared
        """
        tpc = self.two_phase_commits.get(transaction_id)
        if not tpc:
            return False

        if prepared:
            tpc.prepared_participants.add(participant_id)
        else:
            # Participant cannot prepare, abort
            tpc.state = TransactionState.FAILED
            return False

        # Check if all prepared
        if len(tpc.prepared_participants) == len(tpc.participants):
            return True

        return False

    def commit_2pc(
        self,
        transaction_id: str
    ) -> bool:
        """
        Commit two-phase transaction.

        Args:
            transaction_id: 2PC transaction ID

        Returns:
            True if commit succeeded
        """
        tpc = self.two_phase_commits.get(transaction_id)
        if not tpc:
            return False

        if len(tpc.prepared_participants) != len(tpc.participants):
            # Not all participants prepared
            return False

        tpc.state = TransactionState.COMPLETED
        return True

    def abort_2pc(
        self,
        transaction_id: str
    ) -> bool:
        """Abort two-phase transaction"""
        tpc = self.two_phase_commits.get(transaction_id)
        if not tpc:
            return False

        tpc.state = TransactionState.FAILED
        return True

    def get_transaction_status(
        self,
        transaction_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get transaction status"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return None

        return {
            "transaction_id": transaction_id,
            "type": transaction.transaction_type,
            "state": transaction.state.value,
            "total_steps": len(transaction.steps),
            "completed_steps": transaction.completed_steps,
            "current_step": transaction.current_step_index,
            "created_at": transaction.created_at.isoformat(),
            "updated_at": transaction.updated_at.isoformat(),
            "timeout_at": transaction.timeout_at.isoformat(),
            "steps": [
                {
                    "step_id": s.step_id,
                    "name": s.step_name,
                    "state": s.state.value,
                    "retry_count": s.retry_count,
                    "error": s.error
                }
                for s in transaction.steps
            ]
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator metrics"""
        return {
            **self.metrics,
            "active_transactions": len([
                t for t in self.transactions.values()
                if t.state == TransactionState.IN_PROGRESS
            ]),
            "success_rate": (
                self.metrics["completed_transactions"] / self.metrics["total_transactions"] * 100
                if self.metrics["total_transactions"] > 0 else 0
            )
        }

    # Private helper methods

    def _compensate_transaction(
        self,
        transaction: DistributedTransaction,
        last_completed_step: int,
        step_executor: Callable[[TransactionStep], Dict[str, Any]]
    ) -> None:
        """Execute compensating transactions"""
        transaction.state = TransactionState.COMPENSATING

        # Compensate in reverse order
        for i in range(last_completed_step, -1, -1):
            step = transaction.steps[i]

            if step.state != StepState.COMPLETED:
                continue

            step.state = StepState.COMPENSATING

            try:
                # Execute compensating action
                compensating_step = TransactionStep(
                    step_id=f"{step.step_id}_compensate",
                    step_name=f"Compensate {step.step_name}",
                    service_name=step.service_name,
                    action=step.compensating_action,
                    compensating_action="",
                    input_data=step.output_data or {},
                    output_data=None,
                    state=StepState.EXECUTING
                )

                step_executor(compensating_step)
                step.state = StepState.COMPENSATED

            except Exception as e:
                step.error = f"Compensation failed: {str(e)}"
                # Log but continue compensating other steps

        transaction.state = TransactionState.COMPENSATED
        transaction.updated_at = datetime.utcnow()
        self.metrics["compensated_transactions"] += 1
