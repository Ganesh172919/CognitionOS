"""
Revenue-Aware Orchestration - Innovation Feature

Dynamically prioritizes execution paths based on tenant plan, quota state,
and margin optimization. Implements priority queueing, resource allocation,
and revenue-optimized scheduling to maximize platform profitability.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class ExecutionPriority(str, Enum):
    """Execution priority levels"""
    CRITICAL = "critical"        # Highest priority, immediate execution
    HIGH = "high"                # High priority, minimal wait
    NORMAL = "normal"            # Standard priority
    LOW = "low"                  # Lower priority, can wait
    BACKGROUND = "background"    # Lowest priority, best-effort


class ResourceTier(str, Enum):
    """Resource allocation tiers"""
    PREMIUM = "premium"          # Dedicated resources, highest performance
    STANDARD = "standard"        # Shared resources, good performance
    ECONOMY = "economy"          # Minimal resources, cost-optimized


class QuotaStatus(str, Enum):
    """Quota consumption status"""
    AVAILABLE = "available"      # Under quota limits
    WARNING = "warning"          # Approaching limits (80%+)
    CRITICAL = "critical"        # Near limits (95%+)
    EXHAUSTED = "exhausted"      # Over limits
    UNLIMITED = "unlimited"      # No quota restrictions


class RevenueTier(str, Enum):
    """Revenue-based tenant tiers"""
    FREE = "free"                # Free tier
    STARTER = "starter"          # Entry paid tier
    PROFESSIONAL = "professional"  # Mid-tier
    BUSINESS = "business"        # High-value
    ENTERPRISE = "enterprise"    # Highest value


# ==================== Value Objects ====================

@dataclass(frozen=True)
class TenantPlan:
    """Tenant subscription plan details"""
    tier: RevenueTier
    monthly_value_usd: float
    included_quota: Dict[str, float]  # resource_type -> amount
    overage_pricing: Dict[str, float]  # resource_type -> cost per unit
    priority_boost: float  # 0.0 - 1.0
    sla_response_time_ms: Optional[int]
    enable_priority_execution: bool
    enable_dedicated_resources: bool

    def __post_init__(self):
        if self.monthly_value_usd < 0:
            raise ValueError("Monthly value cannot be negative")
        if not 0.0 <= self.priority_boost <= 1.0:
            raise ValueError("Priority boost must be between 0.0 and 1.0")


@dataclass
class QuotaState:
    """Current quota consumption state"""
    tenant_id: UUID
    period_start: datetime
    period_end: datetime
    quotas: Dict[str, float]  # resource_type -> consumed amount
    limits: Dict[str, float]  # resource_type -> limit
    overage: Dict[str, float]  # resource_type -> overage amount
    status: QuotaStatus
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.period_end <= self.period_start:
            raise ValueError("Period end must be after period start")

    def get_usage_percentage(self, resource_type: str) -> float:
        """Get usage percentage for a resource type"""
        consumed = self.quotas.get(resource_type, 0.0)
        limit = self.limits.get(resource_type, float('inf'))
        
        if limit == float('inf'):
            return 0.0
        
        return (consumed / limit * 100) if limit > 0 else 0.0

    def has_capacity(self, resource_type: str, required_amount: float) -> bool:
        """Check if quota has capacity for required amount"""
        consumed = self.quotas.get(resource_type, 0.0)
        limit = self.limits.get(resource_type, float('inf'))
        
        return (consumed + required_amount) <= limit

    def calculate_overage_cost(
        self,
        resource_type: str,
        plan: TenantPlan
    ) -> float:
        """Calculate overage cost for a resource type"""
        overage_amount = self.overage.get(resource_type, 0.0)
        overage_rate = plan.overage_pricing.get(resource_type, 0.0)
        
        return overage_amount * overage_rate

    def is_near_limit(self, resource_type: str, threshold: float = 0.8) -> bool:
        """Check if near quota limit"""
        usage = self.get_usage_percentage(resource_type)
        return usage >= (threshold * 100)


@dataclass
class RevenueMetrics:
    """Revenue metrics for prioritization"""
    tenant_id: UUID
    lifetime_value_usd: float
    current_mrr_usd: float  # Monthly Recurring Revenue
    margin_percentage: float
    payment_status: str  # 'current', 'past_due', 'canceled'
    churn_risk_score: float  # 0.0 - 1.0
    health_score: float  # 0.0 - 1.0
    last_payment_date: Optional[datetime]
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.lifetime_value_usd < 0:
            raise ValueError("Lifetime value cannot be negative")
        if self.current_mrr_usd < 0:
            raise ValueError("MRR cannot be negative")
        if not 0.0 <= self.margin_percentage <= 100.0:
            raise ValueError("Margin percentage must be between 0.0 and 100.0")
        if not 0.0 <= self.churn_risk_score <= 1.0:
            raise ValueError("Churn risk score must be between 0.0 and 1.0")
        if not 0.0 <= self.health_score <= 1.0:
            raise ValueError("Health score must be between 0.0 and 1.0")

    @property
    def is_healthy(self) -> bool:
        """Check if tenant is in good health"""
        return (
            self.payment_status == "current" and
            self.health_score >= 0.7 and
            self.churn_risk_score <= 0.3
        )

    @property
    def revenue_priority_score(self) -> float:
        """Calculate revenue-based priority score (0.0 - 1.0)"""
        # Weighted factors
        mrr_score = min(self.current_mrr_usd / 10000.0, 1.0) * 0.4
        health_score_factor = self.health_score * 0.3
        margin_score = (self.margin_percentage / 100.0) * 0.2
        churn_protection = (1.0 - self.churn_risk_score) * 0.1
        
        return mrr_score + health_score_factor + margin_score + churn_protection


# ==================== Entities ====================

@dataclass
class ExecutionRequest:
    """
    Request for task execution with priority and resource requirements.
    """
    id: UUID
    tenant_id: UUID
    task_id: UUID
    workflow_execution_id: UUID
    requested_priority: ExecutionPriority
    required_resources: Dict[str, float]  # resource_type -> amount
    estimated_cost_usd: float
    estimated_duration_ms: int
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if self.estimated_cost_usd < 0:
            raise ValueError("Estimated cost cannot be negative")
        if self.estimated_duration_ms < 0:
            raise ValueError("Estimated duration cannot be negative")

    @staticmethod
    def create(
        tenant_id: UUID,
        task_id: UUID,
        workflow_execution_id: UUID,
        requested_priority: ExecutionPriority,
        required_resources: Dict[str, float],
        estimated_cost_usd: float,
        estimated_duration_ms: int,
        deadline: Optional[datetime] = None
    ) -> "ExecutionRequest":
        """Create a new execution request"""
        return ExecutionRequest(
            id=uuid4(),
            tenant_id=tenant_id,
            task_id=task_id,
            workflow_execution_id=workflow_execution_id,
            requested_priority=requested_priority,
            required_resources=required_resources,
            estimated_cost_usd=estimated_cost_usd,
            estimated_duration_ms=estimated_duration_ms,
            deadline=deadline
        )

    def has_deadline(self) -> bool:
        """Check if request has a deadline"""
        return self.deadline is not None

    def is_overdue(self) -> bool:
        """Check if request is past deadline"""
        if not self.deadline:
            return False
        return datetime.now(timezone.utc) > self.deadline

    def time_until_deadline(self) -> Optional[timedelta]:
        """Calculate time until deadline"""
        if not self.deadline:
            return None
        return self.deadline - datetime.now(timezone.utc)


@dataclass
class PriorityScore:
    """
    Calculated priority score for execution scheduling.
    
    Combines multiple factors into a single priority score.
    """
    request_id: UUID
    tenant_id: UUID
    composite_score: float  # 0.0 - 1.0
    base_priority_score: float
    revenue_score: float
    quota_penalty: float
    deadline_urgency: float
    resource_availability_score: float
    final_priority: ExecutionPriority
    assigned_tier: ResourceTier
    reasoning: List[str] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        scores = [
            self.composite_score, self.base_priority_score, self.revenue_score,
            self.quota_penalty, self.deadline_urgency, self.resource_availability_score
        ]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError("All scores must be between 0.0 and 1.0")

    @staticmethod
    def calculate(
        request: ExecutionRequest,
        plan: TenantPlan,
        quota_state: QuotaState,
        revenue_metrics: RevenueMetrics
    ) -> "PriorityScore":
        """Calculate priority score for execution request"""
        reasoning = []
        
        # Base priority from request
        priority_map = {
            ExecutionPriority.CRITICAL: 1.0,
            ExecutionPriority.HIGH: 0.8,
            ExecutionPriority.NORMAL: 0.5,
            ExecutionPriority.LOW: 0.3,
            ExecutionPriority.BACKGROUND: 0.1
        }
        base_score = priority_map[request.requested_priority]
        reasoning.append(f"Base priority: {request.requested_priority.value} ({base_score:.2f})")
        
        # Revenue-based boost
        revenue_score = revenue_metrics.revenue_priority_score
        revenue_boost = revenue_score * plan.priority_boost
        reasoning.append(f"Revenue boost: {revenue_boost:.2f} (MRR: ${revenue_metrics.current_mrr_usd:.2f})")
        
        # Quota penalty
        quota_penalty = 0.0
        if quota_state.status == QuotaStatus.CRITICAL:
            quota_penalty = 0.3
            reasoning.append("Quota critical: -30% penalty")
        elif quota_state.status == QuotaStatus.WARNING:
            quota_penalty = 0.15
            reasoning.append("Quota warning: -15% penalty")
        elif quota_state.status == QuotaStatus.EXHAUSTED:
            quota_penalty = 0.5
            reasoning.append("Quota exhausted: -50% penalty")
        
        # Deadline urgency
        deadline_urgency = 0.0
        if request.has_deadline():
            time_left = request.time_until_deadline()
            if time_left and time_left.total_seconds() > 0:
                # Higher urgency as deadline approaches
                hours_left = time_left.total_seconds() / 3600
                if hours_left < 1:
                    deadline_urgency = 0.4
                    reasoning.append("Deadline < 1 hour: +40% urgency")
                elif hours_left < 4:
                    deadline_urgency = 0.3
                    reasoning.append("Deadline < 4 hours: +30% urgency")
                elif hours_left < 24:
                    deadline_urgency = 0.2
                    reasoning.append("Deadline < 24 hours: +20% urgency")
            elif request.is_overdue():
                deadline_urgency = 0.5
                reasoning.append("Overdue: +50% urgency")
        
        # Resource availability (simplified)
        resource_availability = 0.8  # Would check actual system capacity
        
        # Calculate composite score
        composite = (
            base_score * 0.3 +
            revenue_boost * 0.25 +
            (1.0 - quota_penalty) * 0.15 +
            deadline_urgency * 0.2 +
            resource_availability * 0.1
        )
        composite = max(0.0, min(composite, 1.0))
        
        # Determine final priority
        if composite >= 0.8:
            final_priority = ExecutionPriority.CRITICAL
        elif composite >= 0.6:
            final_priority = ExecutionPriority.HIGH
        elif composite >= 0.4:
            final_priority = ExecutionPriority.NORMAL
        elif composite >= 0.2:
            final_priority = ExecutionPriority.LOW
        else:
            final_priority = ExecutionPriority.BACKGROUND
        
        # Assign resource tier
        if plan.enable_dedicated_resources and revenue_metrics.current_mrr_usd >= 1000:
            assigned_tier = ResourceTier.PREMIUM
        elif composite >= 0.5:
            assigned_tier = ResourceTier.STANDARD
        else:
            assigned_tier = ResourceTier.ECONOMY
        
        reasoning.append(f"Final composite score: {composite:.2f}")
        reasoning.append(f"Assigned priority: {final_priority.value}")
        reasoning.append(f"Resource tier: {assigned_tier.value}")
        
        return PriorityScore(
            request_id=request.id,
            tenant_id=request.tenant_id,
            composite_score=composite,
            base_priority_score=base_score,
            revenue_score=revenue_score,
            quota_penalty=quota_penalty,
            deadline_urgency=deadline_urgency,
            resource_availability_score=resource_availability,
            final_priority=final_priority,
            assigned_tier=assigned_tier,
            reasoning=reasoning
        )


@dataclass
class ExecutionAllocation:
    """
    Resource allocation for execution.
    
    Represents allocated resources and scheduling decision.
    """
    id: UUID
    request_id: UUID
    tenant_id: UUID
    priority_score: PriorityScore
    allocated_resources: Dict[str, float]
    estimated_start_time: datetime
    estimated_completion_time: datetime
    resource_tier: ResourceTier
    quota_reserved: bool
    margin_impact_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @staticmethod
    def create(
        request: ExecutionRequest,
        priority_score: PriorityScore,
        queue_position: int,
        estimated_wait_time_ms: int
    ) -> "ExecutionAllocation":
        """Create execution allocation"""
        now = datetime.now(timezone.utc)
        start_time = now + timedelta(milliseconds=estimated_wait_time_ms)
        completion_time = start_time + timedelta(milliseconds=request.estimated_duration_ms)
        
        # Calculate margin impact (simplified)
        # Revenue from execution - cost of execution
        margin_impact = 0.0  # Would calculate based on plan pricing
        
        return ExecutionAllocation(
            id=uuid4(),
            request_id=request.id,
            tenant_id=request.tenant_id,
            priority_score=priority_score,
            allocated_resources=request.required_resources.copy(),
            estimated_start_time=start_time,
            estimated_completion_time=completion_time,
            resource_tier=priority_score.assigned_tier,
            quota_reserved=True,
            margin_impact_usd=margin_impact,
            metadata={
                "queue_position": queue_position,
                "estimated_wait_ms": estimated_wait_time_ms
            }
        )

    @property
    def estimated_wait_time(self) -> timedelta:
        """Calculate estimated wait time"""
        return self.estimated_start_time - self.allocated_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "id": str(self.id),
            "request_id": str(self.request_id),
            "tenant_id": str(self.tenant_id),
            "final_priority": self.priority_score.final_priority.value,
            "composite_score": self.priority_score.composite_score,
            "resource_tier": self.resource_tier.value,
            "estimated_start_time": self.estimated_start_time.isoformat(),
            "estimated_completion_time": self.estimated_completion_time.isoformat(),
            "allocated_resources": self.allocated_resources,
            "margin_impact_usd": self.margin_impact_usd,
            "metadata": self.metadata,
            "allocated_at": self.allocated_at.isoformat()
        }


# ==================== Service ====================

class RevenueOrchestrationService:
    """
    Revenue-aware orchestration service.
    
    Optimizes execution scheduling based on revenue, quotas, and margins.
    """

    def __init__(self):
        """Initialize revenue orchestration service"""
        self._execution_queue: List[ExecutionAllocation] = []
        self._quota_reservations: Dict[UUID, Dict[str, float]] = {}

    async def prioritize_execution(
        self,
        request: ExecutionRequest,
        plan: TenantPlan,
        quota_state: QuotaState,
        revenue_metrics: RevenueMetrics
    ) -> ExecutionAllocation:
        """
        Prioritize and allocate execution request.
        
        Args:
            request: Execution request
            plan: Tenant plan details
            quota_state: Current quota state
            revenue_metrics: Revenue metrics for tenant
            
        Returns:
            Execution allocation with scheduling
        """
        # Calculate priority score
        priority_score = PriorityScore.calculate(
            request, plan, quota_state, revenue_metrics
        )
        
        # Find queue position based on priority
        queue_position = self._find_queue_position(priority_score)
        
        # Estimate wait time
        estimated_wait_ms = self._estimate_wait_time(queue_position)
        
        # Create allocation
        allocation = ExecutionAllocation.create(
            request, priority_score, queue_position, estimated_wait_ms
        )
        
        # Insert into queue at appropriate position
        self._execution_queue.insert(queue_position, allocation)
        
        # Reserve quota
        self._reserve_quota(request.tenant_id, request.required_resources)
        
        return allocation

    async def adjust_priority(
        self,
        allocation_id: UUID,
        new_priority: ExecutionPriority,
        reason: str
    ) -> Optional[ExecutionAllocation]:
        """
        Adjust priority of queued execution.
        
        Args:
            allocation_id: Allocation identifier
            new_priority: New priority level
            reason: Reason for adjustment
            
        Returns:
            Updated allocation or None if not found
        """
        # Find allocation in queue
        allocation = next(
            (a for a in self._execution_queue if a.id == allocation_id),
            None
        )
        
        if not allocation:
            return None
        
        # Update priority score
        old_priority = allocation.priority_score.final_priority
        allocation.priority_score.final_priority = new_priority
        allocation.priority_score.reasoning.append(
            f"Priority adjusted from {old_priority.value} to {new_priority.value}: {reason}"
        )
        
        # Re-sort queue
        self._execution_queue.remove(allocation)
        new_position = self._find_queue_position(allocation.priority_score)
        self._execution_queue.insert(new_position, allocation)
        
        # Update estimated times
        allocation.metadata["queue_position"] = new_position
        estimated_wait_ms = self._estimate_wait_time(new_position)
        allocation.estimated_start_time = datetime.now(timezone.utc) + timedelta(milliseconds=estimated_wait_ms)
        
        return allocation

    async def get_next_execution(
        self,
        resource_capacity: Dict[str, float]
    ) -> Optional[ExecutionAllocation]:
        """
        Get next execution from queue that fits resource capacity.
        
        Args:
            resource_capacity: Available resource capacity
            
        Returns:
            Next execution allocation or None
        """
        for allocation in self._execution_queue:
            # Check if resources are available
            can_execute = all(
                allocation.allocated_resources.get(resource, 0) <= resource_capacity.get(resource, 0)
                for resource in allocation.allocated_resources
            )
            
            if can_execute:
                self._execution_queue.remove(allocation)
                return allocation
        
        return None

    async def calculate_margin_optimization(
        self,
        tenant_id: UUID,
        plan: TenantPlan,
        quota_state: QuotaState,
        execution_cost_usd: float
    ) -> Dict[str, Any]:
        """
        Calculate margin optimization metrics.
        
        Args:
            tenant_id: Tenant identifier
            plan: Tenant plan
            quota_state: Current quota state
            execution_cost_usd: Cost of execution
            
        Returns:
            Margin optimization metrics
        """
        # Calculate revenue from execution
        # For included quota, no direct revenue but maintains customer value
        # For overage, direct revenue from overage charges
        
        overage_revenue = sum(
            quota_state.calculate_overage_cost(resource, plan)
            for resource in quota_state.overage.keys()
        )
        
        # Calculate margin
        margin_usd = overage_revenue - execution_cost_usd
        margin_percentage = (margin_usd / execution_cost_usd * 100) if execution_cost_usd > 0 else 0
        
        # Determine if execution is profitable
        is_profitable = margin_usd >= 0
        
        # Calculate contribution to monthly value
        monthly_contribution = plan.monthly_value_usd / 30  # Daily average
        
        return {
            "tenant_id": str(tenant_id),
            "execution_cost_usd": execution_cost_usd,
            "overage_revenue_usd": overage_revenue,
            "margin_usd": margin_usd,
            "margin_percentage": margin_percentage,
            "is_profitable": is_profitable,
            "monthly_contribution": monthly_contribution,
            "recommendation": self._get_margin_recommendation(
                margin_percentage, quota_state.status
            )
        }

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get execution queue statistics"""
        if not self._execution_queue:
            return {
                "total_queued": 0,
                "by_priority": {},
                "by_tier": {},
                "average_wait_time_ms": 0
            }
        
        by_priority = {}
        by_tier = {}
        total_wait = 0
        
        for allocation in self._execution_queue:
            priority = allocation.priority_score.final_priority.value
            tier = allocation.resource_tier.value
            
            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_tier[tier] = by_tier.get(tier, 0) + 1
            total_wait += allocation.estimated_wait_time.total_seconds() * 1000
        
        return {
            "total_queued": len(self._execution_queue),
            "by_priority": by_priority,
            "by_tier": by_tier,
            "average_wait_time_ms": int(total_wait / len(self._execution_queue))
        }

    # Private helper methods

    def _find_queue_position(self, priority_score: PriorityScore) -> int:
        """Find appropriate queue position based on priority"""
        for i, allocation in enumerate(self._execution_queue):
            if priority_score.composite_score > allocation.priority_score.composite_score:
                return i
        return len(self._execution_queue)

    def _estimate_wait_time(self, queue_position: int) -> int:
        """Estimate wait time based on queue position"""
        if queue_position == 0:
            return 0
        
        # Sum estimated durations of tasks ahead in queue
        total_wait = 0
        for allocation in self._execution_queue[:queue_position]:
            duration = (allocation.estimated_completion_time - allocation.estimated_start_time).total_seconds() * 1000
            total_wait += int(duration)
        
        return total_wait

    def _reserve_quota(self, tenant_id: UUID, resources: Dict[str, float]) -> None:
        """Reserve quota for execution"""
        if tenant_id not in self._quota_reservations:
            self._quota_reservations[tenant_id] = {}
        
        for resource, amount in resources.items():
            current = self._quota_reservations[tenant_id].get(resource, 0.0)
            self._quota_reservations[tenant_id][resource] = current + amount

    def _get_margin_recommendation(
        self,
        margin_percentage: float,
        quota_status: QuotaStatus
    ) -> str:
        """Get recommendation based on margin and quota status"""
        if quota_status == QuotaStatus.EXHAUSTED and margin_percentage < 20:
            return "Consider throttling or upselling - low margin on overage"
        elif margin_percentage < 0:
            return "Execution is unprofitable - review pricing or costs"
        elif margin_percentage < 20:
            return "Low margin - monitor closely"
        elif margin_percentage >= 50:
            return "Healthy margin - good customer value"
        else:
            return "Acceptable margin"
