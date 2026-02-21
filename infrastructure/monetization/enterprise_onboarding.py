"""
Enterprise Onboarding Workflow Automation

Automated onboarding system for enterprise customers with
custom workflows, provisioning, and success tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class OnboardingStepType(str, Enum):
    """Types of onboarding steps"""
    ACCOUNT_SETUP = "account_setup"
    PAYMENT_INFO = "payment_info"
    TEAM_SETUP = "team_setup"
    SSO_CONFIGURATION = "sso_configuration"
    API_KEY_GENERATION = "api_key_generation"
    INTEGRATION_SETUP = "integration_setup"
    TRAINING_SESSION = "training_session"
    SUCCESS_METRICS = "success_metrics"
    COMPLIANCE_REVIEW = "compliance_review"


class StepStatus(str, Enum):
    """Status of onboarding step"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass
class OnboardingStep:
    """Single step in onboarding workflow"""
    id: str
    step_type: OnboardingStepType
    title: str
    description: str
    status: StepStatus = StepStatus.PENDING

    # Dependencies
    depends_on: List[str] = field(default_factory=list)

    # Execution
    auto_execute: bool = False
    requires_manual_approval: bool = False
    estimated_duration_minutes: int = 30

    # Tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None

    # Metadata
    instructions: Optional[str] = None
    resources: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def can_start(self, completed_steps: set[str]) -> bool:
        """Check if step can start"""
        if self.status != StepStatus.PENDING:
            return False
        return all(dep in completed_steps for dep in self.depends_on)


@dataclass
class EnterpriseCustomer:
    """Enterprise customer information"""
    id: str
    company_name: str
    primary_contact_email: str
    primary_contact_name: str

    # Contract details
    contract_value: float
    contract_start_date: datetime
    contract_end_date: datetime

    # Requirements
    required_integrations: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    sso_provider: Optional[str] = None

    # Success metrics
    target_user_count: int = 10
    target_api_calls_per_month: int = 100000
    success_milestones: List[Dict[str, Any]] = field(default_factory=list)

    # Team
    team_members: List[Dict[str, str]] = field(default_factory=list)

    # Status
    onboarding_status: str = "not_started"
    health_score: float = 0.0  # 0-1
    last_activity: Optional[datetime] = None


class OnboardingWorkflow:
    """
    Automated enterprise onboarding workflow

    Manages the complete onboarding process with automation,
    tracking, and success metrics.
    """

    def __init__(self, notification_service: Optional[Any] = None):
        self.notification_service = notification_service
        self._workflows: Dict[str, List[OnboardingStep]] = {}
        self._customers: Dict[str, EnterpriseCustomer] = {}

    async def create_workflow(
        self,
        customer: EnterpriseCustomer,
        custom_steps: Optional[List[OnboardingStep]] = None
    ) -> str:
        """
        Create onboarding workflow for customer

        Args:
            customer: Enterprise customer
            custom_steps: Optional custom steps

        Returns:
            Workflow ID
        """
        logger.info(f"Creating onboarding workflow for {customer.company_name}")

        workflow_id = f"workflow_{customer.id}"

        # Build standard workflow
        steps = custom_steps or self._build_standard_workflow(customer)

        # Store workflow and customer
        self._workflows[workflow_id] = steps
        self._customers[customer.id] = customer
        customer.onboarding_status = "in_progress"

        # Send welcome email
        if self.notification_service:
            await self._send_welcome_email(customer)

        # Auto-start first steps
        await self._auto_start_steps(workflow_id)

        return workflow_id

    def _build_standard_workflow(
        self,
        customer: EnterpriseCustomer
    ) -> List[OnboardingStep]:
        """Build standard enterprise onboarding workflow"""

        steps = []

        # Step 1: Account Setup
        steps.append(OnboardingStep(
            id="step_1",
            step_type=OnboardingStepType.ACCOUNT_SETUP,
            title="Account Setup",
            description="Create enterprise account and configure basic settings",
            auto_execute=True,
            estimated_duration_minutes=15,
            instructions="System will automatically provision your account"
        ))

        # Step 2: Payment Information
        steps.append(OnboardingStep(
            id="step_2",
            step_type=OnboardingStepType.PAYMENT_INFO,
            title="Payment Information",
            description="Add payment method and billing details",
            depends_on=["step_1"],
            estimated_duration_minutes=10,
            instructions="Add credit card or set up invoicing"
        ))

        # Step 3: Team Setup
        steps.append(OnboardingStep(
            id="step_3",
            step_type=OnboardingStepType.TEAM_SETUP,
            title="Invite Team Members",
            description="Invite your team and assign roles",
            depends_on=["step_1"],
            estimated_duration_minutes=20,
            instructions="Invite team members via email"
        ))

        # Step 4: SSO Configuration (if required)
        if customer.sso_provider:
            steps.append(OnboardingStep(
                id="step_4",
                step_type=OnboardingStepType.SSO_CONFIGURATION,
                title="Configure SSO",
                description=f"Set up {customer.sso_provider} SSO integration",
                depends_on=["step_1"],
                estimated_duration_minutes=45,
                instructions="Configure SAML/OIDC settings",
                resources=[
                    {"type": "documentation", "url": "/docs/sso-setup"},
                    {"type": "support", "url": "/support/sso"}
                ]
            ))

        # Step 5: API Keys
        steps.append(OnboardingStep(
            id="step_5",
            step_type=OnboardingStepType.API_KEY_GENERATION,
            title="Generate API Keys",
            description="Create API keys for your integrations",
            depends_on=["step_1", "step_2"],
            auto_execute=True,
            estimated_duration_minutes=5
        ))

        # Step 6: Integrations
        if customer.required_integrations:
            steps.append(OnboardingStep(
                id="step_6",
                step_type=OnboardingStepType.INTEGRATION_SETUP,
                title="Configure Integrations",
                description=f"Set up required integrations: {', '.join(customer.required_integrations)}",
                depends_on=["step_5"],
                estimated_duration_minutes=60,
                metadata={"integrations": customer.required_integrations}
            ))

        # Step 7: Training
        steps.append(OnboardingStep(
            id="step_7",
            step_type=OnboardingStepType.TRAINING_SESSION,
            title="Training Session",
            description="Schedule training for your team",
            depends_on=["step_3"],
            estimated_duration_minutes=90,
            requires_manual_approval=True,
            instructions="Contact success@company.com to schedule"
        ))

        # Step 8: Success Metrics
        steps.append(OnboardingStep(
            id="step_8",
            step_type=OnboardingStepType.SUCCESS_METRICS,
            title="Define Success Metrics",
            description="Set up success metrics and monitoring",
            depends_on=["step_5"],
            estimated_duration_minutes=30
        ))

        # Step 9: Compliance Review (if required)
        if customer.compliance_requirements:
            steps.append(OnboardingStep(
                id="step_9",
                step_type=OnboardingStepType.COMPLIANCE_REVIEW,
                title="Compliance Review",
                description=f"Review {', '.join(customer.compliance_requirements)} compliance",
                depends_on=[s.id for s in steps],
                requires_manual_approval=True,
                estimated_duration_minutes=120,
                metadata={"requirements": customer.compliance_requirements}
            ))

        return steps

    async def _auto_start_steps(self, workflow_id: str):
        """Auto-start eligible steps"""
        steps = self._workflows.get(workflow_id, [])
        completed = {s.id for s in steps if s.status == StepStatus.COMPLETED}

        for step in steps:
            if step.auto_execute and step.can_start(completed):
                await self.execute_step(workflow_id, step.id)

    async def execute_step(
        self,
        workflow_id: str,
        step_id: str,
        executor: Optional[str] = None
    ) -> bool:
        """
        Execute an onboarding step

        Args:
            workflow_id: Workflow identifier
            step_id: Step to execute
            executor: Who is executing the step

        Returns:
            Success status
        """
        steps = self._workflows.get(workflow_id)
        if not steps:
            logger.error(f"Workflow {workflow_id} not found")
            return False

        step = next((s for s in steps if s.id == step_id), None)
        if not step:
            logger.error(f"Step {step_id} not found")
            return False

        # Check if can start
        completed = {s.id for s in steps if s.status == StepStatus.COMPLETED}
        if not step.can_start(completed):
            logger.warning(f"Step {step_id} cannot start yet")
            return False

        logger.info(f"Executing step: {step.title}")

        step.status = StepStatus.IN_PROGRESS
        step.started_at = datetime.utcnow()
        step.assigned_to = executor

        # Execute step based on type
        try:
            success = await self._execute_step_action(workflow_id, step)

            if success:
                step.status = StepStatus.COMPLETED
                step.completed_at = datetime.utcnow()

                # Notify stakeholders
                if self.notification_service:
                    await self._notify_step_completion(workflow_id, step)

                # Auto-start next steps
                await self._auto_start_steps(workflow_id)

                # Check if workflow complete
                await self._check_workflow_completion(workflow_id)
            else:
                step.status = StepStatus.FAILED

            return success

        except Exception as e:
            logger.error(f"Error executing step {step_id}: {e}")
            step.status = StepStatus.FAILED
            return False

    async def _execute_step_action(
        self,
        workflow_id: str,
        step: OnboardingStep
    ) -> bool:
        """Execute the actual step action"""

        if step.step_type == OnboardingStepType.ACCOUNT_SETUP:
            return await self._setup_account(workflow_id)
        elif step.step_type == OnboardingStepType.API_KEY_GENERATION:
            return await self._generate_api_keys(workflow_id)
        elif step.requires_manual_approval:
            # Manual steps need external completion
            step.status = StepStatus.PENDING
            return False

        # Other steps are manual by default
        return True

    async def _setup_account(self, workflow_id: str) -> bool:
        """Set up enterprise account"""
        # Would create tenant, configure settings, etc.
        logger.info(f"Setting up account for workflow {workflow_id}")
        return True

    async def _generate_api_keys(self, workflow_id: str) -> bool:
        """Generate API keys"""
        # Would generate production and sandbox keys
        logger.info(f"Generating API keys for workflow {workflow_id}")
        return True

    async def _check_workflow_completion(self, workflow_id: str):
        """Check if workflow is complete"""
        steps = self._workflows.get(workflow_id, [])

        all_complete = all(
            s.status in [StepStatus.COMPLETED, StepStatus.SKIPPED]
            for s in steps
        )

        if all_complete:
            logger.info(f"Workflow {workflow_id} completed!")

            # Update customer status
            customer_id = workflow_id.replace("workflow_", "")
            customer = self._customers.get(customer_id)
            if customer:
                customer.onboarding_status = "completed"

                # Send completion notification
                if self.notification_service:
                    await self._send_completion_email(customer)

    async def _send_welcome_email(self, customer: EnterpriseCustomer):
        """Send welcome email"""
        logger.info(f"Sending welcome email to {customer.primary_contact_email}")

    async def _notify_step_completion(self, workflow_id: str, step: OnboardingStep):
        """Notify about step completion"""
        logger.info(f"Step completed: {step.title}")

    async def _send_completion_email(self, customer: EnterpriseCustomer):
        """Send onboarding completion email"""
        logger.info(f"Sending completion email to {customer.primary_contact_email}")

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status"""
        steps = self._workflows.get(workflow_id, [])

        completed = sum(1 for s in steps if s.status == StepStatus.COMPLETED)
        total = len(steps)

        return {
            "workflow_id": workflow_id,
            "total_steps": total,
            "completed_steps": completed,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "steps": [
                {
                    "id": s.id,
                    "title": s.title,
                    "status": s.status.value,
                    "started_at": s.started_at.isoformat() if s.started_at else None,
                    "completed_at": s.completed_at.isoformat() if s.completed_at else None
                }
                for s in steps
            ]
        }

    async def calculate_health_score(self, customer_id: str) -> float:
        """
        Calculate customer health score

        Based on:
        - Onboarding progress
        - Usage patterns
        - Engagement level
        - Support interactions
        """
        customer = self._customers.get(customer_id)
        if not customer:
            return 0.0

        score = 0.0

        # Onboarding progress (30%)
        workflow_id = f"workflow_{customer_id}"
        if workflow_id in self._workflows:
            status = self.get_workflow_status(workflow_id)
            score += (status["progress_percent"] / 100) * 0.3

        # Recent activity (20%)
        if customer.last_activity:
            days_since_activity = (datetime.utcnow() - customer.last_activity).days
            if days_since_activity < 7:
                score += 0.2
            elif days_since_activity < 30:
                score += 0.1

        # Team size vs target (20%)
        if customer.target_user_count > 0:
            team_ratio = len(customer.team_members) / customer.target_user_count
            score += min(team_ratio, 1.0) * 0.2

        # Success milestones (30%)
        if customer.success_milestones:
            completed_milestones = sum(
                1 for m in customer.success_milestones
                if m.get("completed", False)
            )
            milestone_score = completed_milestones / len(customer.success_milestones)
            score += milestone_score * 0.3

        customer.health_score = score
        return score
