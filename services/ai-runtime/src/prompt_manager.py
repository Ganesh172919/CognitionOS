"""
Prompt Versioning and A/B Testing Framework.

Manages versioned prompts, A/B testing, and prompt performance tracking.
"""

import json
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum


class PromptVersion(str, Enum):
    """Prompt version identifiers."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    EXPERIMENTAL = "experimental"
    STABLE = "stable"


class ABTestStatus(str, Enum):
    """A/B test status."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class PromptTemplate:
    """
    Represents a versioned prompt template.
    """
    template_id: str
    role: str  # planner, executor, critic, etc.
    version: str
    template: str
    variables: List[str]
    description: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    performance_score: Optional[float] = None
    usage_count: int = 0
    is_active: bool = True


@dataclass
class ABTest:
    """
    Represents an A/B test between prompt variants.
    """
    test_id: str
    name: str
    description: str
    status: ABTestStatus
    variant_a: str  # template_id
    variant_b: str  # template_id
    traffic_split: float  # 0.0 to 1.0 (% going to variant_a)
    created_at: datetime
    started_at: Optional[datetime]
    ended_at: Optional[datetime]
    metrics: Dict[str, Any]
    winner: Optional[str] = None


@dataclass
class PromptExecution:
    """
    Records a prompt execution for analytics.
    """
    execution_id: str
    template_id: str
    template_version: str
    role: str
    rendered_prompt: str
    output: str
    quality_score: float
    confidence_score: float
    tokens_used: int
    cost_usd: float
    duration_seconds: float
    user_id: str
    ab_test_id: Optional[str]
    ab_variant: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]


class PromptManager:
    """
    Manages prompt templates, versioning, and A/B testing.

    Features:
    - Load prompts from versioned directory
    - Render prompts with variables
    - A/B test support
    - Performance tracking
    - Automatic fallback to stable versions
    """

    def __init__(self, prompts_dir: str = "/prompts/versioned"):
        """
        Initialize prompt manager.

        Args:
            prompts_dir: Directory containing versioned prompts
        """
        self.prompts_dir = Path(prompts_dir)
        self.templates: Dict[str, PromptTemplate] = {}
        self.ab_tests: Dict[str, ABTest] = {}
        self.executions: List[PromptExecution] = []

        # Performance tracking
        self.performance_cache: Dict[str, Dict[str, float]] = {}

        # Load prompts from disk
        self._load_prompts()

    def _load_prompts(self):
        """Load all prompt templates from disk."""
        if not self.prompts_dir.exists():
            # Create default prompts directory
            self.prompts_dir.mkdir(parents=True, exist_ok=True)
            self._create_default_prompts()
            return

        # Load all JSON files from prompts directory
        for prompt_file in self.prompts_dir.glob("**/*.json"):
            try:
                with open(prompt_file, 'r') as f:
                    data = json.load(f)
                    template = self._dict_to_template(data)
                    self.templates[template.template_id] = template
            except Exception as e:
                # Log error but continue loading other prompts
                print(f"Error loading prompt {prompt_file}: {e}")

    def _create_default_prompts(self):
        """Create default prompt templates."""
        default_prompts = [
            {
                "template_id": "planner_v1",
                "role": "planner",
                "version": "v1",
                "template": """You are an expert task planner. Break down the following task into clear, actionable steps.

Task: {task_description}

Context:
{context}

Requirements:
- Number each step clearly
- Make steps specific and actionable
- Consider dependencies between steps
- Estimate complexity for each step

Provide a structured plan with numbered steps.""",
                "variables": ["task_description", "context"],
                "description": "Standard task planning prompt",
                "metadata": {"temperature": 0.7, "max_tokens": 1500}
            },
            {
                "template_id": "executor_v1",
                "role": "executor",
                "version": "v1",
                "template": """You are an expert code executor. Implement the following task with high-quality code.

Task: {task_description}

Plan:
{plan}

Requirements:
- Write clean, production-ready code
- Include error handling
- Add brief comments for complex logic
- Follow best practices
- Ensure code is runnable

Provide the implementation:""",
                "variables": ["task_description", "plan"],
                "description": "Standard code execution prompt",
                "metadata": {"temperature": 0.3, "max_tokens": 2000}
            },
            {
                "template_id": "critic_v1",
                "role": "critic",
                "version": "v1",
                "template": """You are an expert code critic. Review the following code and identify issues.

Task: {task_description}

Code:
{code}

Review for:
- Correctness and bugs
- Security vulnerabilities
- Performance issues
- Code quality and style
- Edge cases

Provide constructive feedback:""",
                "variables": ["task_description", "code"],
                "description": "Standard code review prompt",
                "metadata": {"temperature": 0.5, "max_tokens": 1500}
            },
            {
                "template_id": "reasoner_v1",
                "role": "reasoner",
                "version": "v1",
                "template": """You are an expert reasoning engine. Analyze the following problem step by step.

Problem: {problem}

Context:
{context}

Think through this carefully:
1. What are the key facts?
2. What are the constraints?
3. What are possible approaches?
4. What is the most logical solution?

Provide your reasoning:""",
                "variables": ["problem", "context"],
                "description": "Standard reasoning prompt",
                "metadata": {"temperature": 0.6, "max_tokens": 1500}
            },
            {
                "template_id": "summarizer_v1",
                "role": "summarizer",
                "version": "v1",
                "template": """You are an expert summarizer. Create a concise summary of the following content.

Content:
{content}

Requirements:
- Focus on key points
- Maintain accuracy
- Be concise but complete
- Use clear language

Summary:""",
                "variables": ["content"],
                "description": "Standard summarization prompt",
                "metadata": {"temperature": 0.4, "max_tokens": 1000}
            }
        ]

        for prompt_data in default_prompts:
            template = PromptTemplate(
                template_id=prompt_data["template_id"],
                role=prompt_data["role"],
                version=prompt_data["version"],
                template=prompt_data["template"],
                variables=prompt_data["variables"],
                description=prompt_data["description"],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=prompt_data["metadata"],
                is_active=True
            )
            self.templates[template.template_id] = template

            # Save to disk
            self._save_template(template)

    def _save_template(self, template: PromptTemplate):
        """Save template to disk."""
        role_dir = self.prompts_dir / template.role
        role_dir.mkdir(parents=True, exist_ok=True)

        filepath = role_dir / f"{template.template_id}.json"

        with open(filepath, 'w') as f:
            json.dump(self._template_to_dict(template), f, indent=2, default=str)

    def _template_to_dict(self, template: PromptTemplate) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            "template_id": template.template_id,
            "role": template.role,
            "version": template.version,
            "template": template.template,
            "variables": template.variables,
            "description": template.description,
            "created_at": template.created_at.isoformat(),
            "updated_at": template.updated_at.isoformat(),
            "metadata": template.metadata,
            "performance_score": template.performance_score,
            "usage_count": template.usage_count,
            "is_active": template.is_active
        }

    def _dict_to_template(self, data: Dict[str, Any]) -> PromptTemplate:
        """Convert dictionary to template."""
        return PromptTemplate(
            template_id=data["template_id"],
            role=data["role"],
            version=data["version"],
            template=data["template"],
            variables=data["variables"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
            performance_score=data.get("performance_score"),
            usage_count=data.get("usage_count", 0),
            is_active=data.get("is_active", True)
        )

    def get_prompt(
        self,
        role: str,
        variables: Dict[str, Any],
        version: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Tuple[str, PromptTemplate, Optional[ABTest]]:
        """
        Get and render a prompt for a role.

        Supports A/B testing and automatic version selection.

        Args:
            role: Agent role (planner, executor, etc.)
            variables: Variables to render in template
            version: Specific version to use (optional)
            user_id: User ID for A/B test assignment

        Returns:
            (rendered_prompt, template_used, ab_test_if_any)
        """
        # Check if there's an active A/B test for this role
        active_test = self._get_active_ab_test(role)

        if active_test and user_id:
            # Assign user to variant
            template = self._assign_ab_variant(active_test, user_id)
            variant = "A" if template.template_id == active_test.variant_a else "B"
        else:
            # Use specified version or best performing version
            template = self._select_template(role, version)
            active_test = None
            variant = None

        # Render template with variables
        rendered = self._render_template(template, variables)

        # Update usage count
        template.usage_count += 1

        return rendered, template, active_test

    def _get_active_ab_test(self, role: str) -> Optional[ABTest]:
        """Get active A/B test for a role, if any."""
        for test in self.ab_tests.values():
            if test.status == ABTestStatus.ACTIVE:
                # Check if test is for this role
                template_a = self.templates.get(test.variant_a)
                if template_a and template_a.role == role:
                    return test
        return None

    def _assign_ab_variant(self, test: ABTest, user_id: str) -> PromptTemplate:
        """
        Assign user to A/B test variant using consistent hashing.

        Args:
            test: The A/B test
            user_id: User identifier

        Returns:
            Selected template
        """
        # Use hash of user_id + test_id for consistent assignment
        hash_input = f"{user_id}:{test.test_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        assignment = (hash_value % 100) / 100.0

        if assignment < test.traffic_split:
            return self.templates[test.variant_a]
        else:
            return self.templates[test.variant_b]

    def _select_template(self, role: str, version: Optional[str] = None) -> PromptTemplate:
        """
        Select best template for role.

        Args:
            role: Agent role
            version: Specific version (optional)

        Returns:
            Selected template
        """
        # Filter templates by role and active status
        candidates = [
            t for t in self.templates.values()
            if t.role == role and t.is_active
        ]

        if not candidates:
            raise ValueError(f"No active templates found for role: {role}")

        # If version specified, filter by version
        if version:
            versioned = [t for t in candidates if t.version == version]
            if versioned:
                candidates = versioned

        # Select best performing template
        # Sort by: 1) performance_score, 2) usage_count
        candidates.sort(
            key=lambda t: (
                t.performance_score if t.performance_score else 0.0,
                t.usage_count
            ),
            reverse=True
        )

        return candidates[0]

    def _render_template(self, template: PromptTemplate, variables: Dict[str, Any]) -> str:
        """
        Render template with variables.

        Args:
            template: Template to render
            variables: Variables to substitute

        Returns:
            Rendered prompt string
        """
        rendered = template.template

        for var_name, var_value in variables.items():
            placeholder = "{" + var_name + "}"
            rendered = rendered.replace(placeholder, str(var_value))

        return rendered

    def create_ab_test(
        self,
        name: str,
        description: str,
        variant_a_id: str,
        variant_b_id: str,
        traffic_split: float = 0.5
    ) -> ABTest:
        """
        Create a new A/B test.

        Args:
            name: Test name
            description: Test description
            variant_a_id: Template ID for variant A
            variant_b_id: Template ID for variant B
            traffic_split: % traffic to variant A (0.0 to 1.0)

        Returns:
            Created ABTest
        """
        # Validate templates exist
        if variant_a_id not in self.templates:
            raise ValueError(f"Template {variant_a_id} not found")
        if variant_b_id not in self.templates:
            raise ValueError(f"Template {variant_b_id} not found")

        # Ensure same role
        role_a = self.templates[variant_a_id].role
        role_b = self.templates[variant_b_id].role
        if role_a != role_b:
            raise ValueError(f"Templates must have same role: {role_a} != {role_b}")

        test_id = f"test_{hashlib.md5(name.encode()).hexdigest()[:8]}"

        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            status=ABTestStatus.ACTIVE,
            variant_a=variant_a_id,
            variant_b=variant_b_id,
            traffic_split=traffic_split,
            created_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            ended_at=None,
            metrics={
                "variant_a": {"executions": 0, "avg_quality": 0.0, "avg_confidence": 0.0},
                "variant_b": {"executions": 0, "avg_quality": 0.0, "avg_confidence": 0.0}
            }
        )

        self.ab_tests[test_id] = test
        return test

    def record_execution(
        self,
        template: PromptTemplate,
        rendered_prompt: str,
        output: str,
        quality_score: float,
        confidence_score: float,
        tokens_used: int,
        cost_usd: float,
        duration_seconds: float,
        user_id: str,
        ab_test: Optional[ABTest] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a prompt execution for analytics.

        Updates template performance scores and A/B test metrics.
        """
        execution_id = f"exec_{datetime.utcnow().timestamp()}"

        ab_variant = None
        if ab_test:
            ab_variant = "A" if template.template_id == ab_test.variant_a else "B"

        execution = PromptExecution(
            execution_id=execution_id,
            template_id=template.template_id,
            template_version=template.version,
            role=template.role,
            rendered_prompt=rendered_prompt,
            output=output,
            quality_score=quality_score,
            confidence_score=confidence_score,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            duration_seconds=duration_seconds,
            user_id=user_id,
            ab_test_id=ab_test.test_id if ab_test else None,
            ab_variant=ab_variant,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        self.executions.append(execution)

        # Update template performance score
        self._update_template_performance(template, quality_score, confidence_score)

        # Update A/B test metrics
        if ab_test and ab_variant:
            self._update_ab_test_metrics(ab_test, ab_variant, quality_score, confidence_score)

    def _update_template_performance(
        self,
        template: PromptTemplate,
        quality_score: float,
        confidence_score: float
    ):
        """Update template's rolling performance score."""
        # Use weighted average: 70% quality, 30% confidence
        current_score = (quality_score * 0.7) + (confidence_score * 0.3)

        if template.performance_score is None:
            template.performance_score = current_score
        else:
            # Exponential moving average (α = 0.2)
            alpha = 0.2
            template.performance_score = (
                alpha * current_score + (1 - alpha) * template.performance_score
            )

    def _update_ab_test_metrics(
        self,
        test: ABTest,
        variant: str,
        quality_score: float,
        confidence_score: float
    ):
        """Update A/B test metrics for a variant."""
        variant_key = f"variant_{variant.lower()}"
        metrics = test.metrics[variant_key]

        count = metrics["executions"]
        avg_quality = metrics["avg_quality"]
        avg_confidence = metrics["avg_confidence"]

        # Update running averages
        metrics["executions"] = count + 1
        metrics["avg_quality"] = (avg_quality * count + quality_score) / (count + 1)
        metrics["avg_confidence"] = (avg_confidence * count + confidence_score) / (count + 1)

    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """
        Get A/B test results.

        Returns:
            Dictionary with variant metrics and recommendation
        """
        test = self.ab_tests.get(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        variant_a_metrics = test.metrics["variant_a"]
        variant_b_metrics = test.metrics["variant_b"]

        # Calculate combined score
        def combined_score(metrics):
            return metrics["avg_quality"] * 0.7 + metrics["avg_confidence"] * 0.3

        score_a = combined_score(variant_a_metrics)
        score_b = combined_score(variant_b_metrics)

        # Determine winner (need at least 30 executions per variant)
        winner = None
        confidence = None

        if variant_a_metrics["executions"] >= 30 and variant_b_metrics["executions"] >= 30:
            if abs(score_a - score_b) > 0.05:  # 5% difference threshold
                winner = "A" if score_a > score_b else "B"
                confidence = abs(score_a - score_b) / max(score_a, score_b)

        return {
            "test_id": test_id,
            "name": test.name,
            "status": test.status,
            "variant_a": {
                "template_id": test.variant_a,
                "score": score_a,
                **variant_a_metrics
            },
            "variant_b": {
                "template_id": test.variant_b,
                "score": score_b,
                **variant_b_metrics
            },
            "winner": winner,
            "confidence": confidence,
            "recommendation": self._generate_recommendation(test, winner, confidence)
        }

    def _generate_recommendation(
        self,
        test: ABTest,
        winner: Optional[str],
        confidence: Optional[float]
    ) -> str:
        """Generate recommendation based on A/B test results."""
        if not winner:
            return "Continue test - insufficient data or no clear winner yet"

        if confidence and confidence > 0.15:
            return f"Strong winner: Variant {winner} - recommend rollout to 100%"
        elif confidence and confidence > 0.05:
            return f"Moderate winner: Variant {winner} - consider gradual rollout"
        else:
            return "Weak difference - may need more data or test is inconclusive"
