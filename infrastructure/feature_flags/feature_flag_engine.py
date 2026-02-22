"""
Dynamic Feature Flagging and Experimentation Engine

Advanced feature flag system with:
- Dynamic feature toggling
- A/B testing and multivariate experiments
- Gradual rollouts and canary deployments
- User segmentation and targeting
- Real-time metrics collection
"""

import random
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json


class FeatureState(Enum):
    """Feature flag states"""
    OFF = "off"
    ON = "on"
    CONDITIONAL = "conditional"


class RolloutStrategy(Enum):
    """Rollout strategies"""
    PERCENTAGE = "percentage"
    USER_LIST = "user_list"
    ATTRIBUTE_BASED = "attribute_based"
    GRADUAL = "gradual"


@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    flag_id: str
    name: str
    description: str
    state: FeatureState
    rollout_percentage: float = 0.0
    enabled_users: List[str] = field(default_factory=list)
    disabled_users: List[str] = field(default_factory=list)
    targeting_rules: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Experiment:
    """A/B test experiment configuration"""
    experiment_id: str
    name: str
    description: str
    variants: List[str]
    traffic_allocation: Dict[str, float]
    targeting_rules: List[Dict[str, Any]] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_active: bool = True
    metrics: List[str] = field(default_factory=list)


@dataclass
class ExperimentAssignment:
    """User assignment to experiment variant"""
    user_id: str
    experiment_id: str
    variant: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)


class FeatureFlagEngine:
    """
    Dynamic Feature Flagging and Experimentation Engine

    Features:
    - Real-time feature toggling
    - Percentage-based rollouts
    - User-specific overrides
    - Attribute-based targeting
    - A/B testing and multivariate experiments
    - Gradual rollout strategies
    - Environment-based configurations
    - Real-time metrics tracking
    - Kill switches for emergencies
    """

    def __init__(self):
        self.flags: Dict[str, FeatureFlag] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.assignments: Dict[str, ExperimentAssignment] = {}
        self._evaluation_cache: Dict[str, bool] = {}
        self._metrics: Dict[str, Dict[str, int]] = {}

    def create_flag(self, flag: FeatureFlag):
        """Create or update feature flag"""
        flag.updated_at = datetime.utcnow()
        self.flags[flag.flag_id] = flag
        self._clear_cache_for_flag(flag.flag_id)

    def delete_flag(self, flag_id: str):
        """Delete feature flag"""
        if flag_id in self.flags:
            del self.flags[flag_id]
            self._clear_cache_for_flag(flag_id)

    def is_enabled(
        self,
        flag_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if feature is enabled for user

        Args:
            flag_id: Feature flag identifier
            user_id: User identifier
            context: Additional context for evaluation

        Returns:
            True if feature is enabled
        """
        # Check cache
        cache_key = f"{flag_id}:{user_id}"
        if cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]

        # Get flag
        flag = self.flags.get(flag_id)
        if not flag:
            return False

        # Evaluate flag
        result = self._evaluate_flag(flag, user_id, context or {})

        # Cache result (with TTL in production)
        self._evaluation_cache[cache_key] = result

        # Track metric
        self._track_evaluation(flag_id, user_id, result)

        return result

    def _evaluate_flag(
        self,
        flag: FeatureFlag,
        user_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate feature flag for user"""
        # State: OFF - always disabled
        if flag.state == FeatureState.OFF:
            return False

        # State: ON - always enabled
        if flag.state == FeatureState.ON:
            return True

        # Check explicit overrides first
        if user_id in flag.disabled_users:
            return False

        if user_id in flag.enabled_users:
            return True

        # Evaluate targeting rules
        if flag.targeting_rules:
            for rule in flag.targeting_rules:
                if self._evaluate_targeting_rule(rule, user_id, context):
                    return rule.get("enabled", True)

        # Percentage-based rollout
        if flag.rollout_percentage > 0:
            return self._is_in_rollout_percentage(user_id, flag.flag_id, flag.rollout_percentage)

        return False

    def _evaluate_targeting_rule(
        self,
        rule: Dict[str, Any],
        user_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a targeting rule"""
        conditions = rule.get("conditions", [])

        for condition in conditions:
            attribute = condition.get("attribute")
            operator = condition.get("operator")
            value = condition.get("value")

            # Get actual value from context
            actual_value = context.get(attribute)

            # Evaluate condition
            if not self._evaluate_condition(actual_value, operator, value):
                return False

        return True

    def _evaluate_condition(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate a single condition"""
        if operator == "equals":
            return actual == expected
        elif operator == "not_equals":
            return actual != expected
        elif operator == "in":
            return actual in expected
        elif operator == "not_in":
            return actual not in expected
        elif operator == "greater_than":
            return actual > expected
        elif operator == "less_than":
            return actual < expected
        elif operator == "contains":
            return expected in str(actual)
        elif operator == "regex_match":
            import re
            return bool(re.match(expected, str(actual)))

        return False

    def _is_in_rollout_percentage(
        self,
        user_id: str,
        flag_id: str,
        percentage: float
    ) -> bool:
        """Determine if user is in rollout percentage"""
        # Consistent hashing for stable assignments
        hash_input = f"{flag_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        user_percentage = (hash_value % 100) / 100.0

        return user_percentage < percentage

    def create_experiment(self, experiment: Experiment):
        """Create A/B test experiment"""
        self.experiments[experiment.experiment_id] = experiment

    def get_variant(
        self,
        experiment_id: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get experiment variant for user

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            context: Additional context

        Returns:
            Assigned variant name or None
        """
        # Check for existing assignment
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self.assignments:
            return self.assignments[assignment_key].variant

        experiment = self.experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return None

        # Check targeting rules
        if experiment.targeting_rules:
            passes_targeting = any(
                self._evaluate_targeting_rule(rule, user_id, context or {})
                for rule in experiment.targeting_rules
            )
            if not passes_targeting:
                return None

        # Assign variant
        variant = self._assign_variant(experiment, user_id)

        # Store assignment
        assignment = ExperimentAssignment(
            user_id=user_id,
            experiment_id=experiment_id,
            variant=variant
        )
        self.assignments[assignment_key] = assignment

        return variant

    def _assign_variant(self, experiment: Experiment, user_id: str) -> str:
        """Assign user to experiment variant"""
        # Use consistent hashing
        hash_input = f"{experiment.experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        random_value = (hash_value % 10000) / 10000.0

        # Determine variant based on traffic allocation
        cumulative = 0.0
        for variant, allocation in experiment.traffic_allocation.items():
            cumulative += allocation
            if random_value < cumulative:
                return variant

        # Fallback to first variant
        return experiment.variants[0]

    def track_metric(
        self,
        experiment_id: str,
        user_id: str,
        metric_name: str,
        value: float = 1.0
    ):
        """Track experiment metric"""
        assignment = self.assignments.get(f"{experiment_id}:{user_id}")
        if not assignment:
            return

        key = f"{experiment_id}:{assignment.variant}:{metric_name}"

        if key not in self._metrics:
            self._metrics[key] = {"count": 0, "total": 0, "conversions": 0}

        self._metrics[key]["count"] += 1
        self._metrics[key]["total"] += value

        if value > 0:
            self._metrics[key]["conversions"] += 1

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results and statistics"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return {}

        results = {
            "experiment_id": experiment_id,
            "variants": {},
            "total_assignments": 0
        }

        # Count assignments per variant
        for assignment in self.assignments.values():
            if assignment.experiment_id == experiment_id:
                variant = assignment.variant
                if variant not in results["variants"]:
                    results["variants"][variant] = {
                        "assignments": 0,
                        "metrics": {}
                    }
                results["variants"][variant]["assignments"] += 1
                results["total_assignments"] += 1

        # Add metrics
        for key, metrics in self._metrics.items():
            parts = key.split(":")
            if len(parts) >= 3 and parts[0] == experiment_id:
                exp_id, variant, metric_name = parts[0], parts[1], parts[2]

                if variant in results["variants"]:
                    results["variants"][variant]["metrics"][metric_name] = {
                        "count": metrics["count"],
                        "total": metrics["total"],
                        "avg": metrics["total"] / max(metrics["count"], 1),
                        "conversion_rate": metrics["conversions"] / max(metrics["count"], 1)
                    }

        return results

    def gradual_rollout(
        self,
        flag_id: str,
        target_percentage: float,
        increment: float = 0.1,
        interval_hours: int = 24
    ):
        """
        Configure gradual rollout for feature flag

        Args:
            flag_id: Feature flag identifier
            target_percentage: Target rollout percentage (0-1)
            increment: Percentage increment per step
            interval_hours: Hours between increments
        """
        flag = self.flags.get(flag_id)
        if not flag:
            return

        # Store rollout configuration in metadata
        flag.metadata["gradual_rollout"] = {
            "target_percentage": target_percentage,
            "increment": increment,
            "interval_hours": interval_hours,
            "current_percentage": flag.rollout_percentage,
            "started_at": datetime.utcnow().isoformat(),
            "next_increment_at": (
                datetime.utcnow().timestamp() + (interval_hours * 3600)
            )
        }

    def emergency_disable(self, flag_id: str):
        """Emergency kill switch - immediately disable feature"""
        flag = self.flags.get(flag_id)
        if flag:
            flag.state = FeatureState.OFF
            flag.updated_at = datetime.utcnow()
            self._clear_cache_for_flag(flag_id)

    def _track_evaluation(self, flag_id: str, user_id: str, result: bool):
        """Track flag evaluation for analytics"""
        key = f"flag:{flag_id}:{result}"
        if key not in self._metrics:
            self._metrics[key] = {"count": 0}
        self._metrics[key]["count"] += 1

    def _clear_cache_for_flag(self, flag_id: str):
        """Clear evaluation cache for flag"""
        keys_to_remove = [k for k in self._evaluation_cache.keys() if k.startswith(f"{flag_id}:")]
        for key in keys_to_remove:
            del self._evaluation_cache[key]

    def get_flag_metrics(self, flag_id: str) -> Dict[str, Any]:
        """Get metrics for feature flag"""
        enabled_count = self._metrics.get(f"flag:{flag_id}:True", {}).get("count", 0)
        disabled_count = self._metrics.get(f"flag:{flag_id}:False", {}).get("count", 0)
        total = enabled_count + disabled_count

        return {
            "flag_id": flag_id,
            "enabled_count": enabled_count,
            "disabled_count": disabled_count,
            "total_evaluations": total,
            "enabled_percentage": enabled_count / max(total, 1)
        }

    def export_configuration(self) -> str:
        """Export all feature flags and experiments"""
        config = {
            "flags": {
                flag_id: {
                    "name": flag.name,
                    "state": flag.state.value,
                    "rollout_percentage": flag.rollout_percentage,
                    "enabled_users": flag.enabled_users,
                    "targeting_rules": flag.targeting_rules
                }
                for flag_id, flag in self.flags.items()
            },
            "experiments": {
                exp_id: {
                    "name": exp.name,
                    "variants": exp.variants,
                    "traffic_allocation": exp.traffic_allocation,
                    "is_active": exp.is_active
                }
                for exp_id, exp in self.experiments.items()
            }
        }

        return json.dumps(config, indent=2)

    def import_configuration(self, config_json: str):
        """Import feature flags and experiments from JSON"""
        config = json.loads(config_json)

        # Import flags
        for flag_id, flag_data in config.get("flags", {}).items():
            flag = FeatureFlag(
                flag_id=flag_id,
                name=flag_data["name"],
                description="",
                state=FeatureState(flag_data["state"]),
                rollout_percentage=flag_data.get("rollout_percentage", 0.0),
                enabled_users=flag_data.get("enabled_users", []),
                targeting_rules=flag_data.get("targeting_rules", [])
            )
            self.create_flag(flag)

        # Import experiments
        for exp_id, exp_data in config.get("experiments", {}).items():
            experiment = Experiment(
                experiment_id=exp_id,
                name=exp_data["name"],
                description="",
                variants=exp_data["variants"],
                traffic_allocation=exp_data["traffic_allocation"],
                is_active=exp_data.get("is_active", True)
            )
            self.create_experiment(experiment)
