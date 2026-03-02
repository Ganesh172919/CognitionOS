"""
Feature Flags - Dynamic Configuration and Rollout System

Enterprise-grade feature flag evaluation with targeting rules,
percentage rollouts, A/B testing support, and real-time updates.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import hashlib

logger = logging.getLogger(__name__)


class FlagType(str, Enum):
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"


class RuleOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_EQUALS = "greater_than_equals"
    LESS_THAN_EQUALS = "less_than_equals"


@dataclass
class EvaluationContext:
    """Context for evaluating feature flags."""
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def get(self, path: str) -> Any:
        """Get a value by dot-notation path."""
        if path == "user_id":
            return self.user_id
        if path == "tenant_id":
            return self.tenant_id
        if path == "session_id":
            return self.session_id
        if path == "roles":
            return self.roles
        
        # Check custom attributes
        parts = path.split('.')
        curr = self.attributes
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr


@dataclass
class RuleCondition:
    """A single condition within a rule."""
    attribute: str
    operator: RuleOperator
    value: Any

    def evaluate(self, context: EvaluationContext) -> bool:
        context_value = context.get(self.attribute)
        if context_value is None:
            return False

        try:
            if self.operator == RuleOperator.EQUALS:
                return str(context_value) == str(self.value)
            elif self.operator == RuleOperator.NOT_EQUALS:
                return str(context_value) != str(self.value)
            elif self.operator == RuleOperator.IN:
                return context_value in self.value
            elif self.operator == RuleOperator.NOT_IN:
                return context_value not in self.value
            elif self.operator == RuleOperator.CONTAINS:
                return str(self.value) in str(context_value)
            elif self.operator == RuleOperator.NOT_CONTAINS:
                return str(self.value) not in str(context_value)
            elif self.operator == RuleOperator.STARTS_WITH:
                return str(context_value).startswith(str(self.value))
            elif self.operator == RuleOperator.ENDS_WITH:
                return str(context_value).endswith(str(self.value))
            elif self.operator == RuleOperator.MATCHES_REGEX:
                import re
                return bool(re.match(str(self.value), str(context_value)))
            elif self.operator == RuleOperator.GREATER_THAN:
                return float(context_value) > float(self.value)
            elif self.operator == RuleOperator.LESS_THAN:
                return float(context_value) < float(self.value)
            elif self.operator == RuleOperator.GREATER_THAN_EQUALS:
                return float(context_value) >= float(self.value)
            elif self.operator == RuleOperator.LESS_THAN_EQUALS:
                return float(context_value) <= float(self.value)
        except (ValueError, TypeError):
            return False
            
        return False


@dataclass
class RolloutAllocation:
    """A percentage allocation mapping to a specific variation."""
    variation_value: Any
    percentage: float  # 0.0 to 100.0


@dataclass
class FeatureRule:
    """A rule to determine if a specific variation or rollout applies."""
    id: str
    name: str
    conditions: List[RuleCondition] = field(default_factory=list)
    match_all_conditions: bool = True
    
    # If rule matches, return this variation or apply this rollout
    variation_value: Optional[Any] = None
    rollout: List[RolloutAllocation] = field(default_factory=list)

    def evaluate(self, context: EvaluationContext, flag_key: str) -> Optional[Any]:
        # Check conditions
        if not self.conditions:
            return None

        matched = True
        for condition in self.conditions:
            cond_match = condition.evaluate(context)
            if self.match_all_conditions and not cond_match:
                matched = False
                break
            if not self.match_all_conditions and cond_match:
                matched = True
                break

        if not self.match_all_conditions and not any(c.evaluate(context) for c in self.conditions):
            matched = False

        if not matched:
            return None

        # Determine value to return
        if self.variation_value is not None:
            return self.variation_value

        if self.rollout:
            return self._calculate_rollout(context, flag_key)

        return None

    def _calculate_rollout(self, context: EvaluationContext, flag_key: str) -> Optional[Any]:
        # Use user_id, tenant_id, or session_id for stable hashing
        identity = context.user_id or context.tenant_id or context.session_id or "anonymous"
        
        # Calculate consistent hash bucket (0-99.99)
        hash_input = f"{flag_key}:{identity}".encode('utf-8')
        hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        bucket = (hash_val % 10000) / 100.0

        current_threshold = 0.0
        for allocation in self.rollout:
            current_threshold += allocation.percentage
            if bucket < current_threshold:
                return allocation.variation_value

        return None


@dataclass
class FeatureFlag:
    """A feature flag definition."""
    key: str
    name: str
    description: str = ""
    type: FlagType = FlagType.BOOLEAN
    is_enabled: bool = False
    
    default_value: Any = False
    
    # Executed in order until one matches
    rules: List[FeatureRule] = field(default_factory=list)
    
    # If enabled and no rules match
    base_rollout: List[RolloutAllocation] = field(default_factory=list)
    
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1

    def evaluate(self, context: EvaluationContext) -> Any:
        # If master switch is off, return default
        if not self.is_enabled:
            return self.default_value

        # Check explicit rules
        for rule in self.rules:
            result = rule.evaluate(context, self.key)
            if result is not None:
                return result

        # Apply base rollout if present
        if self.base_rollout:
            # Create a dummy rule just to reuse the rollout logic
            dummy_rule = FeatureRule(id="base", name="Base Rollout", rollout=self.base_rollout)
            return dummy_rule._calculate_rollout(context, self.key)

        # Fallback to true if boolean flag is enabled with no rules
        if self.type == FlagType.BOOLEAN:
            return True

        return self.default_value


class FeatureFlagEvaluator:
    """
    Evaluates feature flags against a given context.
    Provides local evaluation capabilities caching flags memory.
    """

    def __init__(self, flags_data: Optional[Dict[str, Any]] = None):
        self._flags: Dict[str, FeatureFlag] = {}
        if flags_data:
            self.load_flags(flags_data)
            
        self._evaluation_hooks: List[Callable] = []

    def register_hook(self, hook: Callable[[str, Any, EvaluationContext], None]) -> None:
        """Register a hook to fire on flag evaluations (e.g. for analytics)."""
        self._evaluation_hooks.append(hook)

    def load_flags(self, flags_data: Dict[str, Any]) -> None:
        """Load flags from JSON dictionary."""
        # In a real system, this would deserialize from the database or CDN
        pass

    def add_flag(self, flag: FeatureFlag) -> None:
        """Add or update a flag."""
        self._flags[flag.key] = flag

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        """Get a flag definition."""
        return self._flags.get(key)

    def evaluate(self, key: str, context: EvaluationContext, default: Any = None) -> Any:
        """Evaluate a specific flag."""
        flag = self._flags.get(key)
        
        if not flag:
            value = default
        else:
            value = flag.evaluate(context)
            
        # Fire hooks (e.g., for experimentation analytics)
        for hook in self._evaluation_hooks:
            try:
                hook(key, value, context)
            except Exception:
                logger.exception("Feature flag evaluation hook failed")
                
        return value

    def is_enabled(self, key: str, context: EvaluationContext, default: bool = False) -> bool:
        """Evaluate a boolean flag."""
        val = self.evaluate(key, context, default)
        return bool(val)

    def get_all_evaluations(self, context: EvaluationContext) -> Dict[str, Any]:
        """Evaluate all flags for the given context (useful for bootstrapping UIs)."""
        return {
            key: flag.evaluate(context)
            for key, flag in self._flags.items()
        }


# Global default instance
_default_evaluator = FeatureFlagEvaluator()


def check_feature(key: str, context: EvaluationContext, default: bool = False) -> bool:
    """Helper to check a feature flag on the default evaluator."""
    return _default_evaluator.is_enabled(key, context, default)


def get_feature_value(key: str, context: EvaluationContext, default: Any = None) -> Any:
    """Helper to get a feature flag value on the default evaluator."""
    return _default_evaluator.evaluate(key, context, default)
