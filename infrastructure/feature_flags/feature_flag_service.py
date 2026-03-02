"""
Feature Flag Service — CognitionOS

Production feature flagging with:
- Boolean, percentage, and multivariate flags
- User/tenant/environment targeting rules
- Gradual rollout with percentage ramps
- A/B experiment assignment
- Audit trail for flag changes
- In-memory cache with TTL
- Analytics integration
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class FlagType(str, Enum):
    BOOLEAN = "boolean"
    PERCENTAGE = "percentage"
    MULTIVARIATE = "multivariate"
    JSON = "json"


class FlagStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class TargetOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    IN = "in"
    NOT_IN = "not_in"
    REGEX = "regex"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class TargetingRule:
    attribute: str  # e.g., "user_id", "tenant_id", "plan", "country"
    operator: TargetOperator
    values: List[Any]
    description: str = ""


@dataclass
class FlagVariant:
    key: str
    value: Any
    weight: float = 0.0  # 0.0-100.0 for percentage
    description: str = ""


@dataclass
class FeatureFlag:
    key: str
    name: str
    flag_type: FlagType
    status: FlagStatus = FlagStatus.ACTIVE
    default_value: Any = False
    description: str = ""
    variants: List[FlagVariant] = field(default_factory=list)
    targeting_rules: List[TargetingRule] = field(default_factory=list)
    percentage_rollout: float = 100.0  # 0-100
    environments: Set[str] = field(default_factory=lambda: {"development", "staging", "production"})
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "type": self.flag_type.value,
            "status": self.status.value,
            "default_value": self.default_value,
            "description": self.description,
            "percentage_rollout": self.percentage_rollout,
            "environments": list(self.environments),
            "variants": [{"key": v.key, "value": v.value, "weight": v.weight} for v in self.variants],
            "targeting_rules": [
                {"attribute": r.attribute, "operator": r.operator.value, "values": r.values}
                for r in self.targeting_rules
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tags": self.tags,
        }


@dataclass
class EvaluationContext:
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = "production"
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def identity_key(self) -> str:
        return self.user_id or self.tenant_id or "anonymous"


@dataclass
class FlagEvaluation:
    flag_key: str
    value: Any
    variant_key: Optional[str] = None
    reason: str = "default"
    evaluation_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Feature Flag Service
# ---------------------------------------------------------------------------


class FeatureFlagService:
    """Production feature flag evaluation engine."""

    def __init__(self, *, cache_ttl_seconds: int = 60, environment: str = "production") -> None:
        self._flags: Dict[str, FeatureFlag] = {}
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = cache_ttl_seconds
        self._environment = environment
        self._evaluation_log: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._change_log: List[Dict[str, Any]] = []
        self._override_flags: Dict[str, Dict[str, Any]] = {}  # user_id -> {flag_key: value}

    # ----- flag management -----

    def register_flag(self, flag: FeatureFlag) -> None:
        self._flags[flag.key] = flag
        self._invalidate_cache(flag.key)
        self._record_change("register", flag.key, None, flag.to_dict())
        logger.info("Feature flag registered: %s (%s)", flag.key, flag.flag_type.value)

    def update_flag(self, key: str, **updates: Any) -> Optional[FeatureFlag]:
        flag = self._flags.get(key)
        if not flag:
            return None
        old_state = flag.to_dict()
        for attr, value in updates.items():
            if hasattr(flag, attr):
                setattr(flag, attr, value)
        flag.updated_at = datetime.now(timezone.utc).isoformat()
        self._invalidate_cache(key)
        self._record_change("update", key, old_state, flag.to_dict())
        return flag

    def delete_flag(self, key: str) -> bool:
        flag = self._flags.pop(key, None)
        if flag:
            self._invalidate_cache(key)
            self._record_change("delete", key, flag.to_dict(), None)
            return True
        return False

    def archive_flag(self, key: str) -> bool:
        return self.update_flag(key, status=FlagStatus.ARCHIVED) is not None

    def get_flag(self, key: str) -> Optional[FeatureFlag]:
        return self._flags.get(key)

    def list_flags(self, *, tag: Optional[str] = None, status: Optional[FlagStatus] = None) -> List[FeatureFlag]:
        flags = list(self._flags.values())
        if tag:
            flags = [f for f in flags if tag in f.tags]
        if status:
            flags = [f for f in flags if f.status == status]
        return flags

    # ----- evaluation -----

    def evaluate(self, key: str, context: Optional[EvaluationContext] = None) -> FlagEvaluation:
        start = time.monotonic()
        ctx = context or EvaluationContext()
        flag = self._flags.get(key)

        if not flag:
            return FlagEvaluation(flag_key=key, value=False, reason="flag_not_found")

        if flag.status != FlagStatus.ACTIVE:
            return FlagEvaluation(flag_key=key, value=flag.default_value, reason="flag_inactive")

        if self._environment not in flag.environments:
            return FlagEvaluation(flag_key=key, value=flag.default_value, reason="environment_excluded")

        # Check user overrides
        if ctx.user_id and ctx.user_id in self._override_flags:
            user_overrides = self._override_flags[ctx.user_id]
            if key in user_overrides:
                elapsed = (time.monotonic() - start) * 1000
                return FlagEvaluation(flag_key=key, value=user_overrides[key], reason="user_override", evaluation_time_ms=elapsed)

        # Check targeting rules
        if flag.targeting_rules:
            for rule in flag.targeting_rules:
                if self._evaluate_rule(rule, ctx):
                    value = self._get_targeting_value(flag, ctx)
                    elapsed = (time.monotonic() - start) * 1000
                    self._record_evaluation(key, ctx, value, "targeting_rule")
                    return FlagEvaluation(flag_key=key, value=value, reason="targeting_rule", evaluation_time_ms=elapsed)

        # Check percentage rollout
        if flag.flag_type == FlagType.PERCENTAGE or flag.percentage_rollout < 100.0:
            in_rollout = self._is_in_percentage(ctx.identity_key, key, flag.percentage_rollout)
            if not in_rollout:
                elapsed = (time.monotonic() - start) * 1000
                self._record_evaluation(key, ctx, flag.default_value, "percentage_excluded")
                return FlagEvaluation(flag_key=key, value=flag.default_value, reason="percentage_excluded", evaluation_time_ms=elapsed)

        # Multivariate selection
        if flag.flag_type == FlagType.MULTIVARIATE and flag.variants:
            variant = self._select_variant(ctx.identity_key, key, flag.variants)
            elapsed = (time.monotonic() - start) * 1000
            self._record_evaluation(key, ctx, variant.value, "variant_selected")
            return FlagEvaluation(flag_key=key, value=variant.value, variant_key=variant.key, reason="variant_selected", evaluation_time_ms=elapsed)

        # Default
        value = True if flag.flag_type == FlagType.BOOLEAN else flag.default_value
        elapsed = (time.monotonic() - start) * 1000
        self._record_evaluation(key, ctx, value, "default")
        return FlagEvaluation(flag_key=key, value=value, reason="default", evaluation_time_ms=elapsed)

    def is_enabled(self, key: str, context: Optional[EvaluationContext] = None) -> bool:
        result = self.evaluate(key, context)
        return bool(result.value)

    # ----- overrides -----

    def set_user_override(self, user_id: str, flag_key: str, value: Any) -> None:
        if user_id not in self._override_flags:
            self._override_flags[user_id] = {}
        self._override_flags[user_id][flag_key] = value

    def remove_user_override(self, user_id: str, flag_key: str) -> None:
        if user_id in self._override_flags:
            self._override_flags[user_id].pop(flag_key, None)

    # ----- internal evaluation helpers -----

    def _evaluate_rule(self, rule: TargetingRule, ctx: EvaluationContext) -> bool:
        value = ctx.attributes.get(rule.attribute)
        if value is None:
            if rule.attribute == "user_id":
                value = ctx.user_id
            elif rule.attribute == "tenant_id":
                value = ctx.tenant_id
            elif rule.attribute == "environment":
                value = ctx.environment

        if value is None:
            return False

        if rule.operator == TargetOperator.EQUALS:
            return value == rule.values[0]
        elif rule.operator == TargetOperator.NOT_EQUALS:
            return value != rule.values[0]
        elif rule.operator == TargetOperator.CONTAINS:
            return any(v in str(value) for v in rule.values)
        elif rule.operator == TargetOperator.IN:
            return value in rule.values
        elif rule.operator == TargetOperator.NOT_IN:
            return value not in rule.values
        elif rule.operator == TargetOperator.GREATER_THAN:
            return float(value) > float(rule.values[0])
        elif rule.operator == TargetOperator.LESS_THAN:
            return float(value) < float(rule.values[0])

        return False

    def _is_in_percentage(self, identity: str, flag_key: str, percentage: float) -> bool:
        hash_input = f"{identity}:{flag_key}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        bucket = hash_value % 100
        return bucket < percentage

    def _select_variant(self, identity: str, flag_key: str, variants: List[FlagVariant]) -> FlagVariant:
        hash_input = f"{identity}:{flag_key}:variant"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)
        bucket = hash_value % 100

        cumulative = 0.0
        for variant in variants:
            cumulative += variant.weight
            if bucket < cumulative:
                return variant
        return variants[-1]

    def _get_targeting_value(self, flag: FeatureFlag, ctx: EvaluationContext) -> Any:
        if flag.flag_type == FlagType.BOOLEAN:
            return True
        if flag.variants:
            return self._select_variant(ctx.identity_key, flag.key, flag.variants).value
        return flag.default_value

    # ----- caching -----

    def _invalidate_cache(self, key: str) -> None:
        keys_to_remove = [k for k in self._cache if k.startswith(f"{key}:")]
        for k in keys_to_remove:
            self._cache.pop(k, None)
            self._cache_timestamps.pop(k, None)

    # ----- analytics -----

    def _record_evaluation(self, flag_key: str, ctx: EvaluationContext, value: Any, reason: str) -> None:
        self._metrics[flag_key][reason] += 1
        if len(self._evaluation_log) < 10000:
            self._evaluation_log.append({
                "flag_key": flag_key,
                "value": value,
                "reason": reason,
                "user_id": ctx.user_id,
                "tenant_id": ctx.tenant_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def _record_change(self, action: str, flag_key: str, old: Any, new: Any) -> None:
        self._change_log.append({
            "action": action,
            "flag_key": flag_key,
            "old_state": old,
            "new_state": new,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def get_metrics(self, flag_key: Optional[str] = None) -> Dict[str, Any]:
        if flag_key:
            return dict(self._metrics.get(flag_key, {}))
        return {k: dict(v) for k, v in self._metrics.items()}

    def get_change_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self._change_log[-limit:]

    def get_evaluation_log(self, *, flag_key: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        logs = self._evaluation_log
        if flag_key:
            logs = [l for l in logs if l["flag_key"] == flag_key]
        return logs[-limit:]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_service: Optional[FeatureFlagService] = None


def get_feature_flag_service() -> FeatureFlagService:
    global _service
    if _service is None:
        _service = FeatureFlagService()
    return _service


def init_feature_flag_service(**kwargs: Any) -> FeatureFlagService:
    global _service
    _service = FeatureFlagService(**kwargs)
    return _service
