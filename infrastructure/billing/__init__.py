"""
Billing Infrastructure Module

Provides billing provider abstraction, entitlement enforcement, and usage tracking.
"""

from infrastructure.billing.provider import (
    BillingProvider,
    BillingProviderError,
    StripeBillingProvider,
    MockBillingProvider,
)

from infrastructure.billing.entitlement_enforcer import (
    EntitlementEnforcer,
    EntitlementEnforcementError,
    EntitlementMiddleware,
    require_entitlement,
    get_entitlement_enforcer,
    check_entitlement_or_raise,
)

from infrastructure.billing.usage_tracker import (
    UsageTracker,
    UsageAggregator,
)

__all__ = [
    # Provider
    "BillingProvider",
    "BillingProviderError",
    "StripeBillingProvider",
    "MockBillingProvider",
    
    # Enforcer
    "EntitlementEnforcer",
    "EntitlementEnforcementError",
    "EntitlementMiddleware",
    "require_entitlement",
    "get_entitlement_enforcer",
    "check_entitlement_or_raise",
    
    # Tracker
    "UsageTracker",
    "UsageAggregator",
]
