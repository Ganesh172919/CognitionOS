"""
Engagement Systems
Recommendation engine and viral referral system for user growth.
"""

from .recommendation_engine import (
    IntelligentRecommendationEngine,
    RecommendationType,
    RecommendationReason,
    UserProfile,
    WorkflowTemplate,
    Recommendation
)
from .referral_system import (
    ViralReferralSystem,
    ReferralStatus,
    RewardType,
    ReferralTier,
    Referral,
    Reward,
    ReferrerProfile
)

__all__ = [
    # Recommendations
    "IntelligentRecommendationEngine",
    "RecommendationType",
    "RecommendationReason",
    "UserProfile",
    "WorkflowTemplate",
    "Recommendation",
    # Referrals
    "ViralReferralSystem",
    "ReferralStatus",
    "RewardType",
    "ReferralTier",
    "Referral",
    "Reward",
    "ReferrerProfile",
]
