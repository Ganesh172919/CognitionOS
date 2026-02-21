"""
Growth Systems - Viral Loops, Referrals, and Network Effects

Mechanisms to drive organic growth through viral features,
referral programs, and network effects.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RewardType(str, Enum):
    """Types of rewards"""
    CREDIT = "credit"
    FREE_MONTH = "free_month"
    FEATURE_UNLOCK = "feature_unlock"
    STORAGE_BONUS = "storage_bonus"
    API_CALLS_BONUS = "api_calls_bonus"


@dataclass
class ReferralProgram:
    """Referral program configuration"""
    program_id: str
    name: str
    active: bool = True

    # Rewards
    referrer_reward: RewardType = RewardType.CREDIT
    referrer_reward_value: float = 10.0
    referee_reward: RewardType = RewardType.FREE_MONTH
    referee_reward_value: float = 29.0

    # Requirements
    requires_payment: bool = True
    minimum_subscription_days: int = 30

    # Limits
    max_referrals_per_user: Optional[int] = None
    max_total_referrals: Optional[int] = None

    # Tracking
    total_referrals: int = 0
    total_rewards_given: float = 0.0


@dataclass
class Referral:
    """Individual referral"""
    referral_id: str
    referrer_id: str
    referee_id: str
    referral_code: str
    created_at: datetime

    # Status
    completed: bool = False
    completed_at: Optional[datetime] = None
    reward_given: bool = False

    # Context
    source: str = "unknown"  # email, social, embed, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


class ViralLoopEngine:
    """
    Viral loop and referral system

    Drives organic growth through incentivized referrals and
    network effects.
    """

    def __init__(self):
        self._programs: Dict[str, ReferralProgram] = {}
        self._referrals: List[Referral] = []
        self._referral_codes: Dict[str, str] = {}  # code -> referrer_id
        self._initialize_default_program()

    def _initialize_default_program(self):
        """Initialize default referral program"""
        self._programs["default"] = ReferralProgram(
            program_id="default",
            name="Standard Referral Program",
            referrer_reward=RewardType.CREDIT,
            referrer_reward_value=25.0,
            referee_reward=RewardType.FREE_MONTH,
            referee_reward_value=29.0,
            requires_payment=True,
            minimum_subscription_days=30
        )

    def generate_referral_code(self, user_id: str) -> str:
        """Generate unique referral code for user"""
        import hashlib
        import secrets

        # Generate code
        random_part = secrets.token_hex(4)
        code = f"{user_id[:4].upper()}{random_part.upper()}"

        self._referral_codes[code] = user_id
        logger.info(f"Generated referral code {code} for user {user_id}")

        return code

    async def create_referral(
        self,
        referral_code: str,
        referee_id: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Referral]:
        """
        Create new referral

        Args:
            referral_code: Referrer's code
            referee_id: New user being referred
            source: Source of referral
            metadata: Additional tracking data

        Returns:
            Referral object if valid
        """
        # Validate code
        referrer_id = self._referral_codes.get(referral_code)
        if not referrer_id:
            logger.warning(f"Invalid referral code: {referral_code}")
            return None

        # Check self-referral
        if referrer_id == referee_id:
            logger.warning("Self-referral attempt blocked")
            return None

        # Check program limits
        program = self._programs["default"]
        if program.max_referrals_per_user:
            referrer_count = len([r for r in self._referrals if r.referrer_id == referrer_id])
            if referrer_count >= program.max_referrals_per_user:
                logger.warning(f"User {referrer_id} exceeded referral limit")
                return None

        # Create referral
        referral = Referral(
            referral_id=f"ref_{len(self._referrals)}_{int(datetime.utcnow().timestamp())}",
            referrer_id=referrer_id,
            referee_id=referee_id,
            referral_code=referral_code,
            created_at=datetime.utcnow(),
            source=source,
            metadata=metadata or {}
        )

        self._referrals.append(referral)
        logger.info(f"Created referral: {referrer_id} -> {referee_id}")

        return referral

    async def complete_referral(
        self,
        referral_id: str
    ) -> bool:
        """
        Complete referral and distribute rewards

        Called when referee meets program requirements (e.g., paid subscription)
        """
        referral = next((r for r in self._referrals if r.referral_id == referral_id), None)
        if not referral:
            return False

        if referral.completed:
            logger.warning(f"Referral {referral_id} already completed")
            return False

        program = self._programs["default"]

        # Mark complete
        referral.completed = True
        referral.completed_at = datetime.utcnow()

        # Give rewards
        await self._give_reward(
            referral.referrer_id,
            program.referrer_reward,
            program.referrer_reward_value
        )

        await self._give_reward(
            referral.referee_id,
            program.referee_reward,
            program.referee_reward_value
        )

        referral.reward_given = True
        program.total_referrals += 1
        program.total_rewards_given += program.referrer_reward_value + program.referee_reward_value

        logger.info(f"Completed referral {referral_id} and distributed rewards")

        return True

    async def _give_reward(
        self,
        user_id: str,
        reward_type: RewardType,
        value: float
    ):
        """Give reward to user"""
        logger.info(f"Giving {reward_type.value} reward of ${value} to {user_id}")
        # Would integrate with billing system

    def get_referral_stats(self, user_id: str) -> Dict[str, Any]:
        """Get referral statistics for user"""

        referrals = [r for r in self._referrals if r.referrer_id == user_id]
        completed = [r for r in referrals if r.completed]

        return {
            "user_id": user_id,
            "total_referrals": len(referrals),
            "completed_referrals": len(completed),
            "pending_referrals": len(referrals) - len(completed),
            "completion_rate": len(completed) / len(referrals) if referrals else 0,
            "estimated_value": len(completed) * self._programs["default"].referrer_reward_value
        }

    def get_viral_coefficient(self) -> float:
        """
        Calculate viral coefficient (K-factor)

        K > 1 means exponential growth
        """
        if not self._referrals:
            return 0.0

        # K = (number of invites sent per user) × (conversion rate)
        unique_referrers = len(set(r.referrer_id for r in self._referrals))
        total_referrals = len(self._referrals)
        completed = len([r for r in self._referrals if r.completed])

        invites_per_user = total_referrals / unique_referrers if unique_referrers > 0 else 0
        conversion_rate = completed / total_referrals if total_referrals > 0 else 0

        k_factor = invites_per_user * conversion_rate

        logger.info(f"Viral coefficient (K-factor): {k_factor:.3f}")

        return k_factor


class SocialSharingEngine:
    """
    Social sharing and viral features

    Makes it easy for users to share their work and invite others.
    """

    def generate_share_url(
        self,
        content_type: str,
        content_id: str,
        user_id: str
    ) -> str:
        """Generate shareable URL with tracking"""

        import hashlib
        tracking_code = hashlib.md5(f"{user_id}:{content_id}".encode()).hexdigest()[:8]

        share_url = f"https://cognitionos.com/share/{content_type}/{content_id}?ref={tracking_code}"

        logger.info(f"Generated share URL: {share_url}")

        return share_url

    def generate_social_preview(
        self,
        title: str,
        description: str,
        image_url: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate social media preview metadata"""

        preview = {
            "og:title": title,
            "og:description": description,
            "og:type": "website",
            "og:site_name": "CognitionOS",
            "twitter:card": "summary_large_image",
            "twitter:title": title,
            "twitter:description": description
        }

        if image_url:
            preview["og:image"] = image_url
            preview["twitter:image"] = image_url

        return preview

    def generate_embed_code(
        self,
        content_id: str,
        width: int = 800,
        height: int = 600
    ) -> str:
        """Generate embeddable iframe code"""

        embed_code = f"""<iframe
  src="https://cognitionos.com/embed/{content_id}"
  width="{width}"
  height="{height}"
  frameborder="0"
  allowfullscreen>
</iframe>"""

        return embed_code


class NetworkEffectEngine:
    """
    Network effect mechanisms

    Creates value that increases with more users.
    """

    def __init__(self):
        self._public_workflows = []
        self._community_plugins = []

    async def publish_workflow_template(
        self,
        workflow_id: str,
        user_id: str,
        name: str,
        description: str,
        tags: List[str]
    ):
        """Publish workflow as reusable template"""

        template = {
            "template_id": f"template_{len(self._public_workflows)}",
            "workflow_id": workflow_id,
            "author_id": user_id,
            "name": name,
            "description": description,
            "tags": tags,
            "published_at": datetime.utcnow(),
            "usage_count": 0,
            "rating": 0.0
        }

        self._public_workflows.append(template)

        logger.info(f"Published workflow template: {name}")

        return template

    async def use_template(
        self,
        template_id: str,
        user_id: str
    ):
        """Use a community template"""

        template = next((t for t in self._public_workflows if t["template_id"] == template_id), None)
        if template:
            template["usage_count"] += 1
            logger.info(f"Template {template_id} used by {user_id}")

    def get_trending_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending templates"""

        # Sort by recent usage and rating
        sorted_templates = sorted(
            self._public_workflows,
            key=lambda t: (t["usage_count"], t["rating"]),
            reverse=True
        )

        return sorted_templates[:limit]

    async def contribute_plugin(
        self,
        plugin_name: str,
        author_id: str,
        description: str,
        source_url: str
    ):
        """Contribute plugin to marketplace"""

        plugin = {
            "plugin_id": f"plugin_{len(self._community_plugins)}",
            "name": plugin_name,
            "author_id": author_id,
            "description": description,
            "source_url": source_url,
            "contributed_at": datetime.utcnow(),
            "downloads": 0,
            "rating": 0.0
        }

        self._community_plugins.append(plugin)

        logger.info(f"Contributed plugin: {plugin_name}")

        return plugin

    def get_network_metrics(self) -> Dict[str, Any]:
        """Get network effect metrics"""

        return {
            "public_templates": len(self._public_workflows),
            "total_template_uses": sum(t["usage_count"] for t in self._public_workflows),
            "community_plugins": len(self._community_plugins),
            "total_downloads": sum(p["downloads"] for p in self._community_plugins),
            "network_value": len(self._public_workflows) * len(self._community_plugins)
        }
