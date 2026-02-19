"""
Viral Referral System
Incentivized referral program to drive viral growth.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ReferralStatus(str, Enum):
    """Referral status"""
    PENDING = "pending"
    SIGNED_UP = "signed_up"
    ACTIVATED = "activated"
    CONVERTED = "converted"


class RewardType(str, Enum):
    """Types of rewards"""
    CREDITS = "credits"
    PREMIUM_TRIAL = "premium_trial"
    STORAGE_BOOST = "storage_boost"
    API_QUOTA = "api_quota"
    DISCOUNT = "discount"
    FEATURE_UNLOCK = "feature_unlock"


class ReferralTier(str, Enum):
    """Referral program tiers"""
    BRONZE = "bronze"  # 1-5 referrals
    SILVER = "silver"  # 6-20 referrals
    GOLD = "gold"      # 21-50 referrals
    PLATINUM = "platinum"  # 51+ referrals


class Referral(BaseModel):
    """Individual referral record"""
    referral_id: str = Field(default_factory=lambda: str(uuid4()))
    referrer_user_id: str
    referred_email: str
    referred_user_id: Optional[str] = None
    status: ReferralStatus = ReferralStatus.PENDING
    referral_code: str
    signed_up_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    converted_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Reward(BaseModel):
    """Reward given to referrer"""
    reward_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    reward_type: RewardType
    value: float
    description: str
    earned_from_referral_id: str
    claimed: bool = False
    claimed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReferrerProfile(BaseModel):
    """Profile of a referrer"""
    user_id: str
    referral_code: str
    total_referrals: int = 0
    successful_referrals: int = 0
    pending_referrals: int = 0
    total_rewards_earned: float = 0.0
    current_tier: ReferralTier = ReferralTier.BRONZE
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ViralReferralSystem:
    """
    Complete referral system with tiered rewards and viral mechanics.
    """

    def __init__(self):
        self.referrals: Dict[str, Referral] = {}
        self.rewards: Dict[str, Reward] = {}
        self.referrer_profiles: Dict[str, ReferrerProfile] = {}
        self.referral_codes: Dict[str, str] = {}  # code -> user_id
        self._initialize_reward_config()

    def _initialize_reward_config(self):
        """Initialize reward configuration"""
        self.reward_config = {
            ReferralStatus.SIGNED_UP: {
                "referrer": [
                    {"type": RewardType.CREDITS, "value": 5.0, "description": "$5 credit"},
                ],
                "referred": [
                    {"type": RewardType.CREDITS, "value": 5.0, "description": "$5 welcome credit"},
                ]
            },
            ReferralStatus.ACTIVATED: {
                "referrer": [
                    {"type": RewardType.CREDITS, "value": 10.0, "description": "$10 bonus credit"},
                ],
                "referred": []
            },
            ReferralStatus.CONVERTED: {
                "referrer": [
                    {"type": RewardType.CREDITS, "value": 50.0, "description": "$50 conversion bonus"},
                    {"type": RewardType.PREMIUM_TRIAL, "value": 30.0, "description": "30-day premium trial"},
                ],
                "referred": [
                    {"type": RewardType.DISCOUNT, "value": 20.0, "description": "20% discount on first month"},
                ]
            }
        }

        # Tier multipliers
        self.tier_multipliers = {
            ReferralTier.BRONZE: 1.0,
            ReferralTier.SILVER: 1.2,
            ReferralTier.GOLD: 1.5,
            ReferralTier.PLATINUM: 2.0
        }

    async def create_referral_code(
        self,
        user_id: str,
        custom_code: Optional[str] = None
    ) -> str:
        """
        Create unique referral code for user
        """
        if user_id in self.referrer_profiles:
            return self.referrer_profiles[user_id].referral_code

        # Generate code
        if custom_code and custom_code not in self.referral_codes:
            code = custom_code.upper()
        else:
            code = self._generate_referral_code(user_id)

        # Create referrer profile
        profile = ReferrerProfile(
            user_id=user_id,
            referral_code=code
        )

        self.referrer_profiles[user_id] = profile
        self.referral_codes[code] = user_id

        return code

    def _generate_referral_code(self, user_id: str) -> str:
        """Generate unique referral code"""
        import hashlib
        import random

        # Generate from user_id + random salt
        salt = str(random.randint(1000, 9999))
        hash_input = f"{user_id}{salt}".encode()
        hash_output = hashlib.sha256(hash_input).hexdigest()

        # Take first 8 characters and make uppercase
        code = hash_output[:8].upper()

        # Ensure uniqueness
        while code in self.referral_codes:
            salt = str(random.randint(1000, 9999))
            hash_input = f"{user_id}{salt}".encode()
            hash_output = hashlib.sha256(hash_input).hexdigest()
            code = hash_output[:8].upper()

        return code

    async def create_referral(
        self,
        referrer_user_id: str,
        referred_email: str,
        referral_code: Optional[str] = None
    ) -> Referral:
        """
        Create a new referral
        """
        # Get or create referrer code
        if referral_code:
            if referral_code not in self.referral_codes:
                return None
            referrer_user_id = self.referral_codes[referral_code]
        else:
            referral_code = await self.create_referral_code(referrer_user_id)

        # Create referral
        referral = Referral(
            referrer_user_id=referrer_user_id,
            referred_email=referred_email.lower(),
            referral_code=referral_code
        )

        self.referrals[referral.referral_id] = referral

        # Update referrer profile
        profile = self.referrer_profiles[referrer_user_id]
        profile.total_referrals += 1
        profile.pending_referrals += 1

        return referral

    async def track_referral_signup(
        self,
        referred_email: str,
        referred_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Track when referred user signs up
        """
        # Find referral by email
        referral = self._find_referral_by_email(referred_email)

        if not referral:
            return None

        # Update referral
        referral.referred_user_id = referred_user_id
        referral.status = ReferralStatus.SIGNED_UP
        referral.signed_up_at = datetime.utcnow()

        # Update referrer profile
        profile = self.referrer_profiles[referral.referrer_user_id]
        profile.pending_referrals -= 1
        profile.successful_referrals += 1

        # Grant rewards
        rewards = await self._grant_rewards(
            referral,
            ReferralStatus.SIGNED_UP
        )

        # Check tier upgrade
        await self._check_tier_upgrade(profile)

        return {
            "referral_id": referral.referral_id,
            "status": referral.status.value,
            "rewards_granted": len(rewards),
            "rewards": [
                {
                    "type": r.reward_type.value,
                    "value": r.value,
                    "description": r.description
                }
                for r in rewards
            ]
        }

    async def track_referral_activation(
        self,
        referred_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Track when referred user activates (e.g., completes onboarding, first action)
        """
        referral = self._find_referral_by_user(referred_user_id)

        if not referral or referral.status != ReferralStatus.SIGNED_UP:
            return None

        # Update referral
        referral.status = ReferralStatus.ACTIVATED
        referral.activated_at = datetime.utcnow()

        # Grant rewards
        rewards = await self._grant_rewards(
            referral,
            ReferralStatus.ACTIVATED
        )

        return {
            "referral_id": referral.referral_id,
            "status": referral.status.value,
            "rewards_granted": len(rewards)
        }

    async def track_referral_conversion(
        self,
        referred_user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Track when referred user converts (e.g., upgrades to paid)
        """
        referral = self._find_referral_by_user(referred_user_id)

        if not referral or referral.status == ReferralStatus.CONVERTED:
            return None

        # Update referral
        referral.status = ReferralStatus.CONVERTED
        referral.converted_at = datetime.utcnow()

        # Grant conversion rewards (highest value)
        rewards = await self._grant_rewards(
            referral,
            ReferralStatus.CONVERTED
        )

        # Update referrer profile
        profile = self.referrer_profiles[referral.referrer_user_id]

        # Check tier upgrade
        await self._check_tier_upgrade(profile)

        return {
            "referral_id": referral.referral_id,
            "status": referral.status.value,
            "rewards_granted": len(rewards),
            "referrer_tier": profile.current_tier.value
        }

    def _find_referral_by_email(self, email: str) -> Optional[Referral]:
        """Find referral by email"""
        email_lower = email.lower()
        for referral in self.referrals.values():
            if referral.referred_email == email_lower:
                return referral
        return None

    def _find_referral_by_user(self, user_id: str) -> Optional[Referral]:
        """Find referral by user ID"""
        for referral in self.referrals.values():
            if referral.referred_user_id == user_id:
                return referral
        return None

    async def _grant_rewards(
        self,
        referral: Referral,
        status: ReferralStatus
    ) -> List[Reward]:
        """Grant rewards for referral milestone"""
        rewards = []

        if status not in self.reward_config:
            return rewards

        config = self.reward_config[status]
        profile = self.referrer_profiles[referral.referrer_user_id]

        # Get tier multiplier
        multiplier = self.tier_multipliers[profile.current_tier]

        # Grant rewards to referrer
        for reward_config in config["referrer"]:
            reward = Reward(
                user_id=referral.referrer_user_id,
                reward_type=RewardType(reward_config["type"]),
                value=reward_config["value"] * multiplier,
                description=reward_config["description"],
                earned_from_referral_id=referral.referral_id,
                expires_at=datetime.utcnow() + timedelta(days=90)
            )

            self.rewards[reward.reward_id] = reward
            rewards.append(reward)

            # Update total rewards earned
            if reward.reward_type == RewardType.CREDITS:
                profile.total_rewards_earned += reward.value

        # Grant rewards to referred user
        if referral.referred_user_id:
            for reward_config in config["referred"]:
                reward = Reward(
                    user_id=referral.referred_user_id,
                    reward_type=RewardType(reward_config["type"]),
                    value=reward_config["value"],
                    description=reward_config["description"],
                    earned_from_referral_id=referral.referral_id,
                    expires_at=datetime.utcnow() + timedelta(days=90)
                )

                self.rewards[reward.reward_id] = reward
                rewards.append(reward)

        return rewards

    async def _check_tier_upgrade(self, profile: ReferrerProfile) -> bool:
        """Check and upgrade referrer tier"""
        successful = profile.successful_referrals

        new_tier = None

        if successful >= 51:
            new_tier = ReferralTier.PLATINUM
        elif successful >= 21:
            new_tier = ReferralTier.GOLD
        elif successful >= 6:
            new_tier = ReferralTier.SILVER
        else:
            new_tier = ReferralTier.BRONZE

        if new_tier != profile.current_tier:
            profile.current_tier = new_tier
            return True

        return False

    async def get_referrer_stats(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get referrer statistics"""
        profile = self.referrer_profiles.get(user_id)

        if not profile:
            return {"error": "User has no referral profile"}

        # Get user's referrals
        user_referrals = [
            r for r in self.referrals.values()
            if r.referrer_user_id == user_id
        ]

        # Count by status
        by_status = {
            ReferralStatus.PENDING: 0,
            ReferralStatus.SIGNED_UP: 0,
            ReferralStatus.ACTIVATED: 0,
            ReferralStatus.CONVERTED: 0
        }

        for referral in user_referrals:
            by_status[referral.status] += 1

        # Get unclaimed rewards
        user_rewards = [
            r for r in self.rewards.values()
            if r.user_id == user_id and not r.claimed
        ]

        unclaimed_value = sum(
            r.value for r in user_rewards
            if r.reward_type == RewardType.CREDITS
        )

        return {
            "user_id": user_id,
            "referral_code": profile.referral_code,
            "current_tier": profile.current_tier.value,
            "tier_multiplier": self.tier_multipliers[profile.current_tier],
            "total_referrals": profile.total_referrals,
            "successful_referrals": profile.successful_referrals,
            "pending_referrals": profile.pending_referrals,
            "by_status": {k.value: v for k, v in by_status.items()},
            "total_rewards_earned": profile.total_rewards_earned,
            "unclaimed_rewards": unclaimed_value,
            "recent_referrals": [
                {
                    "email": r.referred_email,
                    "status": r.status.value,
                    "created_at": r.created_at.isoformat()
                }
                for r in user_referrals[-5:]
            ]
        }

    async def get_user_rewards(
        self,
        user_id: str,
        unclaimed_only: bool = True
    ) -> List[Dict[str, Any]]:
        """Get user's rewards"""
        user_rewards = [
            r for r in self.rewards.values()
            if r.user_id == user_id
        ]

        if unclaimed_only:
            user_rewards = [r for r in user_rewards if not r.claimed]

        return [
            {
                "reward_id": r.reward_id,
                "type": r.reward_type.value,
                "value": r.value,
                "description": r.description,
                "claimed": r.claimed,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "created_at": r.created_at.isoformat()
            }
            for r in user_rewards
        ]

    async def claim_reward(
        self,
        user_id: str,
        reward_id: str
    ) -> Optional[Dict[str, Any]]:
        """Claim a reward"""
        reward = self.rewards.get(reward_id)

        if not reward or reward.user_id != user_id:
            return None

        if reward.claimed:
            return {"error": "Reward already claimed"}

        if reward.expires_at and datetime.utcnow() > reward.expires_at:
            return {"error": "Reward expired"}

        # Mark as claimed
        reward.claimed = True
        reward.claimed_at = datetime.utcnow()

        return {
            "reward_id": reward_id,
            "type": reward.reward_type.value,
            "value": reward.value,
            "claimed": True
        }

    async def get_leaderboard(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get referral leaderboard"""
        profiles = list(self.referrer_profiles.values())

        # Sort by successful referrals
        profiles.sort(key=lambda x: x.successful_referrals, reverse=True)

        return [
            {
                "user_id": p.user_id,
                "referral_code": p.referral_code,
                "successful_referrals": p.successful_referrals,
                "tier": p.current_tier.value,
                "total_rewards": p.total_rewards_earned
            }
            for p in profiles[:limit]
        ]

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide referral metrics"""
        total_referrals = len(self.referrals)
        total_referrers = len(self.referrer_profiles)

        if total_referrals == 0:
            return {
                "total_referrals": 0,
                "total_referrers": 0
            }

        # Count by status
        by_status = {status: 0 for status in ReferralStatus}
        for referral in self.referrals.values():
            by_status[referral.status] += 1

        # Calculate conversion rate
        converted = by_status[ReferralStatus.CONVERTED]
        conversion_rate = (converted / total_referrals * 100) if total_referrals > 0 else 0

        # Count by tier
        by_tier = {tier: 0 for tier in ReferralTier}
        for profile in self.referrer_profiles.values():
            by_tier[profile.current_tier] += 1

        # Total rewards
        total_rewards = sum(p.total_rewards_earned for p in self.referrer_profiles.values())

        return {
            "total_referrals": total_referrals,
            "total_referrers": total_referrers,
            "by_status": {k.value: v for k, v in by_status.items()},
            "conversion_rate": round(conversion_rate, 2),
            "by_tier": {k.value: v for k, v in by_tier.items()},
            "total_rewards_distributed": total_rewards
        }
