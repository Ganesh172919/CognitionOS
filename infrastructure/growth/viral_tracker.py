"""
Viral Loop Tracking — CognitionOS Growth Engine

Track and optimize viral growth mechanisms:
- Referral tracking with attribution
- K-factor calculation
- Invitation funnel tracking
- Network effect measurement
- Growth loop optimization
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ReferralStatus(str, Enum):
    INVITED = "invited"
    CLICKED = "clicked"
    SIGNED_UP = "signed_up"
    ACTIVATED = "activated"
    CONVERTED = "converted"


class ViralChannel(str, Enum):
    DIRECT_LINK = "direct_link"
    EMAIL = "email"
    SOCIAL = "social"
    EMBED = "embed"
    API = "api"
    MARKETPLACE = "marketplace"


@dataclass
class Referral:
    referral_id: str
    referrer_id: str
    invitee_email: str
    channel: ViralChannel
    status: ReferralStatus = ReferralStatus.INVITED
    created_at: float = field(default_factory=time.time)
    signed_up_at: float = 0
    activated_at: float = 0
    converted_at: float = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "referral_id": self.referral_id,
            "referrer_id": self.referrer_id,
            "channel": self.channel.value,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class ViralMetrics:
    """Key viral growth metrics."""
    invitations_sent: int = 0
    sign_ups: int = 0
    activations: int = 0
    conversions: int = 0
    k_factor: float = 0  # Viral coefficient
    cycle_time_hours: float = 0  # Average time from invite to activation
    channel_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "invitations_sent": self.invitations_sent,
            "sign_ups": self.sign_ups,
            "activations": self.activations,
            "conversions": self.conversions,
            "k_factor": round(self.k_factor, 3),
            "cycle_time_hours": round(self.cycle_time_hours, 1),
            "channels": self.channel_breakdown,
            "invite_to_signup_rate": round(
                self.sign_ups / max(self.invitations_sent, 1), 3
            ),
            "signup_to_activation_rate": round(
                self.activations / max(self.sign_ups, 1), 3
            ),
        }


class ViralLoopTracker:
    """
    Growth engine for tracking and optimizing viral loops.

    Measures K-factor, referral funnels, and network effects
    to drive organic platform growth.
    """

    def __init__(self):
        self._referrals: Dict[str, Referral] = {}
        self._user_referrals: Dict[str, List[str]] = defaultdict(list)
        self._total_users: int = 0
        self._organic_users: int = 0
        self._referred_users: int = 0
        logger.info("ViralLoopTracker initialized")

    def create_referral(self, referrer_id: str, invitee_email: str,
                        channel: ViralChannel = ViralChannel.DIRECT_LINK) -> Referral:
        ref_id = hashlib.md5(
            f"{referrer_id}:{invitee_email}:{time.time()}".encode()
        ).hexdigest()[:12]

        referral = Referral(
            referral_id=ref_id, referrer_id=referrer_id,
            invitee_email=invitee_email, channel=channel,
        )
        self._referrals[ref_id] = referral
        self._user_referrals[referrer_id].append(ref_id)
        return referral

    def record_click(self, referral_id: str):
        ref = self._referrals.get(referral_id)
        if ref and ref.status == ReferralStatus.INVITED:
            ref.status = ReferralStatus.CLICKED

    def record_signup(self, referral_id: str):
        ref = self._referrals.get(referral_id)
        if ref:
            ref.status = ReferralStatus.SIGNED_UP
            ref.signed_up_at = time.time()
            self._referred_users += 1

    def record_activation(self, referral_id: str):
        ref = self._referrals.get(referral_id)
        if ref:
            ref.status = ReferralStatus.ACTIVATED
            ref.activated_at = time.time()

    def record_conversion(self, referral_id: str):
        ref = self._referrals.get(referral_id)
        if ref:
            ref.status = ReferralStatus.CONVERTED
            ref.converted_at = time.time()

    def register_organic_user(self):
        self._total_users += 1
        self._organic_users += 1

    def get_k_factor(self) -> float:
        """Calculate viral coefficient (K-factor = invites per user × conversion rate)."""
        if self._total_users == 0:
            return 0

        total_invites = len(self._referrals)
        avg_invites_per_user = total_invites / self._total_users
        signup_rate = self._referred_users / max(total_invites, 1)
        return avg_invites_per_user * signup_rate

    def get_metrics(self) -> ViralMetrics:
        refs = list(self._referrals.values())
        channel_breakdown: Dict[str, int] = defaultdict(int)
        cycle_times = []

        for ref in refs:
            channel_breakdown[ref.channel.value] += 1
            if ref.activated_at and ref.created_at:
                cycle_hours = (ref.activated_at - ref.created_at) / 3600
                cycle_times.append(cycle_hours)

        return ViralMetrics(
            invitations_sent=len(refs),
            sign_ups=sum(1 for r in refs if r.status.value in
                       ("signed_up", "activated", "converted")),
            activations=sum(1 for r in refs if r.status.value in
                           ("activated", "converted")),
            conversions=sum(1 for r in refs if r.status == ReferralStatus.CONVERTED),
            k_factor=self.get_k_factor(),
            cycle_time_hours=sum(cycle_times) / len(cycle_times) if cycle_times else 0,
            channel_breakdown=dict(channel_breakdown),
        )

    def get_user_referrals(self, user_id: str) -> List[Dict[str, Any]]:
        ref_ids = self._user_referrals.get(user_id, [])
        return [
            self._referrals[rid].to_dict()
            for rid in ref_ids if rid in self._referrals
        ]

    def get_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top referrers by number of successful conversions."""
        user_scores: Dict[str, int] = defaultdict(int)
        for ref in self._referrals.values():
            if ref.status in (ReferralStatus.ACTIVATED, ReferralStatus.CONVERTED):
                user_scores[ref.referrer_id] += 1

        top = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"user_id": uid, "successful_referrals": count} for uid, count in top]

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_referrals": len(self._referrals),
            "total_users": self._total_users,
            "organic_users": self._organic_users,
            "referred_users": self._referred_users,
            "k_factor": round(self.get_k_factor(), 3),
            "channels_used": len(set(r.channel for r in self._referrals.values())),
        }
