"""Referral Program Backend"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal
import secrets

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ReferralCode:
    """Referral code"""
    code: str
    tenant_id: str
    created_at: datetime
    uses_count: int
    reward_value: Decimal


class ReferralSystem:
    """Complete referral program backend"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.default_reward = Decimal("50.00")
    
    async def generate_referral_code(self, tenant_id: str) -> ReferralCode:
        """Generate a new referral code"""
        code = f"REF{secrets.token_hex(4).upper()}"
        return ReferralCode(
            code=code,
            tenant_id=tenant_id,
            created_at=datetime.utcnow(),
            uses_count=0,
            reward_value=self.default_reward
        )
