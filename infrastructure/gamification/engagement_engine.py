"""
Gamification and Engagement Engine

Points, badges, levels, and achievements system for user engagement.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession


class BadgeType(str, Enum):
    """Badge types"""
    FIRST_WORKFLOW = "first_workflow"
    TEN_WORKFLOWS = "ten_workflows"
    HUNDRED_WORKFLOWS = "hundred_workflows"
    POWER_USER = "power_user"
    EARLY_ADOPTER = "early_adopter"
    INNOVATOR = "innovator"


@dataclass
class Badge:
    """Achievement badge"""
    badge_id: str
    name: str
    description: str
    icon: str
    earned_at: datetime
    rarity: str  # common, rare, epic, legendary


@dataclass
class UserProgress:
    """User progress and engagement metrics"""
    tenant_id: str
    level: int
    total_points: int
    points_to_next_level: int
    badges_earned: List[Badge]
    current_streak_days: int
    longest_streak_days: int
    workflows_created: int
    executions_count: int
    rank: int  # Global rank


class GamificationEngine:
    """
    Gamification system for user engagement.
    
    Features:
    - Points system
    - Levels and progression
    - Badges and achievements
    - Leaderboards
    - Streaks and challenges
    - Rewards and incentives
    """
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.points_per_level = 1000
    
    async def get_user_progress(self, tenant_id: str) -> UserProgress:
        """Get user progress and achievements"""
        # Would query from database in production
        total_points = 2500
        level = total_points // self.points_per_level
        points_to_next = self.points_per_level - (total_points % self.points_per_level)
        
        badges = [
            Badge(
                badge_id="first_workflow",
                name="First Steps",
                description="Created your first workflow",
                icon="🎯",
                earned_at=datetime.utcnow() - timedelta(days=30),
                rarity="common"
            ),
            Badge(
                badge_id="power_user",
                name="Power User",
                description="Executed 100+ workflows",
                icon="⚡",
                earned_at=datetime.utcnow() - timedelta(days=10),
                rarity="rare"
            )
        ]
        
        return UserProgress(
            tenant_id=tenant_id,
            level=level,
            total_points=total_points,
            points_to_next_level=points_to_next,
            badges_earned=badges,
            current_streak_days=7,
            longest_streak_days=15,
            workflows_created=25,
            executions_count=150,
            rank=42
        )
    
    async def award_points(
        self,
        tenant_id: str,
        points: int,
        reason: str
    ) -> Dict[str, any]:
        """Award points to user"""
        # Would update database in production
        return {
            "points_awarded": points,
            "reason": reason,
            "new_total": 2500 + points,
            "level_up": False
        }
    
    async def check_achievements(
        self,
        tenant_id: str,
        event: str,
        metadata: Dict
    ) -> List[Badge]:
        """Check if user earned new achievements"""
        new_badges = []
        
        # Check for various achievements
        if event == "workflow_created" and metadata.get("count") == 1:
            new_badges.append(Badge(
                badge_id="first_workflow",
                name="First Steps",
                description="Created your first workflow",
                icon="🎯",
                earned_at=datetime.utcnow(),
                rarity="common"
            ))
        
        return new_badges
    
    async def get_leaderboard(
        self,
        category: str = "points",
        limit: int = 10
    ) -> List[Dict]:
        """Get leaderboard for a category"""
        # Mock leaderboard
        return [
            {
                "rank": i + 1,
                "tenant_name": f"User {i+1}",
                "points": 10000 - i * 500,
                "level": 10 - i,
                "badges": 5 + (10 - i)
            }
            for i in range(limit)
        ]
