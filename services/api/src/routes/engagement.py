"""
Gamification and Referral API Routes

Routes for gamification, engagement, and referral systems.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from services.api.src.dependencies.injection import get_db_session
from infrastructure.gamification.engagement_engine import GamificationEngine
from infrastructure.referral.referral_system import ReferralSystem


router = APIRouter(prefix="/api/v3/engagement", tags=["engagement"])


@router.get("/progress")
async def get_user_progress(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Get user progress and achievements"""
    engine = GamificationEngine(session)
    progress = await engine.get_user_progress(tenant_id)
    
    return {
        "tenant_id": progress.tenant_id,
        "level": progress.level,
        "total_points": progress.total_points,
        "points_to_next_level": progress.points_to_next_level,
        "badges_earned": [
            {
                "badge_id": b.badge_id,
                "name": b.name,
                "description": b.description,
                "icon": b.icon,
                "earned_at": b.earned_at.isoformat(),
                "rarity": b.rarity
            }
            for b in progress.badges_earned
        ],
        "current_streak_days": progress.current_streak_days,
        "longest_streak_days": progress.longest_streak_days,
        "workflows_created": progress.workflows_created,
        "executions_count": progress.executions_count,
        "rank": progress.rank
    }


@router.post("/award-points")
async def award_points(
    tenant_id: str,
    points: int,
    reason: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Award points to user"""
    engine = GamificationEngine(session)
    result = await engine.award_points(tenant_id, points, reason)
    
    return result


@router.get("/leaderboard")
async def get_leaderboard(
    category: str = Query("points", regex="^(points|level|badges)$"),
    limit: int = Query(10, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session)
):
    """Get leaderboard"""
    engine = GamificationEngine(session)
    leaderboard = await engine.get_leaderboard(category, limit)
    
    return {
        "category": category,
        "leaderboard": leaderboard
    }


@router.post("/referral/generate")
async def generate_referral_code(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Generate a new referral code"""
    referral_system = ReferralSystem(session)
    code = await referral_system.generate_referral_code(tenant_id)
    
    return {
        "code": code.code,
        "tenant_id": code.tenant_id,
        "created_at": code.created_at.isoformat(),
        "uses_count": code.uses_count,
        "reward_value": float(code.reward_value)
    }


@router.get("/referral/stats")
async def get_referral_stats(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session)
):
    """Get referral statistics"""
    referral_system = ReferralSystem(session)
    stats = await referral_system.get_referral_stats(tenant_id)
    
    return {
        "tenant_id": stats.tenant_id,
        "total_referrals": stats.total_referrals,
        "successful_referrals": stats.successful_referrals,
        "pending_referrals": stats.pending_referrals,
        "total_rewards_earned": float(stats.total_rewards_earned),
        "conversion_rate": stats.conversion_rate,
        "referral_revenue": float(stats.referral_revenue)
    }


@router.get("/referral/leaderboard")
async def get_referral_leaderboard(
    limit: int = Query(10, ge=1, le=100),
    session: AsyncSession = Depends(get_db_session)
):
    """Get top referrers"""
    referral_system = ReferralSystem(session)
    leaderboard = await referral_system.get_referral_leaderboard(limit)
    
    return {
        "leaderboard": leaderboard
    }
