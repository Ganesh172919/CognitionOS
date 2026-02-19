"""
API Routes for Engagement Systems
Exposes recommendation engine and referral system.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from infrastructure.engagement import (
    IntelligentRecommendationEngine,
    ViralReferralSystem,
    RecommendationType,
    ReferralStatus
)

router = APIRouter(prefix="/api/v3/engagement", tags=["Engagement"])

# Initialize systems
recommendation_engine = IntelligentRecommendationEngine()
referral_system = ViralReferralSystem()


# Recommendation endpoints

@router.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    limit: int = Query(5, ge=1, le=20),
    types: Optional[List[RecommendationType]] = Query(None)
):
    """
    Get personalized recommendations for user

    Uses ML-based recommendation engine with:
    - Collaborative filtering
    - Content-based filtering
    - Trending analysis
    - Personalization
    """
    try:
        recommendations = await recommendation_engine.get_recommendations(
            user_id=user_id,
            limit=limit,
            types=types
        )

        return {
            "success": True,
            "user_id": user_id,
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "type": rec.type.value,
                    "item_id": rec.item_id,
                    "title": rec.title,
                    "description": rec.description,
                    "reason": rec.reason.value,
                    "confidence_score": rec.confidence_score,
                    "priority": rec.priority,
                    "metadata": rec.metadata
                }
                for rec in recommendations
            ]
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class TrackInteractionRequest(BaseModel):
    """Request to track recommendation interaction"""
    user_id: str
    recommendation_id: str
    interaction_type: str = Field(..., regex="^(viewed|clicked|converted)$")


@router.post("/recommendations/track")
async def track_recommendation_interaction(request: TrackInteractionRequest):
    """
    Track user interaction with recommendation

    Interaction types:
    - viewed: User saw the recommendation
    - clicked: User clicked on the recommendation
    - converted: User completed the recommended action
    """
    try:
        await recommendation_engine.track_interaction(
            user_id=request.user_id,
            recommendation_id=request.recommendation_id,
            interaction_type=request.interaction_type
        )

        return {
            "success": True,
            "user_id": request.user_id,
            "recommendation_id": request.recommendation_id,
            "interaction_type": request.interaction_type
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class UpdateProfileRequest(BaseModel):
    """Request to update user profile"""
    interests: Optional[List[str]] = None
    skill_level: Optional[str] = Field(None, regex="^(beginner|intermediate|advanced)$")
    completed_workflows: Optional[List[str]] = None


@router.post("/recommendations/profile/{user_id}")
async def update_user_profile(user_id: str, request: UpdateProfileRequest):
    """
    Update user profile for better recommendations
    """
    try:
        updates = {}
        if request.interests is not None:
            updates["interests"] = set(request.interests)
        if request.skill_level is not None:
            updates["skill_level"] = request.skill_level
        if request.completed_workflows is not None:
            updates["completed_workflows"] = request.completed_workflows

        profile = await recommendation_engine.update_user_profile(user_id, updates)

        return {
            "success": True,
            "user_id": profile.user_id,
            "interests": list(profile.interests),
            "skill_level": profile.skill_level,
            "completed_workflows": profile.completed_workflows
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/recommendations/metrics")
async def get_recommendation_metrics():
    """
    Get recommendation system performance metrics
    """
    try:
        metrics = await recommendation_engine.get_recommendation_metrics()

        return {
            "success": True,
            **metrics
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Referral system endpoints

@router.post("/referrals/code/{user_id}")
async def create_referral_code(
    user_id: str,
    custom_code: Optional[str] = Query(None)
):
    """
    Create referral code for user
    """
    try:
        code = await referral_system.create_referral_code(
            user_id=user_id,
            custom_code=custom_code
        )

        return {
            "success": True,
            "user_id": user_id,
            "referral_code": code
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class CreateReferralRequest(BaseModel):
    """Request to create referral"""
    referrer_user_id: str
    referred_email: str
    referral_code: Optional[str] = None


@router.post("/referrals/create")
async def create_referral(request: CreateReferralRequest):
    """
    Create a new referral
    """
    try:
        referral = await referral_system.create_referral(
            referrer_user_id=request.referrer_user_id,
            referred_email=request.referred_email,
            referral_code=request.referral_code
        )

        if not referral:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid referral code"
            )

        return {
            "success": True,
            "referral_id": referral.referral_id,
            "referrer_user_id": referral.referrer_user_id,
            "referred_email": referral.referred_email,
            "status": referral.status.value,
            "created_at": referral.created_at.isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class TrackSignupRequest(BaseModel):
    """Request to track referral signup"""
    referred_email: str
    referred_user_id: str


@router.post("/referrals/track/signup")
async def track_referral_signup(request: TrackSignupRequest):
    """
    Track when a referred user signs up

    Automatically grants rewards to referrer
    """
    try:
        result = await referral_system.track_referral_signup(
            referred_email=request.referred_email,
            referred_user_id=request.referred_user_id
        )

        if not result:
            return {
                "success": True,
                "tracked": False,
                "message": "No referral found for this email"
            }

        return {
            "success": True,
            "tracked": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/referrals/track/activation/{referred_user_id}")
async def track_referral_activation(referred_user_id: str):
    """
    Track when a referred user activates (completes onboarding)
    """
    try:
        result = await referral_system.track_referral_activation(referred_user_id)

        if not result:
            return {
                "success": True,
                "tracked": False,
                "message": "No referral found for this user"
            }

        return {
            "success": True,
            "tracked": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/referrals/track/conversion/{referred_user_id}")
async def track_referral_conversion(referred_user_id: str):
    """
    Track when a referred user converts (upgrades to paid)

    Grants maximum rewards to referrer
    """
    try:
        result = await referral_system.track_referral_conversion(referred_user_id)

        if not result:
            return {
                "success": True,
                "tracked": False,
                "message": "No referral found for this user"
            }

        return {
            "success": True,
            "tracked": True,
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/referrals/stats/{user_id}")
async def get_referrer_stats(user_id: str):
    """
    Get referral statistics for user

    Returns:
    - Total referrals
    - Successful referrals
    - Current tier
    - Total rewards earned
    - Unclaimed rewards
    """
    try:
        stats = await referral_system.get_referrer_stats(user_id)

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/referrals/rewards/{user_id}")
async def get_user_rewards(
    user_id: str,
    unclaimed_only: bool = Query(True)
):
    """
    Get user's referral rewards
    """
    try:
        rewards = await referral_system.get_user_rewards(
            user_id=user_id,
            unclaimed_only=unclaimed_only
        )

        return {
            "success": True,
            "user_id": user_id,
            "rewards": rewards
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/referrals/rewards/{user_id}/{reward_id}/claim")
async def claim_reward(user_id: str, reward_id: str):
    """
    Claim a referral reward
    """
    try:
        result = await referral_system.claim_reward(user_id, reward_id)

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Reward not found"
            )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        return {
            "success": True,
            **result
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/referrals/leaderboard")
async def get_referral_leaderboard(
    limit: int = Query(10, ge=1, le=100)
):
    """
    Get referral leaderboard

    Shows top referrers by successful referrals
    """
    try:
        leaderboard = await referral_system.get_leaderboard(limit=limit)

        return {
            "success": True,
            "leaderboard": leaderboard
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/referrals/metrics")
async def get_referral_metrics():
    """
    Get system-wide referral metrics
    """
    try:
        metrics = await referral_system.get_system_metrics()

        return {
            "success": True,
            **metrics
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
