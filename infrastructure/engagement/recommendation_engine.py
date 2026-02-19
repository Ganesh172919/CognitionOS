"""
Intelligent Recommendation Engine
ML-based recommendations for workflows, plugins, and features to drive engagement.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field


class RecommendationType(str, Enum):
    """Types of recommendations"""
    WORKFLOW_TEMPLATE = "workflow_template"
    PLUGIN = "plugin"
    FEATURE = "feature"
    OPTIMIZATION = "optimization"
    UPGRADE = "upgrade"
    COLLABORATION = "collaboration"


class RecommendationReason(str, Enum):
    """Reason for recommendation"""
    SIMILAR_USERS = "similar_users"
    USAGE_PATTERN = "usage_pattern"
    TRENDING = "trending"
    COMPLEMENTARY = "complementary"
    PERSONALIZED = "personalized"
    NEW_RELEASE = "new_release"


class UserProfile(BaseModel):
    """User profile for personalization"""
    user_id: str
    tenant_id: str
    interests: Set[str] = Field(default_factory=set)
    skill_level: str = "intermediate"  # beginner, intermediate, advanced
    usage_frequency: str = "regular"  # occasional, regular, power_user
    favorite_features: List[str] = Field(default_factory=list)
    completed_workflows: List[str] = Field(default_factory=list)
    installed_plugins: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowTemplate(BaseModel):
    """Workflow template for recommendations"""
    template_id: str
    name: str
    description: str
    category: str
    difficulty: str  # beginner, intermediate, advanced
    popularity_score: float = 0.0
    tags: List[str] = Field(default_factory=list)
    required_plugins: List[str] = Field(default_factory=list)
    estimated_time_minutes: int = 30
    success_rate: float = 0.95
    creator_id: Optional[str] = None


class Recommendation(BaseModel):
    """Individual recommendation"""
    recommendation_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    type: RecommendationType
    item_id: str
    title: str
    description: str
    reason: RecommendationReason
    confidence_score: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    viewed: bool = False
    clicked: bool = False
    converted: bool = False


class IntelligentRecommendationEngine:
    """
    ML-based recommendation engine for driving user engagement and virality.
    """

    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.workflow_templates: Dict[str, WorkflowTemplate] = {}
        self.user_item_interactions: Dict[str, List[Tuple[str, str, datetime]]] = defaultdict(list)
        self.item_similarity_matrix: Dict[str, Dict[str, float]] = {}
        self.trending_items: Dict[RecommendationType, List[str]] = {}
        self.recommendations_history: List[Recommendation] = []

        self._initialize_sample_templates()

    def _initialize_sample_templates(self):
        """Initialize sample workflow templates"""
        templates = [
            WorkflowTemplate(
                template_id="wf_data_pipeline",
                name="Data Pipeline Automation",
                description="Automated data ingestion, transformation, and loading",
                category="data_engineering",
                difficulty="intermediate",
                tags=["data", "etl", "automation", "pipeline"],
                required_plugins=["database", "file_storage"],
                estimated_time_minutes=45,
                popularity_score=0.85
            ),
            WorkflowTemplate(
                template_id="wf_ml_training",
                name="ML Model Training Pipeline",
                description="End-to-end machine learning training workflow",
                category="machine_learning",
                difficulty="advanced",
                tags=["ml", "ai", "training", "models"],
                required_plugins=["ml_framework", "data_processing"],
                estimated_time_minutes=120,
                popularity_score=0.78
            ),
            WorkflowTemplate(
                template_id="wf_api_integration",
                name="API Integration Workflow",
                description="Connect and sync data with external APIs",
                category="integration",
                difficulty="beginner",
                tags=["api", "integration", "sync", "webhooks"],
                required_plugins=["http_client"],
                estimated_time_minutes=20,
                popularity_score=0.92
            ),
            WorkflowTemplate(
                template_id="wf_report_generation",
                name="Automated Report Generation",
                description="Generate and distribute reports automatically",
                category="reporting",
                difficulty="beginner",
                tags=["reporting", "analytics", "automation"],
                required_plugins=["document_generator", "email"],
                estimated_time_minutes=30,
                popularity_score=0.88
            ),
            WorkflowTemplate(
                template_id="wf_sentiment_analysis",
                name="Customer Sentiment Analysis",
                description="Analyze customer feedback and sentiment",
                category="analytics",
                difficulty="intermediate",
                tags=["nlp", "sentiment", "analytics", "ai"],
                required_plugins=["nlp_engine", "database"],
                estimated_time_minutes=40,
                popularity_score=0.75
            )
        ]

        for template in templates:
            self.workflow_templates[template.template_id] = template

    async def get_recommendations(
        self,
        user_id: str,
        limit: int = 5,
        types: Optional[List[RecommendationType]] = None
    ) -> List[Recommendation]:
        """
        Get personalized recommendations for user
        """
        # Get or create user profile
        profile = await self._get_or_create_profile(user_id)

        # Generate recommendations from multiple strategies
        all_recommendations = []

        # Strategy 1: Collaborative Filtering (similar users)
        collab_recs = await self._collaborative_filtering(profile)
        all_recommendations.extend(collab_recs)

        # Strategy 2: Content-Based (similar items)
        content_recs = await self._content_based_filtering(profile)
        all_recommendations.extend(content_recs)

        # Strategy 3: Trending items
        trending_recs = await self._trending_recommendations(profile)
        all_recommendations.extend(trending_recs)

        # Strategy 4: Personalized based on usage
        personalized_recs = await self._personalized_recommendations(profile)
        all_recommendations.extend(personalized_recs)

        # Filter by type if specified
        if types:
            all_recommendations = [
                rec for rec in all_recommendations
                if rec.type in types
            ]

        # Remove duplicates
        seen_items = set()
        unique_recs = []
        for rec in all_recommendations:
            if rec.item_id not in seen_items:
                seen_items.add(rec.item_id)
                unique_recs.append(rec)

        # Sort by priority and confidence
        unique_recs.sort(
            key=lambda x: (x.priority, x.confidence_score),
            reverse=True
        )

        # Store recommendations
        for rec in unique_recs[:limit]:
            self.recommendations_history.append(rec)

        return unique_recs[:limit]

    async def _get_or_create_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                tenant_id=f"tenant_{user_id}"  # Simplified
            )
        return self.user_profiles[user_id]

    async def _collaborative_filtering(
        self,
        profile: UserProfile
    ) -> List[Recommendation]:
        """
        Find recommendations based on similar users
        """
        recommendations = []

        # Find similar users
        similar_users = await self._find_similar_users(profile)

        # Get items liked by similar users
        for similar_user_id, similarity_score in similar_users[:3]:
            similar_profile = self.user_profiles.get(similar_user_id)
            if not similar_profile:
                continue

            # Recommend workflows completed by similar users
            for workflow_id in similar_profile.completed_workflows:
                if workflow_id not in profile.completed_workflows:
                    template = self.workflow_templates.get(workflow_id)
                    if template:
                        recommendations.append(Recommendation(
                            user_id=profile.user_id,
                            type=RecommendationType.WORKFLOW_TEMPLATE,
                            item_id=workflow_id,
                            title=template.name,
                            description=template.description,
                            reason=RecommendationReason.SIMILAR_USERS,
                            confidence_score=similarity_score * 0.8,
                            priority=7,
                            metadata={
                                "category": template.category,
                                "difficulty": template.difficulty,
                                "estimated_time": template.estimated_time_minutes
                            }
                        ))

        return recommendations

    async def _find_similar_users(
        self,
        profile: UserProfile
    ) -> List[Tuple[str, float]]:
        """Find users with similar interests and behavior"""
        similar_users = []

        for other_id, other_profile in self.user_profiles.items():
            if other_id == profile.user_id:
                continue

            # Calculate similarity based on multiple factors
            similarity = 0.0

            # Interest overlap
            if profile.interests and other_profile.interests:
                interest_overlap = len(
                    profile.interests & other_profile.interests
                ) / len(profile.interests | other_profile.interests)
                similarity += interest_overlap * 0.4

            # Workflow overlap
            if profile.completed_workflows and other_profile.completed_workflows:
                workflow_overlap = len(
                    set(profile.completed_workflows) & set(other_profile.completed_workflows)
                ) / max(len(profile.completed_workflows), len(other_profile.completed_workflows))
                similarity += workflow_overlap * 0.4

            # Skill level match
            if profile.skill_level == other_profile.skill_level:
                similarity += 0.2

            if similarity > 0.3:  # Threshold
                similar_users.append((other_id, similarity))

        # Sort by similarity
        similar_users.sort(key=lambda x: x[1], reverse=True)

        return similar_users

    async def _content_based_filtering(
        self,
        profile: UserProfile
    ) -> List[Recommendation]:
        """
        Recommend items similar to what user has liked
        """
        recommendations = []

        # Get user's favorite templates
        favorite_templates = [
            self.workflow_templates[wf_id]
            for wf_id in profile.completed_workflows
            if wf_id in self.workflow_templates
        ]

        if not favorite_templates:
            return recommendations

        # Find similar templates
        for fav_template in favorite_templates[-3:]:  # Last 3
            similar_templates = await self._find_similar_templates(fav_template)

            for template_id, similarity in similar_templates[:2]:
                if template_id not in profile.completed_workflows:
                    template = self.workflow_templates[template_id]
                    recommendations.append(Recommendation(
                        user_id=profile.user_id,
                        type=RecommendationType.WORKFLOW_TEMPLATE,
                        item_id=template_id,
                        title=template.name,
                        description=f"Similar to {fav_template.name}",
                        reason=RecommendationReason.COMPLEMENTARY,
                        confidence_score=similarity * 0.9,
                        priority=8,
                        metadata={
                            "category": template.category,
                            "difficulty": template.difficulty,
                            "similar_to": fav_template.name
                        }
                    ))

        return recommendations

    async def _find_similar_templates(
        self,
        template: WorkflowTemplate
    ) -> List[Tuple[str, float]]:
        """Find similar workflow templates"""
        similar_templates = []

        for other_id, other_template in self.workflow_templates.items():
            if other_id == template.template_id:
                continue

            similarity = 0.0

            # Category match
            if template.category == other_template.category:
                similarity += 0.4

            # Tag overlap
            if template.tags and other_template.tags:
                tag_overlap = len(
                    set(template.tags) & set(other_template.tags)
                ) / len(set(template.tags) | set(other_template.tags))
                similarity += tag_overlap * 0.4

            # Difficulty match
            if template.difficulty == other_template.difficulty:
                similarity += 0.2

            if similarity > 0.3:
                similar_templates.append((other_id, similarity))

        similar_templates.sort(key=lambda x: x[1], reverse=True)

        return similar_templates

    async def _trending_recommendations(
        self,
        profile: UserProfile
    ) -> List[Recommendation]:
        """
        Recommend trending items
        """
        recommendations = []

        # Get top trending workflow templates
        trending_workflows = sorted(
            self.workflow_templates.values(),
            key=lambda x: x.popularity_score,
            reverse=True
        )[:5]

        for template in trending_workflows:
            if template.template_id not in profile.completed_workflows:
                recommendations.append(Recommendation(
                    user_id=profile.user_id,
                    type=RecommendationType.WORKFLOW_TEMPLATE,
                    item_id=template.template_id,
                    title=f"🔥 {template.name}",
                    description=template.description,
                    reason=RecommendationReason.TRENDING,
                    confidence_score=template.popularity_score,
                    priority=6,
                    metadata={
                        "category": template.category,
                        "popularity_score": template.popularity_score,
                        "trending": True
                    }
                ))

        return recommendations[:3]

    async def _personalized_recommendations(
        self,
        profile: UserProfile
    ) -> List[Recommendation]:
        """
        Personalized recommendations based on user behavior
        """
        recommendations = []

        # Recommend based on skill level
        suitable_templates = [
            t for t in self.workflow_templates.values()
            if self._is_suitable_difficulty(profile.skill_level, t.difficulty)
            and t.template_id not in profile.completed_workflows
        ]

        # Recommend based on interests
        if profile.interests:
            for template in suitable_templates:
                interest_match = len(
                    profile.interests & set(template.tags)
                ) / max(len(profile.interests), 1)

                if interest_match > 0.3:
                    recommendations.append(Recommendation(
                        user_id=profile.user_id,
                        type=RecommendationType.WORKFLOW_TEMPLATE,
                        item_id=template.template_id,
                        title=template.name,
                        description=template.description,
                        reason=RecommendationReason.PERSONALIZED,
                        confidence_score=interest_match,
                        priority=9,
                        metadata={
                            "category": template.category,
                            "match_score": interest_match,
                            "matched_interests": list(profile.interests & set(template.tags))
                        }
                    ))

        return recommendations

    def _is_suitable_difficulty(self, skill_level: str, difficulty: str) -> bool:
        """Check if difficulty is suitable for skill level"""
        skill_order = ["beginner", "intermediate", "advanced"]
        difficulty_order = ["beginner", "intermediate", "advanced"]

        try:
            skill_idx = skill_order.index(skill_level)
            diff_idx = difficulty_order.index(difficulty)

            # Allow current level and one above
            return diff_idx <= skill_idx + 1
        except:
            return True

    async def track_interaction(
        self,
        user_id: str,
        recommendation_id: str,
        interaction_type: str  # viewed, clicked, converted
    ) -> None:
        """Track user interaction with recommendation"""
        for rec in self.recommendations_history:
            if rec.recommendation_id == recommendation_id:
                if interaction_type == "viewed":
                    rec.viewed = True
                elif interaction_type == "clicked":
                    rec.clicked = True
                elif interaction_type == "converted":
                    rec.converted = True

                # Update user profile based on conversion
                if rec.converted and rec.type == RecommendationType.WORKFLOW_TEMPLATE:
                    profile = await self._get_or_create_profile(user_id)
                    if rec.item_id not in profile.completed_workflows:
                        profile.completed_workflows.append(rec.item_id)

                    # Update interests from template tags
                    template = self.workflow_templates.get(rec.item_id)
                    if template:
                        profile.interests.update(set(template.tags))

                break

    async def update_user_profile(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> UserProfile:
        """Update user profile"""
        profile = await self._get_or_create_profile(user_id)

        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        return profile

    async def get_recommendation_metrics(self) -> Dict[str, Any]:
        """Get recommendation system metrics"""
        if not self.recommendations_history:
            return {"error": "No recommendation history"}

        total = len(self.recommendations_history)
        viewed = sum(1 for r in self.recommendations_history if r.viewed)
        clicked = sum(1 for r in self.recommendations_history if r.clicked)
        converted = sum(1 for r in self.recommendations_history if r.converted)

        # Calculate rates
        view_rate = viewed / total if total > 0 else 0
        ctr = clicked / viewed if viewed > 0 else 0  # Click-through rate
        conversion_rate = converted / clicked if clicked > 0 else 0

        # Group by type
        by_type = defaultdict(lambda: {"total": 0, "converted": 0})
        for rec in self.recommendations_history:
            by_type[rec.type.value]["total"] += 1
            if rec.converted:
                by_type[rec.type.value]["converted"] += 1

        return {
            "total_recommendations": total,
            "viewed": viewed,
            "clicked": clicked,
            "converted": converted,
            "view_rate": round(view_rate * 100, 2),
            "click_through_rate": round(ctr * 100, 2),
            "conversion_rate": round(conversion_rate * 100, 2),
            "by_type": dict(by_type),
            "active_users": len(self.user_profiles)
        }

    def add_workflow_template(self, template: WorkflowTemplate) -> bool:
        """Add new workflow template"""
        self.workflow_templates[template.template_id] = template
        return True

    def update_template_popularity(
        self,
        template_id: str,
        new_score: float
    ) -> bool:
        """Update template popularity score"""
        if template_id in self.workflow_templates:
            self.workflow_templates[template_id].popularity_score = new_score
            return True
        return False
