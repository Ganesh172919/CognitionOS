"""
Additional Enterprise Systems API Routes

Exposes:
- Webhook Management
- Stream Processing
- AI Code Review
- Cost Optimization
- Plugin Marketplace
"""

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Initialize router
router = APIRouter(prefix="/api/v3/enterprise-ext", tags=["Enterprise Extensions"])


# ==================== Pydantic Models ====================

class WebhookEndpointModel(BaseModel):
    """Webhook endpoint configuration"""
    endpoint_id: str
    url: str
    secret: str
    events: List[str]
    is_active: bool = True


class SendWebhookRequest(BaseModel):
    """Send webhook event request"""
    event_type: str
    payload: Dict[str, Any]
    endpoint_ids: Optional[List[str]] = None


class StreamEventModel(BaseModel):
    """Stream event"""
    event_id: str
    stream_name: str
    data: Dict[str, Any]
    partition_key: Optional[str] = None


class CodeReviewRequest(BaseModel):
    """Code review request"""
    code: str
    file_path: str
    language: str = "python"


class ResourceUsageModel(BaseModel):
    """Resource usage tracking"""
    resource_id: str
    resource_type: str
    cost_usd: float
    usage_amount: float
    usage_unit: str


class PluginModel(BaseModel):
    """Plugin metadata"""
    plugin_id: str
    name: str
    description: str
    author: str
    category: str
    version: str
    price_usd: float = 0.0
    tags: List[str] = Field(default_factory=list)


class PluginReviewModel(BaseModel):
    """Plugin review"""
    plugin_id: str
    user_id: str
    rating: int = Field(ge=1, le=5)
    title: str
    content: str


class PluginSearchRequest(BaseModel):
    """Plugin search"""
    query: str
    category: Optional[str] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    limit: int = 20


# ==================== Webhook System Routes ====================

@router.post("/webhooks/endpoints")
async def register_webhook_endpoint(endpoint: WebhookEndpointModel):
    """Register webhook endpoint"""
    try:
        from infrastructure.webhooks import EnterpriseWebhookSystem, WebhookEndpoint, WebhookEvent

        system = EnterpriseWebhookSystem()

        webhook_endpoint = WebhookEndpoint(
            endpoint_id=endpoint.endpoint_id,
            url=endpoint.url,
            secret=endpoint.secret,
            events=[WebhookEvent(e) for e in endpoint.events],
            is_active=endpoint.is_active
        )

        system.register_endpoint(webhook_endpoint)

        return {
            "status": "success",
            "message": f"Webhook endpoint {endpoint.endpoint_id} registered"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/webhooks/send")
async def send_webhook_event(request: SendWebhookRequest):
    """Send webhook event"""
    try:
        from infrastructure.webhooks import EnterpriseWebhookSystem, WebhookEvent

        system = EnterpriseWebhookSystem()

        delivery_ids = await system.send_event(
            WebhookEvent(request.event_type),
            request.payload,
            request.endpoint_ids
        )

        return {
            "status": "success",
            "delivery_ids": delivery_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks/deliveries/{delivery_id}")
async def get_webhook_delivery_status(delivery_id: str):
    """Get webhook delivery status"""
    try:
        from infrastructure.webhooks import EnterpriseWebhookSystem

        system = EnterpriseWebhookSystem()
        status = system.get_delivery_status(delivery_id)

        if not status:
            raise HTTPException(status_code=404, detail="Delivery not found")

        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/webhooks/statistics")
async def get_webhook_statistics():
    """Get webhook system statistics"""
    try:
        from infrastructure.webhooks import EnterpriseWebhookSystem

        system = EnterpriseWebhookSystem()
        return system.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Stream Processing Routes ====================

@router.post("/streams/publish")
async def publish_stream_event(event: StreamEventModel):
    """Publish event to stream"""
    try:
        from infrastructure.stream_processing import StreamProcessingPipeline, StreamEvent

        pipeline = StreamProcessingPipeline()

        stream_event = StreamEvent(
            event_id=event.event_id,
            stream_name=event.stream_name,
            data=event.data,
            partition_key=event.partition_key
        )

        await pipeline.publish(event.stream_name, stream_event)

        return {
            "status": "success",
            "message": "Event published to stream"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/streams/metrics")
async def get_stream_metrics():
    """Get stream processing metrics"""
    try:
        from infrastructure.stream_processing import StreamProcessingPipeline

        pipeline = StreamProcessingPipeline()
        return pipeline.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AI Code Review Routes ====================

@router.post("/code-review/analyze")
async def review_code(request: CodeReviewRequest):
    """Perform AI code review"""
    try:
        from infrastructure.code_review import AICodeReviewer

        reviewer = AICodeReviewer()

        issues = await reviewer.review_code(
            request.code,
            request.file_path,
            request.language
        )

        report = reviewer.generate_report(issues)

        return {
            "issues": [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "title": issue.title,
                    "description": issue.description,
                    "suggestion": issue.suggestion,
                    "code_snippet": issue.code_snippet,
                    "confidence": issue.confidence
                }
                for issue in issues
            ],
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Cost Optimization Routes ====================

@router.post("/cost-optimization/track-usage")
async def track_resource_usage(usage: ResourceUsageModel):
    """Track resource usage for cost optimization"""
    try:
        from infrastructure.cost_optimization import CostOptimizationEngine, ResourceUsage, ResourceType

        engine = CostOptimizationEngine()

        resource_usage = ResourceUsage(
            resource_id=usage.resource_id,
            resource_type=ResourceType(usage.resource_type),
            cost_usd=usage.cost_usd,
            usage_amount=usage.usage_amount,
            usage_unit=usage.usage_unit
        )

        engine.track_usage(resource_usage)

        return {
            "status": "success",
            "message": "Usage tracked"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cost-optimization/analyze")
async def analyze_costs(time_window_days: int = 30):
    """Analyze costs and generate recommendations"""
    try:
        from infrastructure.cost_optimization import CostOptimizationEngine

        engine = CostOptimizationEngine()
        recommendations = engine.analyze_and_recommend(time_window_days)

        return {
            "recommendations": [
                {
                    "recommendation_id": rec.recommendation_id,
                    "title": rec.title,
                    "description": rec.description,
                    "resource_type": rec.resource_type.value,
                    "priority": rec.priority.value,
                    "estimated_savings_usd": rec.estimated_savings_usd,
                    "estimated_savings_percentage": rec.estimated_savings_percentage,
                    "implementation_effort": rec.implementation_effort,
                    "action_items": rec.action_items,
                    "risk_level": rec.risk_level
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-optimization/forecast")
async def forecast_costs(days_ahead: int = 30):
    """Forecast future costs"""
    try:
        from infrastructure.cost_optimization import CostOptimizationEngine

        engine = CostOptimizationEngine()
        return engine.forecast_costs(days_ahead)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cost-optimization/breakdown")
async def get_cost_breakdown():
    """Get cost breakdown by resource type"""
    try:
        from infrastructure.cost_optimization import CostOptimizationEngine

        engine = CostOptimizationEngine()
        return engine.get_cost_breakdown()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Plugin Marketplace Routes ====================

@router.post("/marketplace/plugins")
async def publish_plugin(plugin: PluginModel):
    """Publish plugin to marketplace"""
    try:
        from infrastructure.plugin_marketplace import PluginMarketplace, Plugin, PluginCategory, PluginStatus

        marketplace = PluginMarketplace()

        plugin_obj = Plugin(
            plugin_id=plugin.plugin_id,
            name=plugin.name,
            description=plugin.description,
            author=plugin.author,
            category=PluginCategory(plugin.category),
            version=plugin.version,
            price_usd=plugin.price_usd,
            status=PluginStatus.ACTIVE,
            tags=plugin.tags
        )

        plugin_id = marketplace.publish_plugin(plugin_obj)

        return {
            "status": "success",
            "plugin_id": plugin_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/marketplace/search")
async def search_plugins(request: PluginSearchRequest):
    """Search plugins in marketplace"""
    try:
        from infrastructure.plugin_marketplace import PluginMarketplace, PluginCategory

        marketplace = PluginMarketplace()

        category = PluginCategory(request.category) if request.category else None

        plugins = marketplace.search_plugins(
            query=request.query,
            category=category,
            max_price=request.max_price,
            min_rating=request.min_rating,
            limit=request.limit
        )

        return {
            "total": len(plugins),
            "plugins": [
                {
                    "plugin_id": p.plugin_id,
                    "name": p.name,
                    "description": p.description,
                    "author": p.author,
                    "category": p.category.value,
                    "version": p.version,
                    "price_usd": p.price_usd,
                    "download_count": p.download_count,
                    "average_rating": p.average_rating,
                    "total_ratings": p.total_ratings
                }
                for p in plugins
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/featured")
async def get_featured_plugins(limit: int = 10):
    """Get featured plugins"""
    try:
        from infrastructure.plugin_marketplace import PluginMarketplace

        marketplace = PluginMarketplace()
        plugins = marketplace.get_featured_plugins(limit)

        return {
            "plugins": [
                {
                    "plugin_id": p.plugin_id,
                    "name": p.name,
                    "description": p.description,
                    "average_rating": p.average_rating,
                    "download_count": p.download_count
                }
                for p in plugins
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/marketplace/reviews")
async def add_plugin_review(review: PluginReviewModel):
    """Add plugin review"""
    try:
        from infrastructure.plugin_marketplace import PluginMarketplace

        marketplace = PluginMarketplace()

        review_obj = marketplace.add_review(
            plugin_id=review.plugin_id,
            user_id=review.user_id,
            rating=review.rating,
            title=review.title,
            content=review.content
        )

        return {
            "status": "success",
            "review_id": review_obj.review_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/marketplace/statistics")
async def get_marketplace_statistics():
    """Get marketplace statistics"""
    try:
        from infrastructure.plugin_marketplace import PluginMarketplace

        marketplace = PluginMarketplace()
        return marketplace.get_marketplace_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== System Health ====================

@router.get("/health")
async def extended_health_check():
    """Health check for extended enterprise systems"""
    return {
        "status": "healthy",
        "systems": {
            "webhooks": "operational",
            "stream_processing": "operational",
            "code_review": "operational",
            "cost_optimization": "operational",
            "plugin_marketplace": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
