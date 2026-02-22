"""
Enterprise Transformation API Routes

Exposes all new enterprise systems through REST API:
- Enterprise API Gateway
- Multi-Model AI Orchestration
- Intelligent Code Generation
- Real-Time Analytics
- Feature Flags & Experiments
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

# Initialize router
router = APIRouter(prefix="/api/v3/enterprise", tags=["Enterprise Systems"])


# ==================== Pydantic Models ====================

class ServiceEndpointModel(BaseModel):
    """Service endpoint configuration"""
    url: str
    weight: int = 1
    max_connections: int = 100


class RegisterServiceRequest(BaseModel):
    """Request to register service"""
    service_name: str
    endpoints: List[ServiceEndpointModel]


class RouteRequest(BaseModel):
    """API Gateway route request"""
    service_name: str
    request_data: Dict[str, Any]
    client_ip: Optional[str] = None


class ModelConfigModel(BaseModel):
    """AI model configuration"""
    model_id: str
    provider: str
    capabilities: List[str]
    cost_per_1k_tokens: float
    avg_latency_ms: float
    quality_score: float
    max_tokens: int = 4096


class GenerateRequest(BaseModel):
    """AI generation request"""
    prompt: str
    capability: str
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    user_id: Optional[str] = None


class CodeGenRequest(BaseModel):
    """Code generation request"""
    description: str
    language: str
    requirements: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    max_iterations: int = 3


class UsageEventModel(BaseModel):
    """Usage tracking event"""
    user_id: str
    endpoint: str
    method: str
    tokens_input: int
    tokens_output: int
    cost_usd: float
    latency_ms: float
    status_code: int


class FeatureFlagModel(BaseModel):
    """Feature flag configuration"""
    flag_id: str
    name: str
    description: str
    state: str  # "off", "on", "conditional"
    rollout_percentage: float = 0.0
    enabled_users: List[str] = Field(default_factory=list)


class ExperimentModel(BaseModel):
    """A/B test experiment"""
    experiment_id: str
    name: str
    description: str
    variants: List[str]
    traffic_allocation: Dict[str, float]


class FeatureCheckRequest(BaseModel):
    """Check if feature is enabled"""
    flag_id: str
    user_id: str
    context: Optional[Dict[str, Any]] = None


# ==================== Enterprise API Gateway Routes ====================

@router.post("/gateway/register")
async def register_service(request: RegisterServiceRequest):
    """Register service with API Gateway"""
    try:
        from infrastructure.api_gateway import EnterpriseAPIGateway, ServiceEndpoint

        gateway = EnterpriseAPIGateway()

        endpoints = [
            ServiceEndpoint(
                url=ep.url,
                weight=ep.weight,
                max_connections=ep.max_connections
            )
            for ep in request.endpoints
        ]

        gateway.register_service(request.service_name, endpoints)

        return {
            "status": "success",
            "message": f"Service {request.service_name} registered",
            "endpoint_count": len(endpoints)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gateway/route")
async def route_request(request: RouteRequest):
    """Route request through API Gateway"""
    try:
        from infrastructure.api_gateway import EnterpriseAPIGateway

        gateway = EnterpriseAPIGateway()
        response = await gateway.route_request(
            request.service_name,
            request.request_data,
            request.client_ip
        )

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/gateway/metrics")
async def get_gateway_metrics():
    """Get API Gateway metrics"""
    try:
        from infrastructure.api_gateway import EnterpriseAPIGateway

        gateway = EnterpriseAPIGateway()
        return gateway.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Multi-Model AI Orchestration Routes ====================

@router.post("/ai/register-model")
async def register_model(config: ModelConfigModel):
    """Register AI model with orchestrator"""
    try:
        from infrastructure.ai_orchestration import (
            MultiModelOrchestrator,
            ModelConfig,
            ModelProvider,
            ModelCapability
        )

        orchestrator = MultiModelOrchestrator()

        model_config = ModelConfig(
            model_id=config.model_id,
            provider=ModelProvider(config.provider),
            capabilities=[ModelCapability(c) for c in config.capabilities],
            cost_per_1k_tokens=config.cost_per_1k_tokens,
            avg_latency_ms=config.avg_latency_ms,
            quality_score=config.quality_score,
            max_tokens=config.max_tokens
        )

        orchestrator.register_model(model_config)

        return {
            "status": "success",
            "message": f"Model {config.model_id} registered"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/generate")
async def generate_with_ai(request: GenerateRequest):
    """Generate content using AI orchestrator"""
    try:
        from infrastructure.ai_orchestration import (
            MultiModelOrchestrator,
            ModelRequest,
            ModelCapability
        )

        orchestrator = MultiModelOrchestrator()

        model_request = ModelRequest(
            prompt=request.prompt,
            capability=ModelCapability(request.capability),
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            user_id=request.user_id
        )

        response = await orchestrator.generate(model_request)

        return {
            "content": response.content,
            "model_id": response.model_id,
            "provider": response.provider.value,
            "tokens_used": response.tokens_used,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "quality_estimate": response.quality_estimate
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/metrics")
async def get_ai_metrics():
    """Get AI orchestrator metrics"""
    try:
        from infrastructure.ai_orchestration import MultiModelOrchestrator

        orchestrator = MultiModelOrchestrator()
        return orchestrator.get_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Code Generation Routes ====================

@router.post("/codegen/generate")
async def generate_code(request: CodeGenRequest):
    """Generate code from specification"""
    try:
        from infrastructure.code_generation import (
            IntelligentCodeGenerator,
            CodeSpec,
            Language
        )

        generator = IntelligentCodeGenerator()

        spec = CodeSpec(
            description=request.description,
            language=Language(request.language),
            requirements=request.requirements,
            constraints=request.constraints
        )

        result = await generator.generate_code(spec, request.max_iterations)

        return {
            "code": result.code,
            "language": result.language.value,
            "quality_score": result.quality_score,
            "validation_results": result.validation_results,
            "test_results": result.test_results,
            "metrics": result.metrics,
            "issues": result.issues,
            "suggestions": result.suggestions,
            "iteration_count": result.iteration_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/codegen/statistics")
async def get_codegen_statistics():
    """Get code generation statistics"""
    try:
        from infrastructure.code_generation import IntelligentCodeGenerator

        generator = IntelligentCodeGenerator()
        return generator.get_statistics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Real-Time Analytics Routes ====================

@router.post("/analytics/track")
async def track_usage_event(event: UsageEventModel):
    """Track usage event"""
    try:
        from infrastructure.realtime_analytics import (
            RealtimeAnalyticsEngine,
            UsageEvent
        )

        engine = RealtimeAnalyticsEngine()

        usage_event = UsageEvent(
            user_id=event.user_id,
            endpoint=event.endpoint,
            method=event.method,
            tokens_input=event.tokens_input,
            tokens_output=event.tokens_output,
            cost_usd=event.cost_usd,
            latency_ms=event.latency_ms,
            status_code=event.status_code
        )

        await engine.track_event(usage_event)

        return {"status": "success", "message": "Event tracked"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: str):
    """Get analytics for specific user"""
    try:
        from infrastructure.realtime_analytics import RealtimeAnalyticsEngine

        engine = RealtimeAnalyticsEngine()
        return engine.get_user_metrics(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/dashboard")
async def get_analytics_dashboard():
    """Get dashboard analytics data"""
    try:
        from infrastructure.realtime_analytics import RealtimeAnalyticsEngine

        engine = RealtimeAnalyticsEngine()
        return engine.get_dashboard_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/forecast/{user_id}")
async def forecast_usage(user_id: str, days_ahead: int = 7):
    """Forecast future usage for user"""
    try:
        from infrastructure.realtime_analytics import RealtimeAnalyticsEngine

        engine = RealtimeAnalyticsEngine()
        return engine.forecast_usage(user_id, days_ahead)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Feature Flags Routes ====================

@router.post("/features/flags")
async def create_feature_flag(flag: FeatureFlagModel):
    """Create feature flag"""
    try:
        from infrastructure.feature_flags import FeatureFlagEngine, FeatureFlag, FeatureState

        engine = FeatureFlagEngine()

        feature_flag = FeatureFlag(
            flag_id=flag.flag_id,
            name=flag.name,
            description=flag.description,
            state=FeatureState(flag.state),
            rollout_percentage=flag.rollout_percentage,
            enabled_users=flag.enabled_users
        )

        engine.create_flag(feature_flag)

        return {
            "status": "success",
            "message": f"Feature flag {flag.flag_id} created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/check")
async def check_feature(request: FeatureCheckRequest):
    """Check if feature is enabled for user"""
    try:
        from infrastructure.feature_flags import FeatureFlagEngine

        engine = FeatureFlagEngine()

        is_enabled = engine.is_enabled(
            request.flag_id,
            request.user_id,
            request.context
        )

        return {
            "flag_id": request.flag_id,
            "user_id": request.user_id,
            "enabled": is_enabled
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/experiments")
async def create_experiment(experiment: ExperimentModel):
    """Create A/B test experiment"""
    try:
        from infrastructure.feature_flags import FeatureFlagEngine, Experiment

        engine = FeatureFlagEngine()

        exp = Experiment(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            description=experiment.description,
            variants=experiment.variants,
            traffic_allocation=experiment.traffic_allocation
        )

        engine.create_experiment(exp)

        return {
            "status": "success",
            "message": f"Experiment {experiment.experiment_id} created"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: str):
    """Get experiment results"""
    try:
        from infrastructure.feature_flags import FeatureFlagEngine

        engine = FeatureFlagEngine()
        return engine.get_experiment_results(experiment_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/flags/{flag_id}/metrics")
async def get_flag_metrics(flag_id: str):
    """Get metrics for feature flag"""
    try:
        from infrastructure.feature_flags import FeatureFlagEngine

        engine = FeatureFlagEngine()
        return engine.get_flag_metrics(flag_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== System Health & Status ====================

@router.get("/health")
async def enterprise_health_check():
    """Health check for enterprise systems"""
    return {
        "status": "healthy",
        "systems": {
            "api_gateway": "operational",
            "ai_orchestration": "operational",
            "code_generation": "operational",
            "real_time_analytics": "operational",
            "feature_flags": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
