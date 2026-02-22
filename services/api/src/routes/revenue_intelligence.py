"""
Revenue & Intelligence API Routes

Advanced revenue management and AI intelligence endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from pydantic import BaseModel, Field

from infrastructure.revenue import (
    DynamicPricingEngine,
    LTVPredictionEngine,
    PaymentOrchestrationEngine,
    RevenueRecognitionSystem,
    CustomerSegment,
    CustomerProfile,
    CustomerTier,
    PaymentMethod,
    PaymentGateway,
    DunningConfig
)
from infrastructure.ai_intelligence import (
    NeuralCodeGenerator,
    CodeGenerationRequest,
    ProgrammingLanguage,
    OptimizationType
)

router = APIRouter(prefix="/api/v3/revenue-intelligence", tags=["Revenue & AI Intelligence"])

# Initialize engines (would be dependency injected in production)
pricing_engine = DynamicPricingEngine(base_prices={"pro": Decimal("99"), "enterprise": Decimal("499")})
ltv_engine = LTVPredictionEngine()
payment_engine = PaymentOrchestrationEngine()
revenue_system = RevenueRecognitionSystem()
code_generator = NeuralCodeGenerator()


# ==================== Pydantic Models ====================

class PricingRequest(BaseModel):
    product_id: str
    customer_segment: str
    current_demand: float = Field(ge=0.0, le=1.0)


class LTVRequest(BaseModel):
    customer_id: str
    time_horizon_months: int = 36


class ChurnPredictionRequest(BaseModel):
    customer_id: str


class PaymentRequest(BaseModel):
    customer_id: str
    amount: float
    currency: str = "USD"
    payment_method: str


class CodeGenerationRequestModel(BaseModel):
    prompt: str
    language: str
    max_tokens: int = 2000
    temperature: float = 0.7
    optimization_goals: List[str] = []


class CodeAnalysisRequest(BaseModel):
    code: str
    language: str


# ==================== Dynamic Pricing Endpoints ====================

@router.post("/pricing/calculate")
async def calculate_dynamic_price(request: PricingRequest) -> Dict[str, Any]:
    """
    Calculate dynamic price based on demand and segment.

    Returns price point with detailed breakdown.
    """
    try:
        segment = CustomerSegment[request.customer_segment.upper()]

        price_point = pricing_engine.calculate_dynamic_price(
            product_id=request.product_id,
            customer_segment=segment,
            current_demand=request.current_demand
        )

        return {
            "product_id": request.product_id,
            "base_price": float(price_point.base_price),
            "final_price": float(price_point.final_price),
            "discount_percentage": float(price_point.discount_percentage),
            "currency": price_point.currency,
            "effective_from": price_point.effective_from.isoformat(),
            "confidence_score": price_point.confidence_score,
            "factors": price_point.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/pricing/forecast-demand")
async def forecast_demand(product_id: str, forecast_days: int = 7) -> Dict[str, Any]:
    """Forecast product demand"""
    try:
        forecast = pricing_engine.forecast_demand(product_id, forecast_days=forecast_days)

        return {
            "product_id": product_id,
            "period_start": forecast.period_start.isoformat(),
            "period_end": forecast.period_end.isoformat(),
            "predicted_demand": forecast.predicted_demand,
            "confidence_interval": forecast.confidence_interval,
            "trend": forecast.trend,
            "factors": forecast.factors
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/pricing/revenue-forecast")
async def get_revenue_forecast(months_ahead: int = 12) -> Dict[str, Any]:
    """Get revenue forecast"""
    try:
        forecast = pricing_engine.get_revenue_forecast(months_ahead=months_ahead)
        return forecast
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== LTV & Churn Prediction ====================

@router.post("/ltv/predict")
async def predict_ltv(request: LTVRequest) -> Dict[str, Any]:
    """
    Predict customer lifetime value.

    Returns LTV prediction with confidence interval.
    """
    try:
        prediction = ltv_engine.predict_ltv(
            customer_id=request.customer_id,
            time_horizon_months=request.time_horizon_months
        )

        return {
            "customer_id": prediction.customer_id,
            "predicted_ltv": float(prediction.predicted_ltv),
            "confidence_interval": {
                "lower": float(prediction.confidence_interval[0]),
                "upper": float(prediction.confidence_interval[1])
            },
            "confidence_score": prediction.confidence_score,
            "predicted_monthly_revenue": float(prediction.predicted_monthly_revenue),
            "predicted_retention_months": prediction.predicted_retention_months,
            "churn_probability": prediction.churn_probability,
            "expansion_potential": float(prediction.expansion_potential),
            "factors": prediction.factors
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ltv/churn-prediction")
async def predict_churn(request: ChurnPredictionRequest) -> Dict[str, Any]:
    """
    Predict customer churn risk.

    Returns churn probability and recommended actions.
    """
    try:
        prediction = ltv_engine.predict_churn(request.customer_id)

        return {
            "customer_id": prediction.customer_id,
            "churn_probability": prediction.churn_probability,
            "risk_level": prediction.risk_level.value,
            "days_to_churn": prediction.days_to_churn,
            "primary_risk_factors": prediction.primary_risk_factors,
            "recommended_actions": prediction.recommended_actions,
            "prevention_value": float(prediction.prevention_value)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ltv/health-score/{customer_id}")
async def get_health_score(customer_id: str) -> Dict[str, Any]:
    """Get customer health score"""
    try:
        metrics = ltv_engine.calculate_health_score(customer_id)

        return {
            "customer_id": metrics.customer_id,
            "overall_score": metrics.overall_score,
            "health_status": metrics.health_status.value,
            "component_scores": metrics.component_scores,
            "trend": metrics.trend,
            "alerts": metrics.alerts,
            "recommendations": metrics.recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Payment Orchestration ====================

@router.post("/payments/process")
async def process_payment(request: PaymentRequest) -> Dict[str, Any]:
    """
    Process payment with intelligent routing.

    Returns transaction result with fraud analysis.
    """
    try:
        payment_method = PaymentMethod[request.payment_method.upper()]

        transaction = payment_engine.process_payment(
            customer_id=request.customer_id,
            amount=Decimal(str(request.amount)),
            currency=request.currency,
            payment_method=payment_method
        )

        return {
            "transaction_id": transaction.transaction_id,
            "status": transaction.status.value,
            "amount": float(transaction.amount),
            "currency": transaction.currency,
            "payment_method": transaction.payment_method.value,
            "gateway": transaction.gateway.value,
            "fraud_score": transaction.fraud_score,
            "gateway_transaction_id": transaction.gateway_transaction_id,
            "failure_reason": transaction.failure_reason,
            "processed_at": transaction.processed_at.isoformat() if transaction.processed_at else None
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/payments/analytics")
async def get_payment_analytics(days: int = 30) -> Dict[str, Any]:
    """Get payment analytics"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        analytics = payment_engine.get_payment_analytics(start_date=start_date)
        return analytics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== Revenue Recognition ====================

@router.get("/revenue/forecast")
async def forecast_revenue(months_ahead: int = 12) -> Dict[str, Any]:
    """Get revenue forecast based on contracts"""
    try:
        forecast = revenue_system.forecast_revenue(months_ahead=months_ahead)
        return forecast
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/revenue/deferred")
async def get_deferred_revenue(customer_id: Optional[str] = None) -> Dict[str, Any]:
    """Get deferred revenue report"""
    try:
        report = revenue_system.get_deferred_revenue_report(customer_id=customer_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== AI Code Generation ====================

@router.post("/ai/generate-code")
async def generate_code(request: CodeGenerationRequestModel) -> Dict[str, Any]:
    """
    Generate code from natural language prompt.

    Returns generated code with quality analysis.
    """
    try:
        language = ProgrammingLanguage[request.language.upper()]

        optimization_goals = [
            OptimizationType[goal.upper()]
            for goal in request.optimization_goals
        ]

        gen_request = CodeGenerationRequest(
            request_id=f"gen_{datetime.utcnow().timestamp()}",
            language=language,
            prompt=request.prompt,
            context={},
            constraints=[],
            optimization_goals=optimization_goals,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        generated = code_generator.generate_code(gen_request)

        return {
            "code": generated.code,
            "language": generated.language.value,
            "quality_score": generated.quality_score,
            "security_score": generated.security_score,
            "performance_score": generated.performance_score,
            "explanation": generated.explanation,
            "imports": generated.imports,
            "complexity_metrics": generated.complexity_metrics,
            "test_code": generated.test_code,
            "documentation": generated.documentation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ai/analyze-code")
async def analyze_code(request: CodeAnalysisRequest) -> Dict[str, Any]:
    """
    Analyze code quality and security.

    Returns detailed analysis with suggestions.
    """
    try:
        language = ProgrammingLanguage[request.language.upper()]

        analysis = code_generator.analyze_code(request.code, language)

        return {
            "quality_level": analysis.quality_level.value,
            "cyclomatic_complexity": analysis.cyclomatic_complexity,
            "lines_of_code": analysis.lines_of_code,
            "code_smells": analysis.code_smells,
            "security_vulnerabilities": analysis.security_vulnerabilities,
            "performance_issues": analysis.performance_issues,
            "refactoring_suggestions": analysis.refactoring_suggestions,
            "maintainability_index": analysis.maintainability_index
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ai/suggest-optimizations")
async def suggest_optimizations(request: CodeAnalysisRequest) -> Dict[str, Any]:
    """Get optimization suggestions for code"""
    try:
        language = ProgrammingLanguage[request.language.upper()]

        suggestions = code_generator.suggest_optimizations(request.code, language)

        return {
            "total_suggestions": len(suggestions),
            "suggestions": suggestions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== System Metrics ====================

@router.get("/metrics/pricing")
async def get_pricing_metrics() -> Dict[str, Any]:
    """Get pricing engine metrics"""
    return {
        "total_price_calculations": len(pricing_engine.price_history),
        "products": list(pricing_engine.base_prices.keys()),
        "ab_tests_active": len([t for t in pricing_engine.ab_tests.values() if t["status"] == "active"])
    }


@router.get("/metrics/ltv")
async def get_ltv_metrics() -> Dict[str, Any]:
    """Get LTV prediction metrics"""
    return {
        "total_customers": len(ltv_engine.customer_profiles),
        "total_predictions": len(ltv_engine.ltv_predictions),
        "churn_predictions": len(ltv_engine.churn_predictions),
        "cohorts_analyzed": len(ltv_engine.cohorts)
    }


@router.get("/metrics/payments")
async def get_payment_metrics() -> Dict[str, Any]:
    """Get payment processing metrics"""
    return payment_engine.payment_metrics


@router.get("/metrics/codegen")
async def get_codegen_metrics() -> Dict[str, Any]:
    """Get code generation metrics"""
    return code_generator.metrics
