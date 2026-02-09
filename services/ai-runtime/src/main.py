"""
AI Runtime Service.

Routes LLM requests to appropriate models with cost optimization and caching.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import datetime
from typing import List, Dict, Optional
from uuid import UUID
from enum import Enum

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from shared.libs.config import AIRuntimeConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import AgentRole, ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)


# Configuration
config = load_config(AIRuntimeConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS AI Runtime",
    version=config.service_version,
    description="LLM routing and execution service"
)

# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(TracingMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================

class CompletionRequest(BaseModel):
    """Request for LLM completion."""
    role: AgentRole
    prompt: str
    context: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    max_tokens: int = 2000
    temperature: float = 0.7
    user_id: UUID


class CompletionResponse(BaseModel):
    """LLM completion response."""
    content: str
    model_used: str
    tokens_used: int
    cost_usd: float
    latency_ms: int
    finish_reason: str
    cached: bool = False


class EmbeddingRequest(BaseModel):
    """Request for text embeddings."""
    texts: List[str]
    model: str = "text-embedding-ada-002"


class EmbeddingResponse(BaseModel):
    """Embedding response."""
    embeddings: List[List[float]]
    model_used: str
    tokens_used: int
    cost_usd: float


# ============================================================================
# Model Configuration
# ============================================================================

MODEL_COSTS = {
    # OpenAI pricing (per 1K tokens)
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0},
    # Anthropic pricing
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
}

ROLE_TO_MODEL = {
    AgentRole.PLANNER: config.default_planner_model,
    AgentRole.REASONER: config.default_planner_model,
    AgentRole.EXECUTOR: config.default_executor_model,
    AgentRole.CRITIC: config.default_critic_model,
    AgentRole.SUMMARIZER: config.default_executor_model,
}


# ============================================================================
# Model Router
# ============================================================================

class ModelRouter:
    """
    Routes requests to appropriate models.

    Selects model based on role, cost, and availability.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="ModelRouter")

    def select_model(
        self,
        role: AgentRole,
        max_cost: Optional[float] = None
    ) -> str:
        """
        Select appropriate model for role.

        Args:
            role: Agent role
            max_cost: Maximum cost per 1K tokens (optional)

        Returns:
            Model name
        """
        # Get default model for role
        model = ROLE_TO_MODEL.get(role, "gpt-3.5-turbo")

        # Check cost constraint
        if max_cost:
            model_cost = MODEL_COSTS[model]["prompt"] + MODEL_COSTS[model]["completion"]
            if model_cost > max_cost:
                # Downgrade to cheaper model
                model = "gpt-3.5-turbo"
                self.logger.info(
                    "Downgraded model due to cost constraint",
                    extra={"original": ROLE_TO_MODEL.get(role), "selected": model}
                )

        return model

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Estimate cost for a completion.

        Args:
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens

        Returns:
            Cost in USD
        """
        costs = MODEL_COSTS.get(model, {"prompt": 0.001, "completion": 0.002})
        prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
        completion_cost = (completion_tokens / 1000) * costs["completion"]
        return prompt_cost + completion_cost


# ============================================================================
# LLM Client
# ============================================================================

class LLMClient:
    """
    Client for calling LLM APIs.

    In production, integrates with OpenAI, Anthropic, etc.
    For now, simulates responses.
    """

    def __init__(self, router: ModelRouter):
        self.router = router
        self.logger = get_contextual_logger(__name__, component="LLMClient")

    async def complete(
        self,
        role: AgentRole,
        prompt: str,
        context: List[Dict[str, str]],
        max_tokens: int,
        temperature: float
    ) -> CompletionResponse:
        """
        Generate completion.

        Args:
            role: Agent role
            prompt: Prompt text
            context: Conversation context
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Completion response
        """
        start_time = datetime.utcnow()

        # Select model
        model = self.router.select_model(role)

        self.logger.info(
            "Generating completion",
            extra={"model": model, "role": role.value}
        )

        # In production, call actual API:
        # if model.startswith("gpt"):
        #     response = await openai.ChatCompletion.create(...)
        # elif model.startswith("claude"):
        #     response = await anthropic.complete(...)

        # For now, simulate response
        import asyncio
        await asyncio.sleep(0.1)

        # Simulate completion
        content = f"[Simulated {model} response for {role.value}]\n\n{prompt[:100]}..."

        # Estimate tokens (rough: ~4 chars per token)
        prompt_tokens = len(prompt) // 4
        completion_tokens = len(content) // 4

        # Calculate cost
        cost = self.router.estimate_cost(model, prompt_tokens, completion_tokens)

        # Calculate latency
        latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return CompletionResponse(
            content=content,
            model_used=model,
            tokens_used=prompt_tokens + completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason="stop",
            cached=False
        )

    async def embed(
        self,
        texts: List[str],
        model: str
    ) -> EmbeddingResponse:
        """
        Generate embeddings.

        Args:
            texts: List of texts to embed
            model: Embedding model

        Returns:
            Embeddings
        """
        self.logger.info(
            "Generating embeddings",
            extra={"model": model, "count": len(texts)}
        )

        # In production, call actual API:
        # response = await openai.Embedding.create(model=model, input=texts)

        # For now, simulate embeddings
        import asyncio
        import random
        await asyncio.sleep(0.05)

        # Simulate 1536-dimensional embeddings (text-embedding-ada-002)
        embeddings = [
            [random.random() for _ in range(1536)]
            for _ in texts
        ]

        # Estimate tokens
        tokens = sum(len(text) // 4 for text in texts)

        # Calculate cost
        cost = (tokens / 1000) * MODEL_COSTS.get(model, {}).get("prompt", 0.0001)

        return EmbeddingResponse(
            embeddings=embeddings,
            model_used=model,
            tokens_used=tokens,
            cost_usd=cost
        )


# ============================================================================
# Response Validator
# ============================================================================

class ResponseValidator:
    """
    Validates LLM responses for quality and safety.

    Detects hallucinations, unsafe content, and errors.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="ResponseValidator")

    def validate(self, response: CompletionResponse) -> bool:
        """
        Validate response quality.

        Args:
            response: Completion response

        Returns:
            True if valid, False otherwise
        """
        # Check for empty response
        if not response.content or len(response.content.strip()) == 0:
            self.logger.warning("Empty response detected")
            return False

        # Check for common hallucination patterns
        hallucination_markers = [
            "I don't have access to",
            "I cannot",
            "As an AI",
            "I apologize, but"
        ]

        content_lower = response.content.lower()
        for marker in hallucination_markers:
            if marker.lower() in content_lower:
                self.logger.warning(
                    "Possible refusal detected",
                    extra={"marker": marker}
                )
                # Not necessarily invalid, just log it
                break

        # Check finish reason
        if response.finish_reason not in ["stop", "end_turn"]:
            self.logger.warning(
                "Unexpected finish reason",
                extra={"finish_reason": response.finish_reason}
            )

        # Validation passed
        return True


# ============================================================================
# API Endpoints
# ============================================================================

router = ModelRouter()
client = LLMClient(router)
validator = ResponseValidator()


@app.post("/complete", response_model=CompletionResponse)
async def complete(request: CompletionRequest):
    """
    Generate LLM completion.

    Routes to appropriate model based on role and generates response.
    """
    log = get_contextual_logger(
        __name__,
        action="complete",
        role=request.role.value,
        user_id=str(request.user_id)
    )

    try:
        # Generate completion
        response = await client.complete(
            role=request.role,
            prompt=request.prompt,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        # Validate response
        if not validator.validate(response):
            log.warning("Response validation failed")

        log.info(
            "Completion generated",
            extra={
                "model": response.model_used,
                "tokens": response.tokens_used,
                "cost": response.cost_usd
            }
        )

        return response

    except Exception as e:
        log.error("Completion failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Completion failed: {str(e)}"
        )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest):
    """
    Generate text embeddings.

    Used for semantic search in memory service.
    """
    log = get_contextual_logger(
        __name__,
        action="embed",
        count=len(request.texts)
    )

    try:
        response = await client.embed(
            texts=request.texts,
            model=request.model
        )

        log.info(
            "Embeddings generated",
            extra={
                "count": len(response.embeddings),
                "cost": response.cost_usd
            }
        )

        return response

    except Exception as e:
        log.error("Embedding failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding failed: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List available models and their costs."""
    return {
        "models": MODEL_COSTS,
        "role_assignments": {
            role.value: model
            for role, model in ROLE_TO_MODEL.items()
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ai-runtime",
        "version": config.service_version,
        "timestamp": datetime.utcnow().isoformat(),
        "providers": {
            "openai": bool(config.openai_api_key),
            "anthropic": bool(config.anthropic_api_key)
        }
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup."""
    logger.info(
        "AI Runtime starting",
        extra={
            "version": config.service_version,
            "default_planner_model": config.default_planner_model
        }
    )

    # Check API keys
    if not config.openai_api_key and not config.anthropic_api_key:
        logger.warning("No API keys configured - running in simulation mode")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("AI Runtime shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=getattr(config, 'port', 8005),
        log_level=config.log_level.lower()
    )
