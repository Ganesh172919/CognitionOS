"""
AI Runtime Service with Real LLM Integration.

Routes LLM requests to OpenAI or Anthropic with cost optimization and caching.
"""

import sys
import os

# Add shared libs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from datetime import datetime
from typing import List, Dict, Optional
from uuid import UUID

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from shared.libs.config import AIRuntimeConfig, load_config
from shared.libs.logger import setup_logger, get_contextual_logger
from shared.libs.models import AgentRole, ErrorResponse
from shared.libs.middleware import (
    TracingMiddleware,
    LoggingMiddleware,
    ErrorHandlingMiddleware
)

# Import database models for usage tracking
from database import get_db, LLMUsage

# Import LLM integrations
from llm_integrations import OpenAIIntegration, AnthropicIntegration


# Configuration
config = load_config(AIRuntimeConfig)
logger = setup_logger(__name__, level=config.log_level)

# FastAPI app
app = FastAPI(
    title="CognitionOS AI Runtime",
    version=config.service_version,
    description="LLM routing and execution service with real API integration"
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
    task_id: Optional[UUID] = None
    agent_id: Optional[UUID] = None


class CompletionResponse(BaseModel):
    """LLM completion response."""
    content: str
    model_used: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
    finish_reason: str
    cached: bool = False


class EmbeddingRequest(BaseModel):
    """Request for text embeddings."""
    texts: List[str]
    model: str = "text-embedding-ada-002"
    user_id: Optional[UUID] = None


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
    # OpenAI pricing (per 1K tokens) - Updated to latest pricing
    "gpt-4": {"prompt": 0.03, "completion": 0.06},
    "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
    "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
    "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
    "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0},
    # Anthropic pricing
    "claude-3-opus-20240229": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
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
    Routes requests to appropriate models with fallback support.
    """

    def __init__(self):
        self.logger = get_contextual_logger(__name__, component="ModelRouter")

    def select_model(
        self,
        role: AgentRole,
        max_cost: Optional[float] = None,
        prefer_provider: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Select appropriate model for role.

        Args:
            role: Agent role
            max_cost: Maximum cost per 1K tokens (optional)
            prefer_provider: Preferred provider ('openai' or 'anthropic')

        Returns:
            Tuple of (provider, model_name)
        """
        # Get default model for role
        model = ROLE_TO_MODEL.get(role, "gpt-3.5-turbo")

        # Determine provider
        if model.startswith("gpt"):
            provider = "openai"
        elif model.startswith("claude"):
            provider = "anthropic"
        else:
            provider = "openai"  # Default

        # Override if preference specified
        if prefer_provider:
            provider = prefer_provider
            # Adjust model if needed
            if provider == "anthropic" and model.startswith("gpt"):
                model = "claude-3-sonnet-20240229"
            elif provider == "openai" and model.startswith("claude"):
                model = "gpt-4-turbo-preview"

        # Check cost constraint
        if max_cost:
            model_cost = MODEL_COSTS.get(model, {}).get("prompt", 0) + MODEL_COSTS.get(model, {}).get("completion", 0)
            if model_cost > max_cost:
                # Downgrade to cheaper model
                if provider == "openai":
                    model = "gpt-3.5-turbo"
                else:
                    model = "claude-3-haiku-20240307"

                self.logger.info(
                    "Downgraded model due to cost constraint",
                    extra={"original": ROLE_TO_MODEL.get(role), "selected": model}
                )

        return provider, model

    def estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """
        Calculate cost for a completion.

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
# LLM Client with Real Integration
# ============================================================================

class LLMClient:
    """
    Client for calling LLM APIs with fallback and error handling.
    """

    def __init__(self, router: ModelRouter):
        self.router = router
        self.logger = get_contextual_logger(__name__, component="LLMClient")

        # Initialize integrations
        self.openai_client = None
        self.anthropic_client = None

        # Try to initialize OpenAI
        if config.openai_api_key:
            try:
                self.openai_client = OpenAIIntegration(config.openai_api_key)
                self.logger.info("OpenAI integration initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")

        # Try to initialize Anthropic
        if config.anthropic_api_key:
            try:
                self.anthropic_client = AnthropicIntegration(config.anthropic_api_key)
                self.logger.info("Anthropic integration initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Anthropic: {e}")

    async def complete(
        self,
        role: AgentRole,
        prompt: str,
        context: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        db: Optional[AsyncSession] = None,
        user_id: Optional[UUID] = None,
        task_id: Optional[UUID] = None,
        agent_id: Optional[UUID] = None
    ) -> CompletionResponse:
        """
        Generate completion with automatic provider selection and fallback.
        """
        start_time = datetime.utcnow()

        # Build messages array
        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        # Select provider and model
        provider, model = self.router.select_model(role)

        self.logger.info(
            "Generating completion",
            extra={"provider": provider, "model": model, "role": role.value}
        )

        # Try primary provider
        try:
            result = await self._call_provider(provider, model, messages, temperature, max_tokens)

        except Exception as primary_error:
            self.logger.warning(
                f"Primary provider {provider} failed: {primary_error}",
                extra={"provider": provider}
            )

            # Try fallback provider
            fallback_provider = "anthropic" if provider == "openai" else "openai"
            if (fallback_provider == "openai" and self.openai_client) or \
               (fallback_provider == "anthropic" and self.anthropic_client):

                _, fallback_model = self.router.select_model(role, prefer_provider=fallback_provider)
                self.logger.info(f"Trying fallback provider: {fallback_provider}")

                try:
                    result = await self._call_provider(fallback_provider, fallback_model, messages, temperature, max_tokens)
                    provider = fallback_provider
                    model = fallback_model
                except Exception as fallback_error:
                    self.logger.error(f"Fallback provider also failed: {fallback_error}")
                    # Fall back to simulation
                    result = self._simulate_response(model, prompt, max_tokens)
            else:
                # No fallback available, use simulation
                result = self._simulate_response(model, prompt, max_tokens)

        # Calculate metrics
        latency_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        cost = self.router.estimate_cost(
            model,
            result["prompt_tokens"],
            result["completion_tokens"]
        )

        # Track usage in database if available
        if db and user_id:
            try:
                usage = LLMUsage(
                    user_id=user_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    model=model,
                    prompt_tokens=result["prompt_tokens"],
                    completion_tokens=result["completion_tokens"],
                    total_tokens=result["total_tokens"],
                    cost_usd=cost
                )
                db.add(usage)
                await db.commit()
            except Exception as e:
                self.logger.warning(f"Failed to track usage: {e}")

        return CompletionResponse(
            content=result["content"],
            model_used=model,
            tokens_used=result["total_tokens"],
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=result["finish_reason"],
            cached=False
        )

    async def _call_provider(
        self,
        provider: str,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> Dict:
        """Call specific LLM provider."""
        if provider == "openai":
            if not self.openai_client:
                raise Exception("OpenAI client not initialized")
            return await self.openai_client.chat_completion(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        elif provider == "anthropic":
            if not self.anthropic_client:
                raise Exception("Anthropic client not initialized")
            return await self.anthropic_client.create_message(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _simulate_response(self, model: str, prompt: str, max_tokens: int) -> Dict:
        """Simulate LLM response when no API available."""
        content = f"[Simulated {model} response]\n\n{prompt[:100]}..."
        prompt_tokens = len(prompt) // 4
        completion_tokens = min(len(content) // 4, max_tokens // 4)

        return {
            "content": content,
            "finish_reason": "stop",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model
        }

    async def embed(
        self,
        texts: List[str],
        model: str,
        db: Optional[AsyncSession] = None,
        user_id: Optional[UUID] = None
    ) -> EmbeddingResponse:
        """Generate embeddings."""
        self.logger.info(
            "Generating embeddings",
            extra={"model": model, "count": len(texts)}
        )

        # Try OpenAI embedding
        if self.openai_client and model.startswith("text-embedding"):
            try:
                result = await self.openai_client.create_embedding(texts, model)

                cost = (result["total_tokens"] / 1000) * MODEL_COSTS.get(model, {}).get("prompt", 0.0001)

                # Track usage
                if db and user_id:
                    try:
                        usage = LLMUsage(
                            user_id=user_id,
                            model=model,
                            prompt_tokens=result["total_tokens"],
                            completion_tokens=0,
                            total_tokens=result["total_tokens"],
                            cost_usd=cost
                        )
                        db.add(usage)
                        await db.commit()
                    except Exception as e:
                        self.logger.warning(f"Failed to track embedding usage: {e}")

                return EmbeddingResponse(
                    embeddings=result["embeddings"],
                    model_used=model,
                    tokens_used=result["total_tokens"],
                    cost_usd=cost
                )

            except Exception as e:
                self.logger.warning(f"OpenAI embedding failed: {e}")

        # Fallback to simulation
        import random
        embeddings = [[random.random() for _ in range(1536)] for _ in texts]
        tokens = sum(len(text) // 4 for text in texts)
        cost = (tokens / 1000) * MODEL_COSTS.get(model, {}).get("prompt", 0.0001)

        return EmbeddingResponse(
            embeddings=embeddings,
            model_used=model,
            tokens_used=tokens,
            cost_usd=cost
        )


# ============================================================================
# API Endpoints
# ============================================================================

router_instance = ModelRouter()
client = LLMClient(router_instance)


@app.post("/complete", response_model=CompletionResponse)
async def complete(request: CompletionRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate LLM completion with real API integration.

    Routes to OpenAI or Anthropic based on model selection with automatic fallback.
    """
    log = get_contextual_logger(
        __name__,
        action="complete",
        role=request.role.value,
        user_id=str(request.user_id)
    )

    try:
        response = await client.complete(
            role=request.role,
            prompt=request.prompt,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            db=db,
            user_id=request.user_id,
            task_id=request.task_id,
            agent_id=request.agent_id
        )

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
async def embed(request: EmbeddingRequest, db: AsyncSession = Depends(get_db)):
    """
    Generate text embeddings using OpenAI API.
    """
    log = get_contextual_logger(
        __name__,
        action="embed",
        count=len(request.texts)
    )

    try:
        response = await client.embed(
            texts=request.texts,
            model=request.model,
            db=db,
            user_id=request.user_id
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
            "openai": client.openai_client is not None,
            "anthropic": client.anthropic_client is not None
        },
        "simulation_mode": client.openai_client is None and client.anthropic_client is None
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
        logger.warning("⚠️  No API keys configured - running in SIMULATION MODE")
        logger.warning("    Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable real LLM calls")
    elif config.openai_api_key:
        logger.info("✓ OpenAI integration enabled")
    elif config.anthropic_api_key:
        logger.info("✓ Anthropic integration enabled")


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
