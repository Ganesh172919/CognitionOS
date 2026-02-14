"""
LLM Provider Abstraction Layer

Provides a unified interface for multiple LLM providers with fallback support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import asyncio
import time


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class LLMRequest:
    """LLM request"""
    messages: List[Dict[str, str]]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """LLM response"""
    content: str
    model: str
    provider: LLMProvider
    usage: Dict[str, int]
    latency_ms: int
    cost_usd: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProviderInterface(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> LLMProvider:
        """Get the provider name"""
        pass


class OpenAIProvider(LLMProviderInterface):
    """OpenAI LLM provider"""
    
    def __init__(self, api_key: str, max_retries: int = 3, timeout: int = 60):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """Get or create OpenAI client"""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            except ImportError:
                raise ImportError("openai package not installed. Install with: pip install openai")
        return self._client
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using OpenAI API"""
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            response = await client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                stream=request.stream,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Calculate cost (approximate)
            cost_usd = self._calculate_cost(
                model=request.model,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=request.model,
                provider=LLMProvider.OPENAI,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                metadata=request.metadata,
            )
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if OpenAI API is accessible"""
        try:
            client = self._get_client()
            # Simple test request
            await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
    
    def get_provider_name(self) -> LLMProvider:
        """Get provider name"""
        return LLMProvider.OPENAI
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate approximate cost in USD"""
        # Pricing as of 2024 (approximate)
        pricing = {
            "gpt-4": {"prompt": 0.03 / 1000, "completion": 0.06 / 1000},
            "gpt-4-turbo": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
            "gpt-3.5-turbo": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
        }
        
        # Find matching model
        for key in pricing:
            if model.startswith(key):
                return (
                    prompt_tokens * pricing[key]["prompt"] +
                    completion_tokens * pricing[key]["completion"]
                )
        
        # Default fallback
        return (prompt_tokens + completion_tokens) * 0.001 / 1000


class AnthropicProvider(LLMProviderInterface):
    """Anthropic Claude LLM provider"""
    
    def __init__(self, api_key: str, max_retries: int = 3, timeout: int = 60):
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
    
    def _get_client(self):
        """Get or create Anthropic client"""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            except ImportError:
                raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        return self._client
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using Anthropic API"""
        start_time = time.time()
        
        try:
            client = self._get_client()
            
            # Convert messages format
            system_message = next(
                (msg["content"] for msg in request.messages if msg["role"] == "system"),
                None
            )
            messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in request.messages
                if msg["role"] != "system"
            ]
            
            response = await client.messages.create(
                model=request.model,
                messages=messages,
                system=system_message,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Calculate cost
            cost_usd = self._calculate_cost(
                model=request.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=request.model,
                provider=LLMProvider.ANTHROPIC,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                metadata=request.metadata,
            )
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def health_check(self) -> bool:
        """Check if Anthropic API is accessible"""
        try:
            client = self._get_client()
            await client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False
    
    def get_provider_name(self) -> LLMProvider:
        """Get provider name"""
        return LLMProvider.ANTHROPIC
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate approximate cost in USD"""
        # Pricing as of 2024 (approximate)
        pricing = {
            "claude-3-opus": {"input": 15 / 1_000_000, "output": 75 / 1_000_000},
            "claude-3-sonnet": {"input": 3 / 1_000_000, "output": 15 / 1_000_000},
            "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
        }
        
        # Find matching model
        for key in pricing:
            if model.startswith(key):
                return (
                    input_tokens * pricing[key]["input"] +
                    output_tokens * pricing[key]["output"]
                )
        
        # Default fallback
        return (input_tokens + output_tokens) * 0.001 / 1000


class LLMRouter:
    """
    LLM Router with fallback support.
    
    Automatically falls back to secondary providers if primary fails.
    """
    
    def __init__(self, providers: List[LLMProviderInterface]):
        if not providers:
            raise ValueError("At least one provider must be configured")
        self.providers = providers
        self.primary_provider = providers[0]
        self.fallback_providers = providers[1:] if len(providers) > 1 else []
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate response with automatic fallback.
        
        Tries primary provider first, then falls back to others if it fails.
        """
        errors = []
        
        # Try primary provider
        try:
            return await self.primary_provider.generate(request)
        except Exception as e:
            errors.append(f"{self.primary_provider.get_provider_name()}: {str(e)}")
        
        # Try fallback providers
        for provider in self.fallback_providers:
            try:
                return await provider.generate(request)
            except Exception as e:
                errors.append(f"{provider.get_provider_name()}: {str(e)}")
        
        # All providers failed
        raise Exception(f"All LLM providers failed: {'; '.join(errors)}")
    
    async def health_check_all(self) -> Dict[LLMProvider, bool]:
        """Check health of all providers"""
        results = {}
        for provider in self.providers:
            try:
                is_healthy = await provider.health_check()
                results[provider.get_provider_name()] = is_healthy
            except Exception:
                results[provider.get_provider_name()] = False
        return results


def create_llm_router(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 60,
) -> LLMRouter:
    """
    Factory function to create LLM router with available providers.
    
    Args:
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key
        max_retries: Max retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        Configured LLM router
    """
    providers = []
    
    if openai_api_key:
        providers.append(OpenAIProvider(
            api_key=openai_api_key,
            max_retries=max_retries,
            timeout=timeout,
        ))
    
    if anthropic_api_key:
        providers.append(AnthropicProvider(
            api_key=anthropic_api_key,
            max_retries=max_retries,
            timeout=timeout,
        ))
    
    if not providers:
        raise ValueError("At least one LLM provider API key must be provided")
    
    return LLMRouter(providers)
