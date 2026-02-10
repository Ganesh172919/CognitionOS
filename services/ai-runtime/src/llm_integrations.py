"""
Real LLM API integrations for OpenAI and Anthropic.

This module provides production-ready integrations with OpenAI and Anthropic APIs.
"""

import os
from typing import List, Dict, Optional, AsyncGenerator
from datetime import datetime
import tiktoken

# OpenAI SDK
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Anthropic SDK
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class OpenAIIntegration:
    """
    OpenAI API integration with support for chat completions and embeddings.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.encodings = {}  # Cache for tokenizers

    def _get_encoding(self, model: str):
        """Get tokenizer for model (cached)."""
        if model not in self.encodings:
            try:
                self.encodings[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                self.encodings[model] = tiktoken.get_encoding("cl100k_base")
        return self.encodings[model]

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Count tokens in text for given model.

        Args:
            text: Text to count tokens for
            model: Model name (determines tokenizer)

        Returns:
            Token count
        """
        encoding = self._get_encoding(model)
        return len(encoding.encode(text))

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            Dict with content, usage, and finish_reason
        """
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )

            if stream:
                # For streaming, return the generator
                return {"stream": response}

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason

            # Token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            return {
                "content": content,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": model
            }

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    async def create_embedding(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> Dict:
        """
        Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            model: Embedding model

        Returns:
            Dict with embeddings and usage info
        """
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=texts
            )

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]

            # Token usage
            total_tokens = response.usage.total_tokens

            return {
                "embeddings": embeddings,
                "model": model,
                "total_tokens": total_tokens
            }

        except Exception as e:
            raise Exception(f"OpenAI Embedding API error: {str(e)}")


class AnthropicIntegration:
    """
    Anthropic Claude API integration.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    def count_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Claude uses a similar tokenization to GPT, so we approximate.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    async def create_message(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system: Optional[str] = None
    ) -> Dict:
        """
        Generate message completion.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Claude model to use
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            system: System prompt (optional)

        Returns:
            Dict with content, usage, and finish_reason
        """
        try:
            # Extract system message if present in messages
            if not system and messages and messages[0]["role"] == "system":
                system = messages[0]["content"]
                messages = messages[1:]

            response = await self.client.messages.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                system=system if system else anthropic.NOT_GIVEN
            )

            # Extract response data
            content = response.content[0].text if response.content else ""
            finish_reason = response.stop_reason

            # Token usage
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            return {
                "content": content,
                "finish_reason": finish_reason,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": model
            }

        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")


# Factory function to get appropriate integration
def get_llm_integration(provider: str, api_key: Optional[str] = None):
    """
    Get LLM integration for provider.

    Args:
        provider: 'openai' or 'anthropic'
        api_key: API key (optional, falls back to env var)

    Returns:
        LLM integration instance
    """
    if provider == "openai":
        return OpenAIIntegration(api_key)
    elif provider == "anthropic":
        return AnthropicIntegration(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")
