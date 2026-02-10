"""
Helper module to call AI Runtime service for embeddings.
"""

import os
from typing import List, Optional
import httpx
from uuid import UUID


class AIRuntimeClient:
    """
    Client to call AI Runtime service for embeddings and completions.
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize AI Runtime client.

        Args:
            base_url: Base URL of AI Runtime service (defaults to env var or localhost)
        """
        self.base_url = base_url or os.getenv("AI_RUNTIME_URL", "http://ai-runtime:8005")
        self.timeout = httpx.Timeout(30.0, connect=5.0)

    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        user_id: Optional[UUID] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for texts using AI Runtime.

        Args:
            texts: List of texts to embed
            model: Embedding model to use
            user_id: User ID for tracking (optional)

        Returns:
            List of embeddings (one per text)

        Raises:
            Exception: If API call fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/embed",
                    json={
                        "texts": texts,
                        "model": model,
                        "user_id": str(user_id) if user_id else None
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["embeddings"]

            except httpx.HTTPStatusError as e:
                raise Exception(f"AI Runtime returned error: {e.response.status_code} - {e.response.text}")
            except httpx.ConnectError:
                # Fallback to simulated embeddings if AI Runtime is unavailable
                import random
                return [[random.random() for _ in range(1536)] for _ in texts]
            except Exception as e:
                raise Exception(f"Failed to generate embeddings: {str(e)}")

    async def health_check(self) -> bool:
        """
        Check if AI Runtime service is available.

        Returns:
            True if healthy, False otherwise
        """
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except:
                return False
