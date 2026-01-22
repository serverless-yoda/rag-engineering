# rag/services/openrouter/openrouter_embedder.py

"""
OpenRouter embedding provider (OpenAI-compatible).
Supports OpenAI, Cohere, VoyageAI embeddings via OpenRouter.
"""

import logging
from typing import List, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ...interfaces import EmbeddingProvider, EmbeddingMatrix
from ...utils.chunk import TokenTracker

class OpenRouterEmbedder(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "voyage/voyage-2",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.token_tracker = token_tracker
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def embed(self, texts: List[str], stage: str = "embedding") -> EmbeddingMatrix:
        if not texts:
            return []
        
        if self.token_tracker:
            self.token_tracker.add_embedding_usage(texts, stage=stage)
        
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                extra_headers={
                    "HTTP-Referer": "https://your-app.com",
                    "X-Title": "RAG Multi-Agent Pipeline",
                }
            )
            
            embeddings = [d.embedding for d in response.data]
            logging.debug(f"Generated {len(embeddings)} embeddings via OpenRouter")
            return embeddings
            
        except Exception as e:
            logging.error(f"OpenRouter embedding failed: {e}")
            raise

    async def close(self) -> None:
        try:
            await self.client.close()
            logging.info(f"{self.__class__.__name__} client closed.")
        except Exception as e:
            logging.error(f"Error closing OpenRouter Embedder: {e}")
