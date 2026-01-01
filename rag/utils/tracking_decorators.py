# utils/tracking_decorators.py

"""Tracking decorators for monitoring function execution."""

from typing import List
from ..abstractions import EmbeddingProvider, EmbeddingMatrix
from ..utils import TokenTracker


class TrackedEmbeddingProvider(EmbeddingProvider):
    """Decorator that adds tracking to any embedder"""
    def __init__(self, embedder: EmbeddingProvider, tracker: TokenTracker):
        self.embedder = embedder
        self.tracker = tracker
    
    async def embed(self, texts: List[str]) -> EmbeddingMatrix:
        # Track before calling
        self.tracker.add_embedding_usage(texts, stage="embedding")
        return await self.embedder.embed(texts)
    
    async def close(self):
        await self.embedder.close()

