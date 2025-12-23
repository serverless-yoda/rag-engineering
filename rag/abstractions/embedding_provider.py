# abstractions/embedding_provider.py

"""
Abstract interface for embedding service providers.

This module defines the contract that all embedding providers must implement,
allowing the system to work with any embedding service (Azure OpenAI, OpenAI, Cohere, etc.).
"""

from abc import ABC, abstractmethod
from typing import List

# Type aliases for clarity
EmbeddingVector = List[float]   # # Single embedding (e.g., [0.1, 0.2, ..., 0.9])
EmbeddingMatrix = List[EmbeddingVector] # Multiple embeddings (e.g., [[0.1, 0.2], [0.3, 0.4], ...]

class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding generation services.
    
    Implementations must provide:
    1. embed() method to convert text into vector embeddings
    2. close() method to cleanup resources
    
    Example implementations:
    - AzureOpenAIEmbedder (Azure OpenAI embeddings)
    - OpenAIEmbedder (OpenAI API embeddings)
    - LocalEmbedder (sentence-transformers, etc.)
    """

    @abstractmethod
    async def embed(self, texts: List[str]) -> EmbeddingMatrix:

        """
        Generate embeddings for a list of text strings.
        
        This method should handle batching internally if needed and return
        embeddings in the same order as the input texts.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors, one per input text
            Each vector is a list of floats (dimensionality depends on model)
        
        Raises:
            Exception: If embedding generation fails (connection error, rate limit, etc.)
        
        Example:
            >>> embedder = AzureOpenAIEmbedder(...)
            >>> embeddings = await embedder.embed(["hello", "world"])
            >>> len(embeddings)  # Should equal len(texts)
            2
            >>> len(embeddings[0])  # Depends on model (e.g., 1536 for text-embedding-ada-002)
            1536
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        
        This method is called when the provider is no longer needed,
        typically in an async context manager's __aexit__ method.
        
        Should handle errors gracefully and not raise exceptions.
        """
        pass
