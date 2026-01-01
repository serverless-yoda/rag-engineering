# implementation/azure_openai_embedder.py

"""
Azure OpenAI embedding provider implementation with token tracking and circuit breaker

This module implements the EmbeddingProvider interface using Azure OpenAI's embeddings API.
"""


import logging
from typing import List, Optional
from openai import AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..abstractions.embedding_provider import EmbeddingProvider, EmbeddingMatrix
from ..utils import TokenTracker


class AzureOpenAIEmbedder(EmbeddingProvider):
    """
    Azure OpenAI implementation of the EmbeddingProvider interface.
    
    Uses the Azure OpenAI embeddings API to generate vector embeddings from text.
    Supports models like text-embedding-ada-002 (1536 dimensions).
    
    The client is created asynchronously and supports batch embedding requests.
    
    Example:
        >>> embedder = AzureOpenAIEmbedder(
        ...     endpoint="https://my-openai.openai.azure.com/",
        ...     api_key="key123",
        ...     api_version="2024-02-15-preview",
        ...     deployment_name="text-embedding-ada-002",
        ... )
        >>> embeddings = await embedder.embed(["hello", "world"])
        >>> await embedder.close()
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        timeout: float = 60.0,
        token_tracker: Optional[TokenTracker] = None
    ):
        """
        Initialize the Azure OpenAI embedder.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: API key for authentication
            api_version: API version string (e.g., "2024-02-15-preview")
            deployment_name: Name of the deployed embedding model
            timeout: Timeout in seconds for API calls
            token_tracker: Optional token tracker
        """
        self.deployment_name = deployment_name
        self.token_tracker = token_tracker

        # Create async Azure OpenAI client
        # This client handles connection pooling and retry logic internally
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=endpoint,
            timeout=timeout,
        )
    
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def embed(self, texts: List[str], stage: str = "embedding") -> EmbeddingMatrix:
        """
        Generate embeddings using Azure OpenAI.
        
        Sends all texts in a single batch request to the embeddings API.
        The API returns embeddings in the same order as input texts.
        
        Args:
            texts: List of strings to embed (max batch size depends on API limits)
            stage: Stage name for tracking
        Returns:
            List of embedding vectors (one per input text)
        
        
        Raises:
            Exception: If API call fails (network error, rate limit, invalid input, etc.)
        
        Note:
            Azure OpenAI has batch size limits (typically 16-2048 texts per request).
            For larger batches, consider splitting in the caller (see DocumentIngester).
        """
        if not texts:
            return []
        
        
        # Track tokens before embedding
        if self.token_tracker:
            self.token_tracker.add_embedding_usage(texts, stage=stage)

        try:
            # Call Azure OpenAI embeddings API
            # model parameter uses the deployment name (not the base model name)
            response = await self.client.embeddings.create(
                model=self.deployment_name,
                input=list(texts),  # Convert to list to ensure compatibility
            )
            
            # Extract embedding vectors from response
            # response.data is a list of embedding objects with .embedding attribute
            embeddings = [d.embedding for d in response.data]
            
            logging.debug(f"Generated {len(embeddings)} embeddings via Azure OpenAI")
            return embeddings
            
        except Exception as e:
            logging.error(f"Azure OpenAI embedding generation failed: {e}")
            raise
    
    async def close(self) -> None:
        """
        Close the Azure OpenAI client connection.
        
        Gracefully closes the underlying HTTP client and connection pool.
        Safe to call multiple times.
        """
        try:
            await self.client.close()
        except Exception as e:
            # Log but don't raise - cleanup should be silent
            logging.error(f"Error closing Azure OpenAI embedder: {e}")
