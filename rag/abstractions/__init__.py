# abstractions/__init__.py

"""
Azure-based implementations of abstract providers.

This package contains concrete implementations of the abstract interfaces
using Azure SDKs (Azure OpenAI and Azure AI Search).
"""

from .embedding_provider import EmbeddingProvider, EmbeddingMatrix
from .llm_provider import LLMProvider
from .vector_store_provider import VectorStoreProvider

__all__ = [
    "EmbeddingProvider",
    "EmbeddingMatrix",
    "LLMProvider",
    "VectorStoreProvider",
]



