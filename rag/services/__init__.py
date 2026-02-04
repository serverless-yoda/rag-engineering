# rag/services/__init__.py

"""
Azure-based implementations of abstract providers.

This package contains concrete implementations of the abstract interfaces
using Azure SDKs (Azure OpenAI and Azure AI Search).
"""


from .azure import AzureOpenAIEmbedder
from .azure import AzureSearchStore
from .azure import AzureOpenAILLM
from .azure import AzureContentSafety
from .openrouter import OpenRouterLLM, OpenRouterEmbedder
from .supabase import SupabaseVectorStore

__all__ = [
    "AzureOpenAIEmbedder",
    "AzureSearchStore",
    "AzureOpenAILLM",
    "AzureContentSafety",
    "OpenRouterLLM",
    "OpenRouterEmbedder",
    "SupabaseVectorStore"
]
