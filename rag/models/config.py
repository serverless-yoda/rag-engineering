# models/config.py

"""
Configuration models for the RAG system.

This module defines the central configuration object that holds all settings
for Azure OpenAI, Azure AI Search, and pipeline behavior.
"""

from dataclasses import dataclass, field
from .types import ChunkingConfig


@dataclass
class RAGConfig:
    """
    Centralized configuration for the RAG pipeline.
    
    This configuration object is passed to the RAGPipeline orchestrator and
    controls all aspects of the B.I.S.A. (Build, Ingest, Search, Answer) flow.
    
    Azure OpenAI Settings:
        azure_openai_endpoint: Base URL for Azure OpenAI service
        azure_openai_api_key: API key for authentication
        azure_openai_api_version: API version (e.g., "2024-02-15-preview")
        embedding_deployment: Deployment name for embeddings model
        model_deployment: Deployment name for chat/completion model
    
    Azure AI Search Settings:
        azure_search_endpoint: Base URL for Azure AI Search service
        azure_search_api_key: API key for authentication
        index_name: Name of the search index to use/create
        vector_dimensions: Dimensionality of embedding vectors (default: 1536 for text-embedding-ada-002)
    
    Pipeline Settings:
        default_namespace: Default namespace for organizing documents
        batch_size: Number of chunks to embed in a single API call
        chunking: ChunkingConfig object controlling text splitting behavior
        llm_timeout: Timeout in seconds for LLM API calls
        llm_retries: Number of retry attempts for failed LLM calls
    """
    # Required Azure OpenAI configuration (no defaults)
    azure_openai_endpoint: str
    azure_openai_api_key: str   
    
    # Required Azure AI Search configuration (no defaults)
    azure_search_endpoint: str
    azure_search_api_key: str   
    index_name: str
    
    # Optional Azure OpenAI configuration (with defaults)
    azure_openai_api_version: str = "2024-02-15-preview"
    embedding_deployment: str = "text-embedding-ada-002"
    model_deployment: str = "gpt-5-nano"
    
    # Optional Azure AI Search configuration (with defaults)
    vector_dimensions: int = 1536  # Default for text-embedding-ada-002
    
    # Optional LLM settings (with defaults)
    llm_timeout: float = 60.0  # seconds
    llm_retries: int = 3
    
    # Optional pipeline settings (with defaults)
    default_namespace: str = "KnowledgeStore"
    batch_size: int = 16
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
