# di/container.py

"""
Dependency injection container using dependency-injector.
"""

from dependency_injector import containers, providers
#from ..models import RAGConfig
from ..implementations import (
    AzureOpenAIEmbedder,
    AzureOpenAILLM,
    AzureSearchStore,
    AzureContentSafety
)

from ..core import (
    IndexManager,
    DocumentIngester,
    SemanticSearcher,
    AnswerGenerator,
)
from ..utils import TokenTracker, TrackedEmbeddingProvider
from ..pipeline.rag_pipeline import RAGPipeline

class Container(containers.DeclarativeContainer):
    """Main DI container for the RAG pipeline."""
    
    config = providers.Configuration()
    
    # Token tracker (singleton)
    token_tracker = providers.Singleton(TokenTracker)

    embedder = providers.Singleton(
        TrackedEmbeddingProvider,
        embedder=providers.Factory(AzureOpenAIEmbedder, ...),
        tracker=token_tracker,
    )

    # Embedding provider
    embedder = providers.Singleton(
        AzureOpenAIEmbedder,
        endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        deployment_name=config.embedding_deployment,
        token_tracker=token_tracker,
    )
    
    # LLM provider
    llm = providers.Singleton(
        AzureOpenAILLM,
        endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        deployment_name=config.model_deployment,
        timeout=config.llm_timeout,
        retries=config.llm_retries,
        token_tracker=token_tracker,
    )
    
    # Vector store provider
    store = providers.Singleton(
        AzureSearchStore,
        endpoint=config.azure_search_endpoint,
        api_key=config.azure_search_api_key,
        index_name=config.index_name,
    )
    
    # Index manager
    index_manager = providers.Singleton(
        IndexManager,
        endpoint=config.azure_search_endpoint,
        api_key=config.azure_search_api_key,
        index_name=config.index_name,
        vector_dimensions=config.vector_dimensions,
    )
    
    # Content safety (optional)
    content_safety = providers.Singleton(
        AzureContentSafety,
        endpoint=config.content_safety_endpoint,
        api_key=config.content_safety_api_key,
        severity_threshold=config.content_moderation_threshold,
        enabled=config.content_moderation_enabled,
    )
    
    # Document ingester
    ingester = providers.Singleton(
        DocumentIngester,
        embedder=embedder,
        store=store,
        index_manager=index_manager,
        batch_size=config.batch_size,
    )
    
    # Semantic searcher
    searcher = providers.Singleton(
        SemanticSearcher,
        embedder=embedder,
        store=store,
        index_manager=index_manager,
    )
    
    # Answer generator
    generator = providers.Singleton(
        AnswerGenerator,
        llm=llm,
    )

    rag_pipeline = providers.Singleton(
        RAGPipeline,
        config=config,
        embedder=embedder,
        llm=llm,
        store=store,
        index_manager=index_manager,
        ingester=ingester,
        searcher=searcher,
        generator=generator,
        token_tracker=token_tracker,
        content_safety=content_safety,
    )
