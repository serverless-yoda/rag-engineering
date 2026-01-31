# rag/di/container.py

"""
Dependency injection container using dependency-injector.
Dynamic provider selection for Azure, OpenRouter, Gemini + Supabase.
"""

from dependency_injector import containers, providers

from rag.models.config import RAGConfig
from ..services import (
    AzureOpenAIEmbedder,
    AzureOpenAILLM,
    AzureSearchStore,
    AzureContentSafety,
    OpenRouterLLM,      
    OpenRouterEmbedder, 
    SupabaseVectorStore, 
)

from ..core import (
    IndexManager,
    DocumentIngester,
    SemanticSearcher,
    AnswerGenerator,
    IndexManagerAdapter,
)

from ..utils.chunk import TokenTracker
from ..utils.token import TrackedEmbeddingProvider
from ..pipeline.rag_pipeline import RAGPipeline

class Container(containers.DeclarativeContainer):
    """Main DI container for the RAG pipeline with dynamic provider selection."""
    
    config = providers.Configuration()
    
    # Token tracker (singleton)
    token_tracker = providers.Singleton(TokenTracker)

    # === LLM PROVIDERS ===
    azure_llm = providers.Singleton(
        AzureOpenAILLM,
        endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        deployment_name=config.model_deployment,
        timeout=config.llm_timeout,
        retries=config.llm_retries,
        token_tracker=token_tracker,
    )
    
    openrouter_llm = providers.Singleton(
        OpenRouterLLM,
        api_key=config.openrouter_api_key,
        model=config.openrouter_model,
        token_tracker=token_tracker,
    )
    
    # LLM Selector
    llm = providers.Selector(
        config.chat_provider_type,
        azure=azure_llm,
        openrouter=openrouter_llm,
    )

    # === EMBEDDING PROVIDERS ===
    azure_embedder = providers.Singleton(
        AzureOpenAIEmbedder,
        endpoint=config.azure_openai_endpoint,
        api_key=config.azure_openai_api_key,
        api_version=config.azure_openai_api_version,
        deployment_name=config.embedding_deployment,
        token_tracker=token_tracker,
    )
    
    openrouter_embedder = providers.Singleton(
        OpenRouterEmbedder,
        api_key=config.openrouter_api_key,
        model="text-embedding-ada-002",
        token_tracker=token_tracker
    )
    
    # Embedding Selector
    embedding_provider = providers.Selector(
        config.provider_stack,
        azure=azure_embedder,
        open=openrouter_embedder,
    )
    
    # Tracked Embedding Provider
    embedder = providers.Singleton(
        TrackedEmbeddingProvider,
        embedder=embedding_provider,
        tracker=token_tracker,
    )

    # === VECTOR STORE PROVIDERS ===
    azure_store = providers.Singleton(
        AzureSearchStore,
        endpoint=config.azure_search_endpoint,
        api_key=config.azure_search_api_key,
        index_name=config.index_name,
    )
    
    supabase_store = providers.Singleton(
        SupabaseVectorStore,
        supabase_url=config.supabase_endpoint_url,
        supabase_key=config.supabase_service_role_key,
        table_name=config.supabase_table_name,
        vector_dimension=config.supabase_vector_dimension,
        token_tracker=token_tracker,
    )
    
    # Vector Store Selector
    store = providers.Selector(
        config.vector_backend,
        azure=azure_store,
        supabase=supabase_store,
    )

    # Index manager
    # index_manager = providers.Singleton(
    #     IndexManager,
    #     endpoint=config.azure_search_endpoint,
    #     api_key=config.azure_search_api_key,
    #     index_name=config.index_name,
    #     vector_dimensions=config.vector_dimensions,
    # )
    index_manager = providers.Factory(
        lambda store, config: IndexManagerAdapter(store, config),
        store=store,  # Uses selected store (Supabase or Azure)
        config=config
    )
    
    # Content safety
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

    rag_config = providers.Factory(
        RAGConfig,
        azure_openai_endpoint=config.azure_openai_endpoint,
        azure_openai_api_key=config.azure_openai_api_key,
        azure_openai_api_version=config.azure_openai_api_version,
        embedding_deployment=config.embedding_deployment,
        model_deployment=config.model_deployment,
        azure_search_endpoint=config.azure_search_endpoint,
        azure_search_api_key=config.azure_search_api_key,
        index_name=config.index_name,
        default_namespace=config.default_namespace,
        chunking=config.chunking,
        provider_stack=config.provider_stack,
        chat_provider_type=config.chat_provider_type,
        openrouter_api_key=config.openrouter_api_key,
        openrouter_model=config.openrouter_model,
        gemini_api_key=config.gemini_api_key,
        gemini_model=config.gemini_model,
        vector_backend=config.vector_backend,
        supabase_endpoint_url=config.supabase_endpoint_url,
        supabase_service_role_key=config.supabase_service_role_key,
        supabase_table_name=config.supabase_table_name,
        supabase_vector_dimension=config.supabase_vector_dimension,
        content_safety_endpoint=config.content_safety_endpoint,
        content_safety_api_key=config.content_safety_api_key,
        content_moderation_enabled=config.content_moderation_enabled,
        content_moderation_threshold=config.content_moderation_threshold,
        llm_timeout=config.llm_timeout,
        llm_retries=config.llm_retries,
        batch_size=config.batch_size,
        vector_dimensions=config.vector_dimensions,
    )
    
    # Main RAG pipeline
    rag_pipeline = providers.Singleton(
        RAGPipeline,
        config=rag_config,
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
