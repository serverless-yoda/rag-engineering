# models/env.py

from typing import Annotated

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    app_name: str = "Generative AI Services"

    azure_endpoint_url: Annotated[
        HttpUrl,
        Field(alias="azure_endpoint_url", description="Azure OpenAI endpoint URL"),
    ]

    azure_deployment_name: Annotated[
        str,
        Field(min_length=5, default="gpt-5-nano", description="Chat/completions deployment name"),
    ]

    azure_openai_api_key: Annotated[
        str,
        Field(min_length=5, description="Azure OpenAI API key"),
    ]

    azure_openai_version: Annotated[
        str,
        Field(min_length=5, default="2024-12-01-preview", description="Azure OpenAI API version"),
    ]

    azure_ai_search_url: Annotated[
        HttpUrl,
        Field(alias="azure_ai_search_url", description="Azure AI Search endpoint URL"),
    ]

    # Keep the existing env var name but provide a clearer field name if you like
    azure_ai_search_api_key: Annotated[
        str,
        Field(
            min_length=5,
            alias="azure_ai_searh_api_key",  # keep backward-compatible alias
            description="Azure AI Search admin key",
        ),
    ]

    text_embedding: Annotated[
        str,
        Field(min_length=5, description="Embedding deployment name"),
    ]

    rag_index_name: Annotated[
        str,
        Field(min_length=5, description="Azure Search index name for RAG"),
    ]

    rag_namespace_knowledge_store: Annotated[
        str,
        Field(min_length=5, description="Default namespace for knowledge store"),
    ]

    rag_namespace_blueprint_context: Annotated[
        str,
        Field(min_length=5, description="Namespace for runtime context chunks"),
    ]

    start_with_clean_index: Annotated[bool, Field(default=True, description="Start from empty index")]

    

    # Azure Content Safety
    content_safety_endpoint: Annotated[
        Optional[HttpUrl],
        Field(
            default=None,
            alias="azure_content_safety_endpoint",
            description="Azure Content Safety endpoint"
        ),
    ]

    content_safety_api_key: Annotated[
        Optional[str],
        Field(
            default=None,
            description="Azure Content Safety API key"
        ),
    ]

    content_moderation_enabled: Annotated[
        bool,
        Field(default=True, description="Enable content moderation")
    ]

    content_moderation_threshold: Annotated[
        int,
        Field(default=2, ge=0, le=6, description="Content severity threshold (0-6)")
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


env_settings = Settings()
