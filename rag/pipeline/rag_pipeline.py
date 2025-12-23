# pipeline/rag_pipeline.py

"""
🎯 RAGPipeline: Orchestrator for the B.I.S.A. framework.

This module coordinates the four core stages of the RAG system:
- Build: Ensures the vector index exists
- Ingest: Processes and stores documents
- Search: Retrieves relevant chunks
- Answer: Generates responses using LLM

It provides high-level workflows and direct access to each stage.
"""

import logging
from typing import List, Union, Dict, Any, Optional
from ..models import RAGConfig, IngestionResult, SearchResult, ChunkingConfig
from ..implementations import AzureOpenAIEmbedder, AzureSearchStore, AzureOpenAILLM
from ..core import IndexManager, DocumentIngester, SemanticSearcher, AnswerGenerator
from ..engine.context_engine import ContextEngine


class RAGPipeline:
    """
    Main orchestrator for the RAG system.

    Responsibilities:
    - Initialize all providers and modules
    - Provide high-level workflows (setup, answer_question, generate_with_context)
    - Expose direct access to each stage for advanced use
    """

    def __init__(self, config: RAGConfig):
        self.config = config

        # Providers
        self.embedder = AzureOpenAIEmbedder(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            deployment_name=config.embedding_deployment,
        )
        self.store = AzureSearchStore(
            endpoint=config.azure_search_endpoint,
            api_key=config.azure_search_api_key,
            index_name=config.index_name,
        )
        self.llm = AzureOpenAILLM(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            deployment_name=config.model_deployment,
            timeout=config.llm_timeout,
            retries=config.llm_retries,
        )

        # Core B.I.S.A. modules
        self.index_manager = IndexManager(
            endpoint=config.azure_search_endpoint,
            api_key=config.azure_search_api_key,
            index_name=config.index_name,
            vector_dimensions=config.vector_dimensions,
        )
        self.ingester = DocumentIngester(
            embedder=self.embedder,
            store=self.store,
            index_manager=self.index_manager,
            batch_size=config.batch_size,
        )
        self.searcher = SemanticSearcher(
            embedder=self.embedder,
            store=self.store,
            index_manager=self.index_manager,
        )
        self.generator = AnswerGenerator(llm=self.llm)

        # Multi-agent context engine
        self.context_engine = ContextEngine(self)

    async def __aenter__(self) -> "RAGPipeline":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Clean up all resources."""
        for client in [self.embedder, self.store, self.llm, self.index_manager]:
            try:
                await client.close()
            except Exception:
                pass

    # === High-Level Workflows ===

    async def setup(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        namespace: Optional[str] = None,
    ) -> IngestionResult:
        """BUILD + INGEST: Setup index and ingest documents."""
        await self.index_manager.create_index()
        return await self.ingester.ingest_documents(
            items=documents,
            namespace=namespace or self.config.default_namespace,
            chunking_config=self.config.chunking,
        )

    async def ingest_blueprints(
        self,
        blueprints: List[Dict[str, Any]],
        *,
        namespace: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> IngestionResult:
        """INGEST: Upload blueprint descriptions (no chunking)."""
        await self.index_manager.create_index()
        return await self.ingester.ingest_blueprints(
            blueprints=blueprints,
            namespace=namespace or self.config.default_namespace,
            extra_meta=extra_meta,
        )

    async def answer_question(
        self,
        question: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> str:
        """SEARCH + ANSWER: End-to-end Q&A workflow."""
        results = await self.searcher.search(
            query=question,
            namespace=namespace,
            top_k=top_k,
        )
        if not results:
            return "I couldn't find any relevant information to answer your question."

        context = "\n\n".join(
            f"[Source: {r.source_id}]\n{r.chunk}" for r in results
        )
        return await self.generator.generate(
            question=question,
            context=context,
            system_prompt=system_prompt,
        )

    async def generate_with_context(self, goal: str) -> str:
        """
        MULTI-AGENT: Execute a contextual generation workflow.

        Args:
            goal: High-level user goal (e.g., "Write a suspenseful story")

        Returns:
            Final generated content
        """
        return await self.context_engine.execute(goal)

    # === Direct Access ===

    async def ingest(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        **kwargs,
    ) -> IngestionResult:
        return await self.ingester.ingest_documents(documents, **kwargs)

    async def search(
        self,
        query: str,
        **kwargs,
    ) -> List[SearchResult]:
        return await self.searcher.search(query, **kwargs)

    async def generate(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> str:
        return await self.generator.generate(question, context, **kwargs)
