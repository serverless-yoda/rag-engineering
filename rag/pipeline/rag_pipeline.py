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
from ..models import RAGConfig, IngestionResult, SearchResult
from ..utils import TokenTracker
from ..engine.context_engine import ContextEngine

class RAGPipeline:
    """
    Main orchestrator for RAG system with DI.
    """
    
    def __init__(
        self,
        config: RAGConfig,
        embedder,
        llm,
        store,
        index_manager,
        ingester,
        searcher,
        generator,
        token_tracker: TokenTracker,
        content_safety=None,
    ):
        """
        Initialize pipeline with injected dependencies.
        
        Args:
            config: RAG configuration
            embedder: Embedding provider
            llm: LLM provider
            store: Vector store provider
            index_manager: Index manager
            ingester: Document ingester
            searcher: Semantic searcher
            generator: Answer generator
            token_tracker: Token usage tracker
            content_safety: Optional content safety
        """
        self.config = config
        self.embedder = embedder
        self.llm = llm
        self.store = store
        self.index_manager = index_manager
        self.ingester = ingester
        self.searcher = searcher
        self.generator = generator
        self.token_tracker = token_tracker
        self.content_safety = content_safety
        
        self.context_engine = ContextEngine(searcher=self.searcher
                                            ,generator=self.generator
                                            ,content_safety=content_safety)
    
    async def __aenter__(self) -> "RAGPipeline":
        return self
    
    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
    
    async def close(self) -> None:
        """Clean up resources."""
        for client in [self.embedder, self.store, self.llm, self.index_manager]:
            try:
                logging.info(f"Closing {self.__class__.__name__} at {hex(id(self))}")
                await client.close()
                logging.info(f"Done closing {client.__class__.__name__}")
                
            except Exception:
                logging.error(f"Error closing {client.__class__.__name__}: {e}")
        
        if self.content_safety:
            try:
                await self.content_safety.close()
            except Exception as e:
                logging.error(f"Error closing content safety: {e}")

    
    # === High-Level Workflows ===
    
    async def setup(
        self,
        documents: List[Union[str, Dict[str, Any]]],
        namespace: Optional[str] = None,
    ) -> IngestionResult:
        """BUILD + INGEST workflow."""
        await self.index_manager.create_index()
        return await self.ingester.ingest_documents(
            items=documents,
            namespace=namespace or self.config.default_namespace,
            chunking_config=self.config.chunking,
        )
    
    async def answer_question(
        self,
        question: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        system_prompt: Optional[str] = None,
    ) -> str:
        """SEARCH + ANSWER workflow."""
        results = await self.searcher.search(
            query=question,
            namespace=namespace,
            top_k=top_k,
        )
        
        if not results:
            return "I couldn't find relevant information."
        
        context = "\n\n".join(
            f"[Source: {r.source_id}]\n{r.chunk}" for r in results
        )
        
        answer = await self.generator.generate(
            question=question,
            context=context,
            system_prompt=system_prompt,
        )
        
        # Log token usage
        #logging.info(self.token_tracker.report())
        
        return answer
    
    async def generate_with_context(self, goal: str) -> str:
        """Multi-agent workflow."""
        result = await self.context_engine.execute(goal)
        
        # Log token usage
        #logging.info(self.token_tracker.report())
        
        return result
    
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
