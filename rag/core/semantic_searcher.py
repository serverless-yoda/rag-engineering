# core/semantic_searcher.py

"""
🔍 SEARCH stage: Semantic vector search using embeddings.

This module handles the retrieval of relevant document chunks from the vector store
based on a query. It embeds the query, performs vector similarity search, and
returns normalized results with metadata.
"""

import logging
from typing import List, Optional
from ..abstractions.embedding_provider import EmbeddingProvider
from ..abstractions.vector_store_provider import VectorStoreProvider
from ..models.types import SearchResult
from ..models import SearchError


class SemanticSearcher:
    """
    SEARCH stage: Performs vector-based semantic retrieval.
    
    Responsibilities:
    - Embed the user's query using the embedding provider
    - Build optional filters (e.g., namespace, tags)
    - Perform vector similarity search using the vector store
    - Normalize and return results as SearchResult objects
    
    Dependencies:
    - EmbeddingProvider: For embedding the query
    - VectorStoreProvider: For performing vector search
    - IndexManager: For checking index existence
    
    Example:
        >>> searcher = SemanticSearcher(embedder, store, index_manager)
        >>> results = await searcher.search("What is RAG?", top_k=5)
        >>> for r in results:
        ...     print(r.chunk, r.score)
    """
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStoreProvider,
        index_manager,  # Type hint avoided to prevent circular import
    ):
        """
        Initialize the searcher with required providers.
        
        Args:
            embedder: Embedding provider for query embedding
            store: Vector store provider for search
            index_manager: Index manager for index existence check
        """
        self.embedder = embedder
        self.store = store
        self.index_manager = index_manager

    def _build_filter(
        self,
        namespace: Optional[str],
        extra_filter: Optional[str],
    ) -> Optional[str]:
        """
        Build a combined OData filter expression for Azure AI Search.
        
        This function combines:
        - Namespace filter: "namespace eq 'value'"
        - Additional filter expression: "(custom filter)"
        
        Returns:
            Combined filter string or None if no filters
        
        Used by:
            - search() method before calling vector_store_provider.vector_search()
        """
        parts = []
        if namespace:
            parts.append(f"namespace eq '{namespace}'")
        if extra_filter:
            parts.append(f"({extra_filter})")
        return " and ".join(parts) if parts else None

    async def search(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        filter_expr: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search for the given query.
        
        Steps:
        1. Check if index exists
        2. Embed the query using EmbeddingProvider
        3. Build filter expression
        4. Perform vector search using VectorStoreProvider
        5. Normalize results into SearchResult objects
        
        Args:
            query: User's natural language query
            namespace: Optional namespace filter
            top_k: Number of top results to return
            filter_expr: Optional additional filter expression
        
        Returns:
            List of SearchResult objects sorted by relevance
        
        Example:
            >>> results = await searcher.search("Azure OpenAI", top_k=3)
            >>> for r in results:
            ...     print(r.chunk, r.score)
        """
        # Step 1: Ensure index exists
        if not await self.index_manager.index_exists():
            logging.warning("Search index does not exist. Returning empty results.")
            raise SearchError(f"Search index does not exist. Returning empty results.")
           
        # Step 2: Embed the query
        try:
            query_embeddings = await self.embedder.embed([query])
            if not query_embeddings:
                logging.warning("Query embedding returned no results.")
                return []
            query_vector = query_embeddings[0]
        except Exception as e:
            logging.error(f"Query embedding failed: {e}")
            raise SearchError(f"Query embedding failed: {e}") from e
            

        # Step 3: Build filter expression
        combined_filter = self._build_filter(namespace, filter_expr)

        # Step 4: Perform vector search
        try:
            raw_results = await self.store.vector_search(
                query_vector=query_vector,
                top_k=top_k,
                filter_expr=combined_filter,
            )
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            raise SearchError(f"Vector search failed: {e}") from e
            

        # Step 5: Normalize results
        results = [SearchResult.from_dict(r) for r in raw_results]
        logging.info(f"Semantic search returned {len(results)} results.")
        return results
