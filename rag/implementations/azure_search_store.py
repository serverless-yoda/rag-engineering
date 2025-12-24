# implementation/azure_search_store.py

"""
Azure AI Search vector store provider implementation.

This module implements the VectorStoreProvider interface using Azure AI Search
for vector storage and similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from ..abstractions.vector_store_provider import VectorStoreProvider
from ..core.vector_store_repository import VectorStoreRepository

class AzureSearchStore(VectorStoreProvider):
    """
    Azure AI Search implementation of the VectorStoreProvider interface.
    
    Provides vector storage and similarity search using Azure AI Search's
    vector search capabilities with HNSW indexing.
    
    The client supports:
    - Document upsert (insert or update)
    - Vector similarity search with filters
    - Document count queries
    
    Example:
        >>> store = AzureSearchStore(
        ...     endpoint="https://my-search.search.windows.net/",
        ...     api_key="key456",
        ...     index_name="rag-index",
        ... )
        >>> await store.upsert_documents([{...}])
        >>> results = await store.vector_search(query_vec, top_k=5)
        >>> await store.close()
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
    ):
        """
        Initialize the Azure AI Search client.
        
        Args:
            endpoint: Azure AI Search endpoint URL
            api_key: API key for authentication
            index_name: Name of the search index to use
        """
        self.index_name = index_name
        
        # Create async search client for document operations
        # This client maintains a connection pool for efficient requests
        self.client = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )
    
        
        # Initialize repository for data access
        self.repository = VectorStoreRepository(self.client)

    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Upload documents via repository."""
        return await self.repository.upsert_documents(documents)

    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector search via repository."""
        return await self.repository.vector_search(
            query_vector=query_vector,
            top_k=top_k,
            filter_expr=filter_expr,
            select_fields=select_fields,
        )

    async def get_document_count(self) -> int:
        """Get document count via repository."""
        return await self.repository.get_document_count()
    
    async def close(self) -> None:
        """Close repository and client."""
        await self.repository.close()
