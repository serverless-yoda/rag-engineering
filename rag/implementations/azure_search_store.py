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
    
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Upload documents to Azure AI Search.
        
        Uses the upload_documents method which performs upsert operations:
        - If document ID exists, it's updated
        - If document ID is new, it's inserted
        
        Args:
            documents: List of document dictionaries matching the index schema
        
        Returns:
            Number of successfully uploaded documents
        
        Note:
            Failed uploads are logged but don't raise exceptions.
            Check the return value to detect partial failures.
        """
        if not documents:
            return 0
        
        try:
            # Upload documents to Azure AI Search
            # Returns a list of IndexingResult objects
            result = await self.client.upload_documents(documents)
            
            # Count successful uploads
            succeeded = sum(1 for r in result if r.succeeded)
            
            # Log any failures for debugging
            failed = [r for r in result if not r.succeeded]
            if failed:
                logging.warning(f"Failed to upload {len(failed)}/{len(documents)} documents")
                # Log details of first few failures
                for f in failed[:3]:
                    logging.warning(
                        f"  Document key: {getattr(f, 'key', '?')}, "
                        f"Error: {getattr(f, 'error_message', '?')}"
                    )
            
            logging.info(
                f"Azure AI Search upload: {succeeded}/{len(documents)} documents succeeded"
            )
            return succeeded
            
        except Exception as e:
            logging.error(f"Azure AI Search document upload failed: {e}")
            raise
    
    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Uses Azure AI Search's vector search with HNSW indexing for fast
        approximate nearest neighbor search. Results are ranked by cosine similarity.
        
        Args:
            query_vector: The embedding vector to search for
            top_k: Maximum number of results to return
            filter_expr: Optional OData filter (e.g., "namespace eq 'KnowledgeBase'")
            select_fields: Optional list of fields to return (default: all standard fields)
        
        Returns:
            List of matching documents with @search.score field indicating relevance
        
        Note:
            The @search.score field contains the similarity score (higher = more relevant).
            Scores depend on the vector search algorithm and normalization settings.
        """
        try:
            # Create vectorized query for Azure AI Search
            # This tells the search engine to use the chunk_vector field for similarity
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,  # How many nearest neighbors to find
                fields="chunk_vector",  # Field containing embedding vectors
            )
            
            # Default fields to return if not specified
            # These match the schema defined in IndexManager
            select = select_fields or [
                "id", "namespace", "source_id", "chunk", "tags",
                "created_at", "source_uri", "metadata_json"
            ]
            
            # Execute the search
            # vector_queries parameter enables vector-based search
            # filter parameter applies OData filtering
            # select parameter limits returned fields
            results_iter = await self.client.search(
                vector_queries=[vector_query],
                select=select,
                filter=filter_expr,
            )
            
            # Collect all results into a list
            # Azure Search returns an async iterator
            results = [r async for r in results_iter]
            
            logging.debug(f"Azure AI Search found {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"Azure AI Search vector search failed: {e}")
            raise
    
    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            Total document count, or 0 if the query fails
        """
        try:
            count = await self.client.get_document_count()
            return count
        except Exception as e:
            logging.warning(f"Failed to get document count from Azure AI Search: {e}")
            return 0
    
    async def close(self) -> None:
        """
        Close the Azure AI Search client connection.
        
        Safe to call multiple times.
        """
        try:
            await self.client.close()
        except Exception as e:
            logging.debug(f"Error closing Azure Search store: {e}")
