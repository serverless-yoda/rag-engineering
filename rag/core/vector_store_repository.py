# core/vector_store_repository.py


"""
Data access layer for vector store operations.
Separates CRUD logic from provider abstraction.
"""

import logging
from typing import List, Dict, Any, Optional
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class VectorStoreRepository:
    """
    Low-level data access for Azure AI Search with circuit breaker.
    """
    
    def __init__(self, client: SearchClient):
        """
        Initialize repository with Azure Search client.
        
        Args:
            client: SearchClient instance
        """
        self.client = client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Upload documents to Azure AI Search.
        Upload documents with retry protection.
        
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
            result = await self.client.upload_documents(documents)
            succeeded = sum(1 for r in result if r.succeeded)
            
            failed = [r for r in result if not r.succeeded]
            if failed:
                logging.warning(f"Failed to upload {len(failed)}/{len(documents)} documents")
                for f in failed[:3]:
                    logging.warning(
                        f"  Document key: {getattr(f, 'key', '?')}, "
                        f"Error: {getattr(f, 'error_message', '?')}"
                    )
            
            logging.info(f"Uploaded {succeeded}/{len(documents)} documents")
            return succeeded
            
        except Exception as e:
            logging.error(f"Document upload failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search with retry protection.
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
            vector_query = VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=top_k,
                fields="chunk_vector",
            )
            
            select = select_fields or [
                "id", "namespace", "source_id", "chunk", "tags",
                "created_at", "source_uri", "metadata_json"
            ]
            
            results_iter = await self.client.search(
                vector_queries=[vector_query],
                select=select,
                filter=filter_expr,
            )
            
            results = [r async for r in results_iter]
            logging.debug(f"Vector search found {len(results)} results")
            return results
            
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((Exception,)),
    )
    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            Total document count, or 0 if the query fails
        """

        try:
            return await self.client.get_document_count()
        except Exception as e:
            logging.warning(f"Failed to get document count: {e}")
            return 0
    
    async def close(self):
        """
        Close the Azure client connection.
        
        Safe to call multiple times.
        """

        try:
            await self.client.close()
        except Exception as e:
            logging.debug(f"Error closing repository: {e}")
