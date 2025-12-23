# abstractions/vector_store_provider.py

"""
Abstract interface for vector storage services.

This module defines the contract that all vector stores must implement,
allowing the system to work with any vector database (Azure AI Search, Pinecone, Weaviate, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStoreProvider(ABC):
    """
    Abstract base class for vector storage and retrieval services.
    
    Implementations must provide:
    1. upsert_documents() to store document chunks with embeddings
    2. vector_search() to find similar documents
    3. get_document_count() to get total indexed documents
    4. close() to cleanup resources
    
    Example implementations:
    - AzureSearchStore (Azure AI Search)
    - PineconeStore (Pinecone vector database)
    - WeaviateStore (Weaviate)
    """

    @abstractmethod
    async def upsert_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Upload or update documents in the vector store.
        
        Documents should follow a consistent schema with fields:
        - id: unique identifier
        - chunk: text content
        - chunk_vector: embedding vector
        - namespace, source_id, tags, metadata_json, etc.
        
        Args:
            documents: List of document dictionaries to upsert
        
        Returns:
            Number of documents successfully uploaded
        
        Raises:
            Exception: If upload fails (connection error, schema mismatch, etc.)
        
        Note:
            "Upsert" = update if exists, insert if new
        """
        pass



    @abstractmethod
    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_expr: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.
        
        Finds the top-K most similar documents to the query vector using
        cosine similarity (or other distance metrics depending on implementation).
        
        Args:
            query_vector: The embedding vector to search for
            top_k: Maximum number of results to return
            filter_expr: Optional filter expression (e.g., "namespace eq 'KnowledgeBase'")
            select_fields: Optional list of fields to return (default: all)
        
        Returns:
            List of matching documents, sorted by relevance (highest score first)
            Each document is a dictionary with all requested fields plus a score
        
        Example:
            >>> results = await store.vector_search(query_vec, top_k=3)
            >>> for result in results:
            ...     print(f"Score: {result['@search.score']}, Chunk: {result['chunk']}")
        """
        pass
    
    @abstractmethod
    async def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        
        Returns:
            Total document count (may return 0 if query fails)
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """
        Cleanup resources (close connections, etc.).
        
        Should handle errors gracefully and not raise exceptions.
        """
        pass
