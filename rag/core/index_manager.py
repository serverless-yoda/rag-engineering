# core/index_manager.py

"""
BUILD stage: Azure AI Search index lifecycle management.

This module handles creation, deletion, and existence checking of Azure AI Search indexes
with vector search capabilities (HNSW algorithm).
"""

import logging
from typing import Optional
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceNotFoundError
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)


class IndexManager:
    """
    BUILD stage: Manages Azure AI Search index lifecycle.
    
    Responsibilities:
    - Check if an index exists
    - Create an index with vector search schema
    - Delete an index
    
    The index schema includes:
    - Metadata fields: id, namespace, source_id, chunk_id, tags, created_at, source_uri
    - Content fields: chunk (searchable text), metadata_json
    - Vector field: chunk_vector (for similarity search)
    - HNSW vector search configuration for fast approximate nearest neighbor search
    
    Example:
        >>> manager = IndexManager(
        ...     endpoint="https://my-search.search.windows.net/",
        ...     api_key="key456",
        ...     index_name="rag-index",
        ... )
        >>> if not await manager.index_exists():
        ...     await manager.create_index()
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        vector_dimensions: int = 1536,
    ):
        """
        Initialize the index manager.
        
        Args:
            endpoint: Azure AI Search endpoint URL
            api_key: API key for authentication
            index_name: Name of the index to manage
            vector_dimensions: Dimensionality of embedding vectors (default: 1536 for text-embedding-ada-002)
        """
        self.index_name = index_name
        self.vector_dimensions = vector_dimensions
        
        # Create async client for index management operations
        self.client = SearchIndexClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key),
        )
    
    async def index_exists(self) -> bool:
        """
        Check if the index exists in Azure AI Search.
        
        Returns:
            True if the index exists, False otherwise
        
        Note:
            Uses get_index() which raises ResourceNotFoundError if index doesn't exist.
            This is the recommended way to check existence (no dedicated exists() method).
        """
        try:
            await self.client.get_index(self.index_name)
            return True
        except ResourceNotFoundError:
            logging.debug(f"Index '{self.index_name}' not found")
            return False
    
    async def create_index(self) -> None:
        """
        Create an Azure AI Search index with vector search capabilities.
        
        The index schema matches the document format produced by make_search_documents():
        - id: Unique key (String, filterable, sortable)
        - namespace: Logical grouping (String, filterable, facetable, sortable)
        - source_id: Source document ID (String, filterable, sortable)
        - chunk_id: Chunk number (Int32, filterable, sortable)
        - chunk: Text content (String, searchable with Lucene analyzer)
        - chunk_vector: Embedding vector (Collection(Single), vector search enabled)
        - tags: Tags for filtering (String, filterable, facetable)
        - created_at: Timestamp (DateTimeOffset, filterable, sortable)
        - source_uri: Source location (String, filterable, sortable)
        - metadata_json: Structured metadata (String, searchable)
        
        Vector Search Configuration:
        - Algorithm: HNSW (Hierarchical Navigable Small World)
        - Profile: myHnswProfile (connects the chunk_vector field to the HNSW algorithm)
        
        If the index already exists, this is a no-op (idempotent).
        
        Raises:
            Exception: If index creation fails
        """
        # Check if index already exists (idempotent behavior)
        if await self.index_exists():
            logging.info(f"Index '{self.index_name}' already exists, skipping creation")
            return
        
        # Define the index schema
        # Each field specifies its type, searchability, and filtering/sorting capabilities
        fields = [
            # Primary key field (must be unique, used as document identifier)
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,  # Marks this as the primary key
                filterable=True,  # Can be used in $filter expressions
                sortable=True,  # Can be used in $orderby
            ),
            
            # Namespace for logical document grouping (e.g., "KnowledgeBase", "Policies")
            SimpleField(
                name="namespace",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,  # Can be used for faceted navigation
                sortable=True,
            ),
            
            # Source document identifier
            SimpleField(
                name="source_id",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
            ),
            
            # Chunk number within source document
            SimpleField(
                name="chunk_id",
                type=SearchFieldDataType.Int32,
                filterable=True,
                sortable=True,
            ),
            
            # Text content (searchable with full-text search)
            # Uses Lucene analyzer for English text
            SearchableField(
                name="chunk",
                type=SearchFieldDataType.String,
                analyzer_name="en.lucene",  # English language analyzer
            ),
            
            # Embedding vector for semantic similarity search
            # This is a Collection (array) of Single (float32) values
            SearchField(
                name="chunk_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,  # Enable vector search on this field
                vector_search_dimensions=self.vector_dimensions,  # Vector size (e.g., 1536)
                vector_search_profile_name="myHnswProfile",  # Links to vector search profile
            ),
            
            # Tags for filtering (e.g., "pdf", "blueprint", "policy")
            SearchableField(
                name="tags",
                type=SearchFieldDataType.String,
                filterable=True,
                facetable=True,
            ),
            
            # Timestamp when document was indexed
            SimpleField(
                name="created_at",
                type=SearchFieldDataType.DateTimeOffset,
                filterable=True,
                sortable=True,
            ),
            
            # Original source location (e.g., file path, URL)
            SimpleField(
                name="source_uri",
                type=SearchFieldDataType.String,
                filterable=True,
                sortable=True,
            ),
            
            # Structured metadata as JSON string
            SearchableField(
                name="metadata_json",
                type=SearchFieldDataType.String,
            ),
        ]
        
        # Configure vector search with HNSW algorithm
        # HNSW (Hierarchical Navigable Small World) is an efficient algorithm
        # for approximate nearest neighbor search in high-dimensional spaces
        vector_search = VectorSearch(
            # Define the algorithm configuration
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw")
            ],
            # Define the profile that links fields to algorithms
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                )
            ],
        )
        
        # Create the index definition
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )
        
        # Create the index in Azure AI Search
        await self.client.create_index(index)
        logging.info(f"Index '{self.index_name}' created successfully")
    
    async def delete_index(self) -> None:
        """
        Delete the Azure AI Search index.
        
        If the index doesn't exist, this is a no-op (idempotent).
        
        Warning:
            This permanently deletes all indexed documents. Use with caution.
        """
        if not await self.index_exists():
            logging.info(f"Index '{self.index_name}' does not exist, skipping deletion")
            return
        
        await self.client.delete_index(self.index_name)
        logging.info(f"Index '{self.index_name}' deleted successfully")
    
    async def close(self) -> None:
        """
        Close the Azure AI Search index client.
        
        Safe to call multiple times.
        """
        try:
            await self.client.close()
        except Exception as e:
            logging.debug(f"Error closing index manager: {e}")
