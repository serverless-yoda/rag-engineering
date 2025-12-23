# models/types.py

"""
Type definitions and result models for the RAG system.

This module contains data classes representing:
- Configuration objects (ChunkingConfig)
- Operation results (IngestionResult, SearchResult)
- Type aliases (JsonDict)
"""

from dataclasses import dataclass, field
from typing import List,Dict,Any,Optional
from datetime import datetime
import json

# Type alias for JSON-compatible dictionaries
JsonDict =Dict[str,Any]

@dataclass
class ChunkingConfig:
    """
    Configuration for text chunking strategy.
    
    Controls how long documents are split into smaller chunks for embedding.
    Two modes are supported:
    - Character-based: Splits by character count (faster, simpler)
    - Token-based: Splits by token count using tiktoken (more accurate for LLMs)
    
    Attributes:
        use_token_chunking: If True, use tiktoken-based chunking; otherwise use character-based
        chunk_size: Maximum size per chunk (characters or tokens depending on mode)
        overlap: Number of characters/tokens to overlap between consecutive chunks
                 (helps maintain context across chunk boundaries)
    
    Example:
        >>> # Token-based chunking (recommended for LLMs)
        >>> config = ChunkingConfig(use_token_chunking=True, chunk_size=400, overlap=50)
        >>> 
        >>> # Character-based chunking (faster for large documents)
        >>> config = ChunkingConfig(use_token_chunking=False, chunk_size=4000, overlap=200)
    """
    use_token_chunking: bool = False
    chunk_size: int = 4000 # chars if False, tokens if True
    overlap: int = 200     


@dataclass
class IngestionResult:
    """
    Result object returned by document ingestion operations.
    
    Provides detailed metrics about the ingestion process, including success status,
    counts of processed items, and any errors encountered.
    
    Attributes:
        success: True if ingestion completed without critical errors
        documents_processed: Number of input documents normalized
        chunks_created: Total number of chunks generated from documents
        documents_uploaded: Number of chunks successfully uploaded to vector store
        errors: List of error messages (empty if success=True)
        duration_seconds: Total time taken for the ingestion operation
    
    Example:
        >>> result = await ingester.ingest_documents(docs)
        >>> if result.success:
        ...     print(f"Uploaded {result.documents_uploaded} chunks in {result.duration_seconds:.2f}s")
        ... else:
        ...     print(f"Errors: {result.errors}")
    """
    success: bool
    documents_processed: int
    chunks_created: int
    documents_uploaded: int
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        """Human-readable summary of ingestion result."""
        if self.success:
            return (f"Ingestion succeeded: {self.documents_processed} documents processed, "
                    f"{self.chunks_created} chunks created, "
                    f"{self.documents_uploaded} chunks uploaded in {self.duration_seconds:.2f}s.")
        else:
            return f"❌ Ingestion failed: {', '.join(self.errors)}"


@dataclass
class SearchResult:
    """
    Normalized result from a vector search operation.
    
    Represents a single matching chunk from the vector store, including its content,
    metadata, and similarity score. The metadata_json field from Azure AI Search
    is parsed into a structured dictionary.
    
    Attributes:
        id: Unique identifier for this chunk (format: "{source_id}-{chunk_id}")
        namespace: Logical grouping for the document (e.g., "KnowledgeBase", "Blueprint")
        source_id: Identifier for the source document
        chunk: The actual text content of this chunk
        score: Similarity score from vector search (higher = more relevant)
        metadata: Parsed metadata dictionary (from metadata_json field)
        tags: Optional tags for filtering (e.g., "pdf", "blueprint")
        created_at: Timestamp when this chunk was indexed
        source_uri: Original source location (e.g., file path, URL)
    
    The from_dict() class method handles parsing raw Azure AI Search results,
    including safe JSON deserialization of the metadata_json field.
    
    Example:
        >>> results = await searcher.search("Azure OpenAI", top_k=3)
        >>> for result in results:
        ...     print(f"Score: {result.score:.4f}")
        ...     print(f"Content: {result.chunk[:100]}...")
        ...     print(f"Metadata: {result.metadata}")
    """
    id: str
    namespace: str
    source_id: str
    chunk: str
    score: float
    metadata: JsonDict = field(default_factory=dict)
    tags: Optional[List[str]] = None
    created_at: Optional[datetime] = None
    source_uri: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """
        Create a SearchResult from a raw Azure AI Search result dictionary.
        
        This method handles:
        1. Extracting all standard fields from the search result
        2. Parsing the metadata_json string into a Python dict
        3. Handling parse errors gracefully (stores raw string with error flag)
        4. Extracting the search score from the @search.score field
        
        Args:
            data: Raw result dictionary from Azure AI Search
        
        Returns:
            SearchResult instance with parsed and normalized fields
        
        Note:
            The @search.score field is a special Azure Search field indicating
            the relevance score (higher = better match).
        """
        # Extract and parse metadata_json field
        # This field contains structured metadata as a JSON string
        metadata_json = data.get("metadata_json", "{}")
        if isinstance(metadata_json, str) and metadata_json:
            try:
                # Attempt to parse JSON string into Python dict
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                # If parsing fails, preserve the raw string and flag the error
                # This allows debugging while preventing data loss
                metadata = {"raw": metadata_json, "parse_error": True}
        else:
            # Handle case where metadata_json is already a dict or is empty
            metadata = metadata_json or {}
        
        return cls(
            id=data.get("id", ""),
            namespace=data.get("namespace", ""),
            source_id=data.get("source_id", ""),
            chunk=data.get("chunk", ""),
            score=data.get("@search.score", 0.0),  # Azure Search's score field
            metadata=metadata,
            tags=data.get("tags"),
            created_at=data.get("created_at"),
            source_uri=data.get("source_uri"),
        )
