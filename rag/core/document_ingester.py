# core/document_ingester.py

"""
INGEST stage: Document processing and storage pipeline.

This module handles the complete ingestion workflow:
1. Normalize input documents to clean text
2. Split text into chunks (character-based or token-based)
3. Generate embeddings for each chunk
4. Upload chunks with embeddings to vector store
"""

import logging
import time
import json
from typing import List, Union, Dict, Any, Optional
from ..abstractions.embedding_provider import EmbeddingProvider
from ..abstractions.vector_store_provider import VectorStoreProvider
from ..models.types import IngestionResult, ChunkingConfig, JsonDict
from ..utils import (
    to_text_content,
    chunk_text,
    chunk_text_tiktoken,
    batched,
    make_search_documents,
    now_iso,
    ensure_namespace,
    normalize_items,
    normalize_file_items,
    file_to_text_content,
    make_item_source_id,
    sanitize_input
)


class DocumentIngester:
    """
    INGEST stage: Process and store documents in the vector store.
    
    The ingestion pipeline follows these steps:
    1. NORMALIZE: Convert input items to clean text (handles str, dict, bytes, etc.)
    2. CHUNK: Split long text into smaller pieces with overlap
    3. EMBED: Generate vector embeddings for each chunk (batched for efficiency)
    4. SHAPE: Create search documents with metadata
    5. STORE: Upload to vector store (Azure AI Search)
    
    Dependencies:
    - EmbeddingProvider: For generating embeddings (see AzureOpenAIEmbedder)
    - VectorStoreProvider: For storing documents (see AzureSearchStore)
    - IndexManager: For ensuring the index exists
    
    Example:
        >>> ingester = DocumentIngester(embedder, store, index_manager)
        >>> result = await ingester.ingest_documents(
        ...     items=["Document 1", "Document 2"],
        ...     namespace="MyDocs",
        ...     chunking_config=ChunkingConfig(use_token_chunking=True, chunk_size=400)
        ... )
        >>> print(result)
        ✅ Ingestion successful: 2 docs → 10 chunks → 10 uploaded (2.5s)
    """
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStoreProvider,
        index_manager,  # Type hint would be circular, keep as Any
        batch_size: int = 16,
    ):
        """
        Initialize the document ingester.
        
        Args:
            embedder: Embedding provider for generating vectors
            store: Vector store provider for document storage
            index_manager: Index manager for ensuring index exists
            batch_size: Number of chunks to embed in a single API call
        """
        self.embedder = embedder
        self.store = store
        self.index_manager = index_manager
        self.batch_size = batch_size
    
    async def ingest_documents(
        self,
        items: List[Union[str, Dict[str, Any]]],
        *,
        namespace: str = "KnowledgeStore",
        source_id: str = "misc",
        chunking_config: Optional[ChunkingConfig] = None,
        extra_meta: Optional[JsonDict] = None,
    ) -> IngestionResult:
        """
        Main ingestion pipeline: normalize → chunk → embed → store.
        
        This method orchestrates the complete ingestion workflow:
        
        1. INDEX CHECK: Ensures the vector store index exists
        2. NORMALIZE: Converts each item to clean text using to_text_content()
        3. CHUNK: Splits text into chunks based on chunking_config
        4. EMBED: Generates embeddings in batches for efficiency
        5. SHAPE: Creates search-ready documents with make_search_documents()
        6. STORE: Uploads documents to vector store
        
        Args:
            items: List of documents to ingest (str, dict, bytes, etc.)
            namespace: Logical namespace for organizing documents
            source_id: Base identifier for source documents
            chunking_config: Chunking strategy configuration (default: ChunkingConfig())
            extra_meta: Additional metadata to attach to all chunks
        
        Returns:
            IngestionResult with success status, counts, errors, and duration
        
        Example:
            >>> # Ingest PDF documents with token-based chunking
            >>> result = await ingester.ingest_documents(
            ...     items=[pdf_text_1, pdf_text_2],
            ...     namespace="PDFs",
            ...     source_id="user_uploads",
            ...     chunking_config=ChunkingConfig(
            ...         use_token_chunking=True,
            ...         chunk_size=400,
            ...         overlap=50,
            ...     ),
            ...     extra_meta={"tags": "pdf", "source_uri": "/uploads/docs/"},
            ... )
        """
        start_time = time.time()
        config = chunking_config or ChunkingConfig()
        
        # === STEP 1: Ensure index exists ===
        # This is idempotent - will skip if index already exists
        if not await self.index_manager.index_exists():
            logging.info("Index does not exist, creating...")
            await self.index_manager.create_index()
        
        # === STEP 2: Normalize and chunk ===
        all_chunks: List[str] = []  # All chunks across all documents
        chunk_map: List[int] = []  # Number of chunks per document (for later document reconstruction)
        normalized: List[str] = []  # Normalized text for each document
        
        # items = normalize_items(items)
        normalized_items = normalize_file_items(items)
        
        for item in normalized_items:
            # print(f"Processing item: {item}")

            # Convert arbitrary input (str, dict, bytes, etc.) to clean text
            # Uses to_text_content() which handles HTML stripping, JSON encoding, etc.
            
            #text = to_text_content(item)
            text = file_to_text_content(item)
            #print(f"Extracted text: {text[:100]}...")
            normalized.append(text)
            
            # Split text into chunks based on configuration
            if config.use_token_chunking:
                # Token-based chunking (better for LLMs, requires tiktoken)
                chunks = chunk_text_tiktoken(
                    text,
                    chunk_size=config.chunk_size,
                    overlap=config.overlap,
                )
            else:
                # Character-based chunking (faster, simpler)
                chunks = chunk_text(
                    text,
                    max_chars=config.chunk_size,
                    overlap=config.overlap,
                )
            
            # Track how many chunks came from this document
            # This lets us later group chunks back to their source documents
            chunk_map.append(len(chunks))
            all_chunks.extend(chunks)
        
        logging.info(
            f"Normalized {len(items)} documents into {len(all_chunks)} chunks"
        )
        
        # Check if we got any chunks
        if not all_chunks:
            return IngestionResult(
                success=False,
                documents_processed=len(items),
                errors=["No chunks generated from input items. Check text extraction."],
            )
        
        # === STEP 3: Embed in batches ===
        # Embedding APIs have batch size limits, so we process in smaller batches
        embeddings: List[List[float]] = []
        try:
            for batch in batched(all_chunks, self.batch_size):
                # Generate embeddings for this batch
                # Uses the EmbeddingProvider interface (e.g., AzureOpenAIEmbedder)
                batch_embeddings = await self.embedder.embed(batch)
                embeddings.extend(batch_embeddings)
            
            logging.info(f"Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            # Embedding generation failed - return detailed error
            return IngestionResult(
                success=False,
                documents_processed=len(items),
                chunks_created=len(all_chunks),
                errors=[f"Embedding generation failed: {str(e)}"],
                duration_seconds=time.time() - start_time,
                documents_uploaded=0,
            )
        
        # Validate embedding count matches chunk count
        # This is a critical invariant - if they don't match, something went wrong
        if len(embeddings) != len(all_chunks):
            return IngestionResult(
                success=False,
                documents_processed=len(items),
                chunks_created=len(all_chunks),
                errors=[
                    f"Embedding count mismatch: expected {len(all_chunks)}, got {len(embeddings)}"
                ],
                duration_seconds=time.time() - start_time,
                documents_uploaded=0,
            )
        
        # === STEP 4: Build search documents ===
        # Group chunks back to their source documents and create search-ready documents
        docs: List[Dict[str, Any]] = []
        cursor = 0  # Current position in all_chunks and embeddings lists
        
        for idx, _ in enumerate(normalized):
            # Get the number of chunks for this document
            cnt = chunk_map[idx]
            
            # Extract this document's chunks and embeddings
            these_chunks = all_chunks[cursor : cursor + cnt]
            these_vecs = embeddings[cursor : cursor + cnt]
            cursor += cnt
                        

            # Build a valid, unique source id based on the corresponding normalized item
            item = normalized_items[idx]
            item_source_id = make_item_source_id(item, idx, base_source_id=source_id)
            print(f"Document source_id: {item_source_id}")

            # Optional: enrich metadata per document with filename/path/mime
            per_doc_meta = dict(extra_meta or {})
            try:
                src = item.get("source", {})
                if src.get("type") == "path":
                    path_str = src.get("value")
                    p = Path(path_str)
                    per_doc_meta.update({
                        "filename": p.name,
                        "filepath": str(p),
                        "fileext": p.suffix.lower(),
                    })
                # Always include mime_type and original item name if present
                if item.get("mime_type"):
                    per_doc_meta["mime_type"] = item["mime_type"]
                if item.get("name"):
                    per_doc_meta["item_name"] = item["name"]
            except Exception:
                # Metadata enrichment is optional; ignore errors
                pass

            
            # Create search documents using the utility function
            # This handles formatting, metadata, timestamps, etc.
            docs.extend(
                make_search_documents(
                    namespace=namespace,
                    source_id=item_source_id,
                    content_chunks=these_chunks,
                    embeddings=these_vecs,
                    extra_meta=per_doc_meta,
                )
            )
        
        logging.info(f"Created {len(docs)} search documents")
        
        # === STEP 5: Upload to vector store ===
        uploaded_count = 0

        try:
            if docs:
                logging.debug(
                    "First doc sample: %s",
                    json.dumps(docs[0], ensure_ascii=False)[:600],
                )
                print(json.dumps(docs[0], ensure_ascii=False)[:600])
                
            # Upload all documents to the vector store
            # Uses the VectorStoreProvider interface (e.g., AzureSearchStore)
            uploaded_count = await self.store.upsert_documents(docs)
            
        except Exception as e:
            # Upload failed - return error with partial progress
            return IngestionResult(
                success=False,
                documents_processed=len(items),
                chunks_created=len(all_chunks),
                errors=[f"Vector store upload failed: {str(e)}"],
                duration_seconds=time.time() - start_time,
                documents_uploaded=0,
            )
        
        # Calculate total duration
        duration = time.time() - start_time
        
        # Return success result with full metrics
        return IngestionResult(
            success=True,
            documents_processed=len(items),
            chunks_created=len(all_chunks),
            documents_uploaded=len(docs),
            duration_seconds=duration
        )

    async def ingest_blueprints(
        self,
        blueprints: List[Dict[str, Any]],
        *,
        namespace: str,
        extra_meta: Optional[JsonDict] = None,
    ) -> IngestionResult:
        """
        Ingest blueprint descriptions only (no chunking).
        Mirrors original upsert_blueprint_context_async.
        """
        start_time = time.time()

        descriptions: List[str] = []
        ids: List[str] = []
        meta_list: List[dict] = []

        uploaded = 0

        for item in blueprints:
            bp_id = str(item["id"])
            desc = str(item["description"])
            blueprint = item.get("blueprint")
            blueprint_json = blueprint if isinstance(blueprint, str) else json.dumps(blueprint, ensure_ascii=False)

            descriptions.append(desc)
            ids.append(bp_id)
            meta_list.append({
                "description": desc,
                "blueprint_json": blueprint_json,
                **(extra_meta or {})
            })

        embeddings: List[List[float]] = []
        for batch in batched(descriptions, self.batch_size):
            embeddings.extend(await self.embedder.embed(batch))

        docs: List[Dict[str, Any]] = []
        timestamp = now_iso()
        safe_namespace = ensure_namespace(namespace)

        for bp_id, desc, vec, meta in zip(ids, descriptions, embeddings, meta_list):
            docs.append({
                "id": bp_id,
                "namespace": safe_namespace,
                "source_id": "blueprint_context",
                "chunk_id": 0,
                "chunk": desc,
                "chunk_vector": vec,
                "tags": "blueprint",
                "created_at": timestamp,
                "source_uri": None,
                "metadata_json": json.dumps(meta, ensure_ascii=False),
            })

        try:
            if docs:
                uploaded = await self.store.upsert_documents(docs)
        finally:
             pass

        # Calculate total duration
        duration = time.time() - start_time
        
        return IngestionResult(
            success=uploaded,
            documents_processed=len(blueprints),
            chunks_created=len(descriptions),
            documents_uploaded=len(docs) | 0,
            duration_seconds=duration
        )
