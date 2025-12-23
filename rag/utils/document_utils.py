# utils/documents_utils.py

"""
Document utilities for creating search-ready documents.

This module provides functions to shape document chunks and embeddings
into the format required by Azure AI Search.
"""

import os
import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from .metadata_utils import ensure_namespace, now_iso

JsonDict = Dict[str, Any]


def make_search_documents(
    namespace: str,
    source_id: str,
    content_chunks: List[str],
    embeddings: List[List[float]],
    extra_meta: Optional[JsonDict] = None,
) -> List[JsonDict]:
    """
    Build Azure AI Search documents from content chunks and embeddings.
    
    This function creates a list of documents that match the Azure AI Search index schema
    defined in IndexManager. Each document represents one chunk with its embedding vector.
    
    Document Schema:
    - id: Unique identifier (format: "{source_id}-{chunk_index}")
    - namespace: Logical grouping (normalized via ensure_namespace)
    - source_id: Identifier for the source document
    - chunk_id: Sequential chunk number (0, 1, 2, ...)
    - chunk: Text content of this chunk
    - chunk_vector: Embedding vector for this chunk
    - tags: Optional tags from extra_meta (for filtering)
    - created_at: ISO 8601 timestamp
    - source_uri: Optional source location from extra_meta
    - metadata_json: JSON-serialized extra_meta dictionary
    
    Args:
        namespace: Namespace for organizing documents (e.g., "KnowledgeBase", "Policies")
        source_id: Unique identifier for the source document
        content_chunks: List of text chunks (must match embeddings length)
        embeddings: List of embedding vectors (must match content_chunks length)
        extra_meta: Optional metadata dictionary (can include tags, source_uri, etc.)
    
    Returns:
        List of document dictionaries ready for Azure AI Search upload
    
    Used by:
        - DocumentIngester.ingest_documents() to prepare documents for upload
    
    Example:
        >>> chunks = ["chunk 1", "chunk 2"]
        >>> embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        >>> docs = make_search_documents(
        ...     namespace="MyDocs",
        ...     source_id="doc123",
        ...     content_chunks=chunks,
        ...     embeddings=embeddings,
        ...     extra_meta={"tags": "important", "source_uri": "/path/to/doc.pdf"}
        ... )
        >>> len(docs)
        2
        >>> docs[0]["id"]
        "doc123-0"
        >>> docs[0]["chunk"]
        "chunk 1"
    """
    # Normalize namespace (handles None, empty string, whitespace)
    safe_namespace = ensure_namespace(namespace)
    
    # Get current timestamp for all documents
    timestamp = now_iso()
    
    # Prepare metadata dictionary
    meta = extra_meta or {}
    
    # Serialize metadata to JSON string for storage
    # Azure AI Search stores this in a String field that we parse later
    metadata_json = json.dumps(meta, ensure_ascii=False) if meta else None
    
    # Build documents by zipping chunks and embeddings
    docs = []
    for idx, (chunk, vec) in enumerate(zip(content_chunks, embeddings)):
        docs.append({
            # Unique identifier: source_id + chunk index
            "id": f"{source_id}-{idx}",
            
            # Namespace for logical grouping
            "namespace": safe_namespace,
            
            # Source document identifier
            "source_id": source_id,
            
            # Sequential chunk number within this source
            "chunk_id": idx,
            
            # Text content
            "chunk": chunk,
            
            # Embedding vector for similarity search
            "chunk_vector": vec,
            
            # Extract tags from metadata if present (used for filtering)
            "tags": meta.get("tags"),
            
            # Timestamp when this document was created
            "created_at": timestamp,
            
            # Extract source URI from metadata if present
            "source_uri": meta.get("source_uri"),
            
            # Full metadata as JSON string
            "metadata_json": metadata_json,
        })
    
    return docs

def normalize_items(items):
    # Accept a single str/bytes/dict and make it a list
    if items is None:
        return []
    if isinstance(items, (str, bytes, dict)):
        return [items]
    # If it's already a sequence/iterable, keep as is
    return list(items)

def list_files_in_folder(folder_path: str) -> List[str]:
    """
    Return a list of full paths for all files in the given folder (non-recursive).
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} is not a valid directory")

    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))]


ID_MAX_LEN = 128  # tighten if your vector store enforces smaller limits

def slugify(text: str) -> str:
    """Letters/numbers/dash/underscore; collapse spaces/punct into dashes."""
    text = text.strip()
    text = re.sub(r"\s+", "-", text)                     # spaces -> dash
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "-", text)        # remove others
    text = re.sub(r"-{2,}", "-", text)                   # collapse dashes
    return text.strip("-").lower()

def short_hash(s: str) -> str:
    """Deterministic 8-char hash from a string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def make_item_source_id(item: Dict[str, Any], idx: int, base_source_id: str) -> str:
    """
    If item is a file (source.type == 'path'), build ID from filename + short hash of path.
    Otherwise, fall back to index-based ID.
    Guaranteed to use only safe characters and be <= ID_MAX_LEN.
    """
    # Default fallback
    fallback = f"{base_source_id}-{idx}"

    src = item.get("source", {})
    name = item.get("name") or ""
    if src.get("type") == "path" and src.get("value"):
        path_str = str(src["value"])
        p = Path(path_str)
        stem = p.stem or name or f"doc-{idx}"
        stem_slug = slugify(stem)
        h = short_hash(path_str.lower())
        candidate = f"{base_source_id}-{stem_slug}-{h}"
    else:
        # Not a path -> raw text, bytes, dict, etc.
        stem_slug = slugify(name) if name else None
        candidate = f"{base_source_id}-{stem_slug}-{idx}" if stem_slug else fallback

    if len(candidate) > ID_MAX_LEN:
        candidate = candidate[:ID_MAX_LEN]
    return candidate
