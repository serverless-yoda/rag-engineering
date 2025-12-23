# utils/metadata_utils.py

"""
Metadata utilities for document management.

This module provides helper functions for timestamps and namespace management.
"""

import time


def now_iso() -> str:
    """
    Get current UTC time in ISO 8601 format.
    
    Returns a timestamp string suitable for storage in Azure AI Search DateTimeOffset fields.
    Format: YYYY-MM-DDTHH:MM:SSZ (e.g., "2024-12-13T20:30:45Z")
    
    Returns:
        ISO 8601 formatted timestamp string
    
    Used by:
        - make_search_documents() to set the created_at field
    
    Example:
        >>> now_iso()
        "2024-12-13T20:30:45Z"
    """
    # Use gmtime() to get UTC time, format using ISO 8601
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_namespace(ns: str, default_ns: str = "KnowledgeStore") -> str:
    """
    Validate and normalize a namespace string.
    
    Namespaces are used to logically group documents in the vector store.
    This function ensures the namespace is valid (non-empty after trimming).
    
    Args:
        ns: Input namespace string (may be None or whitespace)
        default_ns: Default namespace to use if input is empty
    
    Returns:
        Trimmed namespace or default_ns if input is empty
    
    Used by:
        - make_search_documents() to normalize namespace before storage
        - SemanticSearcher._build_filter() to normalize namespace in filters
    
    Example:
        >>> ensure_namespace("  MyNamespace  ")
        "MyNamespace"
        >>> ensure_namespace("")
        "KnowledgeStore"
        >>> ensure_namespace(None)
        "KnowledgeStore"
    """
    # Trim whitespace from input namespace
    # Handle None by treating it as empty string
    ns = (ns or "").strip()
    
    # Return trimmed namespace if non-empty, otherwise use default
    return ns or default_ns
