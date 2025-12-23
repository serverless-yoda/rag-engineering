# utils/__init__.py

"""
Utility functions for the RAG system (Facade pattern).

This module provides a clean import interface for all utility functions,
organized by domain (text processing, chunking, batching, metadata, documents).
"""

from .text_utils import to_text_content, strip_html, sanitize_input
from .chunking_utils import chunk_text, chunk_text_tiktoken
from .batching_utils import batched
from .metadata_utils import ensure_namespace, now_iso
from .document_utils import make_search_documents, normalize_items, list_files_in_folder, make_item_source_id
from .tokens_utils import count_tokens
from .normalize_utils import normalize_file_items
from .generictext_utils import file_to_text_content

__all__ = [
    "to_text_content",
    "strip_html",
    "chunk_text",
    "chunk_text_tiktoken",
    "batched",
    "ensure_namespace",
    "now_iso",
    "make_search_documents",
    "normalize_items",
    "count_tokens",
    "sanitize_input",
    "normalize_file_items",
    "file_to_text_content",
    "list_files_in_folder",
    "make_item_source_id",
]
