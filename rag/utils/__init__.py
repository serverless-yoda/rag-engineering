# utils/__init__.py

"""
Utility functions for the RAG system (Facade pattern).

This module provides a clean import interface for all utility functions,
organized by domain (text processing, chunking, batching, metadata, documents).
"""

from .logging_decorators import with_logging_context


__all__ = [
    "with_logging_context",
]
