# core/__init__.py

"""
Core B.I.S.A. modules for the RAG system.

This package contains the four main stages of the RAG pipeline:
- IndexManager: BUILD stage (create/manage search index)
- DocumentIngester: INGEST stage (process and store documents)
- SemanticSearcher: SEARCH stage (find relevant documents)
- AnswerGenerator: ANSWER stage (generate responses from context)
"""

from .index_manager import IndexManager
from .document_ingester import DocumentIngester
from .semantic_searcher import SemanticSearcher
from .answer_generator import AnswerGenerator

__all__ = [
    "IndexManager",
    "DocumentIngester",
    "SemanticSearcher",
    "AnswerGenerator",
]

