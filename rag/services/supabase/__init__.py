# rag/services/supabase/__init__.py

"""
Docstring for rag.services.supabase
Supabase-based implementations of abstract providers.
This package contains concrete implementations of the abstract interfaces
using Supabase vector database.
"""

from .supabase_vector_store import SupabaseVectorStore

__all__ = [
    "SupabaseVectorStore"
]
