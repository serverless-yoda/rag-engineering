# interfaces/interface.py
"""Base interface for RAG components."""
from typing import Protocol, List, Dict, Optional
from ..models import SearchResult

class SearchProvider(Protocol):
    async def search(self, query: str, namespace: Optional[str] = None, top_k: int = 5) -> List[SearchResult]: ...

class GenerationProvider(Protocol):
    async def generate(self, question: str, context: str, system_prompt: Optional[str] = None) -> str: ...
