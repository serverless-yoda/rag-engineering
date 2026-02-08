# rag/services/gemini/__init__.py

"""
Docstring for rag.services.gemini
Gemini-based implementations of abstract providers.
This package contains concrete implementations of the abstract interfaces
using Google Gemini models via the google-genai library.
"""

from .chat_provider import GeminiChatProvider

__all__ = [
    "GeminiChatProvider"
]
