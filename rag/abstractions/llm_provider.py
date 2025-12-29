# abstractions/llm_provider.py

"""
Abstract interface for Large Language Model services.

This module defines the contract that all LLM providers must implement,
allowing the system to work with any chat/completion API (Azure OpenAI, OpenAI, Anthropic, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class LLMProvider(ABC):
    """
    Abstract base class for Large Language Model services.
    
    Implementations must provide:
    1. generate() method to produce completions from message history
    2. close() method to cleanup resources
    
    Example implementations:
    - AzureOpenAILLM (Azure OpenAI chat completions)
    - OpenAILLM (OpenAI chat completions)
    - AnthropicLLM (Claude)
    """
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a completion from a conversation history.
        
        Messages follow the standard chat format:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is RAG?"},
        ]
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens to generate (None = use model default)
        
        Returns:
            Generated text content from the assistant
        
        Raises:
            Exception: If generation fails (connection error, rate limit, invalid input, etc.)
        
        Example:
            >>> llm = AzureOpenAILLM(...)
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response = await llm.generate(messages)
            >>> print(response)
            "Hello! How can I help you today?"
        """
        pass
    
    @abstractmethod
    async def safe_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Cleanup resources (close connections, etc.).
        
        Should handle errors gracefully and not raise exceptions.
        """
        pass
