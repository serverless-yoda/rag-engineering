# implementation/azure_openai_llm.py

"""
Azure OpenAI LLM provider implementation with retry logic.

This module implements the LLMProvider interface using Azure OpenAI's chat completions API.
Includes robust error handling and automatic retries for transient failures.
"""

import asyncio
import logging
from typing import List, Dict, Optional
from openai import AsyncAzureOpenAI
    
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ..abstractions.llm_provider import LLMProvider
from ..utils import TokenTracker

class AzureOpenAILLM(LLMProvider):
    """
    Azure OpenAI implementation of the LLMProvider interface with retry logic.
    
    Provides chat completion generation using Azure OpenAI's chat models (GPT-3.5, GPT-4, etc.).
    Includes automatic retry logic for common transient errors:
    - Connection errors (exponential backoff)
    - Rate limit errors (longer backoff)
    - API status errors (temporary service issues)
    
    Example:
        >>> llm = AzureOpenAILLM(
        ...     endpoint="https://my-openai.openai.azure.com/",
        ...     api_key="key123",
        ...     api_version="2024-02-15-preview",
        ...     deployment_name="gpt-4",
        ...     timeout=60.0,
        ...     retries=3,
        ... )
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ... ]
        >>> response = await llm.generate(messages)
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str,
        deployment_name: str,
        timeout: float = 60.0,
        retries: int = 3,
        token_tracker: Optional[TokenTracker] = None,
    ):
        """
        Initialize the Azure OpenAI LLM client.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: API key for authentication
            api_version: API version string
            deployment_name: Name of the deployed chat model
            timeout: Timeout in seconds for API calls
            retries: Number of retry attempts for transient errors
            token_tracker: Optional token tracker
        """
        self.deployment_name = deployment_name
        self.timeout = timeout
        self.retries = retries
        self.token_tracker = token_tracker
        
        # Create async Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=endpoint,
            timeout=timeout,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def safe_generate(self, messages, temperature=0.7, max_tokens=None):
        return await self.client.chat.completions.create(
            model=self.deployment_name,
            messages=messages,
            #temperature=temperature,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
    )
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stage: str = "generation",
    ) -> str:
        """
        Generate a chat completion with automatic retry logic.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            temperature: Sampling temperature (0.0-2.0, higher = more creative)
            max_tokens: Maximum tokens to generate (None = model default)
            stage: Stage name for token tracking (if applicable)
        Returns:
            Generated text content from the assistant
        
        Raises:
            RuntimeError: If all retry attempts fail
            ValueError: If LLM returns empty content
        
        Note:
            Empty responses are treated as errors and trigger retries.
        """

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    #temperature=temperature,
                    #max_tokens=max_tokens,
                ),
                timeout=self.timeout,
            )
            
            # Track token usage
            if self.token_tracker and response.usage:
                self.token_tracker.add_llm_usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    stage=stage,
                )
            
            content = (
                response.choices[0].message.content
                if response and response.choices
                else ""
            )
            
            if not content.strip():
                raise ValueError("LLM returned empty content")
            
            return content.strip()
            
        except Exception as e:
            logging.error(f"LLM generation failed: {e}")
            raise
    
    async def close(self) -> None:
        """
        Close the Azure OpenAI client connection.
        
        Safe to call multiple times.
        """
        try:
            await self.client.close()
        except Exception as e:
            logging.error(f"Error closing Azure OpenAI LLM: {e}")
