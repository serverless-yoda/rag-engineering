# implementation/azure_openai_llm.py

"""
Azure OpenAI LLM provider implementation with retry logic.

This module implements the LLMProvider interface using Azure OpenAI's chat completions API.
Includes robust error handling and automatic retries for transient failures.
"""

import asyncio
import logging
from typing import List, Dict, Optional
from openai import (
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
)
from ..abstractions.llm_provider import LLMProvider


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
        """
        self.deployment_name = deployment_name
        self.timeout = timeout
        self.retries = retries
        
        # Create async Azure OpenAI client
        self.client = AsyncAzureOpenAI(
            api_version=api_version,
            api_key=api_key,
            azure_endpoint=endpoint,
            timeout=timeout,
        )
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a chat completion with automatic retry logic.
        
        Implements exponential backoff for transient errors:
        - APIConnectionError: Network issues (backoff: 0.6s * attempt)
        - RateLimitError: Rate limit hit (backoff: 1.5s * attempt)
        - APIStatusError: Service issues (backoff: 0.8s * attempt)
        
        Args:
            messages: List of message dicts with "role" and "content" keys
            temperature: Sampling temperature (0.0-2.0, higher = more creative)
            max_tokens: Maximum tokens to generate (None = model default)
        
        Returns:
            Generated text content from the assistant
        
        Raises:
            RuntimeError: If all retry attempts fail
            ValueError: If LLM returns empty content
        
        Note:
            Empty responses are treated as errors and trigger retries.
        """
        last_error: Optional[Exception] = None
        
        # Retry loop with exponential backoff
        for attempt in range(1, self.retries + 1):
            try:
                # Call Azure OpenAI chat completions API with timeout
                # asyncio.wait_for ensures we respect the timeout setting
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=messages,
                        #temperature=temperature,
                        #max_tokens=max_tokens,
                    ),
                    timeout=self.timeout,
                )
                
                # Extract the generated text from the response
                # response.choices[0].message.content contains the assistant's reply
                content = (
                    response.choices[0].message.content
                    if response and response.choices
                    else ""
                )
                
                # Validate that we got a non-empty response
                if not content.strip():
                    raise ValueError("LLM returned empty content")
                
                return content.strip()
                
            except APIConnectionError as e:
                # Network/connection errors - likely transient
                last_error = e
                logging.error(
                    f"Azure OpenAI connection failed (attempt {attempt}/{self.retries}): {e}"
                )
                await asyncio.sleep(0.6 * attempt)  # Linear backoff
                
            except RateLimitError as e:
                # Rate limit errors - need longer backoff
                last_error = e
                logging.warning(
                    f"Azure OpenAI rate limited (attempt {attempt}/{self.retries}): {e}"
                )
                await asyncio.sleep(1.5 * attempt)  # Longer backoff for rate limits
                
            except APIStatusError as e:
                # HTTP status errors (4xx, 5xx) - might be transient
                last_error = e
                logging.warning(
                    f"Azure OpenAI API status error (attempt {attempt}/{self.retries}): {e}"
                )
                await asyncio.sleep(0.8 * attempt)
                
            except Exception as e:
                # Catch-all for unexpected errors
                last_error = e
                logging.warning(
                    f"Unexpected LLM error (attempt {attempt}/{self.retries}): {e}"
                )
                await asyncio.sleep(0.6 * attempt)
        
        # All retries exhausted - raise error with context
        raise RuntimeError(
            f"LLM generation failed after {self.retries} retries"
        ) from last_error
    
    async def close(self) -> None:
        """
        Close the Azure OpenAI client connection.
        
        Safe to call multiple times.
        """
        try:
            await self.client.close()
        except Exception as e:
            logging.debug(f"Error closing Azure OpenAI LLM: {e}")
