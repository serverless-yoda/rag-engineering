# rag/services/openrouter/openrouter_llm.py

"""
OpenRouter LLM provider (OpenAI-compatible API).
Supports Gemini, GPT models, and 100+ others via single endpoint.
"""

import logging
from typing import List, Dict, Optional, Any
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from ...interfaces import LLMProvider  # Make sure this import exists
from ...utils.chunk import TokenTracker

class OpenRouterLLM(LLMProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-2.5-pro-exp-03-25:free",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 60.0,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.token_tracker = token_tracker
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
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
        """Core generation method."""
        try:
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,
                extra_headers={
                    "HTTP-Referer": "https://your-app.com",
                    "X-Title": "RAG Multi-Agent Pipeline",
                }
            )
            
            # Track token usage
            if self.token_tracker and response.usage:
                self.token_tracker.add_llm_usage(
                    prompt_tokens=response.usage.prompt_tokens or 0,
                    completion_tokens=response.usage.completion_tokens or 0,
                    stage=stage,
                )
            
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise ValueError("LLM returned empty content")
            
            return content.strip()
            
        except Exception as e:
            logging.error(f"OpenRouter generation failed: {e}")
            raise

    # ✅ REQUIRED: Implement ALL abstract methods from LLMProvider
    async def generate_with_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stage: str = "generation",
    ) -> str:
        """Retry wrapper for generate method."""
        return await self.generate(messages, temperature, max_tokens, stage)

    async def close(self) -> None:
        """Close client connection."""
        try:
            await self.client.close()
            logging.info(f"{self.__class__.__name__} client closed.")
        except Exception as e:
            logging.error(f"Error closing OpenRouter LLM: {e}")

    # Additional required methods (implement if your LLMProvider requires them)
    async def generate_stream(self, *args, **kwargs) -> Any:
        """Streaming generation (if required)."""
        raise NotImplementedError("Streaming not implemented")

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model
