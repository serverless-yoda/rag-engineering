# core/answer_generator.py

"""
💬 ANSWER stage: Generate responses using a Large Language Model (LLM).

This module handles the final step of the RAG pipeline: generating a natural language
answer based on retrieved context and a user question. It uses an LLMProvider to
interface with Azure OpenAI or other LLM services.
"""

import logging
from typing import List, Optional
from ..abstractions.llm_provider import LLMProvider
from ..models import GenerationError

class AnswerGenerator:
    """
    ANSWER stage: Generates responses using an LLM.
    
    Responsibilities:
    - Format system and user prompts
    - Send messages to the LLMProvider
    - Return the generated response
    
    Dependencies:
    - LLMProvider: Abstract interface for chat/completion models
    
    Example:
        >>> generator = AnswerGenerator(llm)
        >>> answer = await generator.generate("What is RAG?", context="RAG is...")
        >>> print(answer)
    """
    
    def __init__(self, llm: LLMProvider):
        """
        Initialize the answer generator.
        
        Args:
            llm: LLMProvider instance (e.g., AzureOpenAILLM)
        """
        self.llm = llm

    async def generate(
        self,
        question: str,
        context: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate an answer from the given context and question.
        
        Steps:
        1. Format system and user messages
        2. Send messages to the LLMProvider
        3. Return the generated response
        
        Args:
            question: The user's question
            context: Retrieved document chunks to ground the answer
            system_prompt: Optional system-level instructions for the LLM
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Optional limit on response length
        
        Returns:
            Generated answer string
        
        Raises:
            Exception: If LLMProvider fails to generate a response
        
        Example:
            >>> answer = await generator.generate("What is RAG?", context="RAG is...")
        """
        # Default system prompt if none provided
        default_prompt = (
            "You are a helpful AI assistant. Use the provided context to answer "
            "the user's question. If the context doesn't contain relevant information, "
            "say so clearly."
        )
        prompt = system_prompt or default_prompt

        # Format user message with context and question
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        # Construct message history for LLM
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ]
        logging.info(f"Sending messages to LLM: {messages}")
        try:
            # Send messages to LLMProvider and return response
            answer = await self.llm.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logging.info(f"Answer generated successfully.{answer}")
            
            return answer
        except Exception as e:
            logging.error(f"Answer generation failed: {e}")
            raise GenerationError(f"Answer generation failed: {e}") from e
