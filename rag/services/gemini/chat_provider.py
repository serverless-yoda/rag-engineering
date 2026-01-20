# rag/services/gemini/chat_provider.py

"""
Chat provider for Google Gemini using google-genai library.
Handles generating responses from Gemini models.
"""

from google import genai
from google.genai import types
from ...interfaces import LLMProvider

class GeminiChatProvider(LLMProvider):
    def __init__(self, config):
        self.client = genai.Client(api_key=config.gemini_api_key)
        self.model_id = config.gemini_model_id # e.g., 'gemini-2.0-flash'

    async def generate(self, system_prompt: str, user_query: str) -> str:
        # Gemini 2.0 uses 'system_instruction' as a separate parameter
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.0
            ),
            contents=user_query
        )
        return response.text