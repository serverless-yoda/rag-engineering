# utils/tokens_utils.py

"""Utility functions for token counting with tiktoken."""

import tiktoken

def count_tokens(text: str, model: str = "gpt-4-nano") -> int:
    """Counts tokens using tiktoken for Azure/OpenAI models."""
    import tiktoken

    model_map = {
        "gpt-4": "gpt-4",
        "gpt-4-nano": "gpt-4-nano",
        "gpt-35-turbo": "gpt-3.5-turbo",
        "text-embedding-ada-002": "text-embedding-ada-002",
    }

    base_model = model_map.get(model, model)
    try:
        encoding = tiktoken.encoding_for_model(base_model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))
