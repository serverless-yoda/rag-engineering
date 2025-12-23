# utils/chunking_utils.py

"""
Text chunking utilities for document ingestion.

This module provides two chunking strategies:
1. Character-based: Fast, simple splitting by character count
2. Token-based: LLM-friendly splitting using tiktoken (OpenAI's tokenizer)
"""

from typing import List


def chunk_text(text: str, max_chars: int = 4000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping character-based chunks.
    
    This is a simple, fast chunking strategy that splits text by character count
    with an overlap region to maintain context across chunk boundaries.
    
    Algorithm:
    1. Start at position 0
    2. Take a chunk of max_chars length
    3. Move forward by (max_chars - overlap) to create next chunk
    4. Repeat until end of text
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap: Number of characters to overlap between consecutive chunks
    
    Returns:
        List of text chunks (empty list if input is empty)
    
    Used by:
        - DocumentIngester.ingest_documents() when use_token_chunking=False
    
    Example:
        >>> text = "A" * 10000
        >>> chunks = chunk_text(text, max_chars=4000, overlap=200)
        >>> len(chunks)
        3
        >>> len(chunks[0])
        4000
        >>> chunks[0][-200:] == chunks[1][:200]  # Overlap check
        True
    """
    if not text:
        return []
    
    chunks = []
    n = len(text)
    start = 0
    
    while start < n:
        # Calculate end position for this chunk
        end = min(start + max_chars, n)
        
        # Extract chunk and remove leading/trailing whitespace
        chunk = text[start:end].strip()
        
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Check if we've reached the end
        if end >= n:
            break
        
        # Move start position back by overlap amount to create overlap with next chunk
        # This ensures context continuity across boundaries
        start = end - overlap
    
    return chunks


def chunk_text_tiktoken(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping token-based chunks using tiktoken.
    
    This is a more sophisticated chunking strategy that splits by token count
    rather than character count. This is better for LLM embeddings because:
    - Embedding models have token limits (e.g., 8191 tokens for text-embedding-ada-002)
    - Token-based splitting aligns with how LLMs process text
    - Avoids splitting in the middle of words or multi-byte characters
    
    Algorithm:
    1. Tokenize the entire text using tiktoken (cl100k_base encoding)
    2. Split tokens into overlapping windows of size chunk_size
    3. Decode each token window back to text
    4. Clean up newlines and whitespace
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum tokens per chunk (e.g., 400 tokens ≈ 1600 chars)
        overlap: Number of tokens to overlap between consecutive chunks
    
    Returns:
        List of text chunks (empty list if input is empty)
    
    Raises:
        RuntimeError: If tiktoken is not installed (pip install tiktoken)
    
    Used by:
        - DocumentIngester.ingest_documents() when use_token_chunking=True
    
    Note:
        Requires the tiktoken library. Install with: pip install tiktoken
        The cl100k_base encoding is used by GPT-4 and text-embedding-ada-002.
    
    Example:
        >>> text = "This is a long document..." * 100
        >>> chunks = chunk_text_tiktoken(text, chunk_size=400, overlap=50)
        >>> # Each chunk will be approximately 400 tokens
    """
    if not text:
        return []
    
    # Try to import tiktoken (lazily loaded to make it optional)
    try:
        import tiktoken
    except ImportError as e:
        raise RuntimeError(
            "Token-based chunking requires tiktoken. Install with: pip install tiktoken"
        ) from e
    
    # Get the cl100k_base tokenizer (used by GPT-4 and text-embedding-ada-002)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Tokenize the entire text
    # This converts the text into a list of integer token IDs
    tokens = tokenizer.encode(text)
    
    chunks = []
    
    # Calculate step size (how far to advance between chunks)
    # Ensures overlap between consecutive chunks
    step = max(1, chunk_size - overlap)
    
    # Create overlapping windows of tokens
    for i in range(0, len(tokens), step):
        # Extract chunk_size tokens starting at position i
        chunk_tokens = tokens[i : i + chunk_size]
        
        # Decode tokens back to text
        chunk_text = tokenizer.decode(chunk_tokens)
        
        # Clean up formatting (replace newlines with spaces, strip whitespace)
        chunk_text = chunk_text.replace("\n", " ").strip()
        
        if chunk_text:  # Only add non-empty chunks
            chunks.append(chunk_text)
    
    return chunks
