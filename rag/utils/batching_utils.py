# utils/batching.py

"""
Batching utility for processing sequences in chunks.

This module provides a simple batching function for splitting sequences
into fixed-size batches for efficient API calls.
"""

from typing import Sequence, Any, Iterable


def batched(seq: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    """
    Split a sequence into fixed-size batches.
    
    This is a simple generator that yields consecutive slices of the input sequence.
    Useful for batching API requests to embedding or LLM services.
    
    Args:
        seq: Input sequence (list, tuple, etc.)
        batch_size: Maximum size of each batch
    
    Yields:
        Consecutive slices of the input sequence
        The last batch may be smaller than batch_size
    
    Used by:
        - DocumentIngester.ingest_documents() for batching embedding requests
    
    Example:
        >>> items = list(range(10))
        >>> for batch in batched(items, batch_size=3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    
    Note:
        This function does not modify the input sequence or create copies.
        It yields views (slices) of the original sequence.
    """
    for i in range(0, len(seq), batch_size):
        # Yield a slice from i to i+batch_size
        # Python slicing handles the boundary automatically
        # (last batch may be smaller than batch_size)
        yield seq[i : i + batch_size]
