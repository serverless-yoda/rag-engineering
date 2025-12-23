# utils/text_utils.py

"""
Text normalization and HTML processing utilities.

This module handles conversion of various input types (str, bytes, dict, etc.)
into clean, normalized text suitable for embedding and storage.
"""

import re
import json
from typing import Any


def strip_html(text: str) -> str:
    """
    Remove HTML tags and normalize whitespace from text.
    
    This function performs a lightweight HTML cleanup:
    1. Removes <script> blocks (including content)
    2. Removes <style> blocks (including content)
    3. Removes all HTML tags
    4. Normalizes whitespace (multiple spaces/newlines → single space)
    
    Args:
        text: HTML or plain text string
    
    Returns:
        Clean text with HTML removed and whitespace normalized
    
    Note:
        This is a simple regex-based approach. For complex HTML, consider
        using BeautifulSoup or html2text for more robust parsing.
    
    Example:
        >>> strip_html("<p>Hello <b>world</b>!</p>")
        "Hello world!"
        >>> strip_html("Line 1\n\n\nLine 2")
        "Line 1 Line 2"
    """
    # Remove script tags and their content
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    
    # Remove style tags and their content
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    
    # Remove all remaining HTML tags
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    
    # Normalize whitespace (multiple spaces/newlines → single space)
    text = re.sub(r"\s+", " ", text)
    
    return text

def to_text_content(obj: Any) -> str:
    """
    Convert arbitrary objects to clean text content.
    
    This function handles multiple input types and normalizes them to plain text:
    - None → empty string
    - str → HTML-stripped and trimmed
    - bytes → decoded UTF-8 (errors ignored)
    - dict → JSON string with sorted keys
    - list/tuple → JSON string
    - other → str() conversion then HTML-stripped
    
    Args:
        obj: Any object to convert to text
    
    Returns:
        Normalized text string (may be empty)
    
    Used by:
        - DocumentIngester.ingest_documents() for normalizing input items
    
    Example:
        >>> to_text_content("Hello <b>world</b>")
        "Hello world"
        >>> to_text_content({"key": "value"})
        '{"key": "value"}'
        >>> to_text_content(None)
        ""
    """
    # Handle None
    if obj is None:
        return ""
    
    # Handle strings (most common case)
    if isinstance(obj, str):
        return strip_html(obj).strip()
    
    # Handle bytes (decode to UTF-8)
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="ignore")
        except Exception:
            return ""
    
    # Handle dictionaries (convert to JSON)
    # sort_keys ensures consistent output for same data
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    
    # Handle lists and tuples (convert to JSON)
    if isinstance(obj, (list, tuple)):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            # Fallback to str() if JSON serialization fails
            return strip_html(str(obj))
    
    # Handle all other types (convert to string then strip HTML)
    return strip_html(str(obj))

def sanitize_input(text: str) -> str:
    """
    A simple sanitization function to detect and flag potential prompt injection patterns.
    Returns the text if clean, or raises a ValueError if a threat is detected.
    """
    # List of simple, high-confidence patterns to detect injection attempts
    injection_patterns = [
        r"ignore previous instructions",
        r"ignore all prior commands",
        r"you are now in.*mode",
        r"act as",
        r"print your instructions",
        # A simple pattern to catch attempts to inject system-level commands
        r"sudo|apt-get|yum|pip install"
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError(f"Input sanitization failed. Potential threat detected.")
            
    return text