# utils/normalize_utils.py

"""
Utilities to normalize various input types (paths, bytes, dicts, file-like objects)
into a standard dict format for processing.
This includes MIME type guessing and optional content loading for text files.
"""

import os
import io
import mimetypes
from pathlib import Path
from typing import Any, Iterable, List, Dict, Union, Optional

# Extend common types that you want to recognize
DEFAULT_EXT_MIME_MAP = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md":  "text/markdown",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".rtf": "application/rtf",
    ".csv": "text/csv",
    ".json": "application/json",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

# Set a cap for when we will actually read file content into memory (e.g., for text files)
DEFAULT_MAX_TEXT_READ_BYTES = 2 * 1024 * 1024  # 2 MB

def guess_mime_type(path: Union[str, Path], ext_map: Dict[str, str]) -> str:
    """
    Guess MIME type from extension first (custom map), then fall back to mimetypes.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in ext_map:
        return ext_map[ext]
    # Fall back to Python's mimetypes
    mime, _ = mimetypes.guess_type(str(p))
    return mime or "application/octet-stream"

def is_text_mime(mime_type: str) -> bool:
    """
    Basic heuristic: treat text/* and a few common types as text-like.
    """
    if mime_type.startswith("text/"):
        return True
    return mime_type in {"application/json", "text/markdown", "text/csv"}

def normalize_single_item(
    item: Any,
    *,
    ext_mime_map: Dict[str, str] = DEFAULT_EXT_MIME_MAP,
    max_text_read_bytes: int = DEFAULT_MAX_TEXT_READ_BYTES,
) -> Dict[str, Any]:
    """
    Normalize a single item into a dict with fields:
    - name
    - mime_type
    - source: {type, value}
    - content (optional)
    """
    # Case 1: dict already in target-ish format
    if isinstance(item, dict):
        # We expect either user-provided normalized dict or raw metadata dict
        name = item.get("name") or item.get("filename") or "unnamed"
        mime = item.get("mime_type") or item.get("mime") or "application/octet-stream"
        normalized = {
            "name": name,
            "mime_type": mime,
            "source": {"type": "dict", "value": item},
        }
        # Carry forward optional content if present
        if "content" in item:
            normalized["content"] = item["content"]
        return normalized

    # Case 2: bytes: we can't guess file type reliably; caller should wrap in dict to include mime_type
    if isinstance(item, (bytes, bytearray, memoryview)):
        # Use a conservative default MIME type
        return {
            "name": "bytes",
            "mime_type": "application/octet-stream",
            "source": {"type": "bytes", "value": bytes(item)},
        }

    # Case 3: string — could be a path or a text snippet
    if isinstance(item, str):
        p = Path(item)
        if p.exists() and p.is_file():
            mime = guess_mime_type(p, ext_mime_map)
            entry = {
                "name": p.name,
                "mime_type": mime,
                "source": {"type": "path", "value": str(p)},
            }
            # If the file is text-like and reasonably small, load content
            try:
                size = p.stat().st_size
                if is_text_mime(mime) and size <= max_text_read_bytes:
                    # Try UTF-8 first; fall back to binary read
                    try:
                        entry["content"] = p.read_text(encoding="utf-8")
                        entry["mime_type"] = "text/plain" if mime == "application/octet-stream" else mime
                    except UnicodeDecodeError:
                        entry["content"] = p.read_bytes()
                # For non-text or large files (PDF, DOCX, videos), we keep path only
            except Exception:
                # If stat/read failed, just return path reference
                pass
            return entry
        else:
            # Treat raw string as text content
            return {
                "name": "text",
                "mime_type": "text/plain",
                "source": {"type": "bytes", "value": item.encode("utf-8")},
                "content": item,
            }

    # Case 4: pathlib.Path
    if isinstance(item, Path):
        return normalize_single_item(str(item), ext_mime_map=ext_mime_map, max_text_read_bytes=max_text_read_bytes)

    # Case 5: file-like object
    if hasattr(item, "read") and callable(getattr(item, "read")):
        # Try to get a name if available
        name = getattr(item, "name", None) or "fileobj"
        mime = "application/octet-stream"
        if name and isinstance(name, str):
            mime = guess_mime_type(name, ext_mime_map)
        return {
            "name": os.path.basename(name) if isinstance(name, str) else "fileobj",
            "mime_type": mime,
            "source": {"type": "fileobj", "value": item},
        }

    # Fallback: unknown type
    return {
        "name": "unknown",
        "mime_type": "application/octet-stream",
        "source": {"type": "unknown", "value": item},
    }

def normalize_file_items(
    items: Optional[Union[Any, Iterable[Any]]],
    *,
    ext_mime_map: Dict[str, str] = DEFAULT_EXT_MIME_MAP,
    max_text_read_bytes: int = DEFAULT_MAX_TEXT_READ_BYTES,
) -> List[Dict[str, Any]]:
    """
    Accept a single item or a collection, and normalize each into a dict.
    Handles:
      - str path or raw text
      - bytes
      - dict (already structured data)
      - pathlib.Path
      - file-like objects
      - iterables of any of the above
    """
    if items is None:
        return []
    # Single item case (avoid iterating bytes/str like sequences)
    if isinstance(items, (str, bytes, bytearray, memoryview, dict, Path)) or (
        hasattr(items, "read") and callable(getattr(items, "read"))
    ):
        return [normalize_single_item(items, ext_mime_map=ext_mime_map, max_text_read_bytes=max_text_read_bytes)]

    # Now treat as iterable
    normalized: List[Dict[str, Any]] = []
    for it in list(items):  # materialize generators safely
        normalized.append(normalize_single_item(it, ext_mime_map=ext_mime_map, max_text_read_bytes=max_text_read_bytes))
    return normalized
