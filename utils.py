# utils.py

import re
from typing import List


def simple_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Character-based chunking with overlap.

    Args:
        text (str): Input text.
        chunk_size (int): Max characters per chunk.
        overlap (int): Overlap between consecutive chunks.

    Returns:
        list[str]: List of text chunks.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    # Ensure overlap never causes infinite loop
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == length:
            break

        start = end - overlap

    return chunks


def clean_text(s: str) -> str:
    """
    Cleans input text by normalizing whitespace and removing stray control chars.

    Args:
        s (str): Raw text.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(s, str):
        return ""

    # Remove unwanted control characters (except \n and \t)
    s = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", " ", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)

    return s.strip()
