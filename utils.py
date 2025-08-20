import re

def simple_chunk_text(text, chunk_size=1000, overlap=200):
    """Character-based chunking with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - overlap
    return chunks

def clean_text(s):
    s = re.sub(r"\s+", " ", s)
    return s.strip()
