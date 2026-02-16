from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: str
    text: str


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[Chunk]:
    """
    Split input text into overlapping character-based chunks.

    Args:
        text: Full document text.
        chunk_size: Maximum number of characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    step = chunk_size - overlap
    chunks: List[Chunk] = []

    idx = 0
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(chunk_id=f"chunk_{idx}", text=chunk))
            idx += 1
        if end == len(text):
            break

    return chunks
