from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    char_start: int
    char_end: int


def clean_pdf_text(text: str) -> str:
    """
    Stronger cleanup for PDF-extracted text:
    - remove common boilerplate lines (publisher/footer/header)
    - fix hyphenation across line breaks
    - normalize newlines/spaces
    - remove very "numeric-heavy" lines (tables)
    """
    import re

    if not text:
        return ""

    text = text.replace("\r\n", "\n")

    # Remove lines that are mostly boilerplate / website footer/header noise
    drop_patterns = [
        r"frontiersin\.org",
        r"Frontiers\s+in\s+Physics",
        r"doi:\s*\S+",
        r"OPEN\s+ACCESS",
        r"Received:\s*\d",
        r"Accepted:\s*\d",
        r"Published:\s*\d",
        r"Copyright\s+©",
        r"Edited\s+by",
        r"Reviewed\s+by",
    ]

    cleaned_lines = []
    for line in text.splitlines():
        l = line.strip()
        if not l:
            cleaned_lines.append("")
            continue

        # drop boilerplate lines
        low = l.lower()
        if any(re.search(p.lower(), low) for p in drop_patterns):
            continue

        # drop table-like lines: too many digits/symbols compared to letters
        letters = sum(ch.isalpha() for ch in l)
        digits = sum(ch.isdigit() for ch in l)
        if digits > 25 and letters < 10:
            continue

        cleaned_lines.append(l)

    text = "\n".join(cleaned_lines)

    # Fix hyphenation across line breaks: "seg-\nmentation" -> "segmentation"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Convert single newlines to spaces (keep paragraph breaks)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize multiple newlines to exactly double newline (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def chunk_by_paragraphs(text: str, target_size: int = 800, overlap: int = 120) -> List[Chunk]:
    """
    Build chunks by joining paragraphs until reaching ~target_size chars.
    Produces char offsets relative to the cleaned `text`.
    """
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []

    buffer = ""
    buffer_start = 0
    cursor = 0  # tracks where we are in the cleaned text

    # Build a mapping of paragraph positions by searching sequentially

    pos = 0
    para_positions: List[Tuple[int, int, str]] = []
    for p in paras:
        idx = text.find(p, pos)
        if idx == -1:
            idx = pos
        start = idx
        end = start + len(p)
        para_positions.append((start, end, p))
        pos = end

    i = 0
    while i < len(para_positions):
        p_start, p_end, p = para_positions[i]

        if not buffer:
            buffer = p
            buffer_start = p_start
        else:
            buffer = buffer + "\n\n" + p

        if len(buffer) >= target_size:
            chunk_text = buffer
            c_start = buffer_start
            c_end = c_start + len(chunk_text)

            chunk_id = f"chunk_{len(chunks)}"
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, char_start=c_start, char_end=c_end))

            # overlap: keep last N chars from this chunk as next buffer
            if overlap > 0 and len(chunk_text) > overlap:
                tail = chunk_text[-overlap:]
                buffer = tail
                buffer_start = c_end - overlap
            else:
                buffer = ""
                buffer_start = 0

        i += 1

    # final remainder
    if buffer.strip():
        chunk_id = f"chunk_{len(chunks)}"
        c_start = buffer_start
        c_end = c_start + len(buffer)
        chunks.append(Chunk(chunk_id=chunk_id, text=buffer, char_start=c_start, char_end=c_end))

    return chunks
