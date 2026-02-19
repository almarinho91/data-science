from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import pdfplumber


@dataclass(frozen=True)
class DocumentPage:
    doc: str
    page: Optional[int]  # None for txt
    text: str


def load_txt(path: Path) -> list[DocumentPage]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [DocumentPage(doc=path.name, page=None, text=text)]


def load_pdf(path: Path) -> list[DocumentPage]:
    pages: list[DocumentPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            pages.append(DocumentPage(doc=path.name, page=i, text=t))
    return pages


def iter_document_pages(docs_dir: Path) -> Iterator[DocumentPage]:
    for p in sorted(docs_dir.rglob("*")):
        if p.is_dir():
            continue
        suf = p.suffix.lower()
        if suf == ".txt":
            yield from load_txt(p)
        elif suf == ".pdf":
            yield from load_pdf(p)
