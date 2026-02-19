from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import pdfplumber


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    texts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            texts.append(t)

    text = "\n".join(texts).replace("\r\n", "\n").strip()
    return text


def iter_documents(docs_dir: Path) -> Iterator[Tuple[str, str]]:
    for p in sorted(docs_dir.glob("**/*")):
        if p.is_dir():
            continue
        suf = p.suffix.lower()
        if suf == ".txt":
            yield (p.name, load_txt(p))
        elif suf == ".pdf":
            yield (p.name, load_pdf(p))
