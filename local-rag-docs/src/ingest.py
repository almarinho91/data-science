from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.loaders import iter_document_pages
from src.utils import chunk_by_paragraphs


def main(
    docs_dir: str | Path = "data/docs",
    index_dir: str | Path = "indexes",
    target_chunk_size: int = 800,
    overlap: int = 120,
) -> None:
    docs_dir = Path(docs_dir)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("RAGDOCS_MODEL", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    all_chunks: list[dict] = []

    # 1) Load pages -> chunk
    for page in iter_document_pages(docs_dir):
        if not page.text.strip():
            continue

        chunks = chunk_by_paragraphs(page.text, target_size=target_chunk_size, overlap=overlap)
        for ch in chunks:
            all_chunks.append(
                {
                    "doc": page.doc,
                    "page": page.page,
                    "chunk_id": ch.chunk_id,
                    "text": ch.text,
                    "char_start": ch.char_start,
                    "char_end": ch.char_end,
                }
            )

    if not all_chunks:
        raise RuntimeError(f"No text found under: {docs_dir.resolve()}")

    # 2) Embed
    texts = [c["text"] for c in all_chunks]
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype="float32")

    # 3) Build FAISS index (cosine sim via inner product on normalized vectors)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 4) Persist
    (index_dir / "chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    faiss.write_index(index, str(index_dir / "docs.index"))


if __name__ == "__main__":
    main()