from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.loaders import iter_documents
from src.utils import chunk_text

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks: list[dict] = []

    for filename, text in iter_documents(DOCS_DIR):
        text = (text or "").strip()
        print(f"Loaded {filename}: {len(text)} chars")

        # basic sanity check for PDFs
        if filename.lower().endswith(".pdf") and len(text) < 5000:
            print(f"WARNING: PDF extraction looks small for {filename} ({len(text)} chars).")

        if not text:
            continue

        # smaller chunks -> better retrieval precision
        chunks = chunk_text(text, chunk_size=250, overlap=50)

        for c in chunks:
            all_chunks.append(
                {
                    "doc": filename,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                }
            )

    if not all_chunks:
        raise RuntimeError("No chunks produced. Put .txt/.pdf files into data/docs/ and try again.")

    print(f"Total chunks: {len(all_chunks)}")

    # FAISS index (cosine similarity via inner product on normalized vectors)
    dim = 384  # embedding size for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)

    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding + indexing"):
        batch_texts = [c["text"] for c in all_chunks[i : i + batch_size]]

        batch_emb = model.encode(
            batch_texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        X = np.asarray(batch_emb, dtype="float32")
        index.add(X)

    # Save artifacts
    faiss.write_index(index, str(INDEX_DIR / "docs.index"))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(all_chunks, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Indexed {len(all_chunks)} chunks")
    print(f"Saved: {INDEX_DIR / 'docs.index'}")
    print(f"Saved: {INDEX_DIR / 'chunks.json'}")


if __name__ == "__main__":
    main()
