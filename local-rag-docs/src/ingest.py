from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils import chunk_text

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)


def load_documents() -> list[tuple[str, str]]:
    docs = []
    for p in DOCS_DIR.glob("*.txt"):
        docs.append((p.name, p.read_text(encoding="utf-8")))
    if not docs:
        raise RuntimeError("No .txt files found in data/docs/")
    return docs


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # 1) build chunks metadata (still in memory, but lighter than embeddings)
    all_chunks = []

    for filename, text in load_documents():
        print(filename, len(text))

    for filename, text in load_documents():
        chunks = chunk_text(text, chunk_size=1200, overlap=100)  # fewer chunks
        for c in chunks:
            all_chunks.append(
                {"doc": filename, "chunk_id": c.chunk_id, "text": c.text}
            )

    if not all_chunks:
        raise RuntimeError("No chunks were produced. Check your input files.")

    print(f"Total chunks: {len(all_chunks)}")

    # 2) create FAISS index
    dim = 384  # embedding dimension for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dim)

    # 3) embed + add to FAISS in batches (no huge embeddings list)
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

    # 4) save index + metadata
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
