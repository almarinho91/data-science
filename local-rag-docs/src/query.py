from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("indexes")


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index_path = INDEX_DIR / "docs.index"
    chunks_path = INDEX_DIR / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}. Run `python -m src.ingest` first.")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks metadata: {chunks_path}. Run `python -m src.ingest` first.")

    index = faiss.read_index(str(index_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    print(f"Index vectors: {index.ntotal}")
    print(f"Chunks loaded: {len(chunks)}")

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if not q:
            continue
        if q.lower() == "exit":
            break

        q_emb = model.encode(q, normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32").reshape(1, -1)

        k = min(3, len(chunks), index.ntotal)
        if k <= 0:
            print("No chunks in index. Add documents to data/docs and run ingest.")
            continue

        scores, ids = index.search(q_emb, k=k)

        print("\nTop matches:\n")
        for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
            # FAISS may return -1 if there are no more valid neighbors
            if i < 0:
                continue

            c = chunks[int(i)]
            print(f"{rank}) score={float(s):.4f} | {c['doc']} | {c['chunk_id']}")
            print(c["text"])
            print("-" * 60)


if __name__ == "__main__":
    main()
