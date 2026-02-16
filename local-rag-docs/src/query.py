from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("indexes")


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(str(INDEX_DIR / "docs.index"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        q_emb = model.encode(q, normalize_embeddings=True).astype("float32")
        q_emb = np.expand_dims(q_emb, axis=0)

        scores, ids = index.search(q_emb, k=3)

        print("\nTop answers:\n")

        for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
            c = chunks[int(i)]

            sentences = split_sentences(c["text"])
            sent_embs = model.encode(sentences, normalize_embeddings=True)

            q_vec = q_emb.reshape(-1)

            sims = [
                float(np.dot(q_vec, np.asarray(e, dtype="float32")))
                for e in sent_embs
            ]

            best_idx = int(np.argmax(sims))

            print(f"{rank}) score={s:.4f} | {c['doc']}")
            print(sentences[best_idx])
            print("-" * 60)


if __name__ == "__main__":
    main()
