from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.answering import pick_best_sentences

INDEX_DIR = Path("indexes")


def _load_index_and_chunks():
    index = faiss.read_index(str(INDEX_DIR / "docs.index"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


def _search_global(index, chunks, q_emb: np.ndarray, k: int) -> list[dict]:
    k = min(k, len(chunks))
    scores, ids = index.search(q_emb, k=k)

    results = []
    for i, s in zip(ids[0], scores[0]):
        c = chunks[int(i)]
        results.append(
            {
                "doc": c.get("doc"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text", ""),
                "score": float(s),
            }
        )
    return results


def _search_with_doc_filter(index, chunks, q_emb: np.ndarray, doc_filter: str, k: int) -> list[dict]:
    # portfolio-friendly approach:
    # search a larger K globally, then filter down by doc
    # avoids rebuilding a temporary FAISS index.
    broad_k = min(max(k * 15, 50), len(chunks))
    broad = _search_global(index, chunks, q_emb, k=broad_k)
    filtered = [r for r in broad if r["doc"] == doc_filter]
    return filtered[:k]


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = _load_index_and_chunks()

    docs = sorted({c["doc"] for c in chunks})
    print("Available docs:")
    for d in docs:
        print(" -", d)

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        doc_filter = input("Filter by doc (enter to skip): ").strip()
        k = 5  # retrieve top-k chunks (then we extract best sentences from them)

        q_emb = model.encode(q, normalize_embeddings=True).astype("float32").reshape(1, -1)

        if doc_filter:
            top_chunks = _search_with_doc_filter(index, chunks, q_emb, doc_filter=doc_filter, k=k)
            if not top_chunks:
                print("No matches for that document filter.")
                continue
        else:
            top_chunks = _search_global(index, chunks, q_emb, k=k)

        answer, sources = pick_best_sentences(model, q, top_chunks, top_sentences=2)

        print("\nAnswer:\n")
        print(answer if answer else "(no answer found)")
        print("\nSources:\n")
        for n, s in enumerate(sources, start=1):
            print(
                f"{n}) {s['doc']} | page={s['page']} | {s['chunk_id']} "
                f"| sent_sim={s['sentence_similarity']:.4f} | chunk_score={s['chunk_score']:.4f}"
            )
            snippet = s["sentence"].strip()
            if len(snippet) > 240:
                snippet = snippet[:240] + "..."
            print("   ", snippet)
        print("-" * 60)


if __name__ == "__main__":
    main()
