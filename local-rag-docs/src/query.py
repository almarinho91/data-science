from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("indexes")


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(INDEX_DIR / "docs.index"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))

    # Optional: precompute doc list
    docs = sorted({c["doc"] for c in chunks})
    print("Available docs:")
    for d in docs:
        print(" -", d)

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() == "exit":
            break

        doc_filter = input("Filter by doc (enter to skip): ").strip()
        candidate_idx = list(range(len(chunks)))

        if doc_filter:
            candidate_idx = [i for i, c in enumerate(chunks) if c["doc"] == doc_filter]
            if not candidate_idx:
                print("No chunks for that doc filter.")
                continue

        q_emb = model.encode(q, normalize_embeddings=True).astype("float32").reshape(1, -1)

        # If filtering, we need a smaller temporary index to search only those vectors
        if doc_filter:
            # rebuild a small index from selected vectors
            # (fine for portfolio; later we can optimize)
            dim = q_emb.shape[1]
            tmp = faiss.IndexFlatIP(dim)

            # read vectors from main index by reconstruct (works for Flat indexes)
            vecs = [index.reconstruct(i) for i in candidate_idx]
            X = np.asarray(vecs, dtype="float32")
            tmp.add(X)

            k = min(3, len(candidate_idx))
            scores, ids = tmp.search(q_emb, k=k)

            print("\nTop matches:\n")
            for rank, (local_i, s) in enumerate(zip(ids[0], scores[0]), start=1):
                global_i = candidate_idx[int(local_i)]
                c = chunks[global_i]
                print(f"{rank}) score={float(s):.4f} | {c['doc']} | page={c['page']} | {c['chunk_id']}")
                print(c["text"][:500], "...")
                print("-" * 60)

        else:
            k = min(3, len(chunks))
            scores, ids = index.search(q_emb, k=k)

            print("\nTop matches:\n")
            for rank, (i, s) in enumerate(zip(ids[0], scores[0]), start=1):
                c = chunks[int(i)]
                print(f"{rank}) score={float(s):.4f} | {c['doc']} | page={c['page']} | {c['chunk_id']}")
                snippet = c["text"][:350].replace("\n", " ")
                rint(snippet + ("..." if len(c["text"]) > 350 else ""))
                print("-" * 60)


if __name__ == "__main__":
    main()
