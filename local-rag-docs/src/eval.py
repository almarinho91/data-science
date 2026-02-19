from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.answering import pick_best_sentences

INDEX_DIR = Path("indexes")


QUESTIONS = [
    # general PDF checks
    {
        "q": "What is the main topic of this paper?",
        "doc": "fphy-12-1257512.pdf",
    },
    {
        "q": "Which models are compared in this paper?",
        "doc": "fphy-12-1257512.pdf",
        "must_contain_any": ["U-Net", "nnU-Net", "HR-Net", "U2-Net", "U²-Net"],
    },
    {
        "q": "What metric is used to evaluate segmentation quality?",
        "doc": "fphy-12-1257512.pdf",
        "must_contain_any": ["IoU", "intersection over union", "Dice", "mIoU"],
    },
    {
        "q": "What imaging modality or data is used?",
        "doc": "fphy-12-1257512.pdf",
        "must_contain_any": ["micro", "tomography", "CT", "µCT", "micro-computed"],
    },
    {
        "q": "What is the application domain of the study?",
        "doc": "fphy-12-1257512.pdf",
        "must_contain_any": ["biomedical", "implant", "bone", "segmentation"],
    },
]


def _load():
    index = faiss.read_index(str(INDEX_DIR / "docs.index"))
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    return index, chunks


def _search_global(index, chunks, q_emb: np.ndarray, k: int) -> list[dict]:
    k = min(k, len(chunks))
    scores, ids = index.search(q_emb, k=k)
    out = []
    for i, s in zip(ids[0], scores[0]):
        c = chunks[int(i)]
        out.append(
            {
                "doc": c.get("doc"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text", ""),
                "score": float(s),
            }
        )
    return out


def _search_doc(index, chunks, q_emb: np.ndarray, doc: str, k: int) -> list[dict]:
    broad_k = min(max(k * 15, 50), len(chunks))
    broad = _search_global(index, chunks, q_emb, k=broad_k)
    filtered = [r for r in broad if r["doc"] == doc]
    return filtered[:k]


def _passes_must_contain(answer: str, must_any: list[str]) -> bool:
    a = (answer or "").lower()
    return any(m.lower() in a for m in must_any)


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = _load()

    k = 5
    hit_doc_top1 = 0
    hit_doc_top3 = 0
    must_pass = 0
    must_total = 0

    print(f"Running eval on {len(QUESTIONS)} questions (k={k})\n")

    for i, item in enumerate(QUESTIONS, start=1):
        q = item["q"]
        doc = item.get("doc")
        must_any = item.get("must_contain_any")

        q_emb = model.encode(q, normalize_embeddings=True).astype("float32").reshape(1, -1)

        if doc:
            top_chunks = _search_doc(index, chunks, q_emb, doc=doc, k=k)
        else:
            top_chunks = _search_global(index, chunks, q_emb, k=k)

        answer, sources = pick_best_sentences(model, q, top_chunks, top_sentences=2)

        top_docs = [r["doc"] for r in top_chunks]
        top1_ok = (doc is None) or (top_docs[0] == doc if top_docs else False)
        top3_ok = (doc is None) or (doc in top_docs[:3])

        hit_doc_top1 += 1 if top1_ok else 0
        hit_doc_top3 += 1 if top3_ok else 0

        if must_any:
            must_total += 1
            if _passes_must_contain(answer, must_any):
                must_pass += 1

        print(f"[{i}] Q: {q}")
        print("Answer:", (answer[:220] + "...") if len(answer) > 220 else answer)
        if sources:
            s0 = sources[0]
            doc = s0.get("doc")
            page = s0.get("page")
            chunk_id = s0.get("chunk_id")

            # works for both modes: rerank mode OR targeted extraction mode
            sent_sim = s0.get("sentence_similarity")
            chunk_score = s0.get("chunk_score")
            final_score = s0.get("final_score")

            extra = []
            if sent_sim is not None:
                extra.append(f"sent_sim={float(sent_sim):.3f}")
            if final_score is not None:
                extra.append(f"final={float(final_score):.3f}")
            if chunk_score is not None:
                extra.append(f"chunk_score={float(chunk_score):.3f}")

            extra_str = (" | " + " ".join(extra)) if extra else ""
            print(f"Source: {doc} page={page} chunk={chunk_id}{extra_str}")
        print("Top docs:", top_docs[:3])
        print("-" * 80)

    n = len(QUESTIONS)
    print("\nSummary:")
    print(f"doc hit@1: {hit_doc_top1}/{n} = {hit_doc_top1/n:.2%}")
    print(f"doc hit@3: {hit_doc_top3}/{n} = {hit_doc_top3/n:.2%}")
    if must_total:
        print(f"must-contain pass rate: {must_pass}/{must_total} = {must_pass/must_total:.2%}")


if __name__ == "__main__":
    main()
