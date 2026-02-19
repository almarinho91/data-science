from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.loaders import iter_document_pages
from src.utils import clean_pdf_text, chunk_by_paragraphs

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks: list[dict] = []

    global_chunk_n = 0

    def looks_like_table(t: str) -> bool:
        letters = sum(ch.isalpha() for ch in t)
        digits = sum(ch.isdigit() for ch in t)
        return digits > 80 and letters < 80

    for dp in iter_document_pages(DOCS_DIR):
        raw = (dp.text or "").strip()
        cleaned = clean_pdf_text(raw)

        if dp.page is None:
            # txt still goes through cleanup safely
            pass

        print(f"Loaded {dp.doc} page={dp.page}: raw={len(raw)} chars | cleaned={len(cleaned)} chars")

        if dp.doc.lower().endswith(".pdf") and len(cleaned) < 800:
            print(f"WARNING: very small extracted text for {dp.doc} page {dp.page}")

        if not cleaned:
            continue

        chunks = chunk_by_paragraphs(cleaned, target_size=800, overlap=120)

        for c in chunks:
            if looks_like_table(c.text):
                continue
            all_chunks.append(
                {
                    "doc": dp.doc,
                    "page": dp.page,
                    "chunk_id": f"chunk_{global_chunk_n}",
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                    "text": c.text,
                }
            )

            global_chunk_n += 1

    if not all_chunks:
        raise RuntimeError("No chunks produced. Check data/docs and PDF extraction.")

    print(f"Total chunks: {len(all_chunks)}")

    dim = 384
    index = faiss.IndexFlatIP(dim)

    batch_size = 64
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding + indexing"):
        batch_texts = [c["text"] for c in all_chunks[i : i + batch_size]]
        emb = model.encode(batch_texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        X = np.asarray(emb, dtype="float32")
        index.add(X)

    faiss.write_index(index, str(INDEX_DIR / "docs.index"))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(all_chunks, ensure_ascii=False), encoding="utf-8")

    print(f"Indexed {len(all_chunks)} chunks")
    print(f"Saved: {INDEX_DIR / 'docs.index'}")
    print(f"Saved: {INDEX_DIR / 'chunks.json'}")


if __name__ == "__main__":
    main()
