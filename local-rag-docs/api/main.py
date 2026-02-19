from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.answering import pick_best_sentences

INDEX_DIR = Path("indexes")

app = FastAPI(title="Local RAG Docs", version="0.1.0")

# Lazy-loaded globals
_model: Optional[SentenceTransformer] = None
_index = None
_chunks: Optional[list[dict]] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def load_store():
    global _index, _chunks
    if _index is None:
        _index = faiss.read_index(str(INDEX_DIR / "docs.index"))
    if _chunks is None:
        _chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    k: int = Field(5, ge=1, le=25)
    doc: Optional[str] = None  # optional filter by filename (exact match)


class Source(BaseModel):
    doc: str
    page: Optional[int] = None
    chunk_id: str
    retrieval_score: Optional[float] = None
    sentence_similarity: Optional[float] = None
    final_score: Optional[float] = None
    snippet: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    top_chunks: list[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    load_store()
    model = get_model()

    q_emb = model.encode(req.question, normalize_embeddings=True).astype("float32")
    q_emb = np.expand_dims(q_emb, axis=0)

    scores, ids = _index.search(q_emb, k=req.k)

    # build top chunks
    top_chunks = []
    for i, s in zip(ids[0], scores[0]):
        if int(i) < 0:
            continue
        c = _chunks[int(i)]
        top_chunks.append(
            {
                "doc": c.get("doc"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "retrieval_score": float(s),
                "sentence": None,
                "sentence_similarity": None,
                "final_score": None,
            }
        )

    # optional filter by doc
    filtered = top_chunks
    if req.doc:
        filtered = [c for c in top_chunks if c.get("doc") == req.doc]

    # pass to answer extractor; it expects chunks with text + score
    chunk_dicts = []
    for c in filtered:
        # find original chunk record by (doc, chunk_id, page) match
        # simplest: scan chunks list by index again (small)
        match = None
        for raw in _chunks:
            if raw.get("doc") == c["doc"] and raw.get("chunk_id") == c["chunk_id"] and raw.get("page") == c.get("page"):
                match = raw
                break
        if match is None:
            continue
        chunk_dicts.append(
            {
                "doc": match.get("doc"),
                "page": match.get("page"),
                "chunk_id": match.get("chunk_id"),
                "text": match.get("text"),
                "score": c["retrieval_score"],
            }
        )

    answer, sources = pick_best_sentences(model, req.question, chunk_dicts, top_sentences=2)

    # format response
    out_sources = []
    for s in sources:
        out_sources.append(
            {
                "doc": s.get("doc"),
                "page": s.get("page"),
                "chunk_id": s.get("chunk_id"),
                "retrieval_score": s.get("retrieval_score"),
                "sentence_similarity": s.get("sentence_similarity"),
                "final_score": s.get("final_score"),
                "snippet": (s.get("sentence") or "")[:220] if s.get("sentence") else None,
            }
        )

    return {
        "answer": answer,
        "sources": out_sources,
        "top_chunks": filtered[: req.k],
    }
