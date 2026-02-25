from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.answering import pick_best_sentences
from src.llm_openai import openai_generate
from src.prompting import build_prompt

INDEX_DIR = Path(os.environ.get("RAGDOCS_INDEX_DIR", "indexes"))

app = FastAPI(title="Local RAG Docs", version="0.1.0")

# Lazy-loaded globals
_model: Optional[SentenceTransformer] = None
_index = None
_chunks: Optional[list[dict]] = None
_chunk_by_id: Optional[dict[int, dict]] = None  # idx -> chunk dict


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.environ.get("RAGDOCS_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def load_store():
    global _index, _chunks, _chunk_by_id
    idx_path = INDEX_DIR / "docs.index"
    chunks_path = INDEX_DIR / "chunks.json"

    if _index is None:
        if not idx_path.exists():
            raise FileNotFoundError(f"Index not found: {idx_path}")
        _index = faiss.read_index(str(idx_path))

    if _chunks is None:
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks not found: {chunks_path}")
        _chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    if _chunk_by_id is None:
        # stable mapping: FAISS row id -> chunk payload
        _chunk_by_id = {i: c for i, c in enumerate(_chunks)}


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    k: int = Field(5, ge=1, le=25)
    doc: Optional[str] = None  # optional filter by filename (exact match)

    use_llm: bool = False
    openai_model: str = Field(default_factory=lambda: os.environ.get("RAGDOCS_OPENAI_MODEL", "gpt-4.1-mini"))


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
    try:
        load_store()
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model = get_model()

    q_emb = model.encode(req.question, normalize_embeddings=True).astype("float32")
    q_emb = np.expand_dims(q_emb, axis=0)

    # If doc filter is set, search broader and filter down
    search_k = req.k
    if req.doc:
        search_k = min(max(req.k * 15, 50), len(_chunks))

    scores, ids = _index.search(q_emb, k=min(search_k, len(_chunks)))

    # build retrieved chunk dicts (with text) directly from FAISS ids (no scanning)
    retrieved: list[dict] = []
    for i, s in zip(ids[0], scores[0]):
        i = int(i)
        if i < 0:
            continue
        raw = _chunk_by_id[i]
        if req.doc and raw.get("doc") != req.doc:
            continue
        retrieved.append(
            {
                "doc": raw.get("doc"),
                "page": raw.get("page"),
                "chunk_id": raw.get("chunk_id"),
                "text": raw.get("text", ""),
                "score": float(s),
            }
        )
        if len(retrieved) >= req.k:
            break

    if not retrieved:
        return {"answer": "I don't know.", "sources": [], "top_chunks": []}

    # Answering: either LLM (RAG) or heuristic sentence picker
    if req.use_llm:
        if not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set (use .env or env var).")

        prompt = build_prompt(req.question, retrieved)
        answer = openai_generate(prompt, model=req.openai_model)

        # in LLM mode, we return the retrieved chunks as sources
        sources = [
            {
                "doc": c["doc"],
                "page": c["page"],
                "chunk_id": c["chunk_id"],
                "retrieval_score": c["score"],
                "sentence_similarity": None,
                "final_score": None,
                "snippet": (c.get("text", "")[:220] + "...") if c.get("text") else None,
            }
            for c in retrieved[: min(req.k, len(retrieved))]
        ]
    else:
        answer, picked = pick_best_sentences(model, req.question, retrieved, top_sentences=2)

        sources = []
        for s in picked:
            sources.append(
                {
                    "doc": s.get("doc"),
                    "page": s.get("page"),
                    "chunk_id": s.get("chunk_id"),
                    "retrieval_score": None,  # not carried by pick_best_sentences; retrieval is in `retrieved`
                    "sentence_similarity": s.get("sentence_similarity"),
                    "final_score": s.get("final_score"),
                    "snippet": (s.get("sentence") or "")[:220] if s.get("sentence") else None,
                }
            )

    # top_chunks for debugging/clients
    top_chunks = [
        {
            "doc": c["doc"],
            "page": c["page"],
            "chunk_id": c["chunk_id"],
            "retrieval_score": c["score"],
            "sentence_similarity": None,
            "final_score": None,
            "snippet": (c.get("text", "")[:220] + "...") if c.get("text") else None,
        }
        for c in retrieved[: min(req.k, len(retrieved))]
    ]

    return {"answer": answer, "sources": sources, "top_chunks": top_chunks}