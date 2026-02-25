from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path

from src.llm_openai import openai_generate
from src.prompting import build_prompt

LOG = logging.getLogger("ragdocs")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(name)s | %(message)s")


def _looks_like_references(text: str) -> bool:
    """Heuristic filter to drop bibliography / references-like chunks."""
    t = (text or "").strip()
    if len(t) < 180:
        return True

    low = t.lower()

    # Common signals of references / bibliography
    if "references" in low:
        return True
    if "doi:" in low:
        return True
    if re.search(r"\b(et al\.?|pp\.|vol\.|no\.)\b", low):
        # not always bad, but often references; keep it conservative
        # We'll only drop if also looks citation-heavy
        if re.search(r"\b(19\d{2}|20\d{2})\b", low) and len(re.findall(r"\b\d{4}\b", low)) >= 2:
            return True

    # Citation-heavy: lots of years and commas, few verbs
    years = len(re.findall(r"\b(19\d{2}|20\d{2})\b", low))
    if years >= 3 and "," in t:
        return True

    return False


def cmd_ingest(_: argparse.Namespace) -> int:
    """Build the FAISS index + chunks.json from files in data/docs."""
    from src.ingest import main as ingest_main

    ingest_main()
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the local index from the terminal."""
    import json

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from src.answering import pick_best_sentences

    index_dir = Path(args.index_dir)
    idx_path = index_dir / "docs.index"
    chunks_path = index_dir / "chunks.json"

    if not idx_path.exists() or not chunks_path.exists():
        LOG.error("Index not found. Run: python -m src.cli ingest")
        return 2

    model = SentenceTransformer(args.model)
    index = faiss.read_index(str(idx_path))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    q_emb = model.encode(args.question, normalize_embeddings=True).astype("float32")
    q_emb = np.expand_dims(q_emb, axis=0)

    # If doc filter is set, search broader then filter
    search_k = args.k
    if args.doc:
        search_k = min(max(args.k * 15, 50), len(chunks))

    scores, ids = index.search(q_emb, k=min(search_k, len(chunks)))

    top_chunks: list[dict] = []
    for i, s in zip(ids[0], scores[0]):
        if int(i) < 0:
            continue
        c = chunks[int(i)]
        if args.doc and c.get("doc") != args.doc:
            continue

        top_chunks.append(
            {
                "doc": c.get("doc"),
                "page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "text": c.get("text", ""),
                "score": float(s),
            }
        )
        if len(top_chunks) >= args.k:
            break

    if not top_chunks:
        LOG.warning("No results found.")
        return 0

    # --- Filter low-quality chunks (especially for LLM mode) ---
    filtered_chunks = [c for c in top_chunks if not _looks_like_references(c.get("text", ""))]
    # If we filtered too aggressively, fall back to original
    if len(filtered_chunks) >= max(2, min(3, args.k)):
        top_chunks = filtered_chunks

    # --- Answering ---
    if args.use_llm:
        if not os.environ.get("OPENAI_API_KEY"):
            LOG.error("OPENAI_API_KEY not set. Put it in .env or environment.")
            return 3

        prompt = build_prompt(args.question, top_chunks)
        answer = openai_generate(prompt, model=args.openai_model)
        sources = top_chunks  # in LLM mode, we use retrieved chunks as sources
    else:
        answer, sources = pick_best_sentences(
            model,
            args.question,
            top_chunks,
            top_sentences=args.top_sentences,
        )

    print("\nAnswer:\n")
    print(answer if answer else "(no answer found)")

    print("\nSources:\n")
    for n, s in enumerate(sources or [], start=1):
        doc = s.get("doc", "?")
        page = s.get("page", "?")
        chunk_id = s.get("chunk_id", "?")

        sent_sim = s.get("sentence_similarity", None)
        chunk_score = s.get("chunk_score", None)
        score = s.get("score", None)

        sent_sim_str = f"{sent_sim:.4f}" if isinstance(sent_sim, (int, float)) else "n/a"
        # if chunk_score missing (LLM mode), fall back to FAISS score
        score_val = chunk_score if isinstance(chunk_score, (int, float)) else score
        score_str = f"{score_val:.4f}" if isinstance(score_val, (int, float)) else "n/a"

        print(f"{n}) {doc} | page={page} | {chunk_id} | sent_sim={sent_sim_str} | score={score_str}")

        snippet = (s.get("sentence") or s.get("text") or "").strip()
        if len(snippet) > 240:
            snippet = snippet[:240] + "..."
        if snippet:
            print("   ", snippet)

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="debug" if args.verbose else "info",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragdocs",
        description="Local RAG over your own docs (FAISS + sentence-transformers).",
    )
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")

    sub = p.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest documents from data/docs into indexes/")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="Query the local index")
    p_query.add_argument("--question", "-q", required=True, help="Question to ask")
    p_query.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    p_query.add_argument("--doc", default=None, help="Optional filter by exact filename")
    p_query.add_argument("--top-sentences", type=int, default=2, help="Sentences to return")
    p_query.add_argument(
        "--model",
        default=os.environ.get("RAGDOCS_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformer model name (env: RAGDOCS_MODEL)",
    )
    p_query.add_argument("--index-dir", default="indexes", help="Where docs.index + chunks.json live")

    # LLM options
    p_query.add_argument("--use-llm", action="store_true", help="Use OpenAI LLM to generate final answer")
    p_query.add_argument(
        "--openai-model",
        default=os.environ.get("RAGDOCS_OPENAI_MODEL", "gpt-4.1-mini"),
        help="OpenAI model name (env: RAGDOCS_OPENAI_MODEL)",
    )
    p_query.set_defaults(func=cmd_query)

    p_serve = sub.add_parser("serve", help="Run the FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    p_serve.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())