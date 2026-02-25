from __future__ import annotations

import re

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_TERMS = [
    "U-Net", "nnU-Net", "HR-Net", "U2-Net", "U²-Net", "V-Net", "ResUNet", "UNet++", "DeepLab",
]
METRIC_TERMS = [
    "IoU", "mIoU", "Dice", "F1", "Precision", "Recall", "Accuracy", "MAE", "RMSE",
]
DATA_TERMS = [
    "µCT", "micro-CT", "micro CT", "tomography", "SRµCT", "CT", "micro-computed tomography",
]

STOP_PHRASES = [
    "data availability statement",
    "writing–review and editing",
    "conflict of interest",
    "publisher",
    "funding",
    "horizon 2020",
    "grant",
    "doi:",
]


def _contains_stop_phrase(s: str) -> bool:
    low = s.lower()
    return any(p in low for p in STOP_PHRASES)


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _intent(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ["which models", "what models", "compared models", "models are compared", "architectures"]):
        return "models"
    if any(x in q for x in ["metric", "evaluate", "evaluation", "quality", "score"]):
        return "metrics"
    if any(x in q for x in ["imaging", "modality", "data is used", "what data", "tomography", "µct", "micro-ct"]):
        return "data"
    return "generic"


def _extract_terms_from_text(text: str, terms: list[str]) -> list[str]:
    t = _normalize_space(text)
    found: list[str] = []
    for term in terms:
        # flexible matching (ignore case, allow missing hyphen/space variants)
        pattern = re.escape(term).replace("\\-", "[- ]?")
        if re.search(pattern, t, flags=re.IGNORECASE):
            found.append(term)

    # unique, stable order
    seen = set()
    out: list[str] = []
    for x in found:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def split_sentences(text: str) -> list[str]:
    t = _normalize_space(text)
    if not t:
        return []

    parts = re.split(r"(?<=[.!?])\s+", t)
    out: list[str] = []
    for s in parts:
        s = s.strip()
        if len(s) < 30:
            continue
        if s.count(" ") < 6 and len(s) > 80:
            # likely "glued PDF" noise
            continue
        if _contains_stop_phrase(s):
            continue
        out.append(s)
    return out


def _sentence_rerank(
    model: SentenceTransformer,
    question: str,
    chunks: list[dict],
    top_sentences: int = 2,
    max_sentences_per_chunk: int = 10,
) -> tuple[str, list[dict]]:
    q = question.strip()
    q_emb = model.encode(q, normalize_embeddings=True).astype("float32")

    candidates: list[dict] = []
    for c in chunks:
        sents = split_sentences(c.get("text", ""))[:max_sentences_per_chunk]
        for s in sents:
            candidates.append(
                {
                    "sentence": s,
                    "doc": c.get("doc"),
                    "page": c.get("page"),
                    "chunk_id": c.get("chunk_id"),
                    "chunk_score": float(c.get("score", 0.0)),
                }
            )

    if not candidates:
        if chunks:
            snippet = _normalize_space(chunks[0].get("text", ""))[:260]
            # keep schema consistent even in fallback
            return snippet, [
                {
                    "doc": chunks[0].get("doc"),
                    "page": chunks[0].get("page"),
                    "chunk_id": chunks[0].get("chunk_id"),
                    "chunk_score": float(chunks[0].get("score", 0.0)),
                    "sentence_similarity": None,
                    "final_score": None,
                    "sentence": snippet,
                }
            ]
        return "", []

    sent_texts = [x["sentence"] for x in candidates]
    X = np.asarray(
        model.encode(sent_texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False),
        dtype="float32",
    )

    sims = X @ q_emb
    for x, sim in zip(candidates, sims):
        x["similarity"] = float(sim)
        x["final_score"] = float(sim) + 0.05 * x["chunk_score"]

    candidates.sort(key=lambda r: r["final_score"], reverse=True)
    picked = candidates[:top_sentences]

    answer = " ".join([p["sentence"] for p in picked])

    sources: list[dict] = []
    for p in picked:
        sources.append(
            {
                "doc": p["doc"],
                "page": p["page"],
                "chunk_id": p["chunk_id"],
                "sentence_similarity": p["similarity"],
                "final_score": p["final_score"],
                "chunk_score": p["chunk_score"],
                "sentence": p["sentence"],
            }
        )
    return answer, sources


def pick_best_sentences(
    model: SentenceTransformer,
    question: str,
    chunks: list[dict],
    top_sentences: int = 2,
) -> tuple[str, list[dict]]:
    q = (question or "").strip()
    if not q:
        return "", []

    intent = _intent(q)

    # Targeted extraction: deterministic answers for specific intents
    if intent in {"models", "metrics", "data"}:
        terms = MODEL_TERMS if intent == "models" else METRIC_TERMS if intent == "metrics" else DATA_TERMS

        found: list[str] = []
        best_sources: list[dict] = []

        for c in chunks:
            hits = _extract_terms_from_text(c.get("text", ""), terms)
            if hits:
                found.extend(hits)
                best_sources.append(
                    {
                        "doc": c.get("doc"),
                        "page": c.get("page"),
                        "chunk_id": c.get("chunk_id"),
                        "chunk_score": float(c.get("score", 0.0)),
                        "sentence_similarity": None,
                        "final_score": None,
                        "sentence": _normalize_space(c.get("text", ""))[:220] + "...",
                    }
                )

        # unique found (stable order)
        uniq: list[str] = []
        seen = set()
        for x in found:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(x)

        if uniq:
            if intent == "models":
                answer = "Compared models: " + ", ".join(uniq)
            elif intent == "metrics":
                answer = "Evaluation metrics mentioned: " + ", ".join(uniq)
            else:
                answer = "Data / modality mentioned: " + ", ".join(uniq)

            return answer, best_sources[:2] if best_sources else []

        # fallback to sentence rerank if no terms found
        return _sentence_rerank(model, q, chunks, top_sentences=top_sentences)

    # generic
    return _sentence_rerank(model, q, chunks, top_sentences=top_sentences)