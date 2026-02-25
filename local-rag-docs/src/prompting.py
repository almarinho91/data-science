from __future__ import annotations


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build an LLM prompt with explicit, citeable source IDs [S1], [S2], ...

    Expected chunk dict keys:
      - doc
      - page
      - text
    """
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        doc = c.get("doc", "?")
        page = c.get("page", "?")
        text = c.get("text", "") or ""
        context_blocks.append(f"[S{i}] {doc} | page {page}\n{text}")

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a scientific research assistant.

Answer the question using ONLY the information in the context.
If the answer is not contained in the context, say: "I don't know."

Rules:
- Do NOT copy raw text verbatim; paraphrase into clean, readable English.
- Be concise (2–4 sentences).
- Do not invent details that are not supported by the context.
- Cite sources using the bracket IDs like [S1], [S2] that appear in the context.
- The Sources line must include ONLY these bracket IDs, comma-separated.

Context:
{context}

Question:
{question}

Return format (strict):

Answer:
<your answer>

Sources:
[S#], [S#]
"""
    return prompt.strip()