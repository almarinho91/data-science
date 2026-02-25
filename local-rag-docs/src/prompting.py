def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []

    for c in chunks:
        context_blocks.append(
            f"[{c.get('doc')} | page {c.get('page')}]\n{c.get('text','')}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a scientific research assistant.

Answer the question using ONLY the information provided in the context.
If the answer is not contained in the context, say: "I don't know."

Important rules:
- Do NOT copy raw text verbatim.
- Rewrite the answer in clean, readable English.
- Summarize concisely (2–4 sentences max).
- Do not invent information.
- Use only facts supported by the context.

Context:
{context}

Question:
{question}

Return your answer strictly in this format:

Answer:
<your clean summarized answer>

Sources:
doc:page, doc:page
"""

    return prompt.strip()