from __future__ import annotations

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def openai_generate(prompt: str, model: str = "gpt-4.1-mini") -> str:
    """
    Simple OpenAI completion wrapper.
    """

    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers ONLY using the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()