# Local RAG Docs

Tiny, **fully local** RAG (Retrieval-Augmented Generation) for your own `.pdf` / `.txt` files.

It does **retrieval + sentence selection** (no text generation):
- chunk docs → embed with `sentence-transformers`
- store vectors in FAISS
- query → retrieve top-k chunks → pick best matching sentence(s)

## Install

```bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
```

Optional (nice CLI name):

```bash
pip install -e .
ragdocs --help
```

## Put your docs

Drop files into:
- `data/docs/*.pdf`
- `data/docs/*.txt`

This repo includes:
- `data/docs/sample.txt`
- a small demo PDF (`data/docs/fphy-12-1257512.pdf`)

## CLI usage

Build the index:

```bash
python -m src.cli ingest
```

Ask a question:

```bash
python -m src.cli query -q "Which models are compared?" --doc fphy-12-1257512.pdf
```

Run the API:

```bash
python -m src.cli serve
# open: http://127.0.0.1:8000/docs
```

## API usage (curl)

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Which models are compared?","k":8,"doc":"fphy-12-1257512.pdf"}'
```

## Notes / limitations

- PDF extraction depends on layout. Tables, captions, and multi-column papers can produce messy text.
- This project does **not** generate new text with an LLM. It retrieves and selects the best-matching sentences.
- Retrieval quality depends heavily on:
  - chunking strategy
  - embedding model choice
  - how “clean” the extracted PDF text is
