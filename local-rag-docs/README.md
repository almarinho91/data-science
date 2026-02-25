# Local RAG Docs

Tiny, **fully local** RAG (Retrieval-Augmented Generation) for your own
`.pdf` / `.txt` files.

Pipeline: - chunk docs → embed with `sentence-transformers` - store
vectors in FAISS - query → retrieve top-k chunks - answer via: -
sentence selection (default, no LLM) - optional OpenAI LLM (`--use-llm`)

------------------------------------------------------------------------

## Install

``` bash
python -m venv .venv
# Windows:
#   .venv\Scripts\activate
# macOS/Linux:
#   source .venv/bin/activate

pip install -r requirements.txt
```

Optional (nice CLI name):

``` bash
pip install -e .
ragdocs --help
```

------------------------------------------------------------------------

## Put your docs

Drop files into:

-   `data/docs/*.pdf`
-   `data/docs/*.txt`

This repo includes:

-   `data/docs/sample.txt`
-   a small demo PDF (`data/docs/fphy-12-1257512.pdf`)

------------------------------------------------------------------------

## CLI usage

Build the index:

``` bash
python -m src.cli ingest
```

Ask a question (no LLM, deterministic sentence selection):

``` bash
python -m src.cli query -q "Which models are compared?"
```

Use OpenAI for answer generation (RAG mode):

``` bash
python -m src.cli query -q "Which models are compared?" --use-llm
```

You must set:

``` bash
OPENAI_API_KEY=your_key_here
```

------------------------------------------------------------------------

## Run the API

``` bash
python -m src.cli serve
# open: http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## API usage (curl)

``` bash
curl -X POST "http://127.0.0.1:8000/query"   -H "Content-Type: application/json"   -d '{"question":"Which models are compared?","k":8}'
```

With LLM:

``` bash
curl -X POST "http://127.0.0.1:8000/query"   -H "Content-Type: application/json"   -d '{"question":"Which models are compared?","k":8,"use_llm":true}'
```

------------------------------------------------------------------------

## Notes / limitations

-   PDF extraction depends on layout. Tables, captions, and multi-column
    papers can produce messy text.
-   LLM mode requires an OpenAI API key.
-   Retrieval quality depends heavily on:
    -   chunking strategy
    -   embedding model choice
    -   how clean the extracted PDF text is
