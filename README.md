# DocMind


DocMind is a PDF assistant built with Streamlit, FAISS, sentence-transformers, PyMuPDF, OCR fallback, and either OpenAI for hosted deployment or Ollama for local/private use. Upload one or more PDFs, index their contents, then ask grounded questions, generate summaries, notes, and flashcards.

Features
--------

- Multi-PDF upload and indexing
- Local login and sign-up
- Per-user chat history
- Hybrid semantic and keyword retrieval
- Hosted OpenAI-backed answers for deployment
- Optional local Ollama-backed answers
- OCR fallback for scanned pages
- Structured extraction helpers for form-like PDFs
- Saved chat history and persisted FAISS index

Project Layout
--------------

```text
pdf-ai-assistant/
  app.py              Streamlit UI
  ingest.py           PDF extraction, chunking, FAISS index build
  retriever.py        FAISS and BM25 retrieval
  llm.py              Ollama API wrapper and prompts
  utils.py            Chat/index persistence helpers
  auth.py             Local user creation and login helpers
  config.py           Shared paths and app settings
  data/users.json     Generated local user store
  data/chats/         Generated per-user chat history
  data/indexes/       Generated per-user index files
  data/faiss_index/   Legacy/generated index files
```

Run Locally
-----------

From this directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

For hosted OpenAI answers, add your API key to `.streamlit/secrets.toml`:

```toml
[openai]
api_key = "sk-..."
```

For local Ollama answers, start Ollama separately:

```bash
ollama serve
ollama pull mistral
```

Then open the Streamlit URL, upload PDFs, and click `Index Papers`.

Deploy To Streamlit Community Cloud
-----------------------------------

1. Push this project to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set the app entry point to:

```text
pdf-ai-assistant/app.py
```

4. Add this secret in the Streamlit app settings:

```toml
[openai]
api_key = "sk-..."
```

5. Deploy the app.

The deployed app should use the `openai` provider in the sidebar. Ollama is still available for local use, but it is not required for cloud deployment.

Accounts
--------

DocMind stores local accounts in `data/users.json` with salted password hashes. Chat history is saved per user in `data/chats/`, and document indexes are saved per user in `data/indexes/`. This is suitable for demos and small private deployments, but larger production deployments should use a real identity provider and managed database/object storage.

Run With Docker
---------------

```bash
docker compose up --build
```

The compose file starts both Ollama and the Streamlit app. The FAISS index is persisted under `data/`.

Notes
-----

- Generated files such as `chat_history.json`, `.streamlit/secrets.toml`, `data/users.json`, `data/chats/`, `data/indexes/`, `data/faiss_index/`, `__pycache__/`, and local virtual environments should not be committed.
- If OCR is needed outside Docker, install the Tesseract system package in addition to the `pytesseract` Python package.
- The default embedding model is configured in `config.py`.
