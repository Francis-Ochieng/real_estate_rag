# Real Estate Assistant — Naive RAG Streamlit App

## What this is

A Retrieval-Augmented Generation (RAG) Streamlit app tailored for real estate workflows. It ingests PDFs, DOCX, TXT, CSV, webpages (scraping), and YouTube transcripts, chunks text, creates embeddings, stores them locally in Chroma, and lets you ask questions over your documents.

**Note:** This version uses **Groq** for generation instead of OpenAI.

---

## Features

* Upload multiple files (PDF, TXT, DOCX, CSV).
* Add website URLs and YouTube links (one per line).
* Character-based chunking with adjustable size & overlap.
* Embeddings with sentence-transformers (by default).
* Vector store: Chroma (local).
* Optional reranker (CrossEncoder).
* Streamlit UI: chat history, clear chats, preview documents, download transcript.
* Shows retrieval and reranking steps and scores after ingestion.

---

## Folder layout

```
real_estate_rag/
├─ app.py                # Streamlit app (main)
├─ ingestion.py          # Loaders for files, URLs, YouTube
├─ embeddings.py         # Embedding model wrapper
├─ retriever.py          # Vector store and retrieval + reranker
├─ utils.py              # helpers (chunking, text preview)
├─ requirements.txt
└─ README.md
```

---

## Quick setup (recommended)

Create a Python 3.9+ virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

(Optional) If using **Groq for generation**, make sure you have the credentials set up as per your Groq account.

Run the Streamlit app:

```bash
streamlit run app.py
```

Use the sidebar to upload files, paste URLs/YouTube links, configure chunking, then click **Ingest**. After ingestion you can ask questions in the chat box.

---

## Notes & Limitations

* By default, the app uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings (local/infer). This downloads model weights on first run.
* Generation uses **Groq** if configured. If not set, the app will still show retrieval, but generation will fallback to a very small local HF model if available (not recommended for production).
* Reranking is optional and requires cross-encoder which downloads additional weights.

---

## Example queries

* "Which properties in the document are within 2 km of the CBD?"
* "Summarize the rental trends mentioned in the uploaded reports."
* "List the key amenities for Apartment X and its monthly rent."

---

## Further improvements

* Add authentication before ingesting private docs.
* Add longer-document summarization and caching of embeddings.
* Deploy to Streamlit Cloud or Render for a live demo.
