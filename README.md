# Real Estate Assistant --- RAG Streamlit App (with FAISS + Groq)

## What this is

A Retrieval-Augmented Generation (RAG) Streamlit app tailored for real
estate workflows.\
It ingests PDFs, DOCX, TXT, CSV, webpages (scraping), and YouTube
transcripts, chunks text, creates embeddings, stores them locally in
**FAISS**, and lets you ask questions over your documents.

**Note:**\
- This version uses **Groq** for generation (instead of OpenAI).\
- The vector database is now **FAISS** (instead of Chroma).

------------------------------------------------------------------------

## ✨ Features

-   📂 Upload multiple files (PDF, TXT, DOCX, CSV)\
-   🌐 Add website URLs and YouTube links (one per line)\
-   🧩 Character-based chunking with adjustable size & overlap\
-   🔎 Embeddings with `sentence-transformers` (HuggingFace)\
-   📦 Vector store: **FAISS** (local, persistent)\
-   🤖 Generation with **Groq LLMs** (`llama-3.1-*`)\
-   🖥️ Streamlit UI: chat history, clear chats, preview docs, download
    transcript\
-   📊 Shows retrieved passages & similarity scores

------------------------------------------------------------------------

## 📂 Folder layout

    real_estate_rag/
    ├─ app.py                # Streamlit app (main)
    ├─ ingestion.py          # Loaders for files, URLs, YouTube
    ├─ embeddings.py         # Embedding model wrapper (Groq/HF)
    ├─ retriever.py          # FAISS-based retriever
    ├─ utils.py              # Helpers (chunking, cleaning)
    ├─ requirements.txt
    └─ README.md

------------------------------------------------------------------------

## ⚡ Quick setup

Create a Python 3.11+ virtual environment:

``` bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Set your **Groq API key** in a `.env` file:

    GROQ_API_KEY=your_groq_api_key_here

Run the Streamlit app:

``` bash
streamlit run app.py
```

Use the sidebar to upload files, paste URLs/YouTube links, configure
chunking, then click **Ingest**. After ingestion you can ask questions
in the chat box.

------------------------------------------------------------------------

## 🚀 Notes & Limitations

-   The app uses `sentence-transformers/all-MiniLM-L6-v2` locally for
    embeddings.\
    (Model weights download on first run.)\
-   Groq provides **fast inference** for LLaMA 3 models. Without a valid
    API key, generation is disabled.\
-   FAISS index is stored locally (`faiss_index/`) and persists between
    runs.\
-   Reranker option is present but **not yet supported** with Groq.

------------------------------------------------------------------------

## 🔎 Example queries

-   "Which properties in the document are within 2 km of the CBD?"\
-   "Summarize the rental trends mentioned in the uploaded reports."\
-   "List the key amenities for Apartment X and its monthly rent."

------------------------------------------------------------------------

## 🛠️ Future improvements

-   Add authentication before ingesting private docs\
-   Enable hybrid search (BM25 + FAISS)\
-   Long-document summarization & caching of embeddings\
-   Deploy to Streamlit Cloud or Render for a live demo

------------------------------------------------------------------------
