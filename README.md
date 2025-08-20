# Real Estate Assistant — Naive RAG Streamlit App

## What this is
A Retrieval-Augmented Generation (RAG) Streamlit app tailored for real estate workflows.  
It ingests PDFs, DOCX, TXT, CSV, webpages (scraping), and YouTube transcripts, chunks text, creates embeddings, stores them locally in Chroma, and lets you ask questions over your documents.

---

## Features
- Upload multiple files (PDF, TXT, DOCX, CSV).  
- Add website URLs and YouTube links (one per line).  
- Character-based chunking with adjustable size & overlap.  
- Embeddings with [sentence-transformers](https://www.sbert.net/) (by default).  
- Vector store: Chroma (local).  
- Optional reranker (CrossEncoder).  
- Streamlit UI: chat history, clear chats, preview documents, download transcript.  
- Shows retrieval and reranking steps and scores after ingestion.  

---

## Folder Layout

real_estate_rag/
├─ app.py # Streamlit app (main)
├─ ingestion.py # Loaders for files, URLs, YouTube
├─ embeddings.py # Embedding model wrapper
├─ retriever.py # Vector store and retrieval + reranker
├─ utils.py # Helpers (chunking, text preview)
├─ requirements.txt
└─ README.md

yaml
Copy code

---

## Quick Setup (recommended)

1. Create a Python 3.9+ virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .\.venv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
(Optional) If using OpenAI for generation, set your API key:

bash
Copy code
export OPENAI_API_KEY="sk-..."
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Use the sidebar to upload files, paste URLs/YouTube links, configure chunking, then click Ingest.
After ingestion, you can ask questions in the chat box.

Notes & Limitations
By default, the app uses sentence-transformers/all-MiniLM-L6-v2 for embeddings (local/inference). The model downloads weights on first run.

Generation uses OpenAI if OPENAI_API_KEY is set. If not, the app will still show retrieval, but generation falls back to a very small local Hugging Face model (not recommended for production).

Reranking is optional and requires cross-encoder, which downloads additional weights.

Example Queries
"Which properties in the document are within 2 km of the CBD?"

"Summarize the rental trends mentioned in the uploaded reports."

"List the key amenities for Apartment X and its monthly rent."

Further Improvements
Add authentication before ingesting private documents.

Add longer-document summarization and caching of embeddings.

Deploy to Streamlit Cloud or Render for a live demo.
