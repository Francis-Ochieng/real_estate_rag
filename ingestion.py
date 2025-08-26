# ingestion.py

import os
import csv
import requests
import docx
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
from dotenv import load_dotenv   # ✅ load .env

from utils import clean_text
from retriever import FAISSRetriever  # ✅ now using FAISS retriever


# -------------------
# Load environment
# -------------------
load_dotenv()  # ensures GROQ_API_KEY from .env is available


# -------------------
# File Loaders
# -------------------

def load_pdf(path: str) -> str:
    """Extract and clean text from a PDF file."""
    try:
        reader = PdfReader(path)
        text = [p.extract_text() or "" for p in reader.pages]
        return clean_text("\n".join(text))
    except Exception as e:
        return f"[ERROR reading PDF {path}: {e}]"


def load_docx(path: str) -> str:
    """Extract and clean text from a DOCX file."""
    try:
        doc = docx.Document(path)
        full = [p.text for p in doc.paragraphs if p.text.strip()]
        return clean_text("\n".join(full))
    except Exception as e:
        return f"[ERROR reading DOCX {path}: {e}]"


def load_txt(path: str) -> str:
    """Load and clean text from a TXT file."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return clean_text(f.read())
    except Exception as e:
        return f"[ERROR reading TXT {path}: {e}]"


def load_csv(path: str) -> str:
    """Load and clean text from a CSV file."""
    try:
        rows = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for r in reader:
                if r:
                    rows.append(' | '.join(r))
        return clean_text("\n".join(rows))
    except Exception as e:
        return f"[ERROR reading CSV {path}: {e}]"


# -------------------
# URL & YouTube Loaders
# -------------------

def load_url(url: str, timeout: int = 15) -> str:
    """Fetch and clean text content from a webpage."""
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # remove scripts/styles/navigation clutter
        for s in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            s.decompose()

        text = soup.get_text(separator='\n')
        return clean_text(text)
    except Exception as e:
        return f"[ERROR fetching {url}: {e}]"


def load_youtube_transcript(url_or_id: str, languages: list = ['en']) -> str:
    """Fetch and clean transcript text from a YouTube video."""
    # Extract video ID if URL provided
    if 'youtube' in url_or_id and 'v=' in url_or_id:
        vid = url_or_id.split('v=')[-1].split('&')[0]
    elif 'youtu.be' in url_or_id:
        vid = url_or_id.split('/')[-1]
    else:
        vid = url_or_id

    try:
        tr = YouTubeTranscriptApi.get_transcript(vid, languages=languages)
        parts = [t['text'] for t in tr if t['text'].strip()]
        return clean_text("\n".join(parts))
    except Exception as e:
        return f"[ERROR fetching transcript for {vid}: {e}]"


# -------------------
# Dispatcher
# -------------------

def load_any(source: str) -> str:
    """
    Smart loader:
    - If file path → detect extension and load
    - If YouTube URL → fetch transcript
    - If http(s) → fetch webpage
    """
    if os.path.exists(source):  # Local file
        ext = os.path.splitext(source)[1].lower()
        if ext == '.pdf':
            return load_pdf(source)
        elif ext in ['.docx', '.doc']:
            return load_docx(source)
        elif ext == '.txt':
            return load_txt(source)
        elif ext == '.csv':
            return load_csv(source)
        else:
            return f"[Unsupported file type: {ext}]"

    elif source.startswith("http://") or source.startswith("https://"):
        if "youtube.com" in source or "youtu.be" in source:
            return load_youtube_transcript(source)
        else:
            return load_url(source)

    else:
        return f"[Unknown source type: {source}]"


# -------------------
# Ingestion Pipeline
# -------------------

def ingest(sources: list, collection_name: str = "real_estate"):
    """
    Ingest a list of sources into FAISS.
    - Wipes existing FAISS index before adding
    - Supports PDFs, DOCX, TXT, CSV, URLs, YouTube
    """
    retriever = FAISSRetriever(collection_name=collection_name)

    # ❌ Reset FAISS index
    retriever.reset_collection()

    for i, src in enumerate(sources):
        print(f"📥 Loading: {src}")
        text = load_any(src)
        if text and not text.startswith("[ERROR"):
            retriever.add_documents([text], source=src, ids=[f"doc_{i}"])
            print(f"✅ Added: {src}")
        else:
            print(f"⚠️ Skipped {src} due to error.")

    retriever.persist()
    print("🎉 Ingestion complete!")


if __name__ == "__main__":
    # Example run (replace with your files/URLs)
    sample_sources = [
        "docs/example.pdf",
        "docs/notes.txt",
        "https://en.wikipedia.org/wiki/Real_estate",
        "https://youtu.be/dQw4w9WgXcQ"
    ]
    ingest(sample_sources)
