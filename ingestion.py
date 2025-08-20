import os
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests
from pypdf import PdfReader
import docx
import csv
from .utils import clean_text

def load_pdf(path):
    text = []
    reader = PdfReader(path)
    for p in reader.pages:
        text.append(p.extract_text() or "")
    return clean_text("\n".join(text))

def load_docx(path):
    doc = docx.Document(path)
    full = []
    for p in doc.paragraphs:
        full.append(p.text)
    return clean_text("\n".join(full))

def load_txt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return clean_text(f.read())

def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(' | '.join(r))
    return clean_text("\n".join(rows))

def load_url(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout, headers={'User-Agent':'Mozilla/5.0'})
        soup = BeautifulSoup(r.text, 'html.parser')
        # remove scripts/styles
        for s in soup(['script','style','header','footer','nav','aside']):
            s.decompose()
        text = soup.get_text(separator='\n')
        return clean_text(text)
    except Exception as e:
        return f"[ERROR fetching {url}: {e}]"

def load_youtube_transcript(url_or_id, languages=['en']):
    # accept full url or id
    if 'youtube' in url_or_id and 'v=' in url_or_id:
        vid = url_or_id.split('v=')[-1].split('&')[0]
    else:
        vid = url_or_id
    try:
        tr = YouTubeTranscriptApi.get_transcript(vid, languages=languages)
        parts = [t['text'] for t in tr]
        return clean_text('\n'.join(parts))
    except Exception as e:
        return f"[ERROR fetching transcript for {vid}: {e}]"
