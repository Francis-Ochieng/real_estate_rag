# embeddings.py

import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()


class GroqEmbeddings:
    """
    LangChain-compatible wrapper for Groq embeddings API.
    Implements .embed_documents() and .embed_query().
    Default model: nomic-embed-text
    """

    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "❌ GROQ_API_KEY is not set in environment variables or .env file."
            )
        self.client = Groq(api_key=groq_api_key)

    def _embed(self, texts):
        """Internal helper to call Groq API."""
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        embs = np.array(embeddings, dtype=np.float32)

        # Ensure 2D
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)
        return embs.tolist()

    def embed_documents(self, texts):
        """Embed multiple documents (list of strings)."""
        return self._embed(texts)

    def embed_query(self, text):
        """Embed a single query string."""
        return self._embed([text])[0]


def get_embedding_function(provider: str = "huggingface"):
    """
    Factory to get an embedding function compatible with FAISS & LangChain.
    - "huggingface": Uses all-MiniLM-L6-v2 (local, CPU-safe)
    - "groq": Uses Groq embeddings API (nomic-embed-text)
    """
    if provider == "groq":
        return GroqEmbeddings(model_name="nomic-embed-text")
    elif provider == "huggingface":
        # ✅ Fixed: only pass model_name, not a raw SentenceTransformer instance
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"❌ Unknown embedding provider: {provider}")
