import os
import numpy as np
from dotenv import load_dotenv
from groq import Groq

# Load .env so GROQ_API_KEY is available
load_dotenv()

class Embedder:
    def __init__(self, model_name="nomic-embed-text"):
        """
        Default model: nomic-embed-text (fast embeddings via Groq).
        """
        self.model_name = model_name
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables or .env file.")
        self.client = Groq(api_key=groq_api_key)

    def embed(self, texts):
        """
        Create embeddings for a list of texts using Groq embeddings API.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        embeddings = [item.embedding for item in response.data]
        embs = np.array(embeddings, dtype=np.float32)

        # ensure 2D
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)
        return embs
