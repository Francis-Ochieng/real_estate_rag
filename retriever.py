import os
import chromadb
from groq import Groq  # ✅ Groq client
import numpy as np

# Path where Chroma stores vectors
CHROMA_DIR = os.path.join(os.getcwd(), 'chroma_db')

class ChromaRetriever:
    def __init__(self, collection_name='real_estate',
                 persist=True,
                 embedding_model='text-embedding-3-small',
                 use_reranker=False):
        """
        Wrapper around ChromaDB with Groq embeddings.
        :param collection_name: Name of vector collection
        :param persist: Whether to persist to disk
        :param embedding_model: Groq embedding model name
        :param use_reranker: Placeholder (Groq doesn't provide reranker yet)
        """
        self.should_persist = persist
        self.embedding_model = embedding_model

        # ✅ Persistent client (avoids server mode errors)
        self.client = chromadb.Client(chromadb.config.Settings(
            chroma_api_impl="chromadb.api.local.LocalAPI",
            persist_directory=CHROMA_DIR
        ))

        # ✅ Create or load collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # ✅ Groq client (needs GROQ_API_KEY in env)
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.use_reranker = use_reranker
        if use_reranker:
            print("⚠️ Reranker not available with Groq yet. Using vector similarity only.")

    def _embed(self, texts):
        """Get embeddings from Groq API."""
        try:
            resp = self.groq_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return np.array([d.embedding for d in resp.data])
        except Exception as e:
            print("❌ Embedding error:", e)
            return np.zeros((len(texts), 384))  # fallback

    def add_documents(self, docs, source="unknown", metadatas=None, ids=None):
        """Add documents + embeddings to Chroma."""
        embs = self._embed(docs).tolist()
        ids = ids or [f"id_{i}" for i in range(len(docs))]
        if metadatas:
            for m in metadatas:
                m.setdefault("source", source)
        else:
            metadatas = [{"source": source}] * len(docs)

        self.collection.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids,
            embeddings=embs
        )

    def query(self, query_text, top_k=5, rerank_top_n=10):
        """Query vector DB with light metadata-aware prioritization."""
        q_emb = self._embed([query_text]).tolist()[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=rerank_top_n,
            include=["metadatas", "documents", "distances"]
        )

        docs = res.get("documents", [[]])[0]
        distances = res.get("distances", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]

        items = [
            {"text": d, "distance": dist, "meta": meta}
            for d, dist, meta in zip(docs, distances, metadatas)
        ]

        q_lower = query_text.lower()
        if "youtu" in q_lower:
            items = sorted(
                items,
                key=lambda x: (
                    0 if "youtu" in x["meta"].get("source", "").lower() else 1,
                    x["distance"]
                )
            )
        elif "http" in q_lower or "www" in q_lower:
            items = sorted(
                items,
                key=lambda x: (
                    0 if "url" in x["meta"].get("source", "").lower() else 1,
                    x["distance"]
                )
            )
        else:
            items = sorted(items, key=lambda x: x["distance"])

        return items[:top_k]

    def persist(self):
        """Persist the DB if persistence is enabled."""
        if self.should_persist:
            print("✅ Chroma DB is automatically persisted.")
