# retriever.py

import os
import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer  # ✅ HuggingFace embeddings
from groq import Groq  # ✅ Groq client (for LLM completions)

# ✅ Load environment variables
load_dotenv()

# Path where Chroma stores vectors
CHROMA_DIR = os.path.join(os.getcwd(), 'chroma_db')


class ChromaRetriever:
    def __init__(self, collection_name='real_estate',
                 persist=True,
                 embedding_model="all-MiniLM-L6-v2",
                 use_reranker=False):
        """
        Wrapper around ChromaDB with HuggingFace embeddings + Groq LLM.
        """
        self.should_persist = persist
        self.collection_name = collection_name

        # ✅ Initialize Chroma client
        if persist:
            self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        else:
            self.client = chromadb.EphemeralClient()

        # ✅ Create or load collection
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # ✅ Local HuggingFace embedding model
        print(f"🔎 Loading local embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        # ✅ Groq client for answers
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("❌ Missing GROQ_API_KEY in environment. Please add it to your .env file.")
        self.groq_client = Groq(api_key=api_key)

        self.use_reranker = use_reranker
        if use_reranker:
            print("⚠️ Reranker not available with Groq yet. Using vector similarity only.")

    # ---------------------------
    # Embeddings
    # ---------------------------
    def _embed(self, texts):
        """Generate embeddings with HuggingFace model."""
        try:
            return self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            print("❌ Embedding error:", e)
            return np.zeros((len(texts), 384))  # fallback size for MiniLM

    # ---------------------------
    # Add docs
    # ---------------------------
    def add_documents(self, docs, source="unknown", metadatas=None, ids=None):
        """Add documents + embeddings to Chroma."""
        if not docs:
            print("⚠️ No documents to add.")
            return

        embs = self._embed(docs).tolist()
        ids = ids or [f"{source}_{i}" for i in range(len(docs))]

        # Ensure metadata always has `source`
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
        print(f"✅ Added {len(docs)} docs from {source} into '{self.collection_name}'.")

    # ---------------------------
    # Query
    # ---------------------------
    def query(self, query_text, top_k=5, rerank_top_n=10):
        """Query vector DB with metadata-aware heuristic."""
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

        # Heuristic prioritization
        q_lower = query_text.lower()
        if "youtu" in q_lower:
            items = sorted(items, key=lambda x: (0 if "youtu" in x["meta"].get("source", "").lower() else 1, x["distance"]))
        elif "http" in q_lower or "www" in q_lower:
            items = sorted(items, key=lambda x: (0 if "url" in x["meta"].get("source", "").lower() else 1, x["distance"]))
        else:
            items = sorted(items, key=lambda x: x["distance"])

        return items[:top_k]

    # ---------------------------
    # Reset collection
    # ---------------------------
    def reset_collection(self):
        """Clear all docs from collection safely."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print(f"🗑️ Collection '{self.collection_name}' has been reset.")

    # ---------------------------
    # Persist
    # ---------------------------
    def persist(self):
        """Persist the DB if persistence is enabled."""
        if self.should_persist:
            print("✅ Chroma DB is automatically persisted with DuckDB+Parquet.")

    # ---------------------------
    # Ask Groq
    # ---------------------------
    def ask_groq(self, query_text, top_k=5):
        """
        Run RAG:
        - Query Chroma
        - Feed context into Groq LLM
        """
        results = self.query(query_text, top_k=top_k)
        context = "\n\n".join([f"Source: {r['meta'].get('source','?')}\n{r['text']}" for r in results])

        prompt = f"""You are a helpful real estate RAG assistant.
Use the context below to answer the user’s question.
If the answer is not in the context, say so clearly.

Question: {query_text}

Context:
{context}
"""

        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a real estate RAG assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        return response.choices[0].message["content"].strip(), results
