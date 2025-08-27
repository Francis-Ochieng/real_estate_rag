# retriever.py

import os
import shutil
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from embeddings import get_embedding_function  # Updated embeddings.py

# Load environment variables
load_dotenv()

# Path where FAISS stores vectors
FAISS_DIR = os.path.join(os.getcwd(), "faiss_index")


class FAISSRetriever:
    def __init__(
        self,
        collection_name="real_estate",
        embedding_provider="huggingface",  # "huggingface" or "groq"
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_reranker=False,
    ):
        """
        Wrapper around FAISS with pluggable embeddings (HuggingFace or Groq) + Groq LLM.
        """
        self.collection_name = collection_name
        self.index_path = FAISS_DIR

        # -------------------------
        # Embedding function
        # -------------------------
        if embedding_provider == "huggingface":
            # CPU-only embeddings to prevent Torch meta tensor errors
            self.embedder = get_embedding_function("huggingface")
        elif embedding_provider == "groq":
            self.embedder = get_embedding_function("groq")
        else:
            raise ValueError(f"Unknown embedding provider: {embedding_provider}")

        # -------------------------
        # Groq client
        # -------------------------
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY in environment. Add it to your .env file."
            )
        self.groq_client = Groq(api_key=api_key)

        self.use_reranker = use_reranker
        if use_reranker:
            print("⚠️ Reranker not available with Groq yet. Using vector similarity only.")

        # -------------------------
        # Load FAISS index if exists
        # -------------------------
        self.vs = None
        if os.path.exists(self.index_path):
            try:
                self.vs = FAISS.load_local(
                    self.index_path,
                    self.embedder,
                    allow_dangerous_deserialization=True,
                )
                print(f"✅ Loaded existing FAISS index from {self.index_path}")
            except Exception:
                print("⚠️ Failed to load FAISS index, starting fresh.")
                self.vs = None

    # ---------------------------
    # Add documents
    # ---------------------------
    def add_documents(self, docs, source="unknown", metadatas=None):
        """Add documents + embeddings to FAISS."""
        if not docs:
            print("⚠️ No documents to add.")
            return

        if not self.vs:
            self.vs = FAISS.from_texts(texts=docs, embedding=self.embedder, metadatas=metadatas)
        else:
            new_store = FAISS.from_texts(texts=docs, embedding=self.embedder, metadatas=metadatas)
            self.vs.merge_from(new_store)

        print(f"✅ Added {len(docs)} docs from {source} into FAISS index.")

    # ---------------------------
    # Query
    # ---------------------------
    def query(self, query_text, top_k=5, rerank_top_n=10):
        """Query FAISS vector store with metadata-aware heuristic."""
        if not self.vs:
            print("⚠️ No FAISS index available.")
            return []

        results = self.vs.similarity_search_with_score(query_text, k=rerank_top_n)
        items = []
        for doc, score in results:
            items.append({"text": doc.page_content, "distance": float(score), "meta": doc.metadata or {}})

        # Heuristic prioritization
        q_lower = query_text.lower()
        if "youtu" in q_lower:
            items = sorted(
                items,
                key=lambda x: (0 if "youtu" in x["meta"].get("source", "").lower() else 1, x["distance"]),
            )
        elif "http" in q_lower or "www" in q_lower:
            items = sorted(
                items,
                key=lambda x: (0 if "http" in x["meta"].get("source", "").lower() else 1, x["distance"]),
            )
        else:
            items = sorted(items, key=lambda x: x["distance"])

        return items[:top_k]

    # ---------------------------
    # Reset collection
    # ---------------------------
    def reset_collection(self):
        """Clear all docs from FAISS safely."""
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        self.vs = None
        print(f"🗑️ FAISS index '{self.collection_name}' has been reset.")

    # ---------------------------
    # Persist
    # ---------------------------
    def persist(self):
        """Persist the FAISS index to disk."""
        if self.vs:
            self.vs.save_local(self.index_path)
            print(f"✅ FAISS index saved to {self.index_path}")

    # ---------------------------
    # Ask Groq
    # ---------------------------
    def ask_groq(self, query_text, top_k=5):
        """
        Run RAG:
        - Query FAISS
        - Feed context into Groq LLM
        """
        results = self.query(query_text, top_k=top_k)
        if not results:
            return "⚠️ No documents found in FAISS index.", []

        context = "\n\n".join(
            [f"Source: {r['meta'].get('source','?')}\n{r['text']}" for r in results]
        )

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
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        return response.choices[0].message.content.strip(), results
