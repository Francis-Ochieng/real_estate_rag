import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from .embeddings import Embedder
import numpy as np
from cross_encoder import CrossEncoder

CHROMA_DIR = os.path.join(os.getcwd(), 'chroma_db')

class ChromaRetriever:
    def __init__(self, collection_name='real_estate', persist=True, embedding_model='all-MiniLM-L6-v2', use_reranker=True):
        self.persist = persist
        self.client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=CHROMA_DIR))
        self.collection = self.client.create_collection(collection_name) if collection_name not in [c.name for c in self.client.list_collections()] else self.client.get_collection(collection_name)
        self.embedder = Embedder(embedding_model)
        self.use_reranker = use_reranker
        if use_reranker:
            try:
                self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                print('Failed to load reranker:', e)
                self.reranker = None

    def add_documents(self, docs, metadatas=None, ids=None):
        # docs: list of strings
        embs = self.embedder.embed(docs).tolist()
        self.collection.add(documents=docs, metadatas=metadatas or [{}]*len(docs), ids=ids, embeddings=embs)

    def query(self, query_text, top_k=5, rerank_top_n=10):
        q_emb = self.embedder.embed([query_text]).tolist()[0]
        res = self.collection.query(query_embeddings=[q_emb], n_results=rerank_top_n, include=['metadatas','documents','distances'])
        docs = res['documents'][0]
        distances = res['distances'][0]
        metadatas = res['metadatas'][0]
        items = []
        for d,dist,meta in zip(docs, distances, metadatas):
            items.append({'text':d, 'distance':dist, 'meta':meta})
        # optional rerank
        if self.use_reranker and self.reranker is not None:
            pairs = [[query_text, it['text']] for it in items]
            scores = self.reranker.predict(pairs)
            for it, sc in zip(items, scores):
                it['rerank_score'] = float(sc)
            items = sorted(items, key=lambda x: x['rerank_score'], reverse=True)[:top_k]
        else:
            items = sorted(items, key=lambda x: x['distance'])[:top_k]
        return items

    def persist(self):
        if self.persist:
            self.client.persist()
