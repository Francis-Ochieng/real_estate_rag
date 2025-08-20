from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        # texts: list[str]
        embs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # ensure 2D
        if embs.ndim == 1:
            embs = np.expand_dims(embs, 0)
        return embs
