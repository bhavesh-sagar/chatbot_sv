from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)
