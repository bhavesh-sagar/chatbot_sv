import faiss
import numpy as np

class FaissStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = faiss.IndexFlatL2(384)
        self.docs = []

    def add(self, docs):
        embs = self.embedder.embed(docs).astype("float32")
        self.index.add(embs)
        self.docs.extend(docs)

    def search(self, query, k=3):
        q = self.embedder.embed([query]).astype("float32")
        D, I = self.index.search(q, k)
        return [self.docs[i] for i in I[0] if i < len(self.docs)]
