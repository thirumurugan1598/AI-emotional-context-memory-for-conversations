from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def encode(self, text):
        return self.model.encode(text)

    def fuse_vectors(self, text_vec, emotion_vec):
        return np.concatenate([text_vec, emotion_vec])
