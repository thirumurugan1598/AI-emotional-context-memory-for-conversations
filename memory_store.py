import numpy as np

class MemoryStore:
    def __init__(self):
        self.messages = []

    def add_message(self, speaker, text, emotion, vector):
        self.messages.append({
            "speaker": speaker,
            "text": text,
            "emotion": emotion,
            "vector": vector
        })

    def retrieve_similar(self, query_vector, top_k=3):
        if not self.messages:
            return []

        similarities = []
        for msg in self.messages:
            vec = msg["vector"]
            sim = np.dot(vec, query_vector) / (np.linalg.norm(vec) * np.linalg.norm(query_vector))
            similarities.append((sim, msg["text"]))

        similarities.sort(reverse=True)
        return [text for _, text in similarities[:top_k]]
