from emmotion_model import EmotionModel
from embedder import Embedder
from memory_store import MemoryStore
from generator import Generator

class ChatEngine:
    def __init__(self):
        self.emotion_model = EmotionModel()
        self.embedder = Embedder()
        self.memory = MemoryStore()
        self.generator = Generator()

    def process_message(self, message: str) -> str:
        emotion, emotion_vec = self.emotion_model.predict(message)
        embedding = self.embedder.encode(message)
        fused_vector = self.embedder.fuse_vectors(embedding, emotion_vec)
        past_context = self.memory.retrieve_similar(fused_vector)
        response = self.generator.generate_reply(message, past_context, emotion)
        self.memory.add_message("user", message, emotion, fused_vector)
        return response

    def add_to_memory(self, speaker, text):
        emotion, emotion_vec = self.emotion_model.predict(text)
        embedding = self.embedder.encode(text)
        fused_vector = self.embedder.fuse_vectors(embedding, emotion_vec)
        self.memory.add_message(speaker, text, emotion, fused_vector)
