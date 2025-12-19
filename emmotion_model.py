from transformers import pipeline

class EmotionModel:
    def __init__(self):
        self.model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

    def predict(self, text):
        """
        Predict emotion for a given text
        Returns: (emotion_label, [score]) -> (str, list)
        """
        result = self.model(text)[0]
        emotion = result['label']
        score = result['score']
        return emotion, [score]
