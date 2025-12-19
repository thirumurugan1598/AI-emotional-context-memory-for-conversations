from transformers import AutoModelForCausalLM, AutoTokenizer
from deep_translator import GoogleTranslator
import torch

class Generator:
    def __init__(self):
        # 💡 Use instruction-following, emotion-aware model
        self.model_name = "microsoft/Phi-3-mini-4k-instruct"
        print(f"Loading model: {self.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float32)
        print("Model loaded successfully!")

    def translate_to_english(self, text):
        """Translate any input text to English for consistent understanding."""
        try:
            return GoogleTranslator(source='auto', target='en').translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text  

    def translate_back(self, text, target_lang="auto"):
        """Translate generated reply back to user's original language."""
        try:
            return GoogleTranslator(source='en', target=target_lang).translate(text)
        except Exception as e:
            print(f"Back translation error: {e}")
            return text

    def generate_reply(self, user_message, past_context, emotion, lang="auto"):
        """Generate an empathetic, context-aware reply."""
        user_message_en = self.translate_to_english(user_message)

        # 🧠 Stronger, clearer instruction for emotional empathy
        prompt = (
            f"You are an empathetic and emotionally intelligent AI friend.\n"
            f"The user feels {emotion}. Their message: '{user_message_en}'.\n"
            f"Respond with warmth, encouragement, and emotional understanding. "
            f"Keep the tone kind, short, and conversational."
        )

        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode and clean up
        response_en = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_en = response_en.split("User:")[0].split("Assistant:")[-1].strip()

        # Translate back to user’s language
        final_reply = self.translate_back(response_en, target_lang=lang)
        return final_reply.strip()
