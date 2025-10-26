
# 🧠 Emotional Context Memory AI Chatbot (ECM-AI)

> A next-generation conversational AI that **remembers emotional and semantic context** from past interactions — designed for *empathetic*, *personalized*, and *context-aware* conversations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![AI](https://img.shields.io/badge/AI-Emotional%20Context-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🌍 Overview

Unlike typical chatbots that only remember *text*, **ECM-AI** learns both:
- What you said (**semantic context**)  
- How you felt (**emotional context**)  

By storing these two layers of memory, ECM-AI produces emotionally intelligent and personalized replies over time.

This system combines **Natural Language Understanding**, **Emotion Detection**, **Vector Memory Storage**, and **Generative AI** to simulate a human-like “emotional recall”.

---

## 🎯 Features

✅ Real-time **emotion detection** using fine-tuned transformer  
✅ Stores **semantic + emotional embeddings** for every message  
✅ **Retrieves relevant memories** using FAISS similarity search  
✅ Generates empathetic replies using LLM with contextual prompt building  
✅ Persistent memory across sessions via SQLite + FAISS index  
✅ Built-in **Streamlit interface** for easy interaction  
✅ Modular architecture for extension or fine-tuning  

---

## 🧩 System Architecture

🧩 System Architecture Overview

The ECM-AI architecture is divided into five core layers:

1. User Interaction Layer


2. Emotion & Semantic Processing Layer


3. Memory Storage Layer


4. Memory Retrieval & Context Builder Layer


5. Response Generation Layer

---

## 🧠 How Emotional Context Memory Works

1. **Emotion Recognition** — Every user message is passed through an emotion classifier (e.g. `bhadresh-savani/roberta-base-emotion`)  
   → outputs a vector of emotion probabilities.

2. **Semantic Encoding** — Message text is converted into a 384-dimensional embedding using `sentence-transformers/all-MiniLM-L6-v2`.

3. **Vector Fusion & Storage** — Emotional and semantic vectors are concatenated and stored in **FAISS** and **SQLite**.

4. **Memory Retrieval** — When a new message arrives, FAISS finds top-k past messages with similar emotional + semantic context.

5. **Context-Aware Response Generation** — The retrieved memories and current user input are combined to form a prompt for an LLM (local or API).

6. **Response Delivery** — The AI replies empathetically, referencing previous emotional states naturally.

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Interface | Streamlit |
| Embeddings | Sentence-BERT (MiniLM) |
| Emotion Model | RoBERTa-based Emotion Classifier |
| Memory Store | SQLite + FAISS |
| Generator | Hugging Face LLM / OpenAI API |
| Language | Python 3.10+ |
| Vector Ops | NumPy, FAISS |
| Environment | Virtualenv |

---

## 🚀 Setup & Installation

`bash
# Clone repository
git clone https://github.com/<your-username>/ECM-AI.git
cd ECM-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run src/app.py   

example use cases:
🧍User: I'm really anxious about my exams.
🤖 AI: I can sense the anxiety — exams can feel heavy. Would you like me to help you plan your revision?

🧍User: Yes, I was overwhelmed last week.
🤖 AI: I remember that — you mentioned sleepless nights before. Let’s make this week calmer with short study blocks and breaks.
