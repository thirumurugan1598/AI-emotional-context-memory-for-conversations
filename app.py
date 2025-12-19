import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use: {device}")


model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"Loading model: {model_name} ...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16 if device == "cuda" else torch.float32,  # ✅ replaced torch_dtype
    device_map="auto" if device == "cuda" else None
)

print("Model loaded successfully!\n")
print("Type 'exit' to stop chatting.\n")


while True:
    user_input = input("You: ")
    if user_input.lower().strip() == "exit":
        print("Chatbot: Goodbye! 👋")
        break

    
    inputs = tokenizer(user_input, return_tensors="pt").to(device)

    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    reply = response[len(user_input):].strip() 
    print(f"Chatbot: {reply}\n")
