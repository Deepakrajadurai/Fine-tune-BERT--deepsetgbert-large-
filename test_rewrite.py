import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Load 5 human texts
human_texts = []
with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        human_texts.append(row["text"])
        if len(human_texts) >= 5:
            break

print(f"Loaded {len(human_texts)} human texts. Starting rewriting...")

for i, text in enumerate(human_texts):
    print(f"\nOriginal {i+1}:")
    print(text[:250] + "...")
    
    prompt = f"Schreibe den folgenden Ausschnitt einer Bundestagsrede in deinen eigenen Worten um. Behalte den Sinn bei, aber formuliere es neu. Schreibe nur den neu formulierten Text auf Deutsch und nichts anderes:\n\n{text}"
    messages = [
        {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
    
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    print("Rewritten:")
    print(response)
    print("-" * 50)
