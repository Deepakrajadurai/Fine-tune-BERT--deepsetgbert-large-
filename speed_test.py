import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} in float16...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token

# Load model in FP16 (or BF16 if supported) for maximum speed and lower VRAM usage
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)
model.eval()

# Load 64 human texts
human_texts = []
with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        human_texts.append(row["text"])
        if len(human_texts) >= 64:
            break

print("Creating 64 batched prompts...")
prompts = []
for text in human_texts:
    prompt = f"Schreibe den folgenden Text im typischen Stil von ChatGPT (GPT-4) in eigenen Worten um. Behalte den Sinn bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n{text}"
    messages = [
        {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(formatted_prompt)

# Tokenize batch
inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

print("Running batch generation (Batch Size = 64)...")
start_time = time.time()
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=80, # Cap at 80 tokens to speed up generation
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
elapsed_time = time.time() - start_time

print(f"Generated 64 rewrites in {elapsed_time:.2f} seconds.")
print(f"Generation speed: {64 / elapsed_time:.2f} texts/sec.")
