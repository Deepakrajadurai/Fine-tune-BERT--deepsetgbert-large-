import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = 'left'
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# Load 16 human texts
human_texts = []
with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
    reader = csv.DictReader(f)
    for row in reader:
        human_texts.append(row["text"])
        if len(human_texts) >= 16:
            break

print("Creating batched prompts...")
prompts = []
for text in human_texts:
    prompt = f"Schreibe den folgenden Ausschnitt einer Bundestagsrede in deinen eigenen Worten um. Behalte den Sinn bei, aber formuliere es neu. Schreibe nur den neu formulierten Text auf Deutsch und nichts anderes:\n\n{text}"
    messages = [
        {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
        {"role": "user", "content": prompt}
    ]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts.append(formatted_prompt)

# Tokenize batch
inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

print("Running batch generation...")
start_time = time.time()
with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Slice output and decode
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
elapsed_time = time.time() - start_time

print(f"Generated 16 rewrites in {elapsed_time:.2f} seconds.")
print(f"Generation speed: {16 / elapsed_time:.2f} texts/sec.")

for i, resp in enumerate(responses[:3]):
    print(f"\nSample {i+1} response:")
    print(repr(resp.strip()))
