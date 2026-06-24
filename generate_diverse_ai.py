import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import csv
import random
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Config
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_DIR = pd.io.common.Path("Data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def humanize_text(text: str) -> str:
    """Applies sentence shuffling, German fillers, and minor typo injection."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) >= 3 and random.random() < 0.20:
        idx = random.randint(0, len(sentences) - 2)
        sentences[idx], sentences[idx+1] = sentences[idx+1], sentences[idx]
    text = " ".join(sentences)
    words = text.split()
    if len(words) < 5:
        return text
    
    fillers = ["ja", "also", "halt", "mal", "eben", "sozusagen"]
    if random.random() < 0.15:
        for _ in range(random.randint(1, 2)):
            insert_idx = random.randint(1, len(words) - 2)
            words.insert(insert_idx, random.choice(fillers))
            
    if random.random() < 0.15:
        candidate_indices = [idx for idx, w in enumerate(words) if len(w) > 5 and w.isalpha()]
        if candidate_indices:
            idx = random.choice(candidate_indices)
            w = words[idx]
            typo_type = random.choice(["swap", "drop"])
            if typo_type == "swap" and len(w) > 2:
                char_idx = random.randint(0, len(w) - 2)
                words[idx] = w[:char_idx] + w[char_idx+1] + w[char_idx] + w[char_idx+2:]
            elif typo_type == "drop":
                char_idx = random.randint(0, len(w) - 1)
                words[idx] = w[:char_idx] + w[char_idx+1:]
                
    return " ".join(words)

def generate_batch(model, tokenizer, prompts, max_new_tokens=100):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # Strip prompts from output
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    return [r.strip() for r in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]

def main():
    print(f"Using device: {DEVICE.upper()}")
    
    # 1. Load GNAD news articles
    gnad_path = "Data/gnad_articles.csv"
    if not os.path.exists(gnad_path):
        raise FileNotFoundError(f"{gnad_path} is missing.")
    
    print("Loading GNAD news articles...")
    gnad_df = pd.read_csv(gnad_path, sep=";", header=None, on_bad_lines="skip")
    gnad_texts = gnad_df[1].dropna().tolist()
    random.shuffle(gnad_texts)
    
    # 2. Initialize Qwen
    print(f"Loading {QWEN_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)
    model.eval()
    
    # --- Generate Rewritten News (6,000) ---
    print("\n--- Phase 1: Generating 6,000 Rewritten News Paragraphs ---")
    rewritten_ai = []
    prompts = []
    
    # Prepare GNAD text segments (first 6,000 paragraphs)
    seed_texts = gnad_texts[:6000]
    for text in seed_texts:
        # Take the first ~3-4 sentences of the article to rewrite as a paragraph
        sentences = re.split(r'(?<=[.!?])\s+', text)
        short_text = " ".join(sentences[:3])
        if len(short_text.split()) < 20:
            short_text = text[:300]
            
        messages = [
            {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
            {"role": "user", "content": f"Schreibe den folgenden Text im typischen Stil einer KI (wie ChatGPT oder Gemini) in eigenen Worten um. Behalte den Sinn bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n{short_text}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    batch_size = 128 if DEVICE == "cuda" else 8
    for i in tqdm(range(0, len(prompts), batch_size), desc="Rewriting News"):
        batch = prompts[i:i+batch_size]
        rewritten_ai.extend(generate_batch(model, tokenizer, batch, max_new_tokens=100))
        
    # --- Generate Completely New News (6,000) ---
    print("\n--- Phase 2: Generating 6,000 Completely New News Paragraphs ---")
    new_news_ai = []
    prompts = []
    
    news_topics = [
        "Digitalisierung in deutschen Schulen", "Entwicklungen in der Elektromobilität", 
        "Internationale Klimaschutzverhandlungen", "Wirtschaftswachstum und Inflation in Europa",
        "Wohnungsbau und Mietpreisentwicklung in Großstädten", "Ausbau der erneuerbaren Energien",
        "Neue Entdeckungen in der Astronomie und Raumfahrt", "Technologietrends bei Künstlicher Intelligenz",
        "Reformen im Gesundheitssystem", "Fachkräftemangel in der Industrie", "Tourismus nach der Krise",
        "Forschung im Bereich Fusionsenergie", "Modernisierung des Schienenverkehrs", "Zukunft des Homeoffice",
        "Förderung von Start-ups in Deutschland", "Cybersecurity-Herausforderungen für Unternehmen"
    ]
    
    for _ in range(6000):
        topic = random.choice(news_topics)
        messages = [
            {"role": "system", "content": "Du bist ein KI-Nachrichtenreporter."},
            {"role": "user", "content": f"Schreibe einen kurzen, sachlichen Nachrichtenabschnitt (3 bis 5 Sätze) auf Deutsch im typischen, klaren Stil einer KI über folgendes Thema: {topic}."}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating New News"):
        batch = prompts[i:i+batch_size]
        new_news_ai.extend(generate_batch(model, tokenizer, batch, max_new_tokens=120))
        
    # --- Generate Adversarial/Humanized News (3,000) ---
    print("\n--- Phase 3: Generating 3,000 Adversarial/Humanized News Paragraphs ---")
    adversarial_ai_raw = []
    prompts = []
    
    for _ in range(3000):
        topic = random.choice(news_topics)
        messages = [
            {"role": "system", "content": "Du bist ein KI-Assistent."},
            {"role": "user", "content": f"Schreibe einen kurzen Absatz (3 bis 5 Sätze) auf Deutsch über das Thema: {topic}. Halte den Schreibstil einfach."}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Adversarial"):
        batch = prompts[i:i+batch_size]
        adversarial_ai_raw.extend(generate_batch(model, tokenizer, batch, max_new_tokens=100))
        
    # Humanize them
    adversarial_ai = [humanize_text(t) for t in adversarial_ai_raw]
    
    # --- Generate Casual/Everyday Blogs & Essays (5,000) ---
    print("\n--- Phase 4: Generating 5,000 Casual Blogs, Essays, and Comments ---")
    casual_ai = []
    prompts = []
    
    casual_topics = [
        "Mein liebstes Hobby am Wochenende", "Warum Kochen entspannend ist", 
        "Tipps für eine gute Work-Life-Balance", "Meine Erfahrungen mit Social Media",
        "Reisen im eigenen Land", "Wie Technologie meinen Alltag erleichtert",
        "Warum Lesen wieder im Trend liegt", "Gesunde Ernährung im stressigen Alltag",
        "Sport und Fitness für Einsteiger", "Die Freude an Haustieren",
        "Kinobesuche vs. Streaming zu Hause", "Wie man produktiv im Homeoffice arbeitet",
        "Die Bedeutung von Freundschaften", "Kaffeekultur und Kaffeegenuss",
        "Gartengestaltung im Frühling", "Einfache Tipps zum Umweltschutz im Alltag"
    ]
    
    types = ["Blogbeitrag", "persönlichen Essay", "informellen Forenkommentar"]
    
    for _ in range(5000):
        topic = random.choice(casual_topics)
        text_type = random.choice(types)
        messages = [
            {"role": "system", "content": "Du bist ein Blogger, der im informellen Ton schreibt."},
            {"role": "user", "content": f"Schreibe einen kurzen {text_type} (3 bis 5 Sätze) auf Deutsch im typischen Stil einer KI über das Thema: '{topic}'. Nutze einen lockeren, umgangssprachlichen Ton (Du-Form)."}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating Casual Texts"):
        batch = prompts[i:i+batch_size]
        casual_ai.extend(generate_batch(model, tokenizer, batch, max_new_tokens=120))
        
    # Save datasets
    print("\nSaving generated AI datasets...")
    # Save News AI
    all_news_ai = rewritten_ai + new_news_ai + adversarial_ai
    news_df = pd.DataFrame({
        "text": all_news_ai,
        "source": ["rewritten_news"]*len(rewritten_ai) + ["new_news"]*len(new_news_ai) + ["adversarial_news"]*len(adversarial_ai),
        "label": 1
    })
    news_df.to_csv("Data/ai_generated_news.csv", index=False, encoding="utf-8")
    print(f"Saved {len(news_df):,} AI news rows to Data/ai_generated_news.csv")
    
    # Save Casual AI
    casual_df = pd.DataFrame({
        "text": casual_ai,
        "source": "casual_blog_comment",
        "label": 1
    })
    casual_df.to_csv("Data/ai_generated_casual.csv", index=False, encoding="utf-8")
    print(f"Saved {len(casual_df):,} AI casual rows to Data/ai_generated_casual.csv")

if __name__ == "__main__":
    main()
