import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re
import html
import csv
import random
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = pd.io.common.Path("Data")
DATA_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA SCRAPER
# ──────────────────────────────────────────────────────────────────────────────
def get_wiki_paragraphs(title, num_paragraphs=100):
    url = f"https://de.wikipedia.org/wiki/{urllib.parse.quote(title)}"
    print(f"Scraping Wikipedia page: {title}")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            html_content = response.read().decode('utf-8')
        paragraphs = re.findall(r'<p\b[^>]*>(.*?)</p>', html_content, re.DOTALL)
        cleaned = []
        for p in paragraphs:
            p_clean = re.sub(r'<.*?>', '', p)
            p_clean = html.unescape(p_clean)
            p_clean = re.sub(r'\[\d+\]', '', p_clean)
            p_clean = re.sub(r'\s+', ' ', p_clean).strip()
            if p_clean.count('|') > 2:
                continue
            if len(p_clean.split()) >= 30:
                cleaned.append(p_clean)
                if len(cleaned) >= num_paragraphs:
                    break
        return cleaned
    except Exception as e:
        print(f"Error scraping wiki '{title}': {e}")
        return []

# ──────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL HUMANIZATION LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def humanize_text(text: str) -> str:
    # 1. Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Sentence shuffling (20% chance if >= 3 sentences)
    if len(sentences) >= 3 and random.random() < 0.20:
        idx = random.randint(0, len(sentences) - 2)
        sentences[idx], sentences[idx+1] = sentences[idx+1], sentences[idx]
        
    text = " ".join(sentences)
    words = text.split()
    if len(words) < 5:
        return text
    
    # 2. German filler words insertion (15% chance)
    fillers = ["ja", "also", "halt", "mal", "eben", "sozusagen"]
    if random.random() < 0.15:
        for _ in range(random.randint(1, 2)):
            insert_idx = random.randint(1, len(words) - 2)
            words.insert(insert_idx, random.choice(fillers))
            
    # 3. Typo insertion (20% chance)
    if random.random() < 0.20:
        candidate_indices = [idx for idx, w in enumerate(words) if len(w) > 5 and w.isalpha()]
        if candidate_indices:
            idx = random.choice(candidate_indices)
            w = words[idx]
            typo_type = random.choice(["swap", "drop", "double"])
            if typo_type == "swap" and len(w) > 2:
                char_idx = random.randint(0, len(w) - 2)
                words[idx] = w[:char_idx] + w[char_idx+1] + w[char_idx] + w[char_idx+2:]
            elif typo_type == "drop":
                char_idx = random.randint(0, len(w) - 1)
                words[idx] = w[:char_idx] + w[char_idx+1:]
            elif typo_type == "double":
                char_idx = random.randint(0, len(w) - 1)
                words[idx] = w[:char_idx] + w[char_idx]*2 + w[char_idx+1:]
                
    return " ".join(words)

# ──────────────────────────────────────────────────────────────────────────────
# JACCARD 3-GRAM SIMILARITY FILTER
# ──────────────────────────────────────────────────────────────────────────────
def get_3_grams(text):
    words = text.lower().split()
    if len(words) < 3:
        return set([tuple(words)])
    return set(zip(words[:-2], words[1:-1], words[2:]))

def filter_near_duplicates(train_df, val_df, test_df, ext_val_df, holdout_df, threshold=0.80):
    print("Filtering near-duplicates using Jaccard 3-gram similarity...")
    # Add training set hashes
    train_grams = [get_3_grams(t) for t in train_df["text"]]
    
    def is_near_duplicate(text):
        tg = get_3_grams(text)
        if not tg:
            return False
        for trg in train_grams:
            if not trg:
                continue
            jaccard = len(tg & trg) / len(tg | trg)
            if jaccard > threshold:
                return True
        return False

    # Filter val/test/external dfs
    before_val = len(val_df)
    val_df = val_df[~val_df["text"].apply(is_near_duplicate)].reset_index(drop=True)
    print(f"  Val set: filtered out {before_val - len(val_df)} near-duplicates.")

    before_test = len(test_df)
    test_df = test_df[~test_df["text"].apply(is_near_duplicate)].reset_index(drop=True)
    print(f"  Test set: filtered out {before_test - len(test_df)} near-duplicates.")

    before_ext = len(ext_val_df)
    ext_val_df = ext_val_df[~ext_val_df["text"].apply(is_near_duplicate)].reset_index(drop=True)
    print(f"  External Val set: filtered out {before_ext - len(ext_val_df)} near-duplicates.")

    before_h = len(holdout_df)
    holdout_df = holdout_df[~holdout_df["text"].apply(is_near_duplicate)].reset_index(drop=True)
    print(f"  Holdout set: filtered out {before_h - len(holdout_df)} near-duplicates.")

    return val_df, test_df, ext_val_df, holdout_df

# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 & 2: FETCH HUMAN TEXTS & COMPILE TEST SUITE
# ──────────────────────────────────────────────────────────────────────────────
def step_compile_human_and_test_suite():
    print("\n=== STEP 1: Fetching Wikipedia narrative paragraphs ===")
    news_pool = []
    for title in ["Klimawandel", "Künstliche_Intelligenz", "Deutsche_Wirtschaft", "Klimaschutz", "Europäische_Union", "Digitalisierung"]:
        news_pool.extend(get_wiki_paragraphs(title, 25))
    print(f"Total News paragraphs scraped: {len(news_pool)}")

    legal_pool = []
    for title in ["Grundgesetz_für_die_Bundesrepublik_Deutschland", "Bürgerliches_Gesetzbuch", "Recht_Deutschlands"]:
        legal_pool.extend(get_wiki_paragraphs(title, 40))
    print(f"Total Legal paragraphs scraped: {len(legal_pool)}")

    print("\n=== STEP 2: Compiling human & original AI parts of the test suite ===")
    # Load human Bundestag speeches from unseen indices (index >= 400,000)
    bundestag_speeches = []
    with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 400000:
                bundestag_speeches.append(row["text"])
                if len(bundestag_speeches) >= 100: # get 100 speeches
                    break

    # Sample human groups (50 each)
    human_bundestag = bundestag_speeches[:50]
    human_news = news_pool[:50]
    human_legal = legal_pool[:50]

    # Let's save the pools to use during generation
    return human_bundestag, human_news, human_legal, news_pool[50:], legal_pool[50:]

# ──────────────────────────────────────────────────────────────────────────────
# GENERATION HELPER
# ──────────────────────────────────────────────────────────────────────────────
def generate_batch(model, tokenizer, prompts, max_new_tokens=80, device="cuda"):
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
    return [r.strip() for r in tokenizer.batch_decode(generated_ids, skip_special_tokens=True)]

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Step 1 & 2: Human scraping & initial test suite setup
    test_bt, test_news, test_legal, train_news_pool, train_legal_pool = step_compile_human_and_test_suite()

    # Load seed speeches for rewrites
    print("Loading seed speeches from Human dataset for rewrites...")
    seed_speeches = []
    with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if 300000 <= i < 350000:
                seed_speeches.append(row["text"])
                if len(seed_speeches) >= 10000:
                    break
    random.shuffle(seed_speeches)
    
    # ──────────────────────────────────────────────────────────────────────────
    # GENERATOR 1: QWEN
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=== GENERATOR 1: Loading Qwen/Qwen2.5-1.5B-Instruct ===")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    # Generate 50 samples for each AI model style in our test suite
    print("Generating AI Test Suite samples (ChatGPT, Gemini, Claude, Qwen styles)...")
    test_seeds = test_bt[:20] # use Bundestag seeds to make AI texts stylistically similar
    styles = {
        "ChatGPT": "Schreibe den folgenden Text im typischen Stil von ChatGPT (GPT-4) in eigenen Worten um. Behalte den Sinn bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n",
        "Gemini": "Schreibe den folgenden Text im typischen Stil von Google Gemini in eigenen Worten um. Behalte den Sinn bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n",
        "Claude": "Schreibe den folgenden Text im typischen Stil von Claude 3.5 Sonnet in eigenen Worten um. Behalte den Sinn bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n",
        "Qwen": "Schreibe den folgenden Text im typischen Stil von Qwen in eigenen Worten um. Behalte den Sinn bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n"
    }
    
    test_ai_groups = {style_name: [] for style_name in styles}
    for style_name, prefix in styles.items():
        prompts = []
        for text in test_bt: # use all 50 Bundestag speeches as seeds
            messages = [
                {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
                {"role": "user", "content": prefix + text}
            ]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
        # Batch generate in chunks of 25
        results = []
        for i in range(0, len(prompts), 25):
            results.extend(generate_batch(model, tokenizer, prompts[i:i+25], max_new_tokens=100, device=device))
        test_ai_groups[style_name] = results

    # Generate 5,000 synthetic AI texts for the training set
    print("Generating 5,000 Qwen training samples (2,500 rewrites, 2,500 topic-based)...")
    qwen_ai_texts = []
    
    # 2,500 rewrites
    prompts = []
    for text in seed_speeches[:2500]:
        messages = [
            {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
            {"role": "user", "content": f"Schreibe den folgenden Text im Stil eines Politikers in eigenen Worten um. Behalte die Hauptaussage bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n{text}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    
    for i in range(0, len(prompts), 32):
        qwen_ai_texts.extend(generate_batch(model, tokenizer, prompts[i:i+32], max_new_tokens=80, device=device))

    # 2,500 political topics
    topics = [
        "Digitalisierung der Verwaltung", "Klimaschutz und Kohleausstieg", "Reform der Pflegeversicherung",
        "Wohnungsnot in Großstädten", "Ausbau des deutschen Schienennetzes", "Stärkung der Bundespolizei",
        "Bekämpfung von Altersarmut", "Steuererleichterungen für KMU", "Integrierung von Zuwanderern",
        "Energiewende und Netzausbau", "Förderung von Forschung und KI", "Erneuerbare-Energien-Gesetz",
        "Entlastung der Krankenkassen", "Reform des Bildungssystems", "Subventionierung der Landwirtschaft"
    ]
    prompts = []
    for _ in range(2500):
        topic = random.choice(topics)
        messages = [
            {"role": "system", "content": "Du bist ein Politiker."},
            {"role": "user", "content": f"Schreibe eine kurze, überzeugende Rede (3 bis 5 Sätze) auf Deutsch zum Thema: {topic}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    for i in range(0, len(prompts), 32):
        qwen_ai_texts.extend(generate_batch(model, tokenizer, prompts[i:i+32], max_new_tokens=80, device=device))

    # Clean up Qwen
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # GENERATOR 2: PHI-3
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=== GENERATOR 2: Loading microsoft/Phi-3-mini-4k-instruct ===")
    PHI_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(PHI_MODEL)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PHI_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    print("Generating 5,000 Phi-3 training samples (2,500 rewrites, 2,500 topic-based)...")
    phi_ai_texts = []
    
    # 2,500 rewrites
    prompts = []
    for text in seed_speeches[2500:5000]:
        messages = [
            {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
            {"role": "user", "content": f"Schreibe den folgenden Text im Stil eines Politikers in eigenen Worten um. Behalte die Hauptaussage bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n{text}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    for i in range(0, len(prompts), 16):
        phi_ai_texts.extend(generate_batch(model, tokenizer, prompts[i:i+16], max_new_tokens=80, device=device))

    # 2,500 political topics
    prompts = []
    for _ in range(2500):
        topic = random.choice(topics)
        messages = [
            {"role": "system", "content": "Du bist ein Politiker."},
            {"role": "user", "content": f"Schreibe eine kurze, überzeugende Rede (3 bis 5 Sätze) auf Deutsch zum Thema: {topic}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    for i in range(0, len(prompts), 16):
        phi_ai_texts.extend(generate_batch(model, tokenizer, prompts[i:i+16], max_new_tokens=80, device=device))

    # Clean up Phi-3
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # GENERATOR 3: TINYLLAMA
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=== GENERATOR 3: Loading TinyLlama/TinyLlama-1.1B-Chat-v1.0 ===")
    LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    print("Generating 5,000 TinyLlama training samples (2,500 rewrites, 2,500 topic-based)...")
    llama_ai_texts = []
    
    # 2,500 rewrites
    prompts = []
    for text in seed_speeches[5000:7500]:
        messages = [
            {"role": "system", "content": "Du bist ein KI-Assistent. Schreibe den Text um."},
            {"role": "user", "content": f"Schreibe den folgenden Text im Stil eines Politikers in eigenen Worten um. Behalte die Hauptaussage bei. Antworte nur mit der neuen Formulierung auf Deutsch und sonst nichts:\n\n{text}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    for i in range(0, len(prompts), 32):
        llama_ai_texts.extend(generate_batch(model, tokenizer, prompts[i:i+32], max_new_tokens=80, device=device))

    # 2,500 political topics
    prompts = []
    for _ in range(2500):
        topic = random.choice(topics)
        messages = [
            {"role": "system", "content": "Du bist ein Politiker."},
            {"role": "user", "content": f"Schreibe eine kurze, überzeugende Rede (3 bis 5 Sätze) auf Deutsch zum Thema: {topic}"}
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    for i in range(0, len(prompts), 32):
        llama_ai_texts.extend(generate_batch(model, tokenizer, prompts[i:i+32], max_new_tokens=80, device=device))

    # Clean up TinyLlama
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────────────────────────────────
    # ADVERSARIAL HUMANIZATION & ASSEMBLING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=== STEP 4: Applying adversarial humanization to 20% of generated texts ===")
    synthetic_ai = []
    for model_texts in [qwen_ai_texts, phi_ai_texts, llama_ai_texts]:
        for i, text in enumerate(model_texts):
            # Apply humanization to every 5th text (20%)
            if i % 5 == 0:
                synthetic_ai.append(humanize_text(text))
            else:
                synthetic_ai.append(text)
    
    print(f"Total synthetic AI texts: {len(synthetic_ai)}")

    # Load 5,000 Original template AI speeches (from the 500k file)
    print("Loading 5,000 Original AI template speeches...")
    original_ai = []
    with open("Data/ai_generated_sentences_500k.csv", mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10000: # skip first 10k
                original_ai.append(row["text"])
                if len(original_ai) >= 5000:
                    break

    # Load 20,000 Human speeches (from the 2M file, skip first 100 and final 100)
    print("Loading 20,000 Human Bundestag speeches...")
    human_train_pool = []
    with open("Data/Human_model_ready_dataset.csv", mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if 100 <= i < 300000:
                human_train_pool.append(row["text"])
                if len(human_train_pool) >= 20000:
                    break

    # Construct the training pool
    human_df = pd.DataFrame({"text": human_train_pool, "label": 0})
    ai_df = pd.DataFrame({"text": original_ai + synthetic_ai, "label": 1})
    
    combined_df = pd.concat([human_df, ai_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Clean exact duplicates
    combined_df.drop_duplicates(subset=["text"], inplace=True)
    combined_df = combined_df.reset_index(drop=True)
    print(f"Final balanced dataset size (after exact dedup): {len(combined_df)}")

    # Split into 80/10/10 train/val/test
    train_df, temp_df = train_test_split(combined_df, test_size=0.20, stratify=combined_df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRUCT THE TEST SUITE
    # ──────────────────────────────────────────────────────────────────────────
    print("\n=== Assembling Source-Aware Test Suite (350 samples) ===")
    test_suite_records = []
    
    # 1. Bundestag Speeches (Human)
    for t in test_bt:
        test_suite_records.append({"text": t, "label": 0, "source": "Bundestag Speeches (Human)"})
    # 2. News/Wiki (Human)
    for t in test_news:
        test_suite_records.append({"text": t, "label": 0, "source": "German News/Wiki (Human)"})
    # 3. Legal/Constitutional (Human)
    for t in test_legal:
        test_suite_records.append({"text": t, "label": 0, "source": "Legal/Constitutional (Human)"})
    # 4. ChatGPT style (AI)
    for t in test_ai_groups["ChatGPT"]:
        test_suite_records.append({"text": t, "label": 1, "source": "ChatGPT style (AI)"})
    # 5. Gemini style (AI)
    for t in test_ai_groups["Gemini"]:
        test_suite_records.append({"text": t, "label": 1, "source": "Gemini style (AI)"})
    # 6. Claude style (AI)
    for t in test_ai_groups["Claude"]:
        test_suite_records.append({"text": t, "label": 1, "source": "Claude style (AI)"})
    # 7. Qwen style (AI)
    for t in test_ai_groups["Qwen"]:
        test_suite_records.append({"text": t, "label": 1, "source": "Qwen style (AI)"})

    test_suite_df = pd.DataFrame(test_suite_records)
    # Shuffle
    test_suite_df = test_suite_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Split test suite 50/50 into External Val (175) and Final Holdout (175)
    ext_val_df, holdout_df = train_test_split(test_suite_df, test_size=0.50, stratify=test_suite_df["source"], random_state=SEED)
    ext_val_df = ext_val_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────────────────
    # REMOVE NEAR-DUPLICATES (SEMANTIC LEAKAGE CHECK)
    # ──────────────────────────────────────────────────────────────────────────
    val_df, test_df, ext_val_df, holdout_df = filter_near_duplicates(
        train_df, val_df, test_df, ext_val_df, holdout_df
    )

    # Save all datasets
    train_df.to_csv(DATA_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(DATA_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(DATA_DIR / "test.csv", index=False, encoding="utf-8")
    ext_val_df.to_csv(DATA_DIR / "external_val.csv", index=False, encoding="utf-8")
    holdout_df.to_csv(DATA_DIR / "final_holdout.csv", index=False, encoding="utf-8")

    print("\nDataset preparation and test suite compilation complete!")
    print(f"  Train: {len(train_df)} rows")
    print(f"  Val: {len(val_df)} rows")
    print(f"  Test: {len(test_df)} rows")
    print(f"  External Val: {len(ext_val_df)} rows")
    print(f"  Final Holdout: {len(holdout_df)} rows")

if __name__ == "__main__":
    QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    main()
