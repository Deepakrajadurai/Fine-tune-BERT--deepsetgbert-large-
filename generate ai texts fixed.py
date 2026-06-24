"""
Step 2-gen (FIXED) — AI Text Generation
==========================================
Fixes applied vs the original script:

  FIX A — No metadata artifacts in text:
      Synthetic speaker names, party abbreviations, and session
      numbers are NO LONGER embedded in the generated sentence
      text. They were a trivial classifier shortcut. Speaker/party
      info stays in the metadata columns only.

  FIX B — Artifact-clean prompts:
      Prompts instruct the model to write natural flowing prose
      WITHOUT inserting names, party abbreviations, or session
      numbers mid-sentence (the way real Bundestag transcripts
      actually work — names appear in the header, not inline).

  FIX C — Paragraph-level output:
      Each API call now requests a 3–5 sentence paragraph rather
      than 8 individual sentences. This matches the paragraph-
      level grouping in 01_prepare_dataset.py and makes the
      detection task genuinely hard.

  FIX D — Balanced per-model budget:
      TARGET_SENTENCES is divided equally across all backends so
      no single model dominates the AI class signal.

Usage:
    export GEMINI_API_KEY="your_key"
    export GROQ_API_KEY="your_key"
    export HF_API_KEY="your_key"
    python 02_generate_ai_texts.py
"""

import os
import re
import csv
import json
import time
import random
import logging
import hashlib
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
TARGET_SENTENCES   = 250_000
MIN_WORDS          = 30        # raised to match prep script
OUTPUT_DIR         = Path("data")
OUTPUT_CSV         = OUTPUT_DIR / "ai_generated_sentences.csv"
CHECKPOINT_FILE    = OUTPUT_DIR / "ai_gen_checkpoint.jsonl"

# FIX D: equal budget per backend
BACKENDS = ["gemini", "groq", "ollama", "hf"]
BUDGET_PER_BACKEND = TARGET_SENTENCES // len(BACKENDS)   # 62,500 each

REQUEST_DELAYS = {
    "gemini": 4.5,
    "groq":   2.2,
    "hf":     1.5,
    "ollama": 0.2,
}

OLLAMA_BASE   = "http://localhost:11434"
OLLAMA_MODELS = ["mistral", "llama3", "phi3"]
GROQ_MODELS   = ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]
HF_MODEL      = "mistralai/Mistral-7B-Instruct-v0.3"

# Temperature per style — unchanged
TEMPERATURES = {
    "bundestag_speech":       0.75,
    "landtag_speech":         0.75,
    "law_paragraph":          0.30,
    "admin_press_release":    0.60,
    "coalition_agreement":    0.50,
    "parliamentary_question": 0.65,
    "government_report":      0.40,
}

GERMAN_STATES = [
    "Bayern", "Nordrhein-Westfalen", "Baden-Württemberg", "Niedersachsen",
    "Hessen", "Sachsen", "Berlin", "Brandenburg", "Hamburg", "Thüringen",
    "Sachsen-Anhalt", "Schleswig-Holstein", "Rheinland-Pfalz",
    "Mecklenburg-Vorpommern", "Saarland", "Bremen",
]

PARTIES = ["CDU/CSU", "SPD", "Grüne", "FDP", "AfD", "Linke", "BSW"]

TOPICS = [
    "Klimaschutzgesetz", "Haushaltsdebatte", "Bildungspolitik",
    "Wohnungsbau", "Digitalisierung der Verwaltung", "Innere Sicherheit",
    "Sozialpolitik", "Außenpolitik der Europäischen Union",
    "Verkehrsinfrastruktur", "Gesundheitsversorgung", "Steuerpolitik",
    "Flüchtlings- und Migrationspolitik", "Rentensystem",
    "Energiewende", "Bürokratieabbau", "kommunale Selbstverwaltung",
    "Datenschutz und IT-Sicherheit", "Bundeswehr und Verteidigungspolitik",
    "Kulturförderung", "Landwirtschaftspolitik",
    "Datenschutz und DSGVO-Umsetzung", "Verwaltungsdigitalisierung",
    "kommunales Haushaltsrecht", "Beamtenrecht und öffentlicher Dienst",
]

# ──────────────────────────────────────────────────────────────────────────────
# FIX A + B — Artifact-clean prompt templates
# Key change: prompts NO LONGER ask for speaker names, party tags, or session
# numbers inside the text. The paragraph should read as natural flowing prose.
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(style: str) -> tuple[str, dict]:
    state = random.choice(GERMAN_STATES)
    party = random.choice(PARTIES)
    topic = random.choice(TOPICS)
    wp    = random.choice(["19", "20", "21"])
    year  = random.randint(2018, 2025)
    month = random.randint(1, 12)
    day   = random.randint(1, 28)
    datum = f"{year}-{month:02d}-{day:02d}"

    meta = {
        "wahlperiode": wp,
        "datum":       datum,
        "speaker":     f"Abgeordnete/r ({party})",   # metadata only, not in text
    }

    # FIX C: request a paragraph (3–5 sentences), not a bullet list
    prompts = {

        "bundestag_speech": f"""Schreibe einen sachlichen Absatz im Stil einer Rede im Deutschen Bundestag zum Thema „{topic}".
Der Absatz soll aus 3 bis 5 vollständigen Sätzen bestehen und natürlich klingen, wie ein echter Redebeitrag in einer Plenarsitzung.
Verwende formelle parlamentarische Sprache.
Füge KEINE Sprechernamen, Parteiabkürzungen in Klammern, Sitzungsnummern oder Metainformationen in den Text ein — nur den Redeinhalt.
Gib ausschließlich den Absatz aus, ohne Überschriften, Aufzählungen oder Erklärungen.""",

        "landtag_speech": f"""Schreibe einen Absatz im Stil einer Rede im Landtag von {state} zum Thema „{topic}".
Der Absatz soll aus 3 bis 5 vollständigen Sätzen bestehen und die regionalen Besonderheiten von {state} widerspiegeln.
Verwende formelle politische Sprache wie in echten Landtagsprotokollen.
Füge KEINE Sprechernamen, Parteiabkürzungen oder Metainformationen in den Text ein.
Gib ausschließlich den Absatz aus.""",

        "law_paragraph": f"""Formuliere einen kurzen Absatz im Stil eines deutschen Gesetzestexts oder einer Verwaltungsvorschrift zum Thema „{topic}".
Der Absatz soll aus 3 bis 4 vollständigen Sätzen bestehen und präzise, normative Gesetzessprache verwenden.
Keine Überschriften, keine Paragrafennummern, keine Erklärungen — nur den Gesetzestext selbst.""",

        "admin_press_release": f"""Schreibe einen Absatz im Stil einer offiziellen Pressemitteilung der Landesregierung {state} zum Thema „{topic}".
Der Absatz soll aus 3 bis 5 vollständigen Sätzen bestehen, sachlich und behördlich klingen.
Keine Anreden, keine Unterschriften, keine Metadaten — nur den Inhalt der Meldung.""",

        "coalition_agreement": f"""Schreibe einen Absatz im Stil eines deutschen Koalitionsvertrags zum Politikfeld „{topic}".
Der Absatz soll aus 3 bis 5 vollständigen Sätzen bestehen und den typischen programmatisch-kompromissorientierten Stil verwenden.
Keine Überschriften, keine Parteinamen als Absender — nur den Vertragstext.""",

        "parliamentary_question": f"""Formuliere einen Absatz im Stil einer schriftlichen Anfrage an die Bundesregierung zum Thema „{topic}".
Der Absatz soll aus 3 bis 5 vollständigen Sätzen bestehen und den formell-parlamentarischen Fragenstil verwenden.
Kein Fragensteller, keine Bezugsnummern im Text — nur den Inhalt der Anfrage.""",

        "government_report": f"""Schreibe einen Absatz im Stil eines offiziellen deutschen Regierungsberichts oder Evaluierungsberichts zum Thema „{topic}".
Der Absatz soll aus 3 bis 5 vollständigen Sätzen bestehen, analytisch und sachlich klingen.
Keine Überschriften, keine Quellenangaben — nur den Berichtstext.""",
    }

    return prompts[style], meta


# ──────────────────────────────────────────────────────────────────────────────
# BACKENDS (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
def call_gemini(prompt: str, temperature: float) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    url  = ("https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-1.5-flash:generateContent?key={api_key}")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 512},
    }
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_groq(prompt: str, temperature: float) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    model = random.choice(GROQ_MODELS)
    resp  = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": 512},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_ollama(prompt: str, temperature: float) -> str:
    model = random.choice(OLLAMA_MODELS)
    resp  = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": temperature, "num_predict": 512}},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def call_hf(prompt: str, temperature: float) -> str:
    api_key = os.getenv("HF_API_KEY", "")
    if not api_key:
        raise RuntimeError("HF_API_KEY not set")
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"inputs": f"<s>[INST] {prompt} [/INST]",
              "parameters": {"temperature": temperature,
                             "max_new_tokens": 512,
                             "return_full_text": False}},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()
    return result[0].get("generated_text", "") if isinstance(result, list) else str(result)


BACKEND_FNS = {
    "gemini": call_gemini,
    "groq":   call_groq,
    "ollama": call_ollama,
    "hf":     call_hf,
}


# ──────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING — now validates a paragraph, not a sentence list
# ──────────────────────────────────────────────────────────────────────────────
def validate_paragraph(raw: str) -> str | None:
    """
    Clean raw model output into a single paragraph string.
    Returns None if it fails quality checks.
    """
    text = raw.strip()

    # Strip markdown, list markers, numbering artefacts
    text = re.sub(r"^[#*\-–•\d]+[\.\):]?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*|__|\*|_", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Reject model meta-commentary
    bad = ["als ki", "als sprachmodell", "ich kann leider", "gerne helfe",
           "natürlich, hier", "bitte beachten", "als assistent",
           "hier ist", "hier sind", "selbstverständlich"]
    if any(b in text.lower() for b in bad):
        return None

    # Reject English leakage
    en_markers = ["the ", " of ", " and ", " to ", "this ", "is "]
    if sum(1 for m in en_markers if m in text.lower()) >= 3:
        return None

    # Must contain German-specific chars
    if not re.search(r"[äöüÄÖÜß]", text):
        return None

    # FIX A: reject if synthetic artifacts slipped in despite prompt instruction
    bad_patterns = [
        r"\d+\.\s*Plenarsitzung",
        r"\((?:CDU(?:/CSU)?|SPD|Grüne|FDP|AfD|Linke|BSW|CSU)\)",
    ]
    for p in bad_patterns:
        if re.search(p, text):
            text = re.sub(p, "", text).strip()

    words = text.split()
    if len(words) < MIN_WORDS:
        return None

    return text


def fingerprint(text: str) -> str:
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────────
def load_checkpoint() -> list[dict]:
    if not CHECKPOINT_FILE.exists():
        return []
    rows = []
    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {len(rows):,} rows from checkpoint.")
    return rows


def save_checkpoint(rows: list[dict]):
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    styles = list(TEMPERATURES.keys())

    existing_rows  = load_checkpoint()
    all_rows       = existing_rows.copy()
    existing_texts = {r["text"] for r in existing_rows}

    # Track per-backend count for balanced budget
    backend_counts: dict[str, int] = {}
    for r in existing_rows:
        b = r.get("model", "unknown")
        backend_counts[b] = backend_counts.get(b, 0) + 1

    total_collected = len(all_rows)
    log.info(f"Starting from {total_collected:,} / {TARGET_SENTENCES:,}")
    log.info(f"Budget per backend: {BUDGET_PER_BACKEND:,}")

    pbar = tqdm(total=TARGET_SENTENCES, initial=total_collected,
                desc="Generating")

    while total_collected < TARGET_SENTENCES:
        # Pick the backend furthest below its budget (FIX D)
        backend = min(
            BACKENDS,
            key=lambda b: backend_counts.get(b, 0)
        )
        style       = random.choice(styles)
        temperature = TEMPERATURES[style]
        prompt, meta = build_prompt(style)

        try:
            raw = BACKEND_FNS[backend](prompt, temperature)
        except Exception as e:
            log.warning(f"[{backend}] error: {e}. Sleeping 10s...")
            time.sleep(10)
            continue

        paragraph = validate_paragraph(raw)
        if paragraph is None:
            time.sleep(REQUEST_DELAYS.get(backend, 1.0))
            continue

        fp = fingerprint(paragraph)
        if paragraph in existing_texts:
            time.sleep(REQUEST_DELAYS.get(backend, 1.0))
            continue

        existing_texts.add(paragraph)
        row = {
            "text":        paragraph,
            "label":       1,                          # AI-generated
            "source":      f"ai_{backend}_{style}",
            "wahlperiode": meta["wahlperiode"],
            "datum":       meta["datum"],
            "speaker":     meta["speaker"],            # metadata only
            "style":       style,
            "model":       backend,
            "temperature": temperature,
        }
        save_checkpoint([row])
        all_rows.append(row)
        backend_counts[backend] = backend_counts.get(backend, 0) + 1
        total_collected += 1
        pbar.update(1)

        time.sleep(REQUEST_DELAYS.get(backend, 1.0))

    pbar.close()

    # ── Export ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    df.drop_duplicates(subset=["text"], inplace=True)
    df = df.head(TARGET_SENTENCES).reset_index(drop=True)

    cols = ["text", "label", "source", "wahlperiode", "datum",
            "speaker", "style", "model", "temperature"]
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"\nSaved {len(df):,} rows to {OUTPUT_CSV}")
    log.info(f"Backend distribution:\n{df['model'].value_counts()}")
    log.info(f"Style distribution:\n{df['style'].value_counts()}")
    log.info(f"Mean word count: {df['text'].str.split().apply(len).mean():.1f}")


if __name__ == "__main__":
    main()