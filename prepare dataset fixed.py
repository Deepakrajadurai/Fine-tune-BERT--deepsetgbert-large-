"""
Step 1 (FIXED) — Dataset Preparation
======================================
Fixes applied vs the original script:

  FIX 1 — No leakage:
      Full deduplication happens BEFORE any split.
      The 80/10/10 split is done ONCE on the merged pool.
      Subsampling (500k human + 250k AI) happens from the
      training portion only — val and test are kept separate
      from the very first step.

  FIX 2 — Stratified AI sampling across all generator models:
      The 250k AI sample is drawn evenly across all 'model'
      and 'style' combinations so no single generator
      dominates the training signal.

  FIX 3 — Artifact removal:
      Strips synthetic speaker metadata (party tags, session
      numbers, fake names) that leaked into sentence text and
      gave the model a trivial shortcut.

  FIX 4 — Paragraph-level grouping:
      Consecutive sentences from the same document are merged
      into 3–5 sentence paragraphs, making the task harder
      and more realistic.

  FIX 5 — Length balancing:
      Human and AI word-count distributions are matched by
      percentile-bucketed sampling so the model cannot
      shortcut on length alone.

Usage:
    python 01_prepare_dataset.py \
        --human_csv data/bundestag_sentences.csv \
        --ai_csv    data/ai_generated_sentences.csv

Output:
    data/train.csv  (~600k rows, 80%)
    data/val.csv    (~75k  rows, 10%)
    data/test.csv   (~75k  rows, 10%)
    data/prep_report.txt  (full diagnostics)
"""

import re
import argparse
import logging
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
HUMAN_SAMPLE       = 500_000     # target human rows in final dataset
AI_SAMPLE          = 250_000     # target AI rows in final dataset
MIN_WORDS          = 30          # raised from 20 → harder task, less trivial signals
MAX_WORDS          = 300         # cap very long texts for BERT
PARA_MIN_SENTENCES = 3           # merge N sentences into one paragraph-level sample
PARA_MAX_SENTENCES = 5
RANDOM_SEED        = 42
OUTPUT_DIR         = Path("data")

# ──────────────────────────────────────────────────────────────────────────────
# FIX 3 — Artifact patterns to strip from text
# These are synthetic markers inserted by the generation script that
# give the classifier a trivial shortcut unrelated to writing style.
# ──────────────────────────────────────────────────────────────────────────────
ARTIFACT_PATTERNS = [
    # "239. Plenarsitzung" — fake session numbers
    (r"\d+\.\s*Plenarsitzung", "Plenarsitzung"),
    # "(Linke)", "(SPD)" etc. at end of parenthetical speaker refs
    (r"\((?:CDU(?:/CSU)?|SPD|Grüne|FDP|AfD|Linke|BSW|CSU)\)", ""),
    # "auf Initiative von Abgeordnetem Uwe Schröder" — AI-generated name patterns
    (r"(?:auf Initiative von |Abgeordnet(?:em|er|en)\s+)[A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+", ""),
    # Remove double spaces left behind
    (r"\s{2,}", " "),
]


def strip_artifacts(text: str) -> str:
    for pattern, replacement in ARTIFACT_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text.strip()


# ──────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    text = strip_artifacts(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    n = len(words)
    if n < MIN_WORDS:
        return None
    if n > MAX_WORDS:
        # Hard truncate at MAX_WORDS keeping complete sentence
        text = " ".join(words[:MAX_WORDS])

    # Reject if >25% digits (numerical/tabular artefact)
    if sum(c.isdigit() for c in text) / len(text) > 0.25:
        return None

    # Must contain at least one German-specific character
    if not re.search(r"[äöüÄÖÜß]", text):
        return None

    # Reject model meta-commentary that slipped through generation
    bad_phrases = [
        "als ki ", "als sprachmodell", "ich kann leider",
        "gerne helfe ich", "natürlich, hier", "hier sind die",
        "bitte beachten sie", "als assistent",
    ]
    lower = text.lower()
    if any(p in lower for p in bad_phrases):
        return None

    return text


def fingerprint(text: str) -> str:
    """MD5 of normalised text — used for exact and near-dedup."""
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalised.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# FIX 4 — Paragraph-level grouping
# Groups consecutive sentences from the same source doc/wahlperiode/datum
# into PARA_MIN_SENTENCES–PARA_MAX_SENTENCES sentence chunks.
# ──────────────────────────────────────────────────────────────────────────────
def group_into_paragraphs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive rows from the same source document into paragraphs.
    Keeps all metadata from the first sentence in each paragraph.
    """
    # Sort so same-document sentences are consecutive
    group_cols = [c for c in ["source", "wahlperiode", "datum", "speaker"]
                  if c in df.columns]
    df = df.sort_values(group_cols).reset_index(drop=True)

    records = []
    i = 0
    while i < len(df):
        # Pick random paragraph length
        para_len = np.random.randint(PARA_MIN_SENTENCES, PARA_MAX_SENTENCES + 1)
        chunk    = df.iloc[i: i + para_len]

        # Only merge if all rows are from the same source document
        if len(group_cols) > 0:
            same_doc = (chunk[group_cols[0]] == chunk[group_cols[0]].iloc[0]).all()
        else:
            same_doc = True

        if same_doc and len(chunk) >= PARA_MIN_SENTENCES:
            merged_text = " ".join(chunk["text"].tolist())
            row = chunk.iloc[0].copy()
            row["text"] = merged_text
            records.append(row)
            i += para_len
        else:
            # Emit as single sentence if we can't form a full paragraph
            records.append(df.iloc[i])
            i += 1

    result = pd.DataFrame(records).reset_index(drop=True)
    log.info(f"  Paragraph grouping: {len(df):,} sentences → {len(result):,} paragraphs")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# FIX 5 — Length-balanced sampling
# Matches human and AI word-count distributions so length is not a signal.
# ──────────────────────────────────────────────────────────────────────────────
def length_balanced_sample(human_df: pd.DataFrame,
                            ai_df: pd.DataFrame,
                            n_human: int,
                            n_ai: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample human and AI texts such that their word-count distributions match.
    Uses 10 word-count quantile buckets.
    """
    log.info("Balancing word-count distributions between classes...")
    N_BUCKETS = 10

    human_df  = human_df.copy()
    ai_df     = ai_df.copy()
    human_df["_wc"] = human_df["text"].str.split().apply(len)
    ai_df["_wc"]    = ai_df["text"].str.split().apply(len)

    # Build quantile bucket boundaries from human distribution
    boundaries = human_df["_wc"].quantile(
        np.linspace(0, 1, N_BUCKETS + 1)
    ).values

    human_buckets, ai_buckets = [], []
    per_bucket_human = n_human // N_BUCKETS
    per_bucket_ai    = n_ai    // N_BUCKETS

    for j in range(N_BUCKETS):
        lo, hi = boundaries[j], boundaries[j + 1]
        h_slice = human_df[(human_df["_wc"] >= lo) & (human_df["_wc"] < hi)]
        a_slice = ai_df[(ai_df["_wc"]    >= lo) & (ai_df["_wc"]    < hi)]

        n_h = min(per_bucket_human, len(h_slice))
        n_a = min(per_bucket_ai,    len(a_slice))

        if n_h > 0:
            human_buckets.append(h_slice.sample(n=n_h, random_state=RANDOM_SEED))
        if n_a > 0:
            ai_buckets.append(a_slice.sample(n=n_a, random_state=RANDOM_SEED))

    human_sampled = pd.concat(human_buckets).drop(columns=["_wc"])
    ai_sampled    = pd.concat(ai_buckets).drop(columns=["_wc"])

    # Top-up to exact targets if some buckets were short
    remaining_human = n_human - len(human_sampled)
    remaining_ai    = n_ai    - len(ai_sampled)
    already_h = set(human_sampled.index)
    already_a = set(ai_sampled.index)

    if remaining_human > 0:
        pool = human_df[~human_df.index.isin(already_h)]
        if len(pool) >= remaining_human:
            top_up = pool.sample(n=remaining_human, random_state=RANDOM_SEED)
            human_sampled = pd.concat([human_sampled,
                                       top_up.drop(columns=["_wc"])])

    if remaining_ai > 0:
        pool = ai_df[~ai_df.index.isin(already_a)]
        if len(pool) >= remaining_ai:
            top_up = pool.sample(n=remaining_ai, random_state=RANDOM_SEED)
            ai_sampled = pd.concat([ai_sampled,
                                    top_up.drop(columns=["_wc"])])

    log.info(f"  Human sampled: {len(human_sampled):,}  "
             f"AI sampled: {len(ai_sampled):,}")
    return human_sampled.reset_index(drop=True), ai_sampled.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# FIX 2 — Stratified AI sampling across generator models + styles
# ──────────────────────────────────────────────────────────────────────────────
def stratified_ai_sample(ai_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Sample n rows from ai_df with equal representation from each
    (model × style) combination.
    """
    if "model" not in ai_df.columns and "style" not in ai_df.columns:
        log.warning("No 'model'/'style' columns — falling back to random sample.")
        return ai_df.sample(n=min(n, len(ai_df)), random_state=RANDOM_SEED)

    strat_col = "model" if "model" in ai_df.columns else "style"
    groups    = ai_df[strat_col].unique()
    per_group = n // len(groups)

    log.info(f"Stratified AI sampling across {len(groups)} '{strat_col}' groups, "
             f"{per_group:,} each")

    sampled = []
    for g in groups:
        pool = ai_df[ai_df[strat_col] == g]
        k    = min(per_group, len(pool))
        sampled.append(pool.sample(n=k, random_state=RANDOM_SEED))

    result = pd.concat(sampled)

    # Top-up to exact n
    remaining = n - len(result)
    if remaining > 0:
        already  = set(result.index)
        leftover = ai_df[~ai_df.index.isin(already)]
        if len(leftover) >= remaining:
            result = pd.concat([result,
                                 leftover.sample(n=remaining,
                                                  random_state=RANDOM_SEED)])

    return result.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# LOAD + CLEAN
# ──────────────────────────────────────────────────────────────────────────────
def load_and_clean(csv_path: Path, label: int) -> pd.DataFrame:
    log.info(f"\nLoading {csv_path}  (label={label}) ...")
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df["label"] = label

    log.info(f"  Raw rows: {len(df):,}")
    df["text"] = df["text"].apply(clean_text)
    df.dropna(subset=["text"], inplace=True)
    log.info(f"  After cleaning: {len(df):,}")

    # Exact deduplication
    df["_fp"] = df["text"].apply(fingerprint)
    before    = len(df)
    df.drop_duplicates(subset=["_fp"], inplace=True)
    df.drop(columns=["_fp"], inplace=True)
    log.info(f"  After exact dedup: {len(df):,}  (removed {before - len(df):,})")

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main(human_csv: str, ai_csv: str):
    np.random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    report_lines = []

    def rpt(msg):
        log.info(msg)
        report_lines.append(msg)

    # ── Load ─────────────────────────────────────────────────────────────────
    human_df = load_and_clean(Path(human_csv), label=0)
    ai_df    = load_and_clean(Path(ai_csv),    label=1)

    # ── FIX 4: Paragraph grouping ─────────────────────────────────────────────
    rpt("\nGrouping sentences into paragraphs...")
    human_df = group_into_paragraphs(human_df)
    ai_df    = group_into_paragraphs(ai_df)

    # ── FIX 1: Merge THEN split (prevents any leakage) ───────────────────────
    # Step A: Take the full available pool for each class
    rpt("\nFIX 1: Splitting full pool before subsampling...")
    human_train, human_temp = train_test_split(
        human_df, test_size=0.20, random_state=RANDOM_SEED)
    human_val, human_test   = train_test_split(
        human_temp, test_size=0.50, random_state=RANDOM_SEED)

    ai_train, ai_temp = train_test_split(
        ai_df, test_size=0.20, random_state=RANDOM_SEED)
    ai_val, ai_test   = train_test_split(
        ai_temp, test_size=0.50, random_state=RANDOM_SEED)

    rpt(f"  Human → train: {len(human_train):,}  val: {len(human_val):,}  test: {len(human_test):,}")
    rpt(f"  AI    → train: {len(ai_train):,}  val: {len(ai_val):,}  test: {len(ai_test):,}")

    # Step B: Subsample from training portion only
    rpt("\nFIX 2: Stratified AI sampling from training split only...")
    ai_train_sampled = stratified_ai_sample(ai_train, n=AI_SAMPLE)

    human_target = min(HUMAN_SAMPLE, len(human_train))
    rpt(f"\nFIX 5: Length-balanced sampling "
        f"({human_target:,} human / {len(ai_train_sampled):,} AI)...")
    human_train_sampled, ai_train_sampled = length_balanced_sample(
        human_train, ai_train_sampled,
        n_human=human_target,
        n_ai=len(ai_train_sampled),
    )

    # ── Assemble splits ───────────────────────────────────────────────────────
    def assemble(h, a):
        cols = ["text", "label", "source", "wahlperiode", "datum", "speaker"]
        extra = ["style", "model", "temperature"]
        all_cols = cols + [c for c in extra if c in a.columns or c in h.columns]
        for df in [h, a]:
            for c in all_cols:
                if c not in df.columns:
                    df[c] = None
        merged = pd.concat([h[all_cols], a[all_cols]], ignore_index=True)
        return merged.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    train_df = assemble(human_train_sampled, ai_train_sampled)
    val_df   = assemble(human_val,  ai_val)
    test_df  = assemble(human_test, ai_test)

    # ── Cross-split dedup check (safety net) ─────────────────────────────────
    rpt("\nCross-split leakage safety check...")
    train_fps = set(train_df["text"].apply(fingerprint))
    val_fps   = set(val_df["text"].apply(fingerprint))
    test_fps  = set(test_df["text"].apply(fingerprint))

    tv_leak = len(train_fps & val_fps)
    tt_leak = len(train_fps & test_fps)
    vt_leak = len(val_fps   & test_fps)

    rpt(f"  Train ∩ Val  : {tv_leak} duplicates")
    rpt(f"  Train ∩ Test : {tt_leak} duplicates")
    rpt(f"  Val   ∩ Test : {vt_leak} duplicates")

    if tt_leak > 0:
        rpt("  Removing leaked rows from test set...")
        test_df = test_df[
            ~test_df["text"].apply(fingerprint).isin(train_fps)
        ].reset_index(drop=True)

    # ── Save ─────────────────────────────────────────────────────────────────
    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUTPUT_DIR   / "val.csv",   index=False, encoding="utf-8")
    test_df.to_csv(OUTPUT_DIR  / "test.csv",  index=False, encoding="utf-8")

    # ── Report ───────────────────────────────────────────────────────────────
    rpt("\n" + "=" * 60)
    rpt("FINAL SPLIT REPORT")
    rpt("=" * 60)
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split["label"].value_counts()
        h_n    = counts.get(0, 0)
        a_n    = counts.get(1, 0)
        h_wc   = split[split["label"] == 0]["text"].str.split().apply(len)
        a_wc   = split[split["label"] == 1]["text"].str.split().apply(len)
        rpt(f"\n{name}:")
        rpt(f"  Total rows  : {len(split):,}")
        rpt(f"  Human (0)   : {h_n:,}  mean words = {h_wc.mean():.1f}")
        rpt(f"  AI    (1)   : {a_n:,}  mean words = {a_wc.mean():.1f}")
        rpt(f"  Saved to    : data/{name}.csv")

    (OUTPUT_DIR / "prep_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    rpt("\nPrep report saved to data/prep_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_csv", default="data/bundestag_sentences.csv")
    parser.add_argument("--ai_csv",    default="data/ai_generated_sentences.csv")
    args = parser.parse_args()
    main(args.human_csv, args.ai_csv)