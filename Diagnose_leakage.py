"""
Diagnostic Script — Find Why Test Accuracy is Suspiciously 100%
===================================================================
Run this BEFORE retraining. It checks the four most likely causes
of an unrealistically perfect score:

  1. Train/val/test leakage (duplicate or near-duplicate text)
  2. Source/model imbalance in the AI class
  3. Trivial lexical shortcuts the model could be exploiting
  4. Text length distribution differences between classes

Usage:
    python diagnose_leakage.py
"""

import re
import logging
import pandas as pd
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")


def check_split_leakage():
    log.info("\n" + "=" * 60)
    log.info("CHECK 1: Train / Val / Test Leakage")
    log.info("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["text"])
    val   = pd.read_csv(DATA_DIR / "val.csv",   usecols=["text"])
    test  = pd.read_csv(DATA_DIR / "test.csv",  usecols=["text"])

    train_set = set(train["text"])
    val_set   = set(val["text"])
    test_set  = set(test["text"])

    train_test_overlap = train_set & test_set
    train_val_overlap  = train_set & val_set
    val_test_overlap   = val_set & test_set

    log.info(f"Train size: {len(train_set):,}  Val size: {len(val_set):,}  Test size: {len(test_set):,}")
    log.info(f"Train ∩ Test overlap : {len(train_test_overlap):,} rows  "
             f"({100*len(train_test_overlap)/max(len(test_set),1):.2f}% of test)")
    log.info(f"Train ∩ Val  overlap : {len(train_val_overlap):,} rows")
    log.info(f"Val   ∩ Test overlap : {len(val_test_overlap):,} rows")

    if len(train_test_overlap) > 0:
        log.warning("⚠ LEAKAGE FOUND. This alone can cause artificially perfect scores.")
        log.info(f"Example leaked sentence: {list(train_test_overlap)[0][:150]}")
    else:
        log.info("✓ No exact-duplicate leakage found.")

    # Near-duplicate check (first 8 words match)
    def prefix(t, n=8):
        return " ".join(str(t).split()[:n]).lower()

    train_prefixes = set(train["text"].apply(prefix))
    test_prefixes  = test["text"].apply(prefix)
    near_dup_count = test_prefixes.isin(train_prefixes).sum()
    log.info(f"\nNear-duplicate check (same first 8 words):")
    log.info(f"  Test rows with prefix also seen in train: {near_dup_count:,} "
             f"({100*near_dup_count/len(test):.2f}%)")
    if near_dup_count / len(test) > 0.05:
        log.warning("⚠ High near-duplicate rate — template-based generation is "
                    "likely producing repetitive sentence openings.")


def check_source_imbalance():
    log.info("\n" + "=" * 60)
    log.info("CHECK 2: AI Source / Model Imbalance")
    log.info("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    ai_rows = train[train["label"] == 1]

    if "model" in ai_rows.columns:
        log.info("Distribution of AI generator models in training data:")
        log.info(f"\n{ai_rows['model'].value_counts()}")
    if "style" in ai_rows.columns:
        log.info("\nDistribution of prompt styles in training data:")
        log.info(f"\n{ai_rows['style'].value_counts()}")

    if "model" not in ai_rows.columns and "style" not in ai_rows.columns:
        log.warning("⚠ No 'model' or 'style' column found — cannot check "
                    "generator diversity. Re-export your 100k sample with "
                    "these columns intact from the full dataset.")


def check_lexical_shortcuts():
    log.info("\n" + "=" * 60)
    log.info("CHECK 3: Trivial Lexical Shortcuts")
    log.info("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")

    # Patterns that might be dead giveaways
    patterns = {
        "Plenarsitzung number (e.g. '239. Plenarsitzung')":
            r"\d+\.\s*Plenarsitzung",
        "Party abbreviation in parens (e.g. '(Linke)')":
            r"\((CDU|CSU|SPD|Grüne|FDP|AfD|Linke|BSW)\)",
        "'Meine Damen und Herren' opener":
            r"^Meine Damen und Herren",
        "'Abgeordnetem/Abgeordneter' + name":
            r"Abgeordnet(em|er|en)\s+\w+\s+\w+",
    }

    for label_name, label_val in [("Human", 0), ("AI", 1)]:
        subset = train[train["label"] == label_val]
        log.info(f"\n{label_name} class (n={len(subset):,}):")
        for desc, pattern in patterns.items():
            count = subset["text"].str.contains(pattern, regex=True, na=False).sum()
            pct = 100 * count / len(subset)
            flag = " ⚠ SHORTCUT RISK" if pct > 10 else ""
            log.info(f"  {desc:55s}: {count:>6,} ({pct:5.2f}%){flag}")


def check_length_distribution():
    log.info("\n" + "=" * 60)
    log.info("CHECK 4: Text Length Distribution by Class")
    log.info("=" * 60)

    train = pd.read_csv(DATA_DIR / "train.csv")
    train["word_count"] = train["text"].str.split().apply(len)

    for label_name, label_val in [("Human", 0), ("AI", 1)]:
        subset = train[train["label"] == label_val]["word_count"]
        log.info(f"{label_name}: mean={subset.mean():.1f}  "
                 f"median={subset.median():.1f}  "
                 f"std={subset.std():.1f}  "
                 f"min={subset.min()}  max={subset.max()}")

    human_mean = train[train["label"] == 0]["word_count"].mean()
    ai_mean    = train[train["label"] == 1]["word_count"].mean()
    diff_pct   = abs(human_mean - ai_mean) / human_mean * 100

    if diff_pct > 15:
        log.warning(f"⚠ Length difference between classes is {diff_pct:.1f}% — "
                    f"the model could be using length as a shortcut.")
    else:
        log.info(f"✓ Length distributions are reasonably similar ({diff_pct:.1f}% diff)")


if __name__ == "__main__":
    check_split_leakage()
    check_source_imbalance()
    check_lexical_shortcuts()
    check_length_distribution()
    log.info("\n" + "=" * 60)
    log.info("Diagnosis complete. Review warnings (⚠) above before retraining.")
    log.info("=" * 60)