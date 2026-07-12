"""
Leakage diagnostic for training_pair_v5_clean.csv
---------------------------------------------------
Given a TF-IDF + LogisticRegression model hitting 1.0000 accuracy/AUC,
this script surfaces WHY: top discriminative tokens (likely artifacts),
and checks for near-duplicate / group leakage across train/test.

Usage:
    python leakage_diagnostic.py --csv training_pair_v5_clean.csv \
        --text_col text --label_col label
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--top_n", type=int, default=40, help="How many top tokens per class to show")
    parser.add_argument("--dup_check_n", type=int, default=200000,
                         help="Max rows to use for near-duplicate check (for speed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[info] Loading {args.csv} ...")
    df = pd.read_csv(args.csv)
    df = df[[args.text_col, args.label_col]].dropna()

    # --- 1. Exact duplicate check across whole dataset ---
    n_total = len(df)
    n_unique_text = df[args.text_col].nunique()
    print(f"\n=== EXACT DUPLICATE CHECK ===")
    print(f"Total rows: {n_total:,} | Unique text rows: {n_unique_text:,}")
    if n_unique_text < n_total:
        print(f"[WARNING] {n_total - n_unique_text:,} exact duplicate text rows found.")
        dupe_examples = df[df.duplicated(subset=[args.text_col], keep=False)].sort_values(args.text_col)
        print("Sample duplicate rows:")
        print(dupe_examples.head(10))
    else:
        print("[OK] No exact duplicate text rows.")

    # --- 2. Train/test split + fit (same as baseline) ---
    X_train, X_test, y_train, y_test = train_test_split(
        df[args.text_col].astype(str).to_numpy(),
        df[args.label_col].to_numpy(),
        test_size=0.2,
        random_state=args.seed,
        stratify=df[args.label_col].to_numpy(),
    )

    # --- 3. Cross-split exact/near duplicate check ---
    print(f"\n=== TRAIN/TEST OVERLAP CHECK ===")
    train_set = set(X_train[:args.dup_check_n])
    test_set = set(X_test[:args.dup_check_n])
    overlap = train_set & test_set
    print(f"Exact text overlap between train and test: {len(overlap):,} rows")
    if overlap:
        print("[WARNING] Identical text present in both train and test — this alone can fully explain 100% accuracy.")
        print("Example overlapping text (truncated):")
        for ex in list(overlap)[:3]:
            print(" -", ex[:200])

    # rough near-duplicate proxy: shared first 50 chars (prefix leakage from your known repair bug)
    train_prefixes = set(t[:50] for t in X_train)
    test_prefixes = set(t[:50] for t in X_test)
    prefix_overlap = train_prefixes & test_prefixes
    print(f"\nShared 50-char prefixes between train/test: {len(prefix_overlap):,}")
    if len(prefix_overlap) > 0:
        print("[WARNING] Possible prefix-level near-duplicate leakage (matches your known prefix-dedup issue).")

    # --- 4. Fit model and inspect top coefficients ---
    print(f"\n=== FITTING MODEL FOR COEFFICIENT INSPECTION ===")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
    clf.fit(X_train_tfidf, y_train)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]

    top_ai_idx = np.argsort(coefs)[-args.top_n:][::-1]
    top_human_idx = np.argsort(coefs)[:args.top_n]

    print(f"\n=== TOP {args.top_n} TOKENS PREDICTING 'AI' ===")
    for i in top_ai_idx:
        print(f"  {feature_names[i]:<30} coef={coefs[i]:.4f}")

    print(f"\n=== TOP {args.top_n} TOKENS PREDICTING 'HUMAN' ===")
    for i in top_human_idx:
        print(f"  {feature_names[i]:<30} coef={coefs[i]:.4f}")

    print("\n[info] If these lists show boilerplate artifacts (dates, formatting tokens, source-specific")
    print("       phrases like 'Plenarsitzung', placeholder text, disclaimer strings, etc.) rather than")
    print("       genuinely meaningful linguistic differences, that confirms shortcut/leakage — not real learning.")


if __name__ == "__main__":
    main()
