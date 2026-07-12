"""
TF-IDF + Logistic Regression diagnostic baseline
--------------------------------------------------
Purpose: fast, independent sanity check to isolate whether the
"identical metrics across epochs" bug in the GBERT pipeline lives
in the model/training loop, or in the shared eval/metrics code
(evaluate.py) / data (training_pair_v5.csv).

This script is deliberately dependency-light (sklearn only) and
trains in seconds to minutes, not hours, so you can iterate on the
harness itself.

Usage:
    python tfidf_logreg_baseline.py --csv training_pair_v5.csv \
        --text_col text --label_col label

If your column names differ, pass them explicitly with --text_col / --label_col.
The script will also try to auto-detect common column names if not provided.
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


CANDIDATE_TEXT_COLS = ["text", "content", "sentence", "body"]
CANDIDATE_LABEL_COLS = ["label", "is_ai", "target", "y", "class"]


def autodetect_column(df, candidates, kind):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not auto-detect {kind} column. "
        f"Available columns: {list(df.columns)}. "
        f"Pass --{kind}_col explicitly."
    )


def normalize_labels(series):
    """Map arbitrary label encodings to 0/1 (0=Human, 1=AI)."""
    unique_vals = series.unique()
    if set(unique_vals) <= {0, 1}:
        return series.astype(int)

    # try common string encodings
    mapping_candidates = [
        {"human": 0, "ai": 1},
        {"Human": 0, "AI": 1},
        {"human": 0, "generated": 1},
        {"real": 0, "fake": 1},
    ]
    for mapping in mapping_candidates:
        if set(unique_vals) <= set(mapping.keys()):
            print(f"[info] Mapping labels using: {mapping}")
            return series.map(mapping).astype(int)

    raise ValueError(
        f"Unrecognized label encoding: {unique_vals}. "
        f"Expected 0/1 or one of the known string mappings. "
        f"Edit normalize_labels() to add your mapping."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=None, help="Path to single CSV to split into train/test")
    parser.add_argument("--train_csv", default=None, help="Path to training CSV (use with --test_csv)")
    parser.add_argument("--test_csv", default=None, help="Path to test CSV (use with --train_csv)")
    parser.add_argument("--text_col", default=None, help="Name of text column (auto-detected if omitted)")
    parser.add_argument("--label_col", default=None, help="Name of label column (auto-detected if omitted)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Held-out fraction for evaluation")
    parser.add_argument("--max_features", type=int, default=50000, help="Max TF-IDF vocabulary size")
    parser.add_argument("--ngram_max", type=int, default=2, help="Max n-gram size (1=unigrams, 2=bigrams too)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: subsample N rows for an even faster smoke test")
    args = parser.parse_args()

    t0 = time.time()
    if args.train_csv and args.test_csv:
        print(f"[info] Loading train: {args.train_csv} | test: {args.test_csv} ...")
        df_train = pd.read_csv(args.train_csv)
        df_test = pd.read_csv(args.test_csv)
        print(f"[info] Loaded train: {len(df_train):,} rows, test: {len(df_test):,} rows")

        text_col = args.text_col or autodetect_column(df_train, CANDIDATE_TEXT_COLS, "text")
        label_col = args.label_col or autodetect_column(df_train, CANDIDATE_LABEL_COLS, "label")
        print(f"[info] Using text_col='{text_col}', label_col='{label_col}'")

        df_train = df_train[[text_col, label_col]].dropna()
        df_test = df_test[[text_col, label_col]].dropna()

        df_train[label_col] = normalize_labels(df_train[label_col])
        df_test[label_col] = normalize_labels(df_test[label_col])

        if args.sample:
            df_train = df_train.sample(n=min(args.sample, len(df_train)), random_state=args.seed)
            df_test = df_test.sample(n=min(max(10, args.sample // 4), len(df_test)), random_state=args.seed)
            print(f"[info] Subsampled to train={len(df_train):,}, test={len(df_test):,}")

        print(f"[info] Train class balance:\n{df_train[label_col].value_counts()}")
        print(f"[info] Test class balance:\n{df_test[label_col].value_counts()}")

        X_train = df_train[text_col].astype(str).to_numpy()
        y_train = df_train[label_col].to_numpy()
        X_test = df_test[text_col].astype(str).to_numpy()
        y_test = df_test[label_col].to_numpy()
    else:
        csv_file = args.csv or args.train_csv
        if not csv_file:
            raise ValueError("Must provide either --csv or both --train_csv and --test_csv")
        print(f"[info] Loading {csv_file} ...")
        df = pd.read_csv(csv_file)
        print(f"[info] Loaded {len(df):,} rows, columns: {list(df.columns)}")

        text_col = args.text_col or autodetect_column(df, CANDIDATE_TEXT_COLS, "text")
        label_col = args.label_col or autodetect_column(df, CANDIDATE_LABEL_COLS, "label")
        print(f"[info] Using text_col='{text_col}', label_col='{label_col}'")

        df = df[[text_col, label_col]].dropna()
        df[label_col] = normalize_labels(df[label_col])

        if args.sample:
            df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
            print(f"[info] Subsampled to {len(df):,} rows")

        print(f"[info] Class balance:\n{df[label_col].value_counts()}")

        X_train, X_test, y_train, y_test = train_test_split(
            df[text_col].astype(str).to_numpy(),
            df[label_col].to_numpy(),
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df[label_col].to_numpy(),
        )
    print(f"[info] Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("[info] Fitting TF-IDF vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        sublinear_tf=True,
        min_df=2,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[info] TF-IDF vocab size: {len(vectorizer.vocabulary_):,}")

    print("[info] Training Logistic Regression ...")
    clf = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_tfidf, y_train)

    print("[info] Evaluating ...")
    preds = clf.predict(X_test_tfidf)
    probs = clf.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    macro_precision = precision_score(y_test, preds, average="macro")
    macro_recall = recall_score(y_test, preds, average="macro")
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 60)
    print("TF-IDF + LOGISTIC REGRESSION BASELINE RESULTS")
    print("=" * 60)
    print(f"Accuracy:         {acc:.4f}")
    print(f"Macro F1:         {macro_f1:.4f}")
    print(f"Macro Precision:  {macro_precision:.4f}")
    print(f"Macro Recall:     {macro_recall:.4f}")
    print(f"ROC-AUC:          {auc:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted, [Human, AI]):")
    print(cm)
    print("\nFull classification report:")
    print(classification_report(y_test, preds, target_names=["Human", "AI"]))

    print(f"\n[info] Prediction distribution: {np.bincount(preds)} "
          f"(class 0 / class 1 counts)")
    print(f"[info] Probability stats: mean={probs.mean():.4f}, "
          f"std={probs.std():.4f}, min={probs.min():.4f}, max={probs.max():.4f}")

    n_unique_preds = len(np.unique(preds))
    if n_unique_preds == 1:
        print("\n[WARNING] Model predicted only ONE class for all test samples.")
        print("          This mirrors the GBERT collapse — suggests a real")
        print("          data/label problem (e.g. near-duplicate leakage,")
        print("          degenerate labels, or a genuinely unlearnable split),")
        print("          NOT a GBERT-specific or eval-harness-specific bug.")
    else:
        print("\n[OK] Model produced varying predictions across both classes.")
        print("     If GBERT's per-epoch metrics stayed frozen while this baseline")
        print("     moves normally, that strongly points to a bug in the GBERT")
        print("     eval/checkpoint-loading code, not the data or task itself.")

    print(f"\n[info] Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
