# """
# Step 2 (FIXED) — Model Training with Generalization Tracking
# ============================================================
# - Loads train, in-distribution validation, and external validation sets
# - Employs Hugging Face Trainer with multi-dataset evaluation
# - Calibrates checkpoint selection based on the external validation set's F1 score to prevent overfitting
# - Integrates Early Stopping and prints a source-aware accuracy table at each epoch end
# """

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import argparse
# import logging
# import numpy as np
# import pandas as pd
# import torch
# from datasets import Dataset
# from transformers import (
#     BertTokenizer,
#     BertForSequenceClassification,
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     TrainingArguments,
#     Trainer,
#     EarlyStoppingCallback,
#     TrainerCallback,
#     DataCollatorWithPadding
# )
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# logging.basicConfig(level=logging.INFO,
#                     format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)

# # ──────────────────────────────────────────────────────────────────────────────
# # METRICS COMPUTATION
# # ──────────────────────────────────────────────────────────────────────────────
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     if isinstance(logits, tuple):
#         logits = logits[0]
#     predictions = np.argmax(logits, axis=-1)
    
#     precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
#         labels, predictions, average='macro', zero_division=0
#     )
#     acc = accuracy_score(labels, predictions)
    
#     # Canary metrics to detect model collapse early
#     pct_pred_ai = float(np.mean(predictions == 1))
#     pct_pred_human = float(np.mean(predictions == 0))
    
#     return {
#         'accuracy': acc,
#         'f1': f1_macro,
#         'precision': precision_macro,
#         'recall': recall_macro,
#         'pct_predicted_ai': pct_pred_ai,
#         'pct_predicted_human': pct_pred_human
#     }

# # ──────────────────────────────────────────────────────────────────────────────
# # CUSTOM GENERALIZATION TRACKING CALLBACK
# # ──────────────────────────────────────────────────────────────────────────────
# class GeneralizationTrackerCallback(TrainerCallback):
#     def __init__(self, ext_val_df, tokenizer, device, threshold=0.10):
#         self.ext_val_df = ext_val_df
#         self.tokenizer = tokenizer
#         self.device = device
#         self.threshold = threshold

#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         # Trigger report only when evaluation finishes on the external set
#         # Hugging Face appends dataset name to keys (e.g. eval_external_accuracy)
#         if "eval_external_f1" in metrics:
#             print("\n" + "=" * 70)
#             print(f"EPOCH {state.epoch:.1f} - EXTERNAL VALIDATION SOURCE BREAKDOWN")
#             print("=" * 70)
#             print(f"{'Source':<35} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<8}")
#             print("-" * 70)
            
#             model = kwargs.get("model")
#             model.eval()
            
#             sources = self.ext_val_df["source"].unique()
#             for source in sources:
#                 sub_df = self.ext_val_df[self.ext_val_df["source"] == source]
#                 texts = sub_df["text"].tolist()
#                 labels = sub_df["label"].tolist()
                
#                 correct = 0
#                 for text, label in zip(texts, labels):
#                     enc = self.tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
#                     with torch.no_grad():
#                         logits = model(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device)).logits
#                     probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
#                     pred = 1 if probs[1] >= self.threshold else 0
#                     if pred == label:
#                         correct += 1
                
#                 acc = correct / len(sub_df) if len(sub_df) > 0 else 0
#                 print(f"{source:<35} | {len(sub_df):<8} | {correct:<8} | {acc * 100:.1f}%")
#             print("=" * 70 + "\n")

# # ──────────────────────────────────────────────────────────────────────────────
# # MAIN TRAINING PROCESS
# # ──────────────────────────────────────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(description="Fine-tune German BERT with Generalization Guard")
#     parser.add_argument('--model_name', type=str, default='deepset/gbert-large',
#                         help="HuggingFace model identifier")
#     parser.add_argument('--train_csv', type=str, default='Data/train.csv')
#     parser.add_argument('--val_csv', type=str, default='Data/val.csv')
#     parser.add_argument('--ext_val_csv', type=str, default='Data/external_val.csv')
#     parser.add_argument('--epochs', type=int, default=1)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--lr', type=float, default=1e-5)
#     parser.add_argument('--max_length', type=int, default=256)
#     parser.add_argument('--weight_decay', type=float, default=0.01)
#     parser.add_argument('--warmup_ratio', type=float, default=0.06)
#     parser.add_argument('--output_dir', type=str, default='models/best_model')
#     parser.add_argument('--threshold', type=float, default=0.10, help="Classification decision threshold")
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
#                         help="Number of update steps to accumulate before performing a backward/update pass.")
#     parser.add_argument('--eval_steps', type=int, default=1000,
#                         help="Number of update steps between two evaluations.")
#     parser.add_argument('--max_grad_norm', type=float, default=1.0,
#                         help="Max gradient norm for clipping.")
#     parser.add_argument('--early_stopping_patience', type=int, default=3,
#                         help="Early stopping patience in number of evaluations.")
#     args = parser.parse_args()

#     # Verify input datasets
#     for path in [args.train_csv, args.val_csv, args.ext_val_csv]:
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Required split not found at {path}. Run generate_synthetic_data.py first.")

#     # 1. Load DataFrames
#     log.info("Loading training, in-distribution validation, and external validation sets...")
#     train_df = pd.read_csv(args.train_csv).dropna(subset=['text'])
#     val_df = pd.read_csv(args.val_csv).dropna(subset=['text'])
#     ext_val_df = pd.read_csv(args.ext_val_csv).dropna(subset=['text'])

#     train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
#     val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
#     ext_val_dataset = Dataset.from_pandas(ext_val_df[['text', 'label']])

#     # 2. Initialize Tokenizer
#     log.info(f"Loading tokenizer: {args.model_name}...")
#     if 'gbert' in args.model_name:
#         tokenizer = BertTokenizer.from_pretrained(args.model_name)
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(args.model_name)

#     def preprocess_function(examples):
#         return tokenizer(
#             examples['text'],
#             truncation=True,
#             padding=False,
#             max_length=args.max_length
#         )

#     log.info("Tokenizing datasets...")
#     train_dataset = train_dataset.map(preprocess_function, batched=True)
#     val_dataset = val_dataset.map(preprocess_function, batched=True)
#     ext_val_dataset = ext_val_dataset.map(preprocess_function, batched=True)

#     # 3. Load Pretrained Classification Model
#     log.info(f"Loading classification model: {args.model_name}...")
#     if 'gbert' in args.model_name:
#         model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
#     else:
#         model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     log.info(f"Using device: {device.upper()}")
#     model.to(device)

#     # 4. Configure Training Arguments
#     # Note: We evaluate on both in-distribution (val) and external (ext_val) sets
#     training_args = TrainingArguments(
#         output_dir=args.output_dir,
#         eval_strategy="steps",
#         save_strategy="steps",
#         eval_steps=args.eval_steps,
#         save_steps=args.eval_steps,
#         learning_rate=args.lr,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         max_grad_norm=args.max_grad_norm,
#         num_train_epochs=args.epochs,
#         weight_decay=args.weight_decay,
#         warmup_ratio=args.warmup_ratio,
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_indist_f1",  # Use indist F1 — external eval fires after indist so early stopping can't see it
#         greater_is_better=True,
#         bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
#         fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
#         report_to="none",
#         logging_steps=50,
#         disable_tqdm=False
#     )

#     # 5. Initialize Trainer with Callbacks
#     tracker_callback = GeneralizationTrackerCallback(ext_val_df, tokenizer, device, args.threshold)
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset={
#             "indist": val_dataset,
#             "external": ext_val_dataset
#         },
#         compute_metrics=compute_metrics,
#         data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
#         callbacks=[
#             EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
#             tracker_callback
#         ]
#     )

#     # 6. Execute Fine-Tuning
#     log.info("Starting model fine-tuning with generalization safeguards...")
#     trainer.train()

#     # 7. Save Best Model and Tokenizer
#     log.info(f"Saving best model checkpoint and tokenizer to {args.output_dir}...")
#     trainer.save_model(args.output_dir)
#     tokenizer.save_pretrained(args.output_dir)
#     log.info("Training complete!")

# if __name__ == '__main__':
#     main()
"""
train.py

Full fine-tuning script for deepset/gbert-large on training_pair_v5's
train/val/external_val splits, with:
  - Collapse-detection metrics (pct_predicted_ai / pct_predicted_human)
  - Gradient clipping + warmup to address the prior collapse failure mode
  - Step-based eval/save/early-stopping (not epoch-based)
  - Timestamped progress logging with throughput and ETA

Usage:
    python train.py \
        --train_csv Data/train.csv --val_csv Data/val.csv \
        --ext_val_csv Data/external_val.csv \
        --epochs 1 --batch_size 16 --lr 1e-5 \
        --gradient_accumulation_steps 4 --warmup_ratio 0.06 \
        --max_grad_norm 1.0 --eval_steps 1000 --early_stopping_patience 3 \
        --log_every_n_steps 100
"""

import argparse
import re
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)


# ---------------------------------------------------------------------------
# Whitespace normalization (defense-in-depth against the newline label-leak
# found in training_pair_v5.csv: 66.05% of AI rows carried embedded
# paragraph-break newlines while 0% of human rows did, since human text was
# built by joining sentences with a single space. Applied uniformly to both
# classes so it can never re-emerge as a residual shortcut even if a future
# data refresh reintroduces the asymmetry upstream.)
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


# ---------------------------------------------------------------------------
# Progress logging callback
# ---------------------------------------------------------------------------

class ProgressLoggerCallback(TrainerCallback):
    """Logs a timestamped progress line every `log_every_n_steps` steps:
    elapsed time, steps/sec, and an ETA to run completion based on
    current throughput. Also logs a timestamped summary on every eval."""

    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        now = datetime.now().strftime("%H:%M:%S")
        total_steps = state.max_steps if state.max_steps > 0 else "unknown"
        print(f"[{now}] Training started. Total planned steps: {total_steps}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps != 0:
            return
        elapsed = time.time() - self.start_time
        steps_done = state.global_step
        now = datetime.now().strftime("%H:%M:%S")

        if steps_done > 0 and state.max_steps > 0:
            steps_per_sec = steps_done / elapsed
            remaining_steps = state.max_steps - steps_done
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_time = datetime.now() + timedelta(seconds=eta_seconds)
            pct_done = 100 * steps_done / state.max_steps
            print(f"[{now}] step {steps_done}/{state.max_steps} "
                  f"({pct_done:.1f}%) | "
                  f"elapsed {timedelta(seconds=int(elapsed))} | "
                  f"{steps_per_sec:.2f} steps/s | "
                  f"ETA {eta_time.strftime('%H:%M:%S')} "
                  f"(in {timedelta(seconds=int(eta_seconds))})")
        else:
            print(f"[{now}] step {steps_done} | "
                  f"elapsed {timedelta(seconds=int(elapsed))}")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        now = datetime.now().strftime("%H:%M:%S")
        elapsed = time.time() - self.start_time if self.start_time else 0
        f1 = metrics.get("eval_macro_f1") if metrics else None
        pct_ai = metrics.get("eval_pct_predicted_ai") if metrics else None
        pct_human = metrics.get("eval_pct_predicted_human") if metrics else None
        print(f"[{now}] EVAL at step {state.global_step} "
              f"(elapsed {timedelta(seconds=int(elapsed))}): "
              f"macro_f1={f1}, pct_predicted_ai={pct_ai}, pct_predicted_human={pct_human}")
        if pct_ai is not None and (pct_ai > 0.98 or pct_ai < 0.02):
            print(f"[{now}] WARNING: predictions are >98% one class -- "
                  f"likely collapse in progress. Consider stopping this run.")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time if self.start_time else 0
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] Training complete. Total time: "
              f"{timedelta(seconds=int(elapsed))}")


# ---------------------------------------------------------------------------
# Data loading + tokenization
# ---------------------------------------------------------------------------

def load_and_tokenize(train_csv, val_csv, tokenizer, max_length=512):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)
    train_df["text"] = train_df["text"].apply(normalize_text)
    val_df["text"] = val_df["text"].apply(normalize_text)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length,
                          padding="max_length")

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    val_ds = val_ds.remove_columns(["text"])
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Metrics with collapse detection
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)

    pred_counts = np.bincount(preds, minlength=2)
    pct_predicted_human = pred_counts[0] / len(preds)  # label 0 = human
    pct_predicted_ai = pred_counts[1] / len(preds)     # label 1 = ai
    print(f"[eval] predicted class distribution: "
          f"human={pct_predicted_human:.1%}, ai={pct_predicted_ai:.1%}")
    if pct_predicted_human > 0.98 or pct_predicted_ai > 0.98:
        print("[eval] WARNING: predictions are >98% one class -- "
              "likely collapse in progress.")

    macro_f1 = f1_score(labels, preds, average="macro")
    accuracy = accuracy_score(labels, preds)
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "pct_predicted_ai": pct_predicted_ai,
        "pct_predicted_human": pct_predicted_human,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--ext_val_csv", required=False, default=None,
                         help="External validation set -- evaluated once at the end "
                              "of training as an extra sanity check, not used for "
                              "early stopping or best-model selection.")
    parser.add_argument("--model_name", default="deepset/gbert-large")
    parser.add_argument("--output_dir", default="models/best_model")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--log_every_n_steps", type=int, default=100)

    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)

    args = parser.parse_args()

    print(f"Loading tokenizer/model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"AutoTokenizer failed ({e}). Falling back to BertTokenizer...")
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label={0: "human", 1: "ai"},
            label2id={"human": 0, "ai": 1},
        )
    except Exception as e:
        print(f"AutoModel failed ({e}). Falling back to BertForSequenceClassification...")
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label={0: "human", 1: "ai"},
            label2id={"human": 0, "ai": 1},
        )

    print("Checking sequence length distribution against max_length...")
    sample_df = pd.read_csv(args.train_csv, nrows=5000)
    sample_lengths = sample_df["text"].apply(lambda t: len(tokenizer.encode(str(t))))
    over_limit = (sample_lengths > args.max_length).mean()
    print(f"  sample of {len(sample_df)} rows: "
          f"{over_limit:.2%} exceed {args.max_length} tokens "
          f"(mean={sample_lengths.mean():.0f}, p95={sample_lengths.quantile(0.95):.0f})")
    if over_limit > 0.03:
        print(f"  WARNING: >3% of rows will be truncated. Consider whether this is "
              f"evenly distributed across domains before proceeding "
              f"(state_law is expected to be hit hardest).")

    print("Tokenizing train/val splits...")
    train_ds, val_ds = load_and_tokenize(args.train_csv, args.val_csv, tokenizer,
                                          max_length=args.max_length)
    print(f"  train: {len(train_ds)} rows, val: {len(val_ds)} rows")

    steps_per_epoch = len(train_ds) // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    print(f"  effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  steps/epoch: {steps_per_epoch}, total planned steps: {total_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,

        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,

        logging_strategy="steps",
        logging_steps=args.log_every_n_steps,
        disable_tqdm=False,
        report_to=[],

        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            ProgressLoggerCallback(log_every_n_steps=args.log_every_n_steps),
        ],
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.ext_val_csv:
        print(f"Evaluating on external validation set: {args.ext_val_csv}")
        ext_df = pd.read_csv(args.ext_val_csv)
        ext_df["label"] = ext_df["label"].astype(int)
        ext_df["text"] = ext_df["text"].apply(normalize_text)
        ext_ds = Dataset.from_pandas(ext_df[["text", "label"]], preserve_index=False)

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=args.max_length,
                              padding="max_length")

        ext_ds = ext_ds.map(tokenize_fn, batched=True)
        ext_ds = ext_ds.remove_columns(["text"])
        ext_ds.set_format("torch")

        ext_metrics = trainer.evaluate(eval_dataset=ext_ds, metric_key_prefix="ext_val")
        print("External validation metrics:", ext_metrics)

    print("Done. Run evaluate.py against test.csv and final_holdout.csv next.")


if __name__ == "__main__":
    main()
