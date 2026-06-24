"""
Step 2 (FIXED) — Model Training with Generalization Tracking
============================================================
- Loads train, in-distribution validation, and external validation sets
- Employs Hugging Face Trainer with multi-dataset evaluation
- Calibrates checkpoint selection based on the external validation set's F1 score to prevent overfitting
- Integrates Early Stopping and prints a source-aware accuracy table at each epoch end
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1_macro,
        'precision': precision_macro,
        'recall': recall_macro
    }

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM GENERALIZATION TRACKING CALLBACK
# ──────────────────────────────────────────────────────────────────────────────
class GeneralizationTrackerCallback(TrainerCallback):
    def __init__(self, ext_val_df, tokenizer, device, threshold=0.10):
        self.ext_val_df = ext_val_df
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Trigger report only when evaluation finishes on the external set
        # Hugging Face appends dataset name to keys (e.g. eval_external_accuracy)
        if "eval_external_f1" in metrics:
            print("\n" + "=" * 70)
            print(f"EPOCH {state.epoch:.1f} - EXTERNAL VALIDATION SOURCE BREAKDOWN")
            print("=" * 70)
            print(f"{'Source':<35} | {'Samples':<8} | {'Correct':<8} | {'Accuracy':<8}")
            print("-" * 70)
            
            model = kwargs.get("model")
            model.eval()
            
            sources = self.ext_val_df["source"].unique()
            for source in sources:
                sub_df = self.ext_val_df[self.ext_val_df["source"] == source]
                texts = sub_df["text"].tolist()
                labels = sub_df["label"].tolist()
                
                correct = 0
                for text, label in zip(texts, labels):
                    enc = self.tokenizer(text, max_length=256, padding=True, truncation=True, return_tensors="pt")
                    with torch.no_grad():
                        logits = model(input_ids=enc["input_ids"].to(self.device), attention_mask=enc["attention_mask"].to(self.device)).logits
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                    pred = 1 if probs[1] >= self.threshold else 0
                    if pred == label:
                        correct += 1
                
                acc = correct / len(sub_df) if len(sub_df) > 0 else 0
                print(f"{source:<35} | {len(sub_df):<8} | {correct:<8} | {acc * 100:.1f}%")
            print("=" * 70 + "\n")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING PROCESS
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune German BERT with Generalization Guard")
    parser.add_argument('--model_name', type=str, default='deepset/gbert-large',
                        help="HuggingFace model identifier")
    parser.add_argument('--train_csv', type=str, default='Data/train.csv')
    parser.add_argument('--val_csv', type=str, default='Data/val.csv')
    parser.add_argument('--ext_val_csv', type=str, default='Data/external_val.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--output_dir', type=str, default='models/best_model')
    parser.add_argument('--threshold', type=float, default=0.10, help="Classification decision threshold")
    args = parser.parse_args()

    # Verify input datasets
    for path in [args.train_csv, args.val_csv, args.ext_val_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required split not found at {path}. Run generate_synthetic_data.py first.")

    # 1. Load DataFrames
    log.info("Loading training, in-distribution validation, and external validation sets...")
    train_df = pd.read_csv(args.train_csv).dropna(subset=['text'])
    val_df = pd.read_csv(args.val_csv).dropna(subset=['text'])
    ext_val_df = pd.read_csv(args.ext_val_csv).dropna(subset=['text'])

    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
    ext_val_dataset = Dataset.from_pandas(ext_val_df[['text', 'label']])

    # 2. Initialize Tokenizer
    log.info(f"Loading tokenizer: {args.model_name}...")
    if 'gbert' in args.model_name:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=args.max_length
        )

    log.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    ext_val_dataset = ext_val_dataset.map(preprocess_function, batched=True)

    # 3. Load Pretrained Classification Model
    log.info(f"Loading classification model: {args.model_name}...")
    if 'gbert' in args.model_name:
        model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device.upper()}")
    model.to(device)

    # 4. Configure Training Arguments
    # Note: We evaluate on both in-distribution (val) and external (ext_val) sets
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="eval_external_f1", # Save checkpoint based on external validation macro-F1!
        greater_is_better=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to="none",
        logging_steps=50,
        disable_tqdm=False
    )

    # 5. Initialize Trainer with Callbacks
    tracker_callback = GeneralizationTrackerCallback(ext_val_df, tokenizer, device, args.threshold)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={
            "indist": val_dataset,
            "external": ext_val_dataset
        },
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2),
            tracker_callback
        ]
    )

    # 6. Execute Fine-Tuning
    log.info("Starting model fine-tuning with generalization safeguards...")
    trainer.train()

    # 7. Save Best Model and Tokenizer
    log.info(f"Saving best model checkpoint and tokenizer to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Training complete!")

if __name__ == '__main__':
    main()
