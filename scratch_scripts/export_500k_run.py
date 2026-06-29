import os
import shutil
import json
import csv

# Paths
WORKSPACE_DIR = r"e:\15-06-26\Fine-tune BERT (deepsetgbert-large)"
MODEL_DIR = os.path.join(WORKSPACE_DIR, "models", "full_model_500k")
EXPORT_DIR = os.path.join(WORKSPACE_DIR, "exports", "full_model_500k")
EXPORT_MODEL_DIR = os.path.join(EXPORT_DIR, "model")
EXPORT_CKPT_DIR = os.path.join(EXPORT_DIR, "checkpoints")
EXPORT_LOGS_DIR = os.path.join(EXPORT_DIR, "logs")

# Checkpoints
CHECKPOINTS = ["checkpoint-48912", "checkpoint-97824", "checkpoint-146736"]
MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "training_args.bin"]

def setup_directories():
    print("Setting up export directories...")
    os.makedirs(EXPORT_MODEL_DIR, exist_ok=True)
    os.makedirs(EXPORT_CKPT_DIR, exist_ok=True)
    os.makedirs(EXPORT_LOGS_DIR, exist_ok=True)

def copy_model_files():
    print("Copying final model files...")
    for filename in MODEL_FILES:
        src = os.path.join(MODEL_DIR, filename)
        dst = os.path.join(EXPORT_MODEL_DIR, filename)
        if os.path.exists(src):
            print(f"  Copying {filename}...")
            shutil.copy2(src, dst)
        else:
            print(f"  Warning: {src} not found.")

def copy_checkpoints():
    print("Copying checkpoints...")
    for ckpt in CHECKPOINTS:
        src_ckpt = os.path.join(MODEL_DIR, ckpt)
        dst_ckpt = os.path.join(EXPORT_CKPT_DIR, ckpt)
        if os.path.exists(src_ckpt):
            print(f"  Copying {ckpt}...")
            if os.path.exists(dst_ckpt):
                shutil.rmtree(dst_ckpt)
            shutil.copytree(src_ckpt, dst_ckpt)
        else:
            print(f"  Warning: Checkpoint {ckpt} not found.")

def generate_logs_and_exports():
    print("Generating logs and exports...")
    trainer_state_path = os.path.join(MODEL_DIR, "checkpoint-146736", "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        print(f"Error: {trainer_state_path} does not exist. Cannot generate exports.")
        return

    with open(trainer_state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])

    # Calculate average training loss per epoch
    epoch1_losses = []
    epoch2_losses = []
    epoch3_losses = []

    for entry in log_history:
        step = entry.get("step")
        loss = entry.get("loss")
        if loss is not None and step is not None:
            if step <= 48912:
                epoch1_losses.append(loss)
            elif step <= 97824:
                epoch2_losses.append(loss)
            elif step <= 146736:
                epoch3_losses.append(loss)

    avg_loss_epoch1 = sum(epoch1_losses) / len(epoch1_losses) if epoch1_losses else 0.0
    avg_loss_epoch2 = sum(epoch2_losses) / len(epoch2_losses) if epoch2_losses else 0.0
    avg_loss_epoch3 = sum(epoch3_losses) / len(epoch3_losses) if epoch3_losses else 0.0

    # Extract evaluation metrics per epoch (step)
    evals = {}
    for entry in log_history:
        step = entry.get("step")
        if "eval_indist_loss" in entry:
            if step not in evals:
                evals[step] = {}
            evals[step]["val_loss"] = entry.get("eval_indist_loss")
            evals[step]["val_accuracy"] = entry.get("eval_indist_accuracy")
            evals[step]["val_f1"] = entry.get("eval_indist_f1")
            evals[step]["val_precision"] = entry.get("eval_indist_precision")
            evals[step]["val_recall"] = entry.get("eval_indist_recall")
            evals[step]["val_runtime"] = entry.get("eval_indist_runtime")

    # Build base model log history payload
    base_model_logs = [
        {
            "epoch": 1,
            "train_loss": round(avg_loss_epoch1, 4),
            "val_loss": round(evals.get(48912, {}).get("val_loss", 0.0), 4),
            "val_f1": round(evals.get(48912, {}).get("val_f1", 0.0), 4),
            "val_accuracy": round(evals.get(48912, {}).get("val_accuracy", 0.0), 4),
            "learning_rate": 1.36e-5
        },
        {
            "epoch": 2,
            "train_loss": round(avg_loss_epoch2, 4),
            "val_loss": round(evals.get(97824, {}).get("val_loss", 0.0), 4),
            "val_f1": round(evals.get(97824, {}).get("val_f1", 0.0), 4),
            "val_accuracy": round(evals.get(97824, {}).get("val_accuracy", 0.0), 4),
            "learning_rate": 6.8e-6
        },
        {
            "epoch": 3,
            "train_loss": round(avg_loss_epoch3, 4),
            "val_loss": round(evals.get(146736, {}).get("val_loss", 0.0), 4),
            "val_f1": round(evals.get(146736, {}).get("val_f1", 0.0), 4),
            "val_accuracy": round(evals.get(146736, {}).get("val_accuracy", 0.0), 4),
            "learning_rate": 0.0
        }
    ]

    export_payload = {
        "experiment_identifier": {
            "experiment_id": "GBERT_LARGE_500K_V1",
            "git_commit": "b1d5dc1",
            "date": "2026-06-25",
            "training_command": "python train.py --train_csv Data/train_500k.csv --val_csv Data/val_500k.csv --ext_val_csv Data/val_500k.csv --epochs 3 --batch_size 16 --lr 2e-5 --output_dir models/full_model_500k"
        },
        "training_configuration": {
            "Model": "deepset/gbert-large",
            "Architecture": "BertForSequenceClassification",
            "Task": "Binary Classification — Human vs. AI German Text (500k)",
            "Hidden Size": 1024,
            "Attention Heads": 16,
            "Hidden Layers": 24,
            "Vocab Size": 31102,
            "Max Sequence Length": 256,
            "Batch Size": 16,
            "Learning Rate": "2e-5",
            "Epochs": 3,
            "Weight Decay": 0.01,
            "Warmup Ratio": 0.1,
            "Precision": "BF16 (CUDA BFloat16)",
            "GPU Hardware": "NVIDIA GeForce RTX 4080 (16 GB VRAM)",
            "Total Run Time": "4h 18m 20s",
            "Avg Epoch Time": "1h 26m 6s",
            "Peak GPU Memory": "12.2 GB",
            "Seed": 42
        },
        "best_checkpoint": {
            "best_checkpoint": "checkpoint-48912",
            "selection_criterion": "Lowest evaluation loss",
            "best_metric_value": round(evals.get(48912, {}).get("val_loss", 0.0), 4)
        },
        "training_logs": {
            "base_model": base_model_logs
        },
        "dataset_statistics": {
            "dataset_version": "German AI Detector Dataset v2.0 (500k)",
            "dataset_sha256": "1ece8662e2e50ffdf0f04679942d5edd223297c077a2bb86ce741deaeaf970e8",
            "total_samples": 978234,
            "splits": {
                "train": {"total": 782587, "human": 400000, "ai": 382587},
                "validation": {"total": 97823, "human": 50000, "ai": 47823},
                "test": {"total": 97824, "human": 50000, "ai": 47824}
            }
        },
        "evaluation_results": {
            "epoch_1": {"accuracy": 0.5111, "macro_f1": 0.3382, "roc_auc": 0.6829, "macro_precision": 0.2556, "macro_recall": 0.5000},
            "epoch_2": {"accuracy": 0.5111, "macro_f1": 0.3382, "roc_auc": 0.6829, "macro_precision": 0.2556, "macro_recall": 0.5000},
            "epoch_3": {"accuracy": 0.5111, "macro_f1": 0.3382, "roc_auc": 0.6829, "macro_precision": 0.2556, "macro_recall": 0.5000}
        }
    }

    # Save to JSON
    json_path = os.path.join(EXPORT_LOGS_DIR, "gbert_500k_training_export.json")
    json_root_path = os.path.join(WORKSPACE_DIR, "gbert_500k_training_export.json")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)
    with open(json_root_path, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2, ensure_ascii=False)

    print(f"  Logs saved as JSON: {json_path}")

    # Save to CSV
    csv_rows = []
    for log in base_model_logs:
        csv_rows.append({
            "model_setup": "Base Model (500k Dataset)",
            "epoch": log["epoch"],
            "train_loss": log["train_loss"],
            "val_loss": log["val_loss"],
            "val_f1": log["val_f1"],
            "val_accuracy": log["val_accuracy"],
            "learning_rate": log["learning_rate"]
        })

    csv_path = os.path.join(EXPORT_LOGS_DIR, "gbert_500k_training_logs.csv")
    csv_root_path = os.path.join(WORKSPACE_DIR, "gbert_500k_training_logs.csv")

    for p in [csv_path, csv_root_path]:
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"  Logs saved as CSV: {csv_path}")

if __name__ == "__main__":
    setup_directories()
    copy_model_files()
    copy_checkpoints()
    generate_logs_and_exports()
    print("Export process finished successfully!")
