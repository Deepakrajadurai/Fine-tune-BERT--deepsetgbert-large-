import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    preds = (y_prob >= 0.5).astype(int)
    confs = np.where(preds == 1, y_prob, 1 - y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin] == preds[in_bin])
            avg_confidence_in_bin = np.mean(confs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

def get_length_bin(words):
    if 20 <= words < 40:
        return "20-40"
    elif 40 <= words < 80:
        return "40-80"
    elif 80 <= words < 120:
        return "80-120"
    elif 120 <= words <= 200:
        return "120-200"
    else:
        return "Other"

def compute_dataset_metrics(df):
    y_true = df["label"].astype(int).values
    y_pred = df["prediction"].astype(int).values
    y_prob = df["ai_probability"].astype(float).values
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc": auc,
        "brier": brier,
        "ece": ece
    }

def main():
    print("=== STARTING METRICS COMPUTATION ===")
    
    pred_files = {
        "Original": "results/raw_predictions_original.csv",
        "Rotation 1": "results/raw_predictions_rotation1.csv",
        "Rotation 2": "results/raw_predictions_rotation2.csv"
    }
    
    data = {}
    for name, path in pred_files.items():
        if not os.path.exists(path):
            print(f"Error: {path} not found. Ensure run_inference.py has completed successfully.")
            return
        data[name] = pd.read_csv(path)
    
    # 1. Compute Overall Metrics
    overall_metrics = {}
    for name, df in data.items():
        overall_metrics[name] = compute_dataset_metrics(df)
        
    # 2. Compute Per-Source Metrics
    sources = sorted(data["Original"]["source"].unique())
    per_source_metrics = {name: {} for name in pred_files.keys()}
    for name, df in data.items():
        for source in sources:
            source_df = df[df["source"] == source]
            if not source_df.empty:
                # If there is only one class in this source (e.g. human wiki has label 0 only),
                # standard Macro-F1 needs balanced classes. Let's compute accuracy, and f1 is less meaningful unless combined with humans
                # So we can print per-source Accuracy
                y_true = source_df["label"].astype(int).values
                y_pred = source_df["prediction"].astype(int).values
                acc = accuracy_score(y_true, y_pred)
                per_source_metrics[name][source] = acc
            else:
                per_source_metrics[name][source] = None
                
    # 3. Compute Metrics by Word Count Length Bins
    length_bin_metrics = {}
    for name, df in data.items():
        df = df.copy()
        df["length_bin"] = df["words"].apply(get_length_bin)
        length_bin_metrics[name] = {}
        for lbin in ["20-40", "40-80", "80-120", "120-200", "Other"]:
            bin_df = df[df["length_bin"] == lbin]
            if not bin_df.empty:
                y_true = bin_df["label"].astype(int).values
                y_pred = bin_df["prediction"].astype(int).values
                acc = accuracy_score(y_true, y_pred)
                count = len(bin_df)
                length_bin_metrics[name][lbin] = (acc, count)
            else:
                length_bin_metrics[name][lbin] = (0.0, 0)

    # 4. Generate report
    report_path = "results/evaluation_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Validation & Generalization Evaluation Report\n\n")
        
        f.write("This report summarizes the rigorous validation results of the GBERT-large German AI vs. Human text detector.\n\n")
        
        f.write("## Overall Calibration & Accuracy Diagnostics\n\n")
        f.write("| Model Setup | Accuracy | Macro F1 | ROC-AUC | ECE (10 bins) | Brier Score |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name in pred_files.keys():
            m = overall_metrics[name]
            f.write(f"| {name} | {m['accuracy']*100:.2f}% | {m['f1_macro']*100:.2f}% | {m['roc_auc']:.4f} | {m['ece']:.6f} | {m['brier']:.6f} |\n")
        f.write("\n")
        
        f.write("## Cross-Domain Transfer Matrix (Accuracy by Source)\n\n")
        f.write("| Source Domain | Class | Original Model | Rotation 1 (Gemini Held-Out) | Rotation 2 (Qwen Held-Out) |\n")
        f.write("|---|---|---|---|---|\n")
        for s in sources:
            # Determine class label for this source
            first_row = data["Original"][data["Original"]["source"] == s].iloc[0]
            lbl = "AI" if int(first_row["label"]) == 1 else "Human"
            
            acc_orig = per_source_metrics["Original"].get(s)
            acc_r1 = per_source_metrics["Rotation 1"].get(s)
            acc_r2 = per_source_metrics["Rotation 2"].get(s)
            
            orig_str = f"{acc_orig*100:.2f}%" if acc_orig is not None else "N/A"
            r1_str = f"{acc_r1*100:.2f}%" if acc_r1 is not None else "N/A"
            r2_str = f"{acc_r2*100:.2f}%" if acc_r2 is not None else "N/A"
            
            f.write(f"| {s} | {lbl} | {orig_str} | {r1_str} | {r2_str} |\n")
        f.write("\n")
        
        f.write("## Performance by Text Length Bins\n\n")
        f.write("| Length Bin (Words) | Original (Acc / Count) | Rotation 1 (Acc / Count) | Rotation 2 (Acc / Count) |\n")
        f.write("|---|---|---|---|\n")
        for lbin in ["20-40", "40-80", "80-120", "120-200", "Other"]:
            o_acc, o_cnt = length_bin_metrics["Original"][lbin]
            r1_acc, r1_cnt = length_bin_metrics["Rotation 1"][lbin]
            r2_acc, r2_cnt = length_bin_metrics["Rotation 2"][lbin]
            
            f.write(f"| {lbin} | {o_acc*100:.2f}% ({o_cnt}) | {r1_acc*100:.2f}% ({r1_cnt}) | {r2_acc*100:.2f}% ({r2_cnt}) |\n")
        f.write("\n")
        
    print(f"Metrics report written to {report_path}")
    print("=== METRICS COMPUTATION COMPLETE ===")

if __name__ == "__main__":
    main()
