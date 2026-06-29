import subprocess
import sys
import os

def main():
    print("======================================================================")
    print("STARTING GERMAN LEGAL DATASET FINE-TUNING PIPELINE")
    print("======================================================================")
    
    # 1. Run Fine-Tuning
    train_cmd = [
        sys.executable,
        "train.py",
        "--train_csv", "Data/train_legal.csv",
        "--val_csv", "Data/val_legal.csv",
        "--ext_val_csv", "Data/val_legal.csv",
        "--output_dir", "models/legal_model",
        "--epochs", "2",
        "--lr", "2e-6",
        "--batch_size", "16"
    ]
    
    print("\nExecuting Training command:")
    print(" ".join(train_cmd))
    
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
        sys.exit(1)
        
    # 2. Run Evaluation at default threshold (0.50)
    print("\n======================================================================")
    print("EVALUATING MODEL AT DEFAULT THRESHOLD (0.50)")
    print("======================================================================")
    
    eval_cmd_50 = [
        sys.executable,
        "evaluate .py",
        "--model_dir", "models/legal_model",
        "--test_csv", "Data/test_legal.csv",
        "--holdout_csv", "Data/test_legal.csv",
        "--threshold", "0.50"
    ]
    
    print("\nExecuting Evaluation (0.50) command:")
    print(" ".join(eval_cmd_50))
    
    try:
        subprocess.run(eval_cmd_50, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation (0.50): {e}")
        sys.exit(1)

    # 3. Run Evaluation at threshold (0.10)
    print("\n======================================================================")
    print("EVALUATING MODEL AT CALIBRATED THRESHOLD (0.10)")
    print("======================================================================")
    
    eval_cmd_10 = [
        sys.executable,
        "evaluate .py",
        "--model_dir", "models/legal_model",
        "--test_csv", "Data/test_legal.csv",
        "--holdout_csv", "Data/test_legal.csv",
        "--threshold", "0.10"
    ]
    
    print("\nExecuting Evaluation (0.10) command:")
    print(" ".join(eval_cmd_10))
    
    try:
        subprocess.run(eval_cmd_10, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during evaluation (0.10): {e}")
        sys.exit(1)
        
    print("\nLegal Dataset Fine-Tuning Pipeline Complete!")

if __name__ == "__main__":
    main()
