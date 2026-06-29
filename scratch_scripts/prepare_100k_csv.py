import json
import pandas as pd
from pathlib import Path

def convert_jsonl_to_csv(jsonl_path: Path, csv_path: Path):
    print(f"Converting {jsonl_path.name} to {csv_path.name}...")
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text", "")
            label_str = item.get("label", "")
            source = item.get("source", "unknown")
            
            # Map string label to integer
            if label_str == "human":
                label_val = 0
            elif label_str == "ai":
                label_val = 1
            else:
                # Fallback if label is already integer or something else
                try:
                    label_val = int(label_str)
                except ValueError:
                    continue
            
            data.append({
                "text": text,
                "label": label_val,
                "source": source
            })
            
    df = pd.DataFrame(data)
    df.dropna(subset=["text"], inplace=True)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  Saved {len(df):,} rows. Label distribution: {df['label'].value_counts().to_dict()}")

def main():
    processed_dir = Path("Data/processed")
    output_dir = Path("Data")
    
    splits = [
        ("split_train.jsonl", "train_100k.csv"),
        ("split_val.jsonl", "val_100k.csv"),
        ("split_test.jsonl", "test_100k.csv"),
        ("split_external_benchmark.jsonl", "external_val_100k.csv"),
    ]
    
    for jsonl_name, csv_name in splits:
        jsonl_path = processed_dir / jsonl_name
        csv_path = output_dir / csv_name
        if jsonl_path.exists():
            convert_jsonl_to_csv(jsonl_path, csv_path)
        else:
            print(f"Error: {jsonl_path} does not exist!")

if __name__ == "__main__":
    main()
