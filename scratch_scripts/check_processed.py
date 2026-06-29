import pandas as pd
import json
from pathlib import Path

processed_dir = Path("Data/processed")
files = [
    "human_clean_deduped.jsonl",
    "ai_clean_deduped.jsonl",
    "split_train.jsonl",
    "split_val.jsonl",
    "split_test.jsonl",
    "split_external_benchmark.jsonl",
]

for fname in files:
    path = processed_dir / fname
    if not path.exists():
        print(f"{fname}: Not found")
        continue
    
    labels = []
    sources = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            labels.append(obj.get("label"))
            src = obj.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            
    df_labels = pd.Series(labels)
    print(f"=== {fname} ===")
    print(f"Total samples: {len(labels)}")
    print("Labels distribution:")
    print(df_labels.value_counts())
    print("Sources:")
    for src, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {src}: {count}")
    print()
