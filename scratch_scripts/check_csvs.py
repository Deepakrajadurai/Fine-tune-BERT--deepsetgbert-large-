import pandas as pd
from pathlib import Path

data_dir = Path("Data")
csv_files = [
    "train_legal.csv",
    "val_legal.csv",
    "test_legal.csv",
]

for fname in csv_files:
    path = data_dir / fname
    if not path.exists():
        print(f"{fname}: Not found")
        continue
    
    df = pd.read_csv(path)
    print(f"=== {fname} ===")
    print(f"Total samples: {len(df)}")
    print("Labels distribution:")
    print(df["label"].value_counts())
    print("Sources:")
    print(df["source"].value_counts())
    print()
