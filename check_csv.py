import pandas as pd

for name in ["train", "val", "test"]:
    df = pd.read_csv(f"Data/{name}.csv", nrows=5)
    print(f"File: Data/{name}.csv")
    print("Columns:", df.columns.tolist())
    print("Label value counts for first 5:")
    print(df["label"].value_counts())
    print("First row text:", repr(df["text"].iloc[0][:100]))
    print("-" * 50)
