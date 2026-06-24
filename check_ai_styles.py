import pandas as pd

# Since we want to avoid pandas/pytorch conflict, we can just use pandas here as long as we don't import pytorch/transformers
df = pd.read_csv("Data/test.csv")
print("Total rows:", len(df))
print("Label distribution:")
print(df["label"].value_counts())

ai_df = df[df["label"] == 1]
print("\nUnique models in AI data:")
print(ai_df["model"].value_counts())

print("\nUnique styles in AI data:")
print(ai_df["style"].value_counts())

print("\nUnique sources in AI data:")
print(ai_df["source"].value_counts())

print("\nUnique sources in Human data:")
human_df = df[df["label"] == 0]
print(human_df["source"].value_counts())

print("\nExample AI texts for each model:")
for model in ai_df["model"].dropna().unique():
    sample = ai_df[ai_df["model"] == model].iloc[0]
    print(f"Model: {model} | Style: {sample['style']}")
    print(repr(sample["text"][:200]))
    print("-" * 30)
