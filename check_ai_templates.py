import pandas as pd

# Load AI sentences
df = pd.read_csv("Data/ai_generated_sentences_500k.csv", nrows=10000)
print("Columns:", df.columns.tolist())
print(df.head(10))

print("\nValue counts of model:")
print(df["model"].value_counts())

print("\nValue counts of style:")
print(df["style"].value_counts())

print("\nUnique templates example:")
for style in df["style"].unique():
    subset = df[df["style"] == style]
    print(f"\nStyle: {style} (Total: {len(subset)})")
    for i in range(min(5, len(subset))):
        print(f"  - {subset['text'].iloc[i]}")
