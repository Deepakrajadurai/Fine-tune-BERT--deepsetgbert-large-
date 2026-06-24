import pandas as pd

df = pd.read_csv("Data/ai_generated_sentences_500k.csv")
print("Total AI rows:", len(df))

# Check for presence of common template substrings
patterns = [
    "aufgrund der aktuellen Lage",
    "im Namen der Fraktion",
    "unter Zeichnung von",
    "Plenarsitzung",
    "Rechtsverordnung",
    "Drucksache",
    "Landesgesetz",
    "Auskunftspflichten"
]

print("\nMatching counts for each pattern:")
non_templated = df.copy()
for p in patterns:
    matches = df["text"].str.contains(p, case=False, na=False)
    print(f"  Pattern '{p}': {matches.sum():,} matches ({100 * matches.sum() / len(df):.2f}%)")
    non_templated = non_templated[~non_templated["text"].str.contains(p, case=False, na=False)]

print("\nRows remaining after excluding all patterns:", len(non_templated))
if len(non_templated) > 0:
    print("\nExamples of remaining texts:")
    for i in range(min(10, len(non_templated))):
        print(f"  - {non_templated['text'].iloc[i]}")
