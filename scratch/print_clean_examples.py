import pandas as pd

def export_examples():
    # Read train.csv with UTF-8 encoding
    try:
        df = pd.read_csv("Data/train.csv", encoding="utf-8")
    except Exception:
        df = pd.read_csv("Data/train.csv", encoding="latin-1")
    
    # 5 Human examples (using same random state to keep consistency)
    human_samples = df[df['label'] == 0].sample(n=5, random_state=42)
    # 5 AI examples
    ai_samples = df[df['label'] == 1].sample(n=5, random_state=42)
    
    with open("scratch/sampled_examples.md", "w", encoding="utf-8") as f:
        f.write("# Sampled Examples from Training Dataset (Data/train.csv)\n\n")
        
        f.write("## 🟢 Human-Written Examples (Label: 0)\n\n")
        for i, (idx, row) in enumerate(human_samples.iterrows(), 1):
            f.write(f"### Example {i}\n")
            f.write(f"- **ID**: `{row.get('id', 'N/A')}`\n")
            f.write(f"- **Domain**: `{row.get('domain', 'N/A')}`\n")
            f.write(f"- **Text snippet**:\n  > {row['text']}\n\n")
            f.write("---\n\n")
            
        f.write("## 🔴 AI-Generated Examples (Label: 1)\n\n")
        for i, (idx, row) in enumerate(ai_samples.iterrows(), 1):
            f.write(f"### Example {i}\n")
            f.write(f"- **ID**: `{row.get('id', 'N/A')}`\n")
            f.write(f"- **Domain**: `{row.get('domain', 'N/A')}`\n")
            f.write(f"- **Text snippet**:\n  > {row['text']}\n\n")
            f.write("---\n\n")

if __name__ == "__main__":
    export_examples()
