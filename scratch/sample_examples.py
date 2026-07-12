import pandas as pd

def sample_examples():
    df = pd.read_csv("Data/train.csv")
    
    # 5 Human examples
    human_samples = df[df['label'] == 0].sample(n=5, random_state=42)
    # 5 AI examples
    ai_samples = df[df['label'] == 1].sample(n=5, random_state=42)
    
    print("--- HUMAN SAMPLES ---")
    for idx, row in human_samples.iterrows():
        print(f"\nExample ID: {row.get('id', 'N/A')}")
        print(f"Domain: {row.get('domain', 'N/A')}")
        print(f"Text snippet: {row['text'][:400]}...")
        print("-" * 40)
        
    print("\n--- AI SAMPLES ---")
    for idx, row in ai_samples.iterrows():
        print(f"\nExample ID: {row.get('id', 'N/A')}")
        print(f"Domain: {row.get('domain', 'N/A')}")
        print(f"Text snippet: {row['text'][:400]}...")
        print("-" * 40)

if __name__ == "__main__":
    sample_examples()
