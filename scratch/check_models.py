import os
from pathlib import Path

models_dir = Path("models")
for d in models_dir.iterdir():
    if d.is_dir():
        vocab_exists = (d / "vocab.txt").exists() or (d / "vocab.json").exists()
        config_exists = (d / "config.json").exists()
        weights_exists = (d / "pytorch_model.bin").exists() or (d / "model.safetensors").exists()
        print(f"Directory: {d.name:25s} | Vocab: {vocab_exists} | Config: {config_exists} | Weights: {weights_exists}")
