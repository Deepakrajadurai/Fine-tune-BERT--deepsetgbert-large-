# Implementation Plan: GBERT-Large Fine-Tuning Execution

This plan outlines the steps to set up a CUDA-enabled Python environment and execute the training pipeline for fine-tuning `deepset/gbert-large` on the prepared 400,000-row German text dataset.

## User Review Required

> [!IMPORTANT]
> - **GPU Training**: We detected an NVIDIA GeForce RTX 4080 (16GB VRAM) on this system. Training the 400,000-sample dataset on a CPU is infeasible (it would take days/weeks). 
> - **Python Version**: The system's default environment is using Python 3.14.3. PyTorch does not currently publish official pre-built CUDA/GPU wheels for Python 3.14 on Windows. However, **Python 3.12.7** is installed on the system and has full CUDA PyTorch support.
> - **Action**: We will create a local virtual environment (`venv`) inside this workspace using Python 3.12.7, install PyTorch with CUDA 12.4 support, install the dependencies, and then run the training script on the RTX 4080 GPU.

---

## Proposed Steps

### 1. Create Virtual Environment
Create a local virtual environment named `venv` in the current workspace using Python 3.12:
```powershell
& "C:\Program Files\Python312\python.exe" -m venv venv
```

### 2. Install PyTorch with CUDA Support
Install PyTorch 2.6.0 with CUDA 12.4 support in the virtual environment:
```powershell
.\venv\Scripts\pip install torch==2.6.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### 3. Install Rest of Dependencies
Install other required packages from [requirements.txt](file:///e:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/requirements.txt):
```powershell
.\venv\Scripts\pip install -r requirements.txt
```

### 4. Execute Fine-Tuning
Execute [train.py](file:///e:/15-06-26/Fine-tune%20BERT%20%28deepsetgbert-large%29/train.py) on the full dataset using the GPU:
```powershell
.\venv\Scripts\python train.py
```
This will:
- Load the balanced `train.csv` and `val.csv` datasets.
- Fine-tune `deepset/gbert-large` using Hugging Face `Trainer`.
- Leverage GPU mixed-precision (`fp16`) training.
- Save checkpoints and the best model to `models/best_model`.

---

## Verification Plan

### Automated Verification
* Verify PyTorch CUDA detection in the new virtual environment:
  ```powershell
  .\venv\Scripts\python -c "import torch; print('CUDA active:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
  ```
* Ensure training finishes successfully and saves model outputs to `models/best_model`.
