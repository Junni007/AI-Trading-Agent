# Signal.Engine: Distributed Training Guide (Kaggle/Colab)

**Estimated GPU Time**: 2-3 hours
**Recommended Hardware**: T4 x2 (Kaggle) or A100 (Colab Pro)

This guide provides instructions for training the Reinforcement Learning (PPO) agent and the Supervised Fine-Tuning (SFT) models using free or cheap cloud GPU resources.

---

## 📑 Table of Contents
1. [Pre-Flight Checklist](#1-pre-flight-checklist)
2. [Step 1: Code Upload & Setup](#2-step-1-code-upload--setup)
3. [Step 2: Kaggle Notebook Configuration](#3-step-2-kaggle-notebook-configuration)
4. [Step 3: Execution Script](#4-step-3-execution-script)
5. [Monitoring & Troubleshooting](#5-monitoring--troubleshooting)
6. [Post-Training Evaluation](#6-post-training-evaluation)

---

## 1. Pre-Flight Checklist

Before burning GPU hours, ensure your local code is stable:
- [x] All unit tests passing (`pytest tests/`)
- [x] Feature engineering logic verified (`python -m src.analysis.runner`)
- [x] No hardcoded local file paths in `src/`

---

## 2. Step 1: Code Upload & Setup

You need to get the `src/` directory into the Kaggle environment.

### Option A: Kaggle API (Recommended)
This is the fastest way to sync your local code to Kaggle.

1. Ensure you have your `kaggle.json` credentials configured.
2. Run the provided upload script from the project root:
   ```bash
   ./upload_to_kaggle.sh
   # Or on Windows: .\upload_to_kaggle.ps1
   ```
   *(This script zips the `src` folder and pushes it as a private Kaggle dataset).*

### Option B: Manual Upload
1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets).
2. Click "New Dataset".
3. Upload your entire `src/` folder.
4. Name the dataset `signal-engine-codebase` and set visibility to **Private**.

---

## 3. Step 2: Kaggle Notebook Configuration

1. Create a **New Notebook** on Kaggle.
2. **Settings**:
   - **Accelerator**: GPU T4 x2
   - **Internet**: ON (Required for downloading market data via `yfinance`)
   - **Persistence**: Variables Only
3. **Add Data**:
   - Click "+ Add Data" in the right sidebar.
   - Search for `signal-engine-codebase` (your uploaded dataset) and add it.

---

## 4. Step 3: Execution Script

Copy and paste the following blocks into separate cells in your Kaggle Notebook.

### Cell 1: Install Dependencies & Path Setup
```python
# Install required ML and finance packages
!pip install -q yfinance gymnasium pytorch-lightning torchmetrics
# Note: ta and gudhi may be required depending on your feature set
!pip install -q ta gudhi

import sys
# Update this path based on what you named your dataset
sys.path.append('/kaggle/input/signal-engine-codebase')

print("✅ Dependencies installed and path set.")
```

### Cell 2: Hardware Verification
```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Cell 3: Launch Training
```python
# Import the main training loop
from src.train import main  # Adjust module name if using train_classic.py or train_ppo_optimized.py

# Execute
main()
```

### Cell 4: Save Checkpoints to Output
Kaggle deletes the `/kaggle/working` directory when the session ends unless files are explicitly downloaded or saved as output.
```python
import shutil
import os

checkpoint_path = 'checkpoints/best_ppo.ckpt'
output_path = '/kaggle/working/best_model_run.ckpt'

if os.path.exists(checkpoint_path):
    shutil.copy(checkpoint_path, output_path)
    print(f"✅ Checkpoint successfully moved to {output_path}")
    print("⚠️ IMPORTANT: Download this file from the 'Output' tab on the right sidebar.")
else:
    print("❌ Checkpoint not found. Did training complete successfully?")
```

---

## 5. Monitoring & Troubleshooting

### What a "Good" Training Run Looks Like
Watch the PyTorch Lightning progress bar. Over ~100 epochs, you should see:
- 📈 **Reward**: Steadily increasing (e.g., from 0.5 → 15.0).
- 📉 **Critic Loss**: Decreasing and stabilizing.
- 📉 **Entropy**: Slowly decaying (meaning the agent is becoming more confident in its actions).

### Common Issues

| Issue | Diagnosis & Fix |
| :--- | :--- |
| **CUDA Out of Memory (OOM)** | The batch size or number of parallel environments is too high. Open your training script and reduce `N_ENVS` (e.g., from 256 to 128) or reduce `ROLLOUT_STEPS`. |
| **yfinance Download Errors** | The notebook doesn't have internet access. Check the "Internet" toggle in the notebook settings. |
| **Reward Stagnates at 0** | The model isn't learning. Check your reward function in `env.py`. Ensure penalties (like holding costs) aren't overpowering the profit rewards. |
| **TDA Feature Computation is Slow** | Topological features are CPU intensive. Reduce the `window_size` or train on fewer tickers simultaneously. |

---

## 6. Post-Training Evaluation

Once you have downloaded `best_model_run.ckpt` to your local machine:

1. Move the file into the `checkpoints/` directory.
2. Run the analysis framework against the out-of-sample data (e.g., Nifty 500 validation set):
   ```bash
   python -m src.analysis.runner --checkpoint checkpoints/best_model_run.ckpt
   ```
3. Check the generated `output/edge_validation.png` to verify the model has a statistically significant edge.

---
*End of Guide*
