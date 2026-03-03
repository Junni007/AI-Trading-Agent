# Signal.Engine - Colab Training Guide

This guide is designed for running the `Signal.Engine` training pipeline interactively in **Google Colab** (using a free T4 GPU). It is an alternative to the automated Kaggle script approach.

---

## 📑 Table of Contents
1. [Setup Environment](#1-setup-environment)
2. [Verify GPU Allocation](#2-verify-gpu-allocation)
3. [Execution Options](#3-execution-options)
4. [Live Monitoring](#4-live-monitoring)
5. [Downloading Artifacts](#5-downloading-artifacts)
6. [Troubleshooting Common Colab Errors](#6-troubleshooting-common-colab-errors)

---

## 1. Setup Environment

Open a new notebook in [Google Colab](https://colab.research.google.com/) and ensure you have selected a GPU runtime (`Runtime > Change runtime type > T4 GPU`).

Run the following cell to clone the repository and install dependencies.

```python
# 1. Clone the repository
!git clone https://github.com/your-username/trading-agent.git
%cd trading-agent

# 2. Install dependencies (Gymnasium replaces deprecated Gym)
!pip install -q -r requirements.txt

# 3. Silence common library deprecation warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
print("✅ Environment setup complete.")
```

---

## 2. Verify GPU Allocation

Before starting a long training run, confirm that PyTorch can see the T4 GPU.

```python
import torch

print("GPU Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Show current memory usage
!nvidia-smi
```

---

## 3. Execution Options

We offer two main training scripts. For cloud instances like Colab, we heavily recommend the **Optimized** version.

### Option A: Standard Training (Easier to Debug)
Runs sequentially. Good for verifying logic on a small subset of data, but severely underutilizes the GPU.
```python
!python -m src.train_classic
```

### Option B: GPU-Optimized Training (Recommended)
Uses vectorized environments (`VectorEnv`) to batch multiple parallel rollouts simultaneously. This provides a 10-100x speedup in wall-clock time.
```python
!python -m src.train
```

**Why Optimization Matters:**

| Metric | Standard (Sequential) | Optimized (Vectorized) |
| :--- | :--- | :--- |
| **Parallel Envs** | 1 | 64+ |
| **GPU Utilization** | ~4% | 60% - 90% |
| **Training Speed** | ~0.26 it/s | ~5-10 it/s |
| **Expected Time** | 6-8 hours | **30-60 mins** |

---

## 4. Live Monitoring

You can monitor the Reinforcement Learning metrics in real-time using TensorBoard directly within Colab.

```python
# Load the TensorBoard notebook extension
%load_ext tensorboard

# Point it to the PyTorch Lightning logs directory
%tensorboard --logdir lightning_logs
```

**Key Metrics to Watch:**
- `reward`: Should generally trend upward as the agent learns profitable behaviors.
- `entropy`: Should gradually decrease, indicating the agent's policy is becoming more decisive and confident.

---

## 5. Downloading Artifacts

When training is complete, package and download the best checkpoint.

```python
from google.colab import files
import shutil
import os

checkpoint_dir = 'checkpoints'
checkpoint_file = f'{checkpoint_dir}/best_ppo.ckpt'

if os.path.exists(checkpoint_dir):
    # Zip the entire checkpoints folder
    shutil.make_archive('checkpoints_archive', 'zip', checkpoint_dir)
    print("✅ Created checkpoints_archive.zip")

    # Trigger browser download
    files.download('checkpoints_archive.zip')

    # Alternatively, just download the best model
    if os.path.exists(checkpoint_file):
        files.download(checkpoint_file)
else:
    print("❌ No checkpoints found. Did training complete successfully?")
```

---

## 6. Troubleshooting Common Colab Errors

### ⚠️ Low GPU Utilization (<10%)
- **Cause**: The CPU is bottlenecking the GPU by feeding it data too slowly.
- **Fix**: Ensure you are running the optimized script (`train.py`), not the classic one. If memory permits, increase `N_ENVS` (the number of parallel environments) in your training configuration.

### ⚠️ Out of Memory (CUDA OOM)
- **Cause**: Too much data is loaded onto the VRAM at once.
- **Fix**: Reduce `N_ENVS` (e.g., from 64 to 32) or reduce `ROLLOUT_STEPS` (e.g., from 256 to 128).

### ⚠️ yfinance "Database is Locked" Error
- **Cause**: `yfinance` uses an SQLite cache internally, which crashes when multiple parallel environments try to read/write simultaneously.
- **Fix**: The optimized script should handle this automatically by redirecting the cache. If it still occurs, add this to the top of your script/notebook:
  ```python
  import yfinance as yf
  import tempfile
  yf.set_tz_cache_location(tempfile.mkdtemp())
  ```

### ⚠️ "Not enough training data" / Download Failures
- **Cause**: Rate-limiting from Yahoo Finance, or the notebook lacks internet access.
- **Fix**:
  1. Verify the notebook has internet access.
  2. Wait 15 minutes for the rate limit to reset.
  3. Reduce `NUM_TICKERS` in the training configuration to test with a smaller batch.
