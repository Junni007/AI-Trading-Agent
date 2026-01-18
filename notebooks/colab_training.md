# Signal.Engine - Colab Training Guide

Run this notebook in Google Colab to train the PPO agent using free GPU resources.

## 1. Setup

```python
# Clone your repo (replace with your actual URL or upload files manually)
!git clone https://github.com/your-username/trading-agent.git
%cd trading-agent

# Install Dependencies
!pip install -r requirements.txt
```

## 2. Verify GPU

```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
!nvidia-smi
```

## 3. Train Model (Optimized)

```python
# Run the optimized training script
# Uses mixed precision, gradient clipping, and tuned hyperparameters
!python -m src.train_ppo
```

**Current Optimizations Applied:**
- ✅ 16-bit mixed precision (~2x speedup)
- ✅ Gradient clipping (max_norm=0.5)
- ✅ Learning rate: 3e-4
- ✅ Clip epsilon: 0.1
- ✅ 20 tickers for diverse training
- ✅ 200 epochs (quality over quantity)
- ✅ 500-step rollouts for stable gradients

## 4. Monitor Training

```python
# Launch TensorBoard in Colab
%load_ext tensorboard
%tensorboard --logdir lightning_logs
```

**What to look for:**
- `reward` should increase over time
- `train_loss` should stabilize (not wildly fluctuating)
- Training speed should be ~1.0+ it/s with mixed precision

## 5. Download Results

```python
from google.colab import files
import shutil

# Zip and download checkpoints
shutil.make_archive('checkpoints', 'zip', 'checkpoints')
files.download('checkpoints.zip')

# Download best model
if os.path.exists('checkpoints/best_ppo.ckpt'):
    files.download('checkpoints/best_ppo.ckpt')
```

## Tips

- Use **Runtime → Change runtime type → T4 GPU** for faster training
- If data download fails (yfinance rate limits), upload your local data CSVs
- Training should complete in ~4-6 hours with optimizations
- Stop early if rewards aren't improving after 30+ epochs

