# Signal.Engine - Kaggle/Colab Training Guide

Run this notebook in Kaggle (free T4x2) or Google Colab to train the PPO agent.

## 1. Setup

```python
# Clone your repo
!git clone https://github.com/your-username/trading-agent.git
%cd trading-agent

# Install dependencies (Gymnasium replaces deprecated Gym)
!pip install -q -r requirements.txt

# Silence deprecation warnings
import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')
```

## 2. Verify GPU

```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
!nvidia-smi
```

## 3. Train Model

### Option A: Standard Training (slower, easier to debug)
```python
!python -m src.train_ppo
```

### Option B: GPU-Optimized Training (recommended for Kaggle)
```python
# Uses vectorized environments for 10-100x better GPU utilization
!python -m src.train_ppo_optimized
```

**GPU Optimization Techniques:**
| Feature | Standard | Optimized |
|---------|----------|-----------|
| Model Size | 18K params | 500K params |
| Parallel Envs | 1 | 64 |
| Batch Size | Sequential | 16,384/step |
| GPU Utilization | ~4% | 60-90% |
| Training Speed | ~0.26 it/s | ~5-10 it/s |

## 4. For Multi-GPU (Kaggle T4x2)

```python
# Single GPU is usually better for RL (less communication overhead)
# But if you want to use both:
!python -m src.train_ppo_optimized  # Will auto-use GPU 0 only
```

> **Note**: For PPO, using a single GPU is often faster than DDP because
> RL training involves lots of small, sequential environment interactions.

## 5. Monitor Training

```python
# Launch TensorBoard
%load_ext tensorboard
%tensorboard --logdir lightning_logs
```

**What to look for:**
- `reward` should trend upward
- `entropy` should gradually decrease (policy becomes more confident)
- GPU utilization should be >50% with optimized script

## 6. Download Results

```python
from google.colab import files  # or kaggle
import shutil
import os

# Zip and download checkpoints
shutil.make_archive('checkpoints', 'zip', 'checkpoints')
files.download('checkpoints.zip')

# Download best model
if os.path.exists('checkpoints/best_ppo.ckpt'):
    files.download('checkpoints/best_ppo.ckpt')
```

## Troubleshooting

### Low GPU Utilization (<10%)
- Use `train_ppo_optimized.py` instead of `train_ppo.py`
- Increase `N_ENVS` to 128+ if you have enough GPU memory

### Out of Memory (OOM)
- Reduce `N_ENVS` from 64 to 32
- Reduce `ROLLOUT_STEPS` from 256 to 128

### Gymnasium Warning
The warning "Gym has been unmaintained since 2022" is benign - we're already using Gymnasium, 
but some dependencies still import the old gym. You can safely ignore it.

### Training Time Estimates
| Platform | Script | Expected Time |
|----------|--------|---------------|
| Kaggle T4 | Standard | 6-8 hours |
| Kaggle T4 | Optimized | 30-60 min |
| Colab Free T4 | Optimized | 45-90 min |


