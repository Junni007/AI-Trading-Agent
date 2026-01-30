# Signal.Engine v4.0: Kaggle Training Guide

**Phase**: 3 - Training  
**Estimated Time**: 2-3 hours (GPU time)  
**Date**: 2026-01-30

---

## 🎯 Quick Start

### Pre-Flight Checklist

✅ Phase 2 Backend Implementation complete  
✅ All tests passing (`test_v4_features.py`, `test_v4_integration.py`)  
✅ Bug fixes applied

**Ready to train!** 🚀

---

## 📦 Step 1: Prepare Files for Upload

### Required Files to Upload to Kaggle

Create a **new Kaggle Dataset** with these files:

```
signal-engine-v4/
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # ✅ v4.0 (cross-sectional ranker)
│   ├── env.py                   # ✅ v4.0 (vol-targeting)
│   ├── train_ppo_optimized.py  # ✅ v4.0 (9-dim obs + Sharpe rewards)
│   ├── ticker_utils.py
│   ├── tda_features.py
│   └── data_labeler.py
└── README.md (optional)
```

### Upload Commands

**Option A: Manual Upload**
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the `src/` folder
4. Name it: `signal-engine-v4`
5. Make it **Private** (your code!)

**Option B: Kaggle API** (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Create dataset metadata
# (Use the kaggle_dataset_metadata.json file below)

# Upload
kaggle datasets create -p ./signal-engine-v4
```

---

## 🖥️ Step 2: Create Kaggle Notebook

### Notebook Setup

1. **Create New Notebook**: https://www.kaggle.com/code
2. **Settings**:
   - Accelerator: **GPU T4 x2** (or P100 if available)
   - Internet: **ON** (for yfinance)
   - Persistence: **Variables Only** (faster startup)

### Add Your Dataset
- Click "+ Add Data"
- Search for your dataset: `signal-engine-v4`
- Add it to notebook

---

## 📝 Step 3: Training Code (Copy-Paste into Kaggle Notebook)

### Cell 1: Install Dependencies
```python
# Install required packages
!pip install -q yfinance ta gymnasium pytorch-lightning torchmetrics

import sys
sys.path.append('/kaggle/input/signal-engine-v4')

print("✅ Dependencies installed")
```

### Cell 2: Import & Verify
```python
from src.train_ppo_optimized import main
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Cell 3: Run Training
```python
# Start training
main()
```

### Cell 4: Download Checkpoint
```python
# After training completes, download the checkpoint
import shutil
import os

# Copy checkpoint to output
if os.path.exists('checkpoints/best_ppo.ckpt'):
    shutil.copy('checkpoints/best_ppo.ckpt', '/kaggle/working/best_ppo_v4.ckpt')
    print("✅ Checkpoint saved to /kaggle/working/best_ppo_v4.ckpt")
    print("Download it from the 'Output' tab after notebook finishes")
else:
    print("❌ No checkpoint found. Check training logs above.")
```

---

## 📊 Step 4: Monitor Training

### Expected Output

```
🏋️ Signal.Engine v4.0 - Training Started
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Training Data: 20 tickers, 1500+ days
🚀 Environments: 256 parallel
🧠 Model: RecurrentActorCritic (9-dim input)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Epoch 0: 100%|██████████| 50/50 [02:15<00:00]
├─ reward: 12.45
├─ critic_loss: 0.234
├─ entropy: 0.045
└─ train_loss: 1.23

...

Epoch 99: 100%|██████████| 50/50 [02:10<00:00]
├─ reward: 45.67  ⬆️ (Good sign!)
├─ critic_loss: 0.089
├─ entropy: 0.012
└─ train_loss: 0.56

✅ Training complete!
Best checkpoint: checkpoints/best_ppo.ckpt
```

### Success Indicators

✅ **Reward increasing** over epochs (12 → 45+)  
✅ **Critic loss decreasing** (0.2 → 0.08)  
✅ **Entropy decaying** (0.05 → 0.01)  
❌ **Reward stuck/decreasing** → Might need hyperparameter tuning

---

## ⚙️ Step 5: Troubleshooting

### Common Issues

#### 1. **Out of Memory (OOM)**
```python
# In train_ppo_optimized.py, reduce N_ENVS
N_ENVS = 128  # Instead of 256
```

#### 2. **Training Too Slow**
```python
# Reduce tickers or rollout steps
NUM_TICKERS = 10  # Instead of 20
ROLLOUT_STEPS = 128  # Instead of 256
```

#### 3. **Model Not Learning (Reward Stuck)**
- Check if data is loading correctly
- Verify 9-dim observations (should print in logs)
- Try lower learning rate: `lr=1e-4` instead of `3e-4`

#### 4. **yfinance Download Fails**
- Enable internet in notebook settings
- Add retry logic (already in data_loader.py)
- Use cached data if available

---

## 📥 Step 6: After Training

### Download Files

1. **Checkpoint**: `/kaggle/working/best_ppo_v4.ckpt`
2. **Training Logs**: `/kaggle/working/lightning_logs/`

### Local Setup

```bash
# On your local machine
# Place checkpoint in project root
cp ~/Downloads/best_ppo_v4.ckpt ~/trading-agent/checkpoints/
# Or set CHECKPOINT_DIR environment variable
# cp ~/Downloads/best_ppo_v4.ckpt "${CHECKPOINT_DIR}/best_ppo_v4.ckpt"

# Verify it works
python -c "import torch; ckpt = torch.load('checkpoints/best_ppo_v4.ckpt', weights_only=True); print('✅ Checkpoint loaded')"
```

---

## 🔍 Step 7: Next Steps (Phase 4)

After downloading the checkpoint:

1. **Run Evaluation**: `python -m src.evaluate_nifty500 --checkpoint checkpoints/best_ppo_v4.ckpt`
2. **Compare v3 vs v4**: Check Sharpe Ratio, Drawdown, Win Rate
3. **Update Documentation**: Create `NIFTY500_EVALUATION_v4.md`

---

## 📋 Training Hyperparameters (Reference)

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **N_ENVS** | 256 | Parallel environments |
| **NUM_TICKERS** | 20 | Training stocks |
| **ROLLOUT_STEPS** | 256 | Steps per update |
| **PPO_EPOCHS** | 4 | Gradient updates per batch |
| **LEARNING_RATE** | 3e-4 | Adam optimizer |
| **GAMMA** | 0.99 | Discount factor |
| **CLIP_EPS** | 0.2 | PPO clipping range |

---

## 🎯 Expected Training Time

- **Setup + Data Download**: 5-10 min
- **Training (100 epochs)**: 2-3 hours
- **Total**: ~3 hours

**Tip**: Set it and forget it! The notebook will auto-save progress.

---

## ✅ Success Criteria

After training, your v4.0 model should show:

- ✅ **Positive expectancy** (Avg PnL > 0%)
- ✅ **Sharpe Ratio > 1.2** (vs. v3.0's ~0.8)
- ✅ **Max Drawdown < 20%** (vs. v3.0's -34%)
- ✅ **Smooth equity curve** (vol-targeting effect)

If these metrics are met, **v4.0 is a success!** 🎉

---

*Ready to train? Copy the code above into your Kaggle notebook and let it run!*
