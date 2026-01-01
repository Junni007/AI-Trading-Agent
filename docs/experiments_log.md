# 🧪 AI Trading Experiments & Findings Log

This document tracks the progress, experiments, and key learnings during the scaling and development of the AI Trading Agent.

## Experiment 01: Massive Scale Training (550+ Tickers)
**Date**: 2026-01-01
**Objective**: Train a single LSTM model on ~550 tickers (S&P 500 + Nifty 50) to learn universal market patterns.

### ⚙️ Configuration
- **Model**: LSTM (Input: ?, Hidden: 256, Layers: 3)
- **Data**: ~550 Tickers, 2019-2022 (Train), 2023 (Val), 2024 (Test).
- **Features (Initial)**: `['Close', 'RSI', 'MACD', 'Log_Return']`
- **Batch Size**: 64
- **Precision**: 16-mixed

### 📉 Observation: Model Collapse
During the first massive run, the model exhibited "Collapse" behavior:
- **Training Accuracy**: ~44% (stuck).
- **Validation Accuracy**: ~42% (flatline).
- **Test Precision**: 
    - DOWN: 0.00
    - NEUTRAL: 0.00
    - UP: 0.42 (100% Recall)
    
**Diagnosis**: 
The model learned to predict "UP" for every single sample.
Why? The **`Close`** price feature. 
- In a mixed dataset, Asset A has `Close=$150` (AAPL) and Asset B has `Close=$2000` (GOOGL/Booking). 
- A simple `StandardScaler` fitted mainly on lower-priced assets (or averaged) causes massive outliers for high-priced assets.
- The model treats `Close` as a raw value rather than a relative trend, confusing the gradients.

### 🛠️ The Fix (Hypothesis)
**Action**: Remove `Close` price from the input features.
**New Features**: `['RSI', 'MACD', 'MACD_Signal', 'Log_Return']`
- **Rationale**: All selected features are **stationary** or bounded. 
    - `RSI` is always 0-100. 
    - `MACD` behaves similarly relative to price action momentum.
    - `Log_Return` is a percentage change (scale invariant).
    
**Next Steps**: Retrain with the clean feature set.

### 📊 Result (Experiment 01b)
**Config**: Removed 'Close'. Features: `[RSI, MACD, Returns]`. No Optuna.
**Outcome**:
- **Accuracy**: ~41.5%
- **Precision (DOWN)**: 0.18 (First sign of learning!)
- **Recall (UP)**: 0.99 (Still heavily biased to Bull Market)
**Insight**: The model is safer but still lazy (predicts UP 90% of time). This is because 2024 was a Bull Market. We need **Class Balancing** or **Hyperparameter Tuning** to force it to learn "DOWN" signals better.

## Experiment 02: Bayesian Architecture Search
**Date**: 2026-01-01
**Objective**: Optimize LSTM hyperparameters using Optuna (Automated Tuning).

### 🔍 Search Space
- **Hidden Dim**: `[64, 128, 256, 512]`
- **Num Layers**: `[1, 2, 3, 4]`
- **Dropout**: `0.1` - `0.5`
- **Learning Rate**: `1e-5` - `1e-2` (Log Scale)
- **Batch Size**: `[32, 64, 128]`

### 🧠 Hypothesis
- A **Deep but Narrow** network (e.g., 3 Layers, 128 Dim) might generalize better on Multi-Ticker data than a Shallow Wide one.
- **Pruning**: Using `MedianPruner` to kill bad trials early (e.g., if Loss > Median at epoch 2, stop).

### 🧪 Execution
Running `src/tune.py` on Colab (GPU).
Results will be saved to `best_hyperparameters.json`.
