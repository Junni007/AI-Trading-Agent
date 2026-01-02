# Project Manual & Architecture Guide

## 1. Project Overview
**Goal**: Build a scalable AI Trading Agent capable of predicting market direction (Up/Down/Neutral) using deep learning (LSTM) and potentially Topological Data Analysis (TDA) features.
**Status**: Prototyping / MVP Phase.
**Tech Stack**: PyTorch, PyTorch Lightning, Optuna, FastAPI, React.

## 2. Architecture

### Core Modules (`src/`)
- **`tune.py`**: Hyperparameter optimization using Optuna.
    - *Key Features*: SQLite Checkpointing, Error Handling, Pruning.
    - *Output*: `best_hyperparameters.json`.
- **`lstm_model.py`**: The neural network definition.
    - *Type*: 2-Layer LSTM with Dropout and fully connected head.
    - *Framework*: PyTorch LightningModule.
- **`data_loader.py`**: Handles data fetching (yfinance), splitting (Train/Val/Test), and windowing.
    - *Strict Splitting*: Train (2018-2022), Val (2023), Test (2024).
- **`agent.py`**: Reinforcement Learning (PPO) agent implementation (Prototype stage).
- **`ticker_utils.py`**: Utilities for fetching ticker symbols (S&P 500, Nifty 50).

### Pipelines
1. **tuning**: `src/tune.py` -> Optimizes params -> `best_hyperparameters.json`
2. **training**: `train.py` -> Loads data + best params -> Trains LSTM -> Saves `final_lstm_model.pth`
3. **serving**: `app/main.py` -> Serves model predictions via API.
4. **dashboard**: `frontend/` -> Visualizes predictions.

## 3. Development Rules (Strict Adherence)
1.  **Root Cause Analysis**: No patch work. Fix problems permanently at the source.
2.  **Simplicity**: Find the easiest, most viable route. Do not overcomplicate.
3.  **Documentation & Truth**: Do not hallucinate. Verify facts. Update documentation (`docs/`) to keep context aligned.
4.  **senior Quality**: Write viable, long-lasting, production-grade code.

## 4. Current State
- `tune.py` has been patched to support resuming (checkpoints) and robust error handling.
- `train.py` is ready for massive training but requires careful resource management on local/Colab.
- `optuna` dependency is required for tuning.

## 5. Theory References
- `docs/TDA_THEORY.md`: Explains the use of Algebraic Topology for feature extraction.
