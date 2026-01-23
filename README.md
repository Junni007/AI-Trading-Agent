# 🦅 Signal.Engine

### Generative Algorithmic Intelligence for the Nifty 500

**Signal.Engine** is an autonomous market analysis system that moves beyond simple price prediction. It uses **Generative AI (Sequence Modeling)** to understand market context and **Reinforcement Learning (PPO)** to execute precision trades.

It combines **Heuristic Experts**, **Deep Learning (LSTM + PPO)**, and **Quantitative Risk Modeling** into a unified "Brain" that trades live via Alpaca.

---

## 🧠 The "v3.0" Architecture (Generative Agent)

The system has evolved from a simple scanner to a reasoning agent:

### 1. The Brain (Deep Learning)
*   **Sequence Modeling (LSTM)**: Processes the last 50 candles as a time-series sequence (not a snapshot).
*   **Supervised Fine-Tuning (SFT)**: Pre-trained on a "Golden Dataset" of perfect hindsight trades (ZigZag labeled) to learn simple "Common Sense" (77% Accuracy).
*   **RL Fine-Tuning (PPO)**: Fine-tuned in a vectorized GPU environment to learn risk management and complex trend following.

### 2. The Deployment (Live Trading)
*   **Alpaca Integration**: Connects directly to the Alpaca Paper Trading API.
*   **Real-Time Execution**: Fetches live data, processes features, and executes Buy/Sell/Hold orders autonomously.

---

## 📚 Documentation
*   **[Project Manual](docs/PROJECT_MANUAL.md)**: Detailed guide on setup, architecture, and modules.
*   **[Architecture Roadmap](docs/ARCHITECTURE_ROADMAP.md)**: The evolution path (Phase 1 to Phase 8).
*   **[Research Logs](docs/research/experiments_log.md)**: History of training experiments (MLP vs LSTM, Reward Shaping).

---

## 🚀 Quick Start (Deployment)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file with your Alpaca API credentials:
```env
APCA_API_KEY_ID="your_key"
APCA_API_SECRET_KEY="your_secret"
APCA_API_BASE_URL="https://paper-api.alpaca.markets"
```

### 3. Run the Trader (Dry Run)
Verify the Brain is working without placing real orders:
```bash
python -m src.trader_alpaca --symbol RELIANCE.NS --qty 1
```

### 4. Go Live (Paper Trading)
Allow the agent to execute trades:
```bash
python -m src.trader_alpaca --symbol RELIANCE.NS --qty 1 --live
```

---

## 📂 Project Structure

*   `src/`
    *   **Agent Logic**:
        *   `train_ppo_optimized.py`: The main RL Training Loop (Vectorized PPO + LSTM).
        *   `train_sft.py`: The Teacher (Supervised Fine-Tuning).
        *   `sft_dataset.py`: The Data Loader for Golden Labels.
        *   `data_labeler.py`: The Hindsight Labeler (ZigZag).
    *   **Deployment**:
        *   `trader_alpaca.py`: **Live Trading Script**.
    *   **Core**:
        *   `data_loader.py`: Nifty 500 Data Fetcher.
*   `frontend/`: React Dashboard (Archives/Visualizer).
*   `checkpoints/`: RL Model Weights (`best_ppo.ckpt`).
*   `checkpoints_sft/`: SFT Model Weights (`final_sft_model.pth`).
*   `docs/`: detailed documentation.

---

*Signal.Engine is an experimental research project. Use at your own risk.*
