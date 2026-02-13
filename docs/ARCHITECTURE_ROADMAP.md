# 🧠 Architecture Roadmap: The "Golden Mean"

This document outlines the strategic evolution of the Signal.Engine trading agent, moving from a black-box PPO implementation to a verifiable, reasoning-based Generative Trading Agent.

---

## Phase 1: Verifiable Rewards (Verified ✅)
**Goal:** Fix the "Blind Gambler" problem where the agent learns to overfit noise.
**Status:** **COMPLETE** (2026-01-23)
**Key Discovery:** "Micro-Training" (1 Epoch, 50 Iterations) is sufficient to master Trend Following logic. 100 Epochs is overfitting.
**Mechanism:** 
- **Component-Based Reward Function:** $ R = R_{trend} + R_{risk} + R_{pnl} $.
    - **Trend Reward:** Positive if long/short aligns with SMA50.
- **Entropy Scheduling:** Linear decay (0.68 -> 0.09) forces rapid convergence.
- **Optimization:** Micro-Mode + Graph Compilation (disabled on Windows) = < 2 min training.

## Phase 2: Sequence Modeling (Verified ✅)
**Goal:** Capture long-term market dependencies and context.
**Status:** **COMPLETE** (2026-01-23)
**Key Discovery:** LSTM architecture (`RecurrentActorCritic`) achieves 0.827 reward vs 0.80 MLP. Broadcasting `(Position, Balance)` across sequence is critical for state awareness.
**Mechanism:**
- **Architecture Shift:** Replace MLP with **LSTM** (Recurrent Neural Network).
- **State Representation:** Input is 3D Tensor `(Batch, Window, Features)` instead of flattened vector.

## Phase 3: Generative Trading (Verified ✅)
**Goal:** Encode "Expert Knowledge" before RL training.
**Status:** **COMPLETE** (2026-01-23)
**Results:** 
- **SFT Accuracy:** 76.9% (Imitating Golden Labels).
- **RL Improvement:** Bootstrapped agent achieves positive PnL (`0.125` reward) immediately vs random start.
**Mechanism:**
- **Supervised Fine-Tuning (SFT):** Train on `GoldenDataset` (ZigZag Hindsight).
- **RL Fine-Tuning:** Load SFT weights -> PPO with State-Based Rewards.

## Phase 4: Live Deployment (Verified ✅)
**Goal:** Bridge the "Brain" to the "Market".
**Status:** **OPERATIONAL** (2026-01-23)
**Mechanism:**
- **Alpaca API:** Fetches live bars, processes features, acts.
- **Dry-Run Verified:** `AAPL` test run confirmed pipeline integrity.

## Phase 5: Analytics & Validation (Verified ✅)
**Goal:** Prove statistical edge and track expert performance.
**Status:** **COMPLETE** (2026-02-13)
**Results:**
- **Edge Validation**: 21% win rate improvement for high confidence trades
- **Expert Tracking**: Quant most confident (88%), RL most active (180 signals)
- **Statistical Testing**: Chi-square tests, p-value tracking for significance
**Mechanism:**
- **Modular Analysis Framework**: Plugin-based system (`src/analysis/`)
- **Enhanced Metrics**: Sortino, Win Rate, Profit Factor, Calmar ratios
- **Analytics Dashboard**: React UI (`/analytics`) with statistical visualizations
- **Auto-generation**: PNG charts (300 DPI) + JSON data exports

---
**Project Status:** 🟢 **Active / Live**
The agent is now a fully autonomous entity capable of:
1.  **Thinking** (LSTM Sequence Modeling)
2.  **Learning** (PPO Reinforcement Learning)
3.  **Acting** (Alpaca API Execution)

---

## Implementation Details (Phase 1)

### verifiable_reward_function
```python
def calculate_reward(action, price, trend_indicator, volatility):
    # Trend Component
    if action == BUY and price > trend_indicator:
        r_trend = +0.1
    elif action == SELL and price < trend_indicator:
        r_trend = +0.1
    else:
        r_trend = -0.05
        
    return r_trend + pnl_scaled
```
