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

## Phase 3: Generative Trading (SFT + LoRA)
**Goal:** Encode "Expert Knowledge" before RL training.
**Status:** **PLANNED**

**Mechanism:**
- **Supervised Fine-Tuning (SFT):** Train a Foundation Model (Time-Series Transformer) on a "Golden Dataset" of perfect hindsight trades.
- **LoRA (Low-Rank Adaptation):** Efficiently fine-tune this pre-trained "Expert" using RL (PPO) on live market data.
- **Result:** An agent that starts with "Common Sense" and adapts to "Alpha".

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
