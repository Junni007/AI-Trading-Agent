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

## Phase 2: Sequence Modeling
**Goal:** Capture long-term market dependencies and context.
**Mechanism:**
- **Architecture Shift:** Replace MLP (Linear) Actor-Critic with **LSTM** or **Transformer** (Decision Transformer).
- **State Representation:** $ S_t $ involves a history window $ [t-k, t] $ explicitly processed as a sequence.

## Phase 3: Generative Trading (SFT + LoRA)
**Goal:** Encode "Expert Knowledge" before RL training.
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
