# Theoretical Background: Topological Data Analysis (TDA) in Finance

This document explains the mathematical foundations behind the **TDA Trading Agent**. We utilize **Topological Data Analysis (TDA)** to extract shape-based features from financial time series, which act as robust, noise-resistant inputs for our Reinforcement Learning model.

---

## 📑 Table of Contents

1. [The Problem with Standard Indicators](#1-the-problem-with-standard-indicators)
2. [Time Delay Embedding (Takens' Theorem)](#2-time-delay-embedding-takens-theorem)
3. [Persistent Homology](#3-persistent-homology)
4. [Translating TDA to Trading Signals](#4-translating-tda-to-trading-signals)
5. [Implementation in Signal.Engine](#5-implementation-in-signalengine)

---

## 1. The Problem with Standard Indicators

Traditional technical indicators (RSI, MACD, Bollinger Bands) suffer from a fundamental flaw: **they are point-in-time derivatives of price that lag behind real market dynamics.**

They analyze the *amplitude* of data, making them highly susceptible to market noise, volatility spikes, and regime changes.

**The TDA Approach:**
Instead of asking "What is the average price over the last 14 days?", TDA asks:
*"What is the underlying geometric shape of the market behavior over the last 50 candles?"*

By focusing on the *topology* (shape) rather than the exact coordinates (price), the model becomes invariant to small fluctuations and learns to recognize true structural patterns (like a coiled spring before a breakout).

---

## 2. Time Delay Embedding (Takens' Theorem)

Financial data typically comes as a single 1D time series (e.g., Closing Prices). However, the underlying market dynamics are complex and multidimensional (driven by millions of interacting agents).

**Takens' Theorem** allows us to reconstruct the "phase space" (the hidden, multi-dimensional state space) of a dynamic system from a single observable variable.

### The Math
Given a time series $X = [x_1, x_2, ..., x_n]$, we create a point cloud in $d$-dimensional space by taking delayed copies of the series:

$V_t = [x_t, x_{t-\tau}, x_{t-2\tau}, ..., x_{t-(d-1)\tau}]$

Where:
- $d$ is the **Embedding Dimension** (how many axes we project into).
- $\tau$ (Tau) is the **Time Delay** (the lag between points).

```mermaid
graph LR
    A[1D Price Time Series] -->|Time Delay Embedding| B[3D Point Cloud]
    B -->|Phase Space Reconstruction| C[Hidden Market Structure]

    style A fill:#2a1a24,stroke:#775,color:#fff
    style B fill:#1a1a24,stroke:#555,color:#fff
    style C fill:#111,stroke:#444,stroke-width:2px,color:#fff
```

In `Signal.Engine`, we typically use $d=3$ or $d=4$, turning a simple line chart into a multi-dimensional scatter plot where structural patterns become visible.

---

## 3. Persistent Homology

Once we have our point cloud in the phase space, we need to measure its "shape." We do this using **Persistent Homology**.

Imagine drawing a circle (radius $\epsilon$) around every point in the cloud. As we slowly increase $\epsilon$, the circles start to merge, forming connected components, loops, and voids.

### Betti Numbers
We count these topological features using **Betti Numbers**:
- **Betti-0 ($H_0$)**: The number of connected components. (Measures clustering / trending behavior).
- **Betti-1 ($H_1$)**: The number of 1-dimensional loops. (Measures cyclicality / mean-reversion behavior).
- **Betti-2 ($H_2$)**: The number of 2-dimensional voids. (Rarely used in finance due to noise).

### Persistence
A feature that appears and disappears quickly is considered "noise." A feature that *persists* across a large range of $\epsilon$ is considered a true, structural "signal."

*The lifespan of a feature (Death - Birth) is its **Persistence**.*

---

## 4. Translating TDA to Trading Signals

How do these abstract mathematical concepts translate to buying and selling stocks?

| Topological Feature | Market Interpretation | Trading Signal |
| :--- | :--- | :--- |
| **High Betti-0 Persistence** | Data points are tightly clustered along a line. | **Strong Trend.** Trend-following strategies (Momentum) perform best here. |
| **High Betti-1 Persistence** | Data points form a loop in the phase space. | **Mean Reversion / Consolidation.** The market is cycling. Breakout or Iron Condor strategies perform best here. |
| **Rapid Topology Change** | The structural shape suddenly fractures or changes Betti numbers. | **Regime Shift.** A market shock or trend reversal is occurring. High probability of volatility. |

---

## 5. Implementation in Signal.Engine

In `src/tda_features.py`, we utilize the `gudhi` library to compute these features efficiently in real-time.

1. **Windowing**: We take the last $W$ candles (e.g., 50).
2. **Embedding**: We create the point cloud.
3. **Simplicial Complex**: We build a Vietoris-Rips complex to calculate homology.
4. **Feature Extraction**: We extract `max(persistence(H0))` and `max(persistence(H1))` and pass them as features to the Deep Learning model (PPO).

By feeding the Neural Network these structural TDA features alongside traditional technical indicators, the AI gains a "sixth sense" for market regimes that standard quantitative models miss.
