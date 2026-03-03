# 🧪 AI Trading Experiments & Findings Log

This document tracks the progress, experiments, and key learnings during the development of the Signal.Engine trading agent. It serves as a historical record of why certain architectural decisions were made.

---

## 📑 Table of Contents
1. [Experiment 01-05: The Failure of Pure Directional ML](#experiment-01-05-the-failure-of-pure-directional-ml)
2. [Experiment 06: The "Shootout" (Binary & Features)](#experiment-06-the-shootout-binary--features)
3. [Experiment 08: The "Sniper" Check (Managed Trades)](#experiment-08-the-sniper-check-managed-trades)
4. [Experiment 09: The Hybrid "Thinking" Engine (Final Architecture)](#experiment-09-the-hybrid-thinking-engine-final-architecture)
5. [Experiment 10: The "Funnel" Architecture (Quantitative Validation)](#experiment-10-the-funnel-architecture-quantitative-validation)
6. [Experiment 11-12: The Brain Transplant (RL Optimization)](#experiment-11-12-the-brain-transplant-rl-optimization)

---

## Experiment 01-05: The Failure of Pure Directional ML
**Dates**: Early January 2026
**Objective**: Train an LSTM model to predict the next-day price direction (Up, Down, Neutral) using standard technical indicators (RSI, MACD, Returns).

### 🧪 The Journey
- **Exp 01-03 (Model Collapse)**: The model kept predicting "Up" 100% of the time because the market drifts upward over long horizons.
- **Exp 04 (Dynamic Labeling)**: We forced perfectly balanced classes (33% Down, 33% Neutral, 33% Up) using dynamic quantiles.
  - *Result*: The model memorized the training set (56% accuracy) but failed completely on unseen data (32% accuracy - random chance).
- **Exp 05 (Regularization)**: We drastically shrank the model to prevent overfitting.
  - *Result*: The model ignored the "Neutral" class entirely and just guessed Up/Down.

### 🏁 Conclusion
**Financial data is too noisy for simple sequence prediction.** The features (`RSI`, `MACD`, `Returns`) have insufficient signal to distinguish a "Neutral" day from a "Weak Trend".

---

## Experiment 06: The "Shootout" (Binary & Features)
**Date**: 2026-01-02
**Objective**: Break the 32% accuracy ceiling by simplifying the problem to binary classification (UP vs DOWN) and adding rule-based indicators (ATR, EMA deviation).

### 🧪 Outcome
We benchmarked multiple models on 50 tickers (50k samples):
- **MLP (Neural Net)**: 51.05%
- **Logistic Regression**: 50.86%
- **Gradient Boosting**: 50.21%
- **Random Forest**: 50.06%
- **Random Guess**: 50.00%

### 🏁 Conclusion
**The "Model" is not the problem; the market is efficient.**
Standard technical indicators on daily data have near-zero predictive power for next-day direction. The Signal-to-Noise ratio is too low. We must stop trying to predict *direction* and start predicting *volatility* or shift to intraday execution.

---

## Experiment 08: The "Sniper" Check (Managed Trades)
**Objective**: Can we extract profit using strict risk management (Take Profit +2%, Stop Loss -1%) instead of predicting the daily close?

### 🧪 Outcome
- **RSI < 30 strategy**: 22.7% Hit TP (Avg PnL +0.08% per trade).
- **Hammer pattern**: 16.8% Hit TP (Avg PnL +0.07% per trade).

### 🏁 Conclusion
While there is a slightly positive expectancy, the "Take Profit" hit rate is terrible (~20%). 80% of the time, the price just drifts or hits the stop loss.
**Verdict**: Daily charts are too slow for directional scalping. We must move to Intraday data (15m charts).

---

## Experiment 09: The Hybrid "Thinking" Engine (Final Architecture)
**Date**: 2026-01-03
**Objective**: Combine Intraday Momentum and Volatility Regimes into a "Mixture of Experts" system.

### 🧠 The Logic
Instead of a black-box ML model, we use a structured decision tree:
1. **Volatility Expert**: Is IV Rank > 80%? → Sell Premium (Iron Condor).
2. **Sniper Expert**: Is Price > VWAP + RSI Breakout on 15m? → Buy Direction.

### 🏁 Final Verdict
**This is the superior solution.**
By targeting Volatility (which is mean-reverting) instead of Direction (which is a random walk), and using 15m data for entry, we avoid the "Daily Chart Noise". This architecture is now deployed.

---

## Experiment 10: The "Funnel" Architecture (Quantitative Validation)
**Date**: 2026-01-17
**Objective**: Integrate heavy probabilistic models (Monte Carlo, Heston Stochastic Volatility) without destroying real-time performance.

### 🧠 The Solution
Running 10,000 Monte Carlo paths for 500 stocks takes too long.
- **Level 1 (Fast)**: Heuristics and RL Agent scan the 500 stocks to find the top 10 setups.
- **Level 2 (Deep)**: The `QuantExpert` runs the heavy mathematical models *only* on those 10 candidates.

**Result**: Real-time performance is sub-2 seconds for the full Nifty 500 pipeline.

---

## Experiment 11-12: The Brain Transplant (RL Optimization)
**Date**: 2026-01-23
**Objective**: Train the Reinforcement Learning (PPO) agent efficiently.

### 🧪 Exp 11: Micro-Training
We hypothesized that simple financial concepts (like Trend Following) are "low complexity".
- *Result*: The agent solved "Trend Following" almost instantly (< 50 steps) when given explicit rewards. Training for 100 epochs caused catastrophic overfitting. We established a new standard of 1 Epoch / 50 Iterations.

### 🧪 Exp 12: MLP vs LSTM
We replaced the flat Neural Network (MLP) with a Recurrent Network (LSTM) that looks at the last 50 candles.
- *Result*: The LSTM achieved a **higher peak reward (+3.3%)** and maintained better exploration. It successfully learned to use "Memory" to inform its trading decisions.

**Status**: The Recurrent PPO agent is now the core AI engine.
