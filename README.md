# 🦅 AI Trading Agent (The "Thinker" Engine)

## Overview
This project is an **Autonomous Trading Agent** built on a **Mixture of Experts (MoE)** architecture.
It moved away from "Black Box" Neural Networks (which drift with the market) to a **Hybrid Thinking Model** that dynamically switches between "Sniper" (Intraday Momentum) and "Income" (Volatility Selling) strategies.

## 🚀 Core Philosophy
- **Thinking, Not Guessing**: The system does not output a price. It outputs a **Vote** and a **Reason**.
- **Context is King**: High Volatility = Option Selling. Low Volatility = Directional Sniping.
- **Managed Outcomes**: Every trade has a specific Regime context.

## 🛠️ Modules (The Brain)

### 1. Hybrid Brain (`scan_hybrid.py`)
- **Purpose**: The Master Controller. Aggregates votes from Experts 1 & 2.
- **Output**: A "Thinking Process" report (e.g., "High Volatility -> Conflict -> Bull Put Spread").

### 2. Expert 1: Sniper Engine (`scan_intraday.py`)
- **Purpose**: Scans 15-minute candles for Momentum (VWAP Cross + Volume Spike).
- **Goal**: Catch the +2% Intraday Moves.

### 3. Expert 2: Income Engine (`scan_volatility.py`)
- **Purpose**: Scans Daily History for Volatility Rank (HV).
- **Goal**: Sell Premium (Iron Condors) when Volatility is High (>80%).

### 4. Daily Strategy Scanner (`scan_strategies.py`)
- **Purpose**: Legacy Daily Chart patterns (Hammer, Engulfing).
- **Use**: Best for finding long-term candidates.

## 📦 Installation
```bash
pip install -r requirements.txt
```

## 🏃 Usage
Run the specific engine you need, or the Master Brain:
```bash
python scan_hybrid.py
```

## 📊 Performance (Backtest 2018-2025)
- **Neural Networks (LSTM)**: Failed (51% Accuracy).
- **Daily Strategy**: Positive Expectancy (0.07%) but low hit rate.
- **Hybrid Engine**: Successfully identifies Regime-Specific trades (Validated Jan 2026).

## 📂 Project Structure
- `src/`: Data Loaders (Daily & Intraday 15m).
- `scan_*.py`: The Execution Engines.
- `archive/`: Legacy ML models (LSTM, XGBoost) - *Deprecated*.

---
*Built with Python, Pandas, and yfinance.*
