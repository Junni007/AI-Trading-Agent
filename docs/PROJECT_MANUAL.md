# Project Manual & Architecture Guide (Phase 2)

## 1. Project Overview
**Goal**: Build a "High Precision" Autonomous Trading Agent ("The Sniper").
**Philosophy**: Shift from "Predicting Direction" (50% Accuracy) to "Reacting to Volatility" (High Confidence Setups).
**Core Metrics**:
- **Win Rate**: >60% (Target).
- **Profit Factor**: >1.5.
- **Method**: Intraday Momentum & Options Income.

## 2. Architecture

### Core Engines
1.  **Strategy Scanner (`scan_strategies.py`)**:
    - *Timeframe*: Daily.
    - *Purpose*: Filters the "Universe" (500+ stocks) for specific setups (Hammer, Engulfing).
    - *Output*: Watchlist for the next day.
2.  **Intraday Sniper (`scan_intraday.py` - Active)**:
    - *Timeframe*: 15-minute / 5-minute.
    - *Purpose*: Executing the "Sniper" trade.
    - *Triggers*: VWAP cross, RSI Momentum, Volume Spikes.
3.  **Derivatives Engine (`scan_volatility.py` - Active)**:
    - *Purpose*: Generating income from "Neutral" stocks via Option Selling (HV Rank).

### Data Pipeline (`src/`)
- **`data_loader.py`**: Handles Daily data fetching & Feature Engineering (RSI, EMA, Patterns).
- **`data_loader_intraday.py`**: Handles Live 15m/5m data fetching (Robust w/ Auto-Retry).
- **`patterns.py`**: Pure Python implementation of Candlestick Patterns (No `talib` dependency).
- **`ticker_utils.py`**: Manages S&P 500 & Nifty 50 ticker lists.

## 3. Development Rules
1.  **Deterministic Logic**: No "Black Box" Neural Networks. All rules must be explainable (e.g., "Bought because RSI < 30").
2.  **Managed Outcomes**: Every trade *must* have a systematic Stop Loss and Take Profit in the backtest logic.
3.  **Clean Code**: Keep modules focused. One script = One job.

## 4. Current State (Jan 2026)
- **Daily Scanner**: **Active**. Scans 550 tickers.
- **Intraday Engine**: **Active**. (Expert 1).
- **Volatility Engine**: **Active**. (Expert 2).
- **Hybrid Brain**: **Active**. (`scan_hybrid.py` Aggregator).
- **Legacy ML**: **Deprecated**. All LSTM/XGBoost models moved to `archive/`.

## 5. Strategic Roadmap
- **Step 1**: Build Intraday Scanner for "Momentum" (The Snipe). [COMPLETED]
- **Step 2**: Build Volatility Scanner for "Income" (The Hedge). [COMPLETED]
- **Step 3**: Combine into a "Live Dashboard" (`scan_hybrid.py`). [COMPLETED]

---
*Reference: See `experiments_log.md` for the failure analysis of the LSTM approach.*
