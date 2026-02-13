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
2.  **Intraday Sniper (`Expert 1`)**:
    - *Purpose*: Executing the "Sniper" trade.
    - *Triggers*: VWAP cross, RSI Momentum, Volume Spikes.
3.  **Derivatives Engine (`Expert 2`)**:
    - *Purpose*: Generating income from "Neutral" stocks via Option Selling (HV Rank).
4.  **RL Agent (`Expert 3`)**:
    - *Purpose*: Deep Learning (PPO) signal used as a "Booster" for high-confidence setups.
5.  **Quant Expert (`Expert 4`)**:
    - *Purpose*: The "Validator". Runs Monte Carlo simulations (Heston Model) on shortlisted candidates to calculate mathematical Win Probability.
6.  **Analysis Framework (`src/analysis/`)**:
    - *Purpose*: Statistical validation of trading edge and expert performance tracking.
    - *Features*: Automated chi-square tests, enhanced metrics (Sortino, Win Rate, Profit Factor).

### Data Pipeline (`src/`)
- **`data_loader.py`**: Handles Daily data fetching & Feature Engineering (RSI, EMA, Patterns).
- **`data_loader_intraday.py`**: Handles Live 15m/5m data fetching (Robust w/ Auto-Retry).
- **`patterns.py`**: Pure Python implementation of Candlestick Patterns (No `talib` dependency).
- **`ticker_utils.py`**: Manages S&P 500 & Nifty 50 ticker lists.
- **`metrics_enhanced.py`**: Enhanced performance metrics (Sortino, Win Rate, Profit Factor, Calmar).
- **`analysis/`**: Modular analysis framework (expert_performance, edge_validation).

## 3. Development Rules
1.  **Deterministic Logic**: No "Black Box" Neural Networks. All rules must be explainable (e.g., "Bought because RSI < 30").
2.  **Managed Outcomes**: Every trade *must* have a systematic Stop Loss and Take Profit in the backtest logic.
3.  **Clean Code**: Keep modules focused. One script = One job.

## 4. Current State (Feb 2026)
- **Daily Scanner**: **Active**. Scans 550 tickers.
- **Intraday Engine**: **Active**. (Expert 1).
- **Volatility Engine**: **Active**. (Expert 2).
- **RL Agent**: **Active**. (Expert 3).
- **Quant Expert**: **Active**. (Expert 4).
- **Hybrid Brain**: **Active**. (`scan_hybrid.py` Aggregator).
- **Analysis Framework**: **Active**. Statistical validation and performance tracking.
- **Analytics Dashboard**: **Active**. React UI with insights (`/analytics` route).
- **Legacy ML**: **Deprecated**. All LSTM/XGBoost models moved to `archive/`.

## 5. Strategic Roadmap
- **Step 1**: Build Intraday Scanner for "Momentum" (The Snipe). [COMPLETED]
- **Step 2**: Build Volatility Scanner for "Income" (The Hedge). [COMPLETED]
- **Step 3**: Combine into a "Live Dashboard" (`scan_hybrid.py`). [COMPLETED]
- **Step 4**: Add Statistical Validation & Analytics Framework. [COMPLETED]

## 6. Analysis & Validation

### Run Performance Analysis
```bash
python -m src.analysis.runner
```

### Available Analyses
1. **Expert Performance**: Compare confidence and activity across all 4 experts
2. **Edge Validation**: Statistical significance testing (chi-square) of trading edge

### Enhanced Metrics
- **Sortino Ratio**: Downside-adjusted returns
- **Win Rate**: % of profitable trades
- **Profit Factor**: Gross profit/loss ratio  
- **Calmar Ratio**: Return per unit of drawdown

### View Results
- **Charts**: `output/expert_performance.png`, `output/edge_validation.png`
- **Data**: `output/*.json`
- **Dashboard**: `http://localhost:5173/analytics`

---

## 7. Documentation

For detailed information, see:
- **[Data Schemas](DATA_SCHEMAS.md)**: Complete data structure reference
- **[Analysis Guide](ANALYSIS_GUIDE.md)**: How to use the analysis framework
- **[Architecture Roadmap](ARCHITECTURE_ROADMAP.md)**: Evolution path and phases
- **[Experiments Log](research/experiments_log.md)**: Failure analysis of LSTM approach

---

*Updated: February 2026*
