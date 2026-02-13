# Data Schemas Reference

Complete documentation of all data structures used in Signal.Engine.

---

## Signal Objects

### Expert Vote
Each expert (`SniperEngine`, `VolatilityEngine`, `RLExpert`, `QuantExpert`) returns a vote with this structure:

| Field | Type | Description |
|-------|------|-------------|
| `Signal` | string | One of: `BUY`, `SELL`, `WAIT`, `NEUTRAL` |
| `Confidence` | float | Confidence level (0.0 - 1.0) |
| `Reason` | string | Human-readable explanation |
| `Ticker` | string | Stock ticker (e.g., `RELIANCE.NS`) |
| `Price` | float | Current price (optional) |

**Example:**
```json
{
  "Ticker": "INFY.NS",
  "Signal": "BUY",
  "Confidence": 0.85,
  "Reason": "Price > VWAP + Volume Spike (Z=2.1)",
  "Price": 1450.50
}
```

**Source Files:**
- [intraday.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/brain/intraday.py) - Sniper Expert
- [volatility.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/brain/volatility.py) - Volatility Expert
- [rl_expert.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/brain/rl_expert.py) - RL Expert
- [quant_expert.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/brain/quant_expert.py) - Quant Expert

---

### Hybrid Decision
The `HybridBrain` aggregates expert votes into an enriched decision:

| Field | Type | Description |
|-------|------|-------------|
| `Ticker` | string | Stock ticker |
| `Action` | string | Trading strategy (see below) |
| `Confidence` | float | Aggregated confidence (0.0 - 1.0) |
| `Rational` | list[string] | Multi-line reasoning from experts |
| `History` | list[dict] | Price history for charts (optional) |

**Action Types:**
- `LONG_CALL_SNIPER` - High conviction directional play
- `IRON_CONDOR` - High volatility income strategy
- `BULL_PUT_SPREAD` - Defined risk bullish setup
- `WAIT` - No actionable setup

**Example:**
```json
{
  "Ticker": "TATASTEEL.NS",
  "Action": "LONG_CALL_SNIPER",
  "Confidence": 0.95,
  "Rational": [
    "Sniper: Price > VWAP + RSI Bullish (62.3)",
    "Volatility: Low IV (Preparing for breakout)",
    "Quant: Win Rate 0.78 | EV +12.5"
  ],
  "History": []
}
```

**Source File:**
- [hybrid.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/brain/hybrid.py)

---

## Training Data

### SFT Dataset (Golden Labels)
Supervised fine-tuning data from ZigZag hindsight labeling:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `state` | np.ndarray | `[window_size, 6]` | TDA feature matrix |
| `action` | int | `()` | 0=Hold, 1=Buy, 2=Sell |
| `ticker` | string | `()` | Source ticker |
| `timestamp` | datetime | `()` | Data point timestamp |

**Generation Process:**
1. Download OHLCV data
2. Apply ZigZag indicator (5% threshold)
3. Label peaks as "Sell", troughs as "Buy"
4. Extract TDA features for each labeled point

**Source Files:**
- [data_labeler.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/data_labeler.py)
- [sft_dataset.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/sft_dataset.py)

---

### PPO Episode Data
Reinforcement learning training rollouts:

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `obs` | torch.Tensor | `[batch, 8]` | TDA + account state |
| `action` | torch.Tensor | `[batch]` | Agent action (0-2) |
| `reward` | torch.Tensor | `[batch]` | Verifiable reward |
| `done` | torch.Tensor | `[batch]` | Episode terminal flag |
| `log_prob` | torch.Tensor | `[batch]` | Action log probability |
| `value` | torch.Tensor | `[batch]` | State value estimate |

**Reward Components:**
```python
reward = r_pnl + r_trend + r_holding_cost
# r_pnl: Realized profit/loss
# r_trend: +0.05 if aligned with SMA50
# r_holding_cost: -0.001 per step in position
```

**Source Files:**
- [agent.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/agent.py) - TradingAgent class
- [env.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/env.py) - TradingEnv

---

## Feature Vectors

### TDA Features
Topological Data Analysis features from `tda_features.py`:

| Index | Feature | Range | Description |
|-------|---------|-------|-------------|
| 0 | Betti-0 | [0, ∞) | Connected components (trend patterns) |
| 1 | Betti-1 | [0, ∞) | Loops/cycles (mean reversion signals) |
| 2 | Persistence | [0, ∞) | Pattern strength/lifetime |
| 3 | Norm Returns | [-1, 1] | Scaled log returns |
| 4 | Vol Ratio | [0, ∞) | Current vol / historical avg |
| 5 | Volume Shock | (-∞, ∞) | Z-score of volume |

**Feature Engineering:**
```python
from src.tda_features import FeatureProcessor

processor = FeatureProcessor()
features = processor.compute(price_array)  # Shape: [window, 6]
```

**Source File:**
- [tda_features.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/tda_features.py)

---

### Traditional Indicators
Standard technical indicators in DataFrames:

| Column | Type | Range | Calculation |
|--------|------|-------|-------------|
| `Close` | float | (0, ∞) | Closing price |
| `RSI` | float | [0, 100] | Relative Strength Index (14-period) |
| `VWAP` | float | (0, ∞) | Volume-Weighted Average Price |
| `MACD` | float | (-∞, ∞) | 12-26 EMA difference |
| `MACD_Signal` | float | (-∞, ∞) | 9-period EMA of MACD |
| `ATR` | float | [0, ∞) | Average True Range (14-period) |
| `Vol_Z` | float | (-∞, ∞) | Volume Z-Score (20-period) |
| `Log_Return` | float | (-∞, ∞) | ln(Close[t] / Close[t-1]) |

**Data Loading:**
```python
from src.data_loader import MVPDataLoader

loader = MVPDataLoader(ticker="RELIANCE.NS")
df = loader.fetch_data()  # Returns engineered DataFrame
```

**Source Files:**
- [data_loader.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/data_loader.py)
- [data_loader_intraday.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/data_loader_intraday.py)

---

## Backtest Results

### Performance Metrics
From `backtest.py`:

```python
{
    "Total Return": 0.127,      # 12.7% return
    "Sharpe Ratio": 1.82,       # Risk-adjusted return
    "Max Drawdown": -0.08,      # -8% worst drop
    "Win Rate": 0.58,           # 58% winning trades  
    "Profit Factor": 1.45,      # Gross profit / Gross loss
    "Sortino Ratio": 2.1,       # Downside-adjusted return
    "Calmar Ratio": 1.59        # Return / Max Drawdown
}
```

**Calculation Details:**
- **Sharpe**: `mean(returns) / std(returns) * sqrt(252)`
- **Sortino**: `mean(returns) / std(negative_returns) * sqrt(252)`
- **Max Drawdown**: `min((cumulative - running_max) / running_max)`
- **Win Rate**: `count(returns > 0) / count(returns)`
- **Profit Factor**: `sum(positive_returns) / abs(sum(negative_returns))`

**Source File:**
- [backtest.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/backtest.py)

---

## Storage Formats

### Current Implementation

| Data Type | Format | Location | Notes |
|-----------|--------|----------|-------|
| Market tickers | CSV | `src/nifty500.csv` | 500+ Indian stocks |
| Signals | JSON | In-memory | No persistence |
| Model checkpoints | `.ckpt` | `checkpoints/` | PyTorch Lightning |
| SFT weights | `.pth` | `checkpoints_sft/` | PyTorch state dict |
| Simulation state | JSON | `frontend/simulation_state.json` | Gamification data |
| Analysis outputs | PNG/JSON | `output/` | From analysis framework |

### Recommended Future Enhancement

**SQLite for Lightweight Persistence:**
```sql
-- Expert votes history
CREATE TABLE votes (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    ticker TEXT,
    expert TEXT,
    signal TEXT,
    confidence REAL,
    reason TEXT
);

-- Backtest results
CREATE TABLE backtests (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    model_version TEXT,
    sharpe REAL,
    max_dd REAL,
    win_rate REAL
);
```

**Benefits:**
- Track expert accuracy over time
- Compare model versions
- No memory overhead (disk-based)
- Standard SQL queries
- ~10MB database for 1000s of records

---

## API Endpoints (Future)

When the FastAPI backend is activated:

### GET /api/signals
Returns current active signals from HybridBrain

**Response:**
```json
[
  {
    "Ticker": "RELIANCE.NS",
    "Action": "WAIT",
    "Confidence": 0.65,
    "Rational": ["Market: Choppy regime"]
  }
]
```

### POST /api/backtest
Runs backtest on specified ticker

**Request:**
```json
{
  "ticker": "INFY.NS",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

**Response:**
```json
{
  "metrics": {
    "Sharpe": 1.82,
    "Win Rate": 0.58
  }
}
```

**Source File:**
- [src/api/main.py](file:///c:/Users/Junni-Adi/Documents/GitHub/tradeing%20agent/src/api/main.py)

---

## Query Examples

### Load and Analyze Expert Votes
```python
import json
from pathlib import Path

# Assuming votes are logged to JSON
votes_file = Path("logs/expert_votes.json")
if votes_file.exists():
    with open(votes_file) as f:
        votes = json.load(f)
    
    # Analyze Sniper accuracy
    sniper_votes = [v for v in votes if v['expert'] == 'Sniper']
    avg_confidence = sum(v['confidence'] for v in sniper_votes) / len(sniper_votes)
    print(f"Sniper avg confidence: {avg_confidence:.2f}")
```

### Filter High-Confidence Signals
```python
from src.brain.hybrid import HybridBrain

brain = HybridBrain()
thoughts = brain.think()

# Only high conviction trades
high_conf = [t for t in thoughts if t['Confidence'] > 0.8]
print(f"Found {len(high_conf)} high-confidence setups")
```

### Backtest with Enhanced Metrics
```python
from src.backtest import calculate_metrics_enhanced
import pandas as pd

returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
metrics = calculate_metrics_enhanced(returns)

print(f"Sharpe: {metrics['Sharpe Ratio']:.2f}")
print(f"Sortino: {metrics['Sortino Ratio']:.2f}")
print(f"Win Rate: {metrics['Win Rate']:.1%}")
```

---

## Notes

> [!NOTE]
> All monetary values are in INR for Indian stocks (`.NS` suffix) and USD for US stocks.

> [!TIP]
> Use `pd.DataFrame.to_json()` to serialize DataFrames for frontend consumption.

> [!IMPORTANT]
> TDA features require a minimum window of 50 candles. Shorter windows will fail.

> [!WARNING]
> Model checkpoints from different architectures are incompatible. Always check `input_dim` matches.
