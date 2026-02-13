# Analysis Framework Quick Reference

## Overview

The analysis framework (`src/analysis/`) provides statistical validation and performance tracking for Signal.Engine's trading system.

---

## Quick Start

### Run All Analyses
```bash
python -m src.analysis.runner
# Select 0 to run all
```

### Individual Analysis
```bash
python -m src.analysis.runner
# Select 1 for edge_validation
# Select 2 for expert_performance
```

---

## Available Analyses

### 1. Expert Performance Analysis
**File**: `src/analysis/expert_performance.py`

**Purpose**: Compare confidence levels and activity across all 4 experts (Sniper, Volatility, RL, Quant).

**Outputs**:
- `output/expert_performance.png` - Dual-panel chart
- `output/expert_performance.json` - Raw data

**Key Metrics**:
- Average confidence by expert
- Total signals generated
- Most confident expert (badge)
- Most active expert (badge)

### 2. Edge Validation Analysis
**File**: `src/analysis/edge_validation.py`

**Purpose**: Prove statistical significance of trading edge using chi-square tests.

**Outputs**:
- `output/edge_validation.png` - Win rate comparison chart
- `output/edge_validation.json` - Statistical test results

**Key Metrics**:
- Win rate by confidence bucket (Low/Medium/High)
- Chi-square statistic & p-value
- Average PnL by confidence
- Statistical significance (p < 0.05)

---

## Enhanced Metrics

### File: `src/metrics_enhanced.py`

**Standard Metrics**:
- Total Return
- Sharpe Ratio
- Max Drawdown

**Enhanced Metrics**:
- **Sortino Ratio**: Downside-adjusted return (better for asymmetric strategies)
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss ratio (>1.0 = profitable)
- **Calmar Ratio**: Return / Max Drawdown (risk-adjusted)

**Usage**:
```python
from src.metrics_enhanced import calculate_metrics_enhanced
import pandas as pd

returns = pd.Series([0.01, -0.005, 0.02])
metrics = calculate_metrics_enhanced(returns)
print(metrics['Sortino Ratio'])
```

---

## Analytics Dashboard

### Access
```bash
cd frontend && npm run dev
# Navigate to http://localhost:5173/analytics
```

### Features

1. **Statistical Edge Panel**:
   - Confidence bucket cards
   - Significance indicator (✅/⚠️)
   - Win rate comparison

2. **Expert Performance Grid**:
   - 4 expert cards with badges
   - Confidence bars
   - Signal counts

3. **Data Exports**:
   - Links to PNG charts
   - JSON data downloads

---

## Creating Custom Analyses

### 1. Create New Analysis File

```python
# src/analysis/my_analysis.py
from .base import Analysis
import matplotlib.pyplot as plt

class MyCustomAnalysis(Analysis):
    name = "my_analysis"
    description = "My custom analysis"
    
    def run(self):
        # Your analysis logic
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        
        data = {"result": 123}
        return fig, data
```

### 2. Run It
```bash
python -m src.analysis.runner
# It will auto-discover your new analysis
```

---

## Data Schemas

All data structures are documented in:
- [docs/DATA_SCHEMAS.md](../DATA_SCHEMAS.md)

Key schemas:
- Expert Vote structure
- Hybrid Decision format
- TDA Features
- Training data formats

---

## Testing

### Run Analysis Tests
```bash
# If pytest available
pytest tests/test_analysis_framework.py -v

# Manual verification
python -m src.analysis.runner
ls output/  # Check files generated
```

### Run Metrics Tests
```bash
pytest tests/test_backtest_metrics.py -v

# Or run directly
python src/metrics_enhanced.py
```

---

## Best Practices

1. **Use Mock Data During Development**: Replace with real trade logs in production
2. **Keep Analyses Focused**: One analysis = one insight
3. **Document Assumptions**: Use docstrings to explain statistical methods
4. **Version Your Data**: Save JSON with timestamps for historical tracking

---

## Troubleshooting

**Analysis runner shows no analyses:**
- Ensure `__init__.py` exists in `src/analysis/`
- Check your analysis inherits from `Analysis` base class

**PNG files have white backgrounds:**
- Update `fig.patch.set_facecolor('#1a1a1a')` in your analysis

**Dashboard shows "No data":**
- Run `python -m src.analysis.runner` first
- Check `output/` directory has JSON files
- Verify frontend can access `/output/` route

---

For complete documentation, see:
- [Implementation Plan](../../.gemini/antigravity/brain/1a16b873-eeb1-4783-b7d8-4e86518abbc4/implementation_plan.md)
- [Walkthrough](../../.gemini/antigravity/brain/1a16b873-eeb1-4783-b7d8-4e86518abbc4/walkthrough.md)
