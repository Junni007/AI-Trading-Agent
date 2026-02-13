# Signal.Engine Changelog

## [v3.5] - 2026-02-13

### Added
- **Modular Analysis Framework** (`src/analysis/`)
  - Base `Analysis` class with auto-discovery and multi-format output
  - `expert_performance.py`: Compare confidence and activity across all 4 experts
  - `edge_validation.py`: Statistical significance testing with chi-square
  - Interactive CLI runner (`python -m src.analysis.runner`)

- **Enhanced Backtest Metrics** (`src/metrics_enhanced.py`)
  - Sortino Ratio (downside-adjusted returns)
  - Win Rate (% profitable trades)
  - Profit Factor (gross profit/loss ratio)
  - Calmar Ratio (return per unit of drawdown)

- **Data Schema Documentation** (`docs/DATA_SCHEMAS.md`)
  - Complete reference for all data structures
  - Expert votes, hybrid decisions, TDA features
  - Training data formats, backtest results
  - Query examples and usage notes

- **Analytics Dashboard** (`frontend/src/pages/Analytics.tsx`)
  - Statistical edge validation panel
  - Expert performance grid with badges
  - Confidence bucket win rate comparison
  - Direct links to PNG charts and JSON exports

- **Analysis Guide** (`docs/ANALYSIS_GUIDE.md`)
  - Quick reference for analysis framework
  - Usage examples and best practices
  - Custom analysis creation guide
  - Troubleshooting tips

- **Test Suites**
  - `tests/test_analysis_framework.py`: Analysis framework tests
  - `tests/test_backtest_metrics.py`: Enhanced metrics validation

### Changed
- Updated `README.md` with Analysis & Metrics section
- Updated `PROJECT_MANUAL.md` with 6th expert (Analysis Framework)
- Updated `ARCHITECTURE_ROADMAP.md` with Phase 5 (Analytics & Validation)
- Enhanced project structure documentation

### Technical Details
- **Memory Overhead**: <20 MB
- **New Dependencies**: 0 (all using existing libraries)
- **Lines Added**: ~1,400
- **Execution Time**: <5 seconds for all analyses

### Outputs
- `output/expert_performance.png` - Expert comparison chart (300 DPI)
- `output/expert_performance.json` - Expert statistics
- `output/edge_validation.png` - Win rate validation chart (300 DPI)
- `output/edge_validation.json` - Statistical test results

---

## [v3.0] - 2026-01-23

### Added
- LSTM-based recurrent trading agent
- Supervised Fine-Tuning (SFT) with ZigZag labels
- PPO Reinforcement Learning
- Alpaca live trading integration
- Hybrid Brain with 4 experts

### Changed
- Deprecated MLP architecture in favor of LSTM
- Moved legacy models to `archive/`

---

## [v2.0] - 2026-01-17

### Added
- Premium React frontend with glassmorphism
- Multiple expert systems (Sniper, Volatility, Quant)
- Gamified simulation system

---

## [v1.0] - 2026-01-01

### Added
- Initial LSTM model
- Basic backtesting infrastructure
- Data loading pipeline
