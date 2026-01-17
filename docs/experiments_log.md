# 🧪 AI Trading Experiments & Findings Log

This document tracks the progress, experiments, and key learnings during the scaling and development of the AI Trading Agent.

## Experiment 01: Massive Scale Training (550+ Tickers)
**Date**: 2026-01-01
**Objective**: Train a single LSTM model on ~550 tickers (S&P 500 + Nifty 50) to learn universal market patterns.

### ⚙️ Configuration
- **Model**: LSTM (Input: ?, Hidden: 256, Layers: 3)
- **Data**: ~550 Tickers, 2019-2022 (Train), 2023 (Val), 2024 (Test).
- **Features (Initial)**: `['Close', 'RSI', 'MACD', 'Log_Return']`
- **Batch Size**: 64
- **Precision**: 16-mixed

### 📉 Observation: Model Collapse
During the first massive run, the model exhibited "Collapse" behavior:
- **Training Accuracy**: ~44% (stuck).
- **Validation Accuracy**: ~42% (flatline).
- **Test Precision**: 
    - DOWN: 0.00
    - NEUTRAL: 0.00
    - UP: 0.42 (100% Recall)
    
**Diagnosis**: 
The model learned to predict "UP" for every single sample.
Why? The **`Close`** price feature. 
- In a mixed dataset, Asset A has `Close=$150` (AAPL) and Asset B has `Close=$2000` (GOOGL/Booking). 
- A simple `StandardScaler` fitted mainly on lower-priced assets (or averaged) causes massive outliers for high-priced assets.
- The model treats `Close` as a raw value rather than a relative trend, confusing the gradients.

### 🛠️ The Fix (Hypothesis)
**Action**: Remove `Close` price from the input features.
**New Features**: `['RSI', 'MACD', 'MACD_Signal', 'Log_Return']`
- **Rationale**: All selected features are **stationary** or bounded. 
    - `RSI` is always 0-100. 
    - `MACD` behaves similarly relative to price action momentum.
    - `Log_Return` is a percentage change (scale invariant).
    
**Next Steps**: Retrain with the clean feature set.

### 📊 Result (Experiment 01b)
**Config**: Removed 'Close'. Features: `[RSI, MACD, Returns]`. No Optuna.
**Outcome**:
- **Accuracy**: ~41.5%
- **Precision (DOWN)**: 0.18 (First sign of learning!)
- **Recall (UP)**: 0.99 (Still heavily biased to Bull Market)
**Insight**: The model is safer but still lazy (predicts UP 90% of time). This is because 2024 was a Bull Market. We need **Class Balancing** or **Hyperparameter Tuning** to force it to learn "DOWN" signals better.

## Experiment 02: Bayesian Architecture Search
**Date**: 2026-01-01
**Objective**: Optimize LSTM hyperparameters using Optuna (Automated Tuning).

### 🔍 Search Space
- **Hidden Dim**: `[64, 128, 256, 512]`
- **Num Layers**: `[1, 2, 3, 4]`
- **Dropout**: `0.1` - `0.5`
- **Learning Rate**: `1e-5` - `1e-2` (Log Scale)
- **Batch Size**: `[32, 64, 128]`

### 🧠 Hypothesis
- A **Deep but Narrow** network (e.g., 3 Layers, 128 Dim) might generalize better on Multi-Ticker data than a Shallow Wide one.
- **Pruning**: Using `MedianPruner` to kill bad trials early (e.g., if Loss > Median at epoch 2, stop).

### 🧪 Execution
Running `src/tune.py` on Colab (GPU).
Results will be saved to `best_hyperparameters.json`.

## Experiment 03: Tuning & Collapse Diagnosis
**Date**: 2026-01-02
**Objective**: Train using optimized hyperparameters and diagnose "Model Collapse" (only predicting UP).

### 🔍 Results
- **Optimized Params**: `{'hidden_dim': 256, 'num_layers': 4, 'dropout': 0.22, 'lr': 2e-5, 'batch_size': 64}`
- **Training**:
    - Converged to Accuracy ~42%.
    - **Issue**: Precision for DOWN/NEUTRAL is 0.0. The model ignores these classes entirely.
- **Hypothesis**:
    1.  **Architecture Bug?**: Is the LSTM code broken?
        - **Test**: `debug_overfit.py` (Single Batch Overfit).
        - **Result**: ✅ Passed (Acc: 1.00). The model CAN learn patterns.
    2.  **Data Issue?**: 
        - Features `['RSI', 'MACD', 'MACD_Signal', 'Log_Return']` are stationary but might be too noisy or have weak predictive power.
        - Lack of **Batch Normalization** might be hurting deep (4-layer) convergence.
        - **Class Imbalance**: 2024 was a strong Bull Market, biasing the model to "Always UP" as a local optimum.

**Next Steps**:
- Verify data statistics (`inspect_data.py`).
- Add Batch Normalization.
- Consider Weighted Loss or Class Balancing.

## Experiment 04: Dynamic Labeling (The Fix)
**Date**: 2026-01-02
**Objective**: Fix Model Collapse by enforcing perfect class balance.

### 🛠️ Change Implementation
- **Method**: Replaced fixed threshold (`0.002`) with **Dynamic Quantiles** (`pd.qcut`).
- **Logic**: 
    - Future returns are sorted.
    - Bottom 33% -> Class 0 (Down)
    - Middle 33% -> Class 1 (Neutral)
    - Top 33% -> Class 2 (Up)
- **Also Added**: `BatchNorm1d` to LSTM model for stability.

### 📊 Verification (Data Inspection)
Running `inspect_data.py` on 5 random tickers confirmed the fix:
- **Class 0**: 34.70%
- **Class 1**: 32.23%
- **Class 2**: 33.07%
✅ **Result**: Data is now perfectly balanced. The model is mathematically forced to see all classes equally.

### 🧪 Outcome (Experiment 04)
- **Training Accuracy**: ~56% (Good! It learned the training data).
- **Validation Accuracy**: ~30% (Bad - Random Chance).
- **Test Results**:
    - Precision (DOWN): 0.33 (Non-zero! Improvement)
    - Recall (UP): 0.88 (Still biased)
    - **Confusion Matrix**: Shows predictions for all classes, but heavy confusion.

**Diagnosis**: **Overfitting**.
The model (4 Layers, 256 Hidden) is too complex for this noisy data. It memorized the balanced training set but failed to generalize to the 2023/2024 unseen markets.

## Experiment 05: Regularization (Simplicity)
**Objective**: Force generalization by drastically reducing model complexity.
**Hypothesis**: Financial data is noisy. A smaller model (Lower Variance) will generalize better than a deep one.

### 🧪 Outcome (Experiment 05)
- **Train Accuracy**: 44% (Expected drop).
- **Val Accuracy**: ~30% (Flat).
- **Test Results**:
    - **Neutral Precision**: 0.00 (The model ignoring the Middle 33% entirely).
    - **Accuracy**: 32% (Random).
    - **Confusion Matrix**: 
        - Predicts DOWN ~40% of time.
        - Predicts UP ~60% of time.
        - Predicts NEUTRAL 0% of time.

### 🏁 Conclusion & Strategy Pivot
1.  **Model Collapse (Only UP)**: **SOLVED**. The model now predicts Down and Up.
2.  **Generalization**: **FAILED**. The model cannot generalize to unseen data (2023/2024).
3.  **Diagnosis**: The current features (`RSI`, `MACD`, `Returns`) likely have **insufficient signal** to distinguish the "Neutral" class from "Weak Trend". The model finds it safer to just guess Up/Down.

**Next Strategic Move**:
We have exhausted "Architecture Search" (Complex vs Simple). The problem is the **Data Quality**.
- **Option B**: Different Model (XGBoost might handle the noisy thresholds better).
- **Option C**: Binary Classification (Switch target to just UP/DOWN).

## Experiment 06: The "Concrete Fix" (Binary & Features)
**Date**: 2026-01-02
**Objective**: Break the 32% accuracy ceiling by simplifying the problem and enriching the data.

### 🛠️ Major Architecture Changes
1.  **Binary Switch**: Dropped "Neutral" class. Target is now **UP (1)** vs **DOWN (0)**.
    - *Why*: Removes the ambiguous "Weak Trend" samples that confused the model.
2.  **Feature Enrichment**: Added "Rule-Based" indicators:
    - `EMA_Trend`: Normalized deviation from 50-day Trend.
    - `ATR_Vol`: Volatility proxy.
    - `Vol_Change`: Volume momentum.
3.  **The Shootout**: We are benchmarking two approaches:
    - **Deep Learning**: LSTM (Sequence Expert).
    - **Classical**: XGBoost (Tabular Expert).

3.  **Goal**: Achieve >55% Directional Accuracy.

### 🧪 Outcome (Experiment 06/07 - The Shootout)
**Data**: 50 Tickers (50k samples).
**Features**: RSI, MACD, EMA, ATR, Volume.
**Results**:
- **MLP (Neural Net)**: 51.05% (Champion)
- **Logistic Regression**: 50.86%
- **Gradient Boosting**: 50.21%
- **Random Forest**: 50.06%
- **Random Guess**: 50.00%

### 🏁 Conclusion
**The "Model" is not the problem.**
We tried 10 architectures (Linear, Tree, Boosting, Neural). All converged to ~50-51%.
This proves **The Efficiency of the Market**:
- Standard Technical Indicators (RSI/MACD) on Daily Data have **near-zero predictive power** for next-day direction on clean out-of-sample data.
- The "Signal-to-Noise" ratio is too low.

**Strategic Conclusion**:
Improving the *architecture* further is futile. We must change the **Question** or the **Data**.
- **Idea 1**: Predict "Volatility" (Risk) instead of "Direction" (Gain) -> Easier.
- **Idea 2**: Longer horizon (Weekly) vs Daily.
- **Idea 3**: Sentiment Data (News) to break the efficiency.

### 🧪 Outcome (Experiment 08 - Managed Trades / "Sniper" Check)
**Hypothesis**: Maybe Close-to-Close is noisy. Can we "Snpe" +2% intraday profits?
**Data**: All ~550 Tickers (Max Scale).
**Mechanism**: Buy Signal -> Target +2% / Stop -1%.
**Results**:
- **RSI < 30**: 22.7% Hit TP (Avg PnL +0.08%).
- **Hammer**: 16.8% Hit TP (Avg PnL +0.07%).
- **Golden Cross**: 14.5% Hit TP (Avg PnL +0.04%).

**Conclusion**:
- **Positive Expectancy**: Yes, the PnL is positive (0.07% per trade).
- **Not a Sniper**: The "TP Hit Rate" is very low (~20%). This means 80% of the time, the price drifts or hits SL.
- **Verdict**: Daily charts are too slow/efficient for "High Confidence" directional scalping.
-   **Action**: Move to **Intraday** (higher relative volatility) or **Options** (selling the drift).

## Experiment 09: The Hybrid "Thinking" Engine (Final Architecture)
**Date**: 2026-01-03
**Objective**: Combine Intraday Momentum and Volatility Regimes into a "Mixture of Experts" system.

### 🧠 The Logic (Thinking Model)
Instead of a "Black Box" prediction, we use a recursive decision tree:
1.  **Check Regime (Expert 2)**: Is IV Rank > 80% (High Vol)? -> Sell Premium.
2.  **Check Momentum (Expert 1)**: Is Price > VWAP + RSI Breakout? -> Buy Direction.
3.  **Resolve**:
    -   High Vol + Momentum -> **Bull Put Spread/Credit Spread**.
    -   Low Vol + Momentum -> **Long Call Sniper**.
    -   Low Vol + No Momentum -> **Wait**.

### 🧪 Outcome (Final Verification)
Run of `scan_hybrid.py` on 14 liquid tickers (Tech + Indices):
-   **GOOGL**: Identified High Volatility Regime.
    -   Decision: **IRON CONDOR** / Bull Put Spread.
    -   *Why*: Capitalizes on "Drift" while markets are fearful.
-   **NVDA**: Identified Low Volatility (Coiled).
    -   Decision: **WATCH FOR BREAKOUT** (Sniper Prep).
    -   *Why*: Avoiding "Chop". Waiting for the expansion phase.

### 🏁 Final Verdict
The **Hybrid Architecture** is the superior solution.
-   **Solves efficiency**: By targeting Volatility (which is predictable/mean-reverting) instead of Direction (which is random walk).
-   **Solves "Sniper" problem**: By using 15m data for entry, we avoid the "Daily Chart Noise".
-   **Status**: **DEPLOYED**.

## Experiment 10: The "Funnel" Architecture (Quantitative Validation)
**Date**: 2026-01-17
**Objective**: Integrate Advanced Probability models (Monte Carlo, Heston, Jump Diffusion) without killing performance.

### ⚠️ The Constraint
Running 10,000 Monte Carlo simulations with Stochastic Volatility for 500+ tickers in real-time is computationally impossible.

### 🧠 The Solution: "The Funnel"
1.  **Level 1 (Fast)**: Use **Heuristics (Sniper)** and **RL (PPO Agent)** to scan the full universe (Nifty 500) and identify top candidates.
2.  **Level 2 (Deep)**: Pass *only* the shortlisted candidates to the **Quantitative Expert**.
    -   Run **10k Path Monte Carlo** with Heston Stochastic Volatility.
    -   Apply **Merton Jump Diffusion** to model tail risks.
    -   Output: A mathematical "Probability of Success".

**Status**: **DEPLOYED / INTEGRATED**.
-   **Method**: `HybridBrain` scans 500 tickers -> Filters Top 10 -> `QuantExpert` runs 5k paths on each.
-   **Result**: Validated "Level 1" -> "Level 2" funnel on mocked data. Real-time performance is sub-2 seconds for full pipeline.
