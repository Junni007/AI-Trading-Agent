# AI Trading Agent: Project Deep Dive 🦅

## 1. The Thought Process (Why this exists)

### The Failure of "Black Box" Models
The project started with a simple goal: *Can an AI predict the price of a stock 15 minutes from now?*
I spent weeks training **LSTM (Long Short-Term Memory)** neural networks on price and volume data.
*   **Input:** Past 60 candles.
*   **Output:** Next Close Price.
*   **Result:** 51% Accuracy.

The model wasn't "learning"; it was just outputting the current price +/- a small random noise. In trending markets, it trailed. In choppy markets, it got slaughtered. It lacked **Context**.

### The Pivot: Mixture of Experts (MoE)
I realized that human traders don't just "guess numbers." They follow a **Decision Tree**:
1.  *What is the Market Regime?* (High Volatility? Trending? Choppy?)
2.  *What Strategy fits this Regime?* (Sniping? Selling Options? Sitting out?)

So I deleted the Neural Network and built a **"Council of Experts"**.
Instead of one "Black Box" predicting a number, I built independent modules that "Vote" based on specific logic.

---

## 2. Architecture & The "Brain"

The Core System is a **Hybrid Thinking Engine** (`scan_hybrid.py`). It coordinates two sub-agents:

### Expert A: The "Sniper" (Intraday Momentum)
*   **File:** `scan_intraday.py`
*   **Personality:** Aggressive, Impatient.
*   **Logic:**
    *   **VWAP Support:** Price MUST be above the Volume Weighted Average Price (Institutions are buying).
    *   **RSI:** Must be between 55 (Bullish) and 75 (Not Overbought).
    *   **Volume Z-Score:** Needs a massive spike (>1.5 Std Dev) to confirm interest.
*   **Output:** "BUY" only if all 3 align.

### Expert B: The "Strategist" (Volatility/Income)
*   **File:** `scan_volatility.py`
*   **Personality:** Conservative, Patient.
*   **Logic:**
    *   checks **Historical Volatility (HV)**.
    *   If HV > 80th Percentile → Premiums are expensive → **Signal: SELL OPTIONS (Iron Condor)**.
    *   If HV < 20th Percentile → Market is coiled → **Signal: PREPARE FOR BREAKOUT**.

### The Meta-Brain (Conflict Resolution)
The `HybridBrain` class takes votes from both.
*   *Scenario 1:* Sniper says "BUY", Strategist says "High Volatility".
    *   **Brain Decision:** Do not buy naked stock (too risky). **Buy a Bull Put Spread** (Defined Risk).
*   *Scenario 2:* Sniper says "WAIT", Strategist says "Low Volatility".
    *   **Brain Decision:** **WATCHLIST**. A big move is coming, but no trigger yet.

---

## 3. Data Structures

The system relies on structured JSON-like dictionaries to pass information between Experts and the Brain.

### The Vote Object
Every Expert returns a standard "Vote" structure:
```json
{
  "Ticker": "RELIANCE.NS",
  "Signal": "BUY",
  "Confidence": 0.85,
  "Reason": "Price > VWAP + Volume Spike (Z=2.1)"
}
```

### The Decision Object (Final Output)
The Brain enriches this into a "Decision" used by the Frontend:
```json
{
  "Ticker": "TATASTEEL.NS",
  "Action": "LONG_CALL_SNIPER",
  "Confidence": 0.95,
  "Rational": [
    "Regime: Low Volatility (Coiled)",
    "Solution: PERFECT SETUP. Vol Expansion + Trend."
  ],
  "History": [ ... ] // 60 datapoints for Sparkline charts
}
```

### The Simulation State (`simulation_state.json`)
For the Gamification engine, we persist state in a flat JSON file:
```json
{
    "balance": 9850.50,
    "current_level": "Apprentice",
    "status": "ALIVE",
    "positions": {
        "INFY.NS": { "qty": 10, "avg_price": 1450.00 }
    },
    "history": [
        "SOLD_TP INFY @ 1460 (+1.2%) Reward: +1"
    ]
}
```

---

## 4. Key Features

### A. The "Thinking" Logs
Unlike standard trading bots that just say "BUY", this Agent explains itself.
The Frontend displays the `Rational` list, so you can see *why* it made a decision.
> *"Market is efficient. No edge detected."*
> *"Vol is high, but momentum is weak. Rejecting trade."*

### B. Gamified Simulation (Permadeath) ☠️
To solve the "Paper Trading has no stakes" problem, I introduced Game Mechanics.
*   **Survival Mode:** If Balance hits $0, the simulation **LOCKS**. The Agent is marked `DEAD`. You cannot restart without manually hacking the database.
*   **Leveling Up:** You start as a **Novice (Risk Taker)**. You only become a **Grandmaster** by accumulating consistent "Score" (Points for profitable exits, huge penalty for Stop Losses).

### C. Premium AI Interface 🎨
To match the intelligence of the backend, the Frontend has been upgraded to a **"Living Interface"**:
*   **Neural Particle Background:** A floating network of connected nodes runs in the background, simulating the AI's "Brain" searching for patterns.
*   **Holographic Focus:** When the AI finds a High-Confidence trade (>80%), the card glows with a spinning **Holographic Gradient Border**.
*   **Real-Time Reasoning:** The "Thinking Process" in the details view uses a **Typewriter Effect**, simulating the AI writing its report live for you.

---

## 5. Technology Stack

*   **Language:** Python 3.10+
*   **Data Source:** `yfinance` (Live feed emulation)
*   **Backend:** FastAPI (planned for Phase 3) / Direct Script Execution (Current)
*   **Frontend:** React + Vite + TailwindCSS
    *   **UI Library:** Framer Motion (for the "Glassmorphism" animations)
    *   **State Management:** React Hooks
