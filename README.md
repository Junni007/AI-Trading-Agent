# 🦅 Signal.Engine

### Hybrid Algorithmic Intelligence for the Nifty 500

**Signal.Engine** is an autonomous market analysis system that moves beyond simple price prediction. Instead of asking "Where will the price go?", it asks **"What is the current Regime?"** and adapts its strategy accordingly.

It combines **Heuristic Experts**, **Deep Reinforcement Learning (PPO)**, and **Quantitative Risk Modeling (Monte Carlo)** into a unified "Brain" that scans the market in real-time.

---

## 🧠 The "Funnel" Architecture

The system operates like a funnel, processing thousands of data points to find the few actionable opportunities.

### 1. Level 1: The Scanners (Fast)
The **Hybrid Brain** aggregates votes from three distinct experts to shortlist candidates:
*   **🎯 Sniper Expert**: Detects Intraday Momentum (VWAP Cross + Volume Spikes).
*   **🛡️ Income Expert**: Analyzes Volatility Regimes (IV Rank) to identify neutral/credit-spread setups.
*   **🤖 RL Expert (PPO)**: A Deep Neural Network trained to spot non-linear patterns.

### 2. Level 2: The Validator (Deep)
Shortlisted candidates are passed to the **Quant Expert**:
*   **🧪 Monte Carlo Simulation**: Runs **5,000 paths** using the **Heston Stochastic Volatility** model with **Jump Diffusion**.
*   **Risk Scoring**: Calculates Win Probability and Expected Value (EV).
*   **Final Decision**: If the Math agrees with the Signal, the confidence is boosted.

---

## ✨ Features (v2.1)

*   **Real-Time Dashboard**: A futuristic, distraction-free UI built with React & Tailwind.
    *   **"Void" Aesthetic**: Deep dark mode with holographic accents.
    *   **Live Sparklines**: Visual price history for every signal.
    *   **Signal Badges**: visual indicators for "AI" approval and "Monte Carlo" win rates.
*   **Nifty 500 Universe**: Scans the entire broad market index.
*   **Simulation Engine**: Built-in paper trading to track performance ("Novice" to "Grandmaster" levels).
*   **Adaptive Regimes**: Switches logic between **Bullish**, **Bearish**, and **High Volatility**.

---

## 🛠️ Tech Stack

*   **Brain (Backend)**: Python, FastAPI, Pandas, NumPy, PyTorch (RL).
*   **Face (Frontend)**: React, TypeScript, Vite, Framer Motion, Recharts.
*   **Data**: `yfinance` (Real-time), `ta` (Technical Analysis).

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Engine (Full Stack)
We use a convenient start script to launch both Backend and Frontend:
```bash
cd frontend
npm install
npm run start
```
*Access the Dashboard at `http://localhost:5173`*

---

## 📂 Project Structure

*   `src/brain/`: The core logic.
    *   `hybrid.py`: The Meta-Controller (The Funnel).
    *   `intraday.py`: Sniper Expert.
    *   `volatility.py`: Income Expert.
    *   `rl_expert.py`: PPO Agent Wrapper.
    *   `quant_expert.py`: Heston/Monte Carlo Engine.
*   `src/api/`: FastAPI Backend.
*   `frontend/`: The React Visualizer.
*   `docs/`: Detailed architectural documentation.

---

*Signal.Engine is an experimental research project. Use at your own risk.*
