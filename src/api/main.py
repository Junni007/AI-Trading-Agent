"""
Standalone Demo API for Koyeb Free Tier
- Zero heavy dependencies at import time
- Returns mock/demo data for frontend testing
- Can be upgraded later when on better hosting
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import random
from datetime import datetime

app = FastAPI(title="Signal Engine API (Demo)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Demo tickers
DEMO_TICKERS = ["RELIANCE", "TCS", "INFY", "HDFC", "ICICIBANK", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"]

# Simulation state
sim_state = {
    "balance": 100000,
    "holdings": {},
    "level": 1,
    "total_trades": 0,
    "wins": 0,
    "losses": 0,
    "status": "ALIVE",
    "portfolio_history": [100000]
}

def generate_demo_signals():
    """Generate realistic-looking demo trading signals"""
    signals = []
    actions = ["BUY", "SELL", "HOLD"]
    
    for ticker in random.sample(DEMO_TICKERS, min(5, len(DEMO_TICKERS))):
        confidence = random.uniform(0.5, 0.95)
        action = random.choice(actions)
        
        rationales = []
        if confidence > 0.8:
            rationales.append("Strong momentum detected")
        if action == "BUY":
            rationales.append("RSI showing oversold conditions")
            rationales.append("MACD bullish crossover")
        elif action == "SELL":
            rationales.append("RSI overbought")
            rationales.append("Resistance level reached")
        else:
            rationales.append("Consolidation phase")
        
        signals.append({
            "Ticker": ticker,
            "Action": action,
            "Confidence": round(confidence * 100, 2),
            "Rational": rationales,
            "QuantRisk": {
                "WinRate": round(random.uniform(0.55, 0.75), 2),
                "EV": round(random.uniform(0.5, 2.5), 2),
                "VaR95": round(random.uniform(-5, -2), 2),
                "MaxDrawdown": round(random.uniform(-10, -3), 2)
            }
        })
    
    signals.sort(key=lambda x: x["Confidence"], reverse=True)
    return signals

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Signal Engine (Demo Mode) is Ready.",
        "mode": "demo",
        "note": "This is a demo API. Full features require upgraded hosting."
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/api/scan")
def run_scan():
    return {"status": "started", "message": "Scan triggered (demo mode)."}

@app.get("/api/results")
def get_results():
    signals = generate_demo_signals()
    
    # Simulate portfolio changes
    change = random.uniform(-500, 1000)
    sim_state["balance"] = max(0, sim_state["balance"] + change)
    sim_state["portfolio_history"].append(sim_state["balance"])
    if len(sim_state["portfolio_history"]) > 50:
        sim_state["portfolio_history"] = sim_state["portfolio_history"][-50:]
    
    return {
        "status": "success",
        "data": signals,
        "simulation": sim_state,
        "logs": [
            f"[{datetime.now().strftime('%H:%M:%S')}] Demo scan completed",
            f"[{datetime.now().strftime('%H:%M:%S')}] Generated {len(signals)} signals"
        ],
        "is_thinking": False
    }

@app.get("/api/simulation/state")
def get_sim_state():
    return sim_state

@app.post("/api/simulation/reset")
def reset_sim():
    global sim_state
    sim_state = {
        "balance": 100000,
        "holdings": {},
        "level": 1,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "status": "ALIVE",
        "portfolio_history": [100000]
    }
    return {"status": "reset", "state": sim_state}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
