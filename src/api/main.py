"""
Lightweight API for Koyeb Free Tier (256MB RAM limit)
- Lazy loads heavy modules only when needed
- Minimal startup memory footprint
"""
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Signal Engine API (Lite)", version="1.0")

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State - Lazy initialized
brain = None
sim_engine = None
LATEST_DECISIONS = []
LATEST_LOGS = []
IS_SCANNING = False

def get_brain():
    """Lazy load the brain to avoid startup memory issues"""
    global brain
    if brain is None:
        logger.info("🧠 Loading HybridBrain...")
        from src.brain.hybrid import HybridBrain
        brain = HybridBrain()
    return brain

def get_sim_engine():
    """Lazy load the simulation engine"""
    global sim_engine
    if sim_engine is None:
        logger.info("🎮 Loading SimulationEngine...")
        from src.simulation.engine import SimulationEngine
        sim_engine = SimulationEngine()
    return sim_engine

@app.get("/")
def home():
    return {"status": "Online", "message": "Signal Engine (Lite) is Ready."}

@app.get("/health")
def health():
    """Health check endpoint for Koyeb"""
    return {"status": "healthy"}

def background_scan():
    global LATEST_DECISIONS, LATEST_LOGS, IS_SCANNING
    try:
        logger.info("🧠 Brain started thinking...")
        _brain = get_brain()
        _sim = get_sim_engine()
        
        decisions = _brain.think()
        
        # Determine Regime
        regime = "NEUTRAL"
        vol_count = sum(1 for d in decisions if "High Volatility" in str(d.get('Rational', [])))
        if vol_count > len(decisions) * 0.3:
            regime = "HIGH_VOLATILITY"
        
        # Simulation Tick
        logs = _sim.process_tick(decisions, regime=regime)
        
        # Sort by Confidence
        decisions.sort(key=lambda x: x['Confidence'], reverse=True)
        
        LATEST_DECISIONS = decisions
        LATEST_LOGS = logs
        logger.info("✅ Brain finished thinking.")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        IS_SCANNING = False

@app.get("/api/scan")
def run_scan(background_tasks: BackgroundTasks):
    """Triggers the Hybrid Brain to think in the background."""
    global IS_SCANNING
    if IS_SCANNING:
        return {"status": "busy", "message": "Brain is already thinking."}
    
    IS_SCANNING = True
    background_tasks.add_task(background_scan)
    return {"status": "started", "message": "Scan triggered in background."}

@app.get("/api/results")
def get_results():
    """Returns the latest available scan results."""
    _sim = get_sim_engine() if sim_engine else None
    return {
        "status": "success", 
        "data": LATEST_DECISIONS, 
        "simulation": _sim.get_portfolio() if _sim else {},
        "logs": LATEST_LOGS,
        "is_thinking": IS_SCANNING
    }

@app.get("/api/simulation/state")
def get_sim_state():
    _sim = get_sim_engine() if sim_engine else None
    return _sim.get_portfolio() if _sim else {"status": "not_loaded"}

@app.post("/api/simulation/reset")
def reset_sim():
    _sim = get_sim_engine()
    _sim.reset()
    return {"status": "reset", "state": _sim.get_portfolio()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
