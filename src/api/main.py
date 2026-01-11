from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from src.brain.hybrid import HybridBrain
from src.simulation.engine import SimulationEngine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title="Sniper Trading Agent API", version="1.0")

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Brain & Sim Engine
brain = HybridBrain()
sim_engine = SimulationEngine()

@app.get("/")
def home():
    return {"status": "Online", "message": "Sniper Agent is Ready."}

# Global State for Async Scanning
LATEST_DECISIONS = []
LATEST_LOGS = []
IS_SCANNING = False

def background_scan():
    global LATEST_DECISIONS, LATEST_LOGS, IS_SCANNING
    try:
        logger.info("🧠 Brain started thinking...")
        decisions = brain.think()
        
        # Determine Regime (Simple Heuristic: majority of income votes)
        # In a real system, the Brain would output a global 'regime' field
        regime = "NEUTRAL"
        vol_count = sum(1 for d in decisions if "High Volatility" in str(d.get('Rational', [])))
        if vol_count > len(decisions) * 0.3:
            regime = "HIGH_VOLATILITY"
        
        # RL Simulation Tick (Auto-Run)
        logs = sim_engine.process_tick(decisions, regime=regime)
        
        # Sort by Confidence
        decisions.sort(key=lambda x: x['Confidence'], reverse=True)
        
        # Update Global State
        LATEST_DECISIONS = decisions
        LATEST_LOGS = logs
        logger.info("✅ Brain finished thinking.")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
    finally:
        IS_SCANNING = False

@app.get("/api/scan")
def run_scan(background_tasks: BackgroundTasks):
    """
    Triggers the Hybrid Brain to think in the background.
    """
    global IS_SCANNING
    if IS_SCANNING:
        return {"status": "busy", "message": "Brain is already thinking."}
    
    IS_SCANNING = True
    background_tasks.add_task(background_scan)
    return {"status": "started", "message": "Scan triggered in background."}

@app.get("/api/results")
def get_results():
    """
    Returns the latest available scan results.
    """
    return {
        "status": "success", 
        "data": LATEST_DECISIONS, 
        "simulation": sim_engine.get_portfolio(),
        "logs": LATEST_LOGS,
        "is_thinking": IS_SCANNING
    }

@app.get("/api/simulation/state")
def get_sim_state():
    return sim_engine.get_portfolio()

@app.post("/api/simulation/reset")
def reset_sim():
    sim_engine.reset()
    return {"status": "reset", "state": sim_engine.get_portfolio()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
