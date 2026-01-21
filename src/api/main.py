from fastapi import FastAPI, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import logging
import asyncio
import time
from collections import defaultdict
from src.brain.hybrid import HybridBrain
from src.simulation.engine import SimulationEngine
from src.api.websocket import router as ws_router, broadcast_update

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

# Simple Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean old requests
        self.requests[client_ip] = [t for t in self.requests[client_ip] if now - t < 60]
        
        if len(self.requests[client_ip]) >= self.calls_per_minute:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Try again later."}
            )
        
        self.requests[client_ip].append(now)
        return await call_next(request)

app = FastAPI(title="Signal.Engine API", version="2.0")

# Add Rate Limiting
app.add_middleware(RateLimitMiddleware, calls_per_minute=120)

# Allow CORS for Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount WebSocket router
app.include_router(ws_router)

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
        
        # Broadcast to WebSocket clients
        try:
            asyncio.get_event_loop().run_until_complete(
                broadcast_update({
                    "type": "scan_complete",
                    "data": decisions,
                    "simulation": sim_engine.get_portfolio(),
                    "logs": logs
                })
            )
        except Exception as ws_err:
            logger.warning(f"WebSocket broadcast failed: {ws_err}")
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
