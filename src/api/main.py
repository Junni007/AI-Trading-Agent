import os
import json
import logging
import asyncio
import time
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Request, Security
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from src.brain.hybrid import HybridBrain
from src.simulation.engine import SimulationEngine
from src.api.websocket import router as ws_router, broadcast_update
from src.api.schemas import (
    HealthResponse, HomeResponse, ScanTriggerResponse,
    ResultsResponse, ResetResponse, SettingsPayload,
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")
START_TIME = time.time()

# Simple Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
            
        client_ip = request.client.host if request.client else "unknown"
        # ... (rest of logic)

# ...

app = FastAPI(title="Signal.Engine API", version="2.0")

# Trusted Host Middleware (Fix for 400 Bad Request on Host Header)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Middleware stack (order: request ID → security headers → rate limit → CORS)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, calls_per_minute=120)


# Global exception handler (Step 4.2)
# ...

# CORS — restricted to configured origins (Step 2.1)
# Robust parsing: handle commas, spaces, and accidental quotes
headers_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,http://localhost:8000")
CORS_ORIGINS = [origin.strip().strip('"').strip("'") for origin in headers_env.split(",") if origin.strip()]

logger.info(f"🔧 Raw CORS_ORIGINS env: {headers_env}")
logger.info(f"✅ Parsed CORS_ORIGINS: {CORS_ORIGINS}")

# Trusted Host Middleware (Fix for 400 Bad Request on Host Header)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

logger.info(f"🔧 Raw CORS_ORIGINS env: {headers_env}")
logger.info(f"✅ Parsed CORS_ORIGINS: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication (Step 2.2)
# When API_KEY env is not set, auth is disabled (dev mode)
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Security(api_key_header)):
    """Validate API key. Disabled when API_KEY env var is not set."""
    if not API_KEY:
        return  # Dev mode — no auth required
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# Mount WebSocket router
app.include_router(ws_router)

# Initialize Brain & Sim Engine
brain = HybridBrain()
sim_engine = SimulationEngine()

@app.get("/", response_model=HomeResponse)
def home():
    return {"status": "Online", "message": "Sniper Agent is Ready."}

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for monitoring and Settings page connectivity."""
    return {
        "status": "healthy",
        "version": app.version,
        "uptime": round(time.time() - START_TIME, 1),
    }


# ─── Thread-Safe State Store (Step 3.1) ──────────────────────────────────────

@dataclass
class ScanState:
    """Thread-safe container for scan results. All access goes through lock."""
    decisions: List = field(default_factory=list)
    logs: List = field(default_factory=list)
    is_scanning: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, decisions: list, logs: list):
        with self._lock:
            self.decisions = decisions
            self.logs = logs

    def read(self):
        with self._lock:
            return self.decisions.copy(), self.logs.copy(), self.is_scanning

    def set_scanning(self, value: bool):
        with self._lock:
            self.is_scanning = value


scan_state = ScanState()


# ─── Async Background Scan (Step 3.2) ────────────────────────────────────────

async def background_scan():
    """Run brain.think() in a thread pool and broadcast results via WS."""
    try:
        logger.info("🧠 Brain started thinking...")
        # Off-load CPU-heavy work to thread pool
        decisions = await asyncio.to_thread(brain.think)

        # Determine Regime (Simple Heuristic)
        regime = "NEUTRAL"
        vol_count = sum(1 for d in decisions if "High Volatility" in str(d.get('Rational', [])))
        if vol_count > len(decisions) * 0.3:
            regime = "HIGH_VOLATILITY"

        # RL Simulation Tick
        logs = sim_engine.process_tick(decisions, regime=regime)

        # Sort by Confidence
        decisions.sort(key=lambda x: x['Confidence'], reverse=True)

        # Thread-safe update
        scan_state.update(decisions, logs)
        logger.info("✅ Brain finished thinking.")

        # Broadcast to WebSocket clients (direct await — no more run_until_complete)
        try:
            await broadcast_update({
                "type": "scan_complete",
                "data": decisions,
                "simulation": sim_engine.get_portfolio(),
                "logs": logs
            })
        except Exception as ws_err:
            logger.warning(f"WebSocket broadcast failed: {ws_err}")
    except Exception as e:
        logger.error(f"Scan failed: {e}")
    finally:
        scan_state.set_scanning(False)


@app.get("/api/scan", dependencies=[Depends(verify_api_key)], response_model=ScanTriggerResponse)
async def run_scan():
    """Triggers the Hybrid Brain to think in the background."""
    if scan_state.is_scanning:
        return {"status": "busy", "message": "Brain is already thinking."}

    scan_state.set_scanning(True)
    asyncio.create_task(background_scan())
    return {"status": "started", "message": "Scan triggered in background."}

@app.get("/api/results", dependencies=[Depends(verify_api_key)], response_model=ResultsResponse)
def get_results():
    """Returns the latest available scan results."""
    decisions, logs, is_thinking = scan_state.read()
    return {
        "status": "success",
        "data": decisions,
        "simulation": sim_engine.get_portfolio(),
        "logs": logs,
        "is_thinking": is_thinking
    }

@app.get("/api/simulation/state", dependencies=[Depends(verify_api_key)])
def get_sim_state():
    """Returns the current simulation portfolio state."""
    return sim_engine.get_portfolio()

@app.post("/api/simulation/reset", dependencies=[Depends(verify_api_key)], response_model=ResetResponse)
def reset_sim():
    """Resets the simulation to initial state."""
    sim_engine.reset()
    return {"status": "reset", "state": sim_engine.get_portfolio()}

# ─── Settings Sync (Step 3.3) ────────────────────────────────────────────────

SETTINGS_FILE = Path(__file__).parent.parent.parent / "settings.json"
DEFAULT_SETTINGS = SettingsPayload()


def _load_settings() -> dict:
    """Load settings from JSON file, falling back to defaults."""
    try:
        if SETTINGS_FILE.exists():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load settings: {e}")
    return DEFAULT_SETTINGS.model_dump()


def _save_settings(data: dict) -> None:
    """Persist settings to JSON file."""
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


@app.get("/api/settings", dependencies=[Depends(verify_api_key)])
def get_settings():
    """Returns stored settings."""
    return _load_settings()


@app.post("/api/settings", dependencies=[Depends(verify_api_key)])
def save_settings(payload: SettingsPayload):
    """Save user settings to disk."""
    data = payload.model_dump()
    _save_settings(data)
    return {"status": "saved", "settings": data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
