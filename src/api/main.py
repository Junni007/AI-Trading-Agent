import os
import json
import hmac
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
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from contextlib import asynccontextmanager

from src.brain.hybrid import HybridBrain
from src.simulation.engine import SimulationEngine
from src.api.websocket import router as ws_router, broadcast_update
from src.scheduler import start_scheduler, stop_scheduler
from src.api.schemas import (
    HealthResponse, HomeResponse, ScanTriggerResponse,
    ResultsResponse, ResetResponse, SettingsPayload, ErrorResponse,
)

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")
START_TIME = time.time()


# ─── Middleware: Request ID ──────────────────────────────────────────────────

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Adds a unique X-Request-ID header to every request/response."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ─── Middleware: Security Headers ────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Adds standard security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# ─── Middleware: Rate Limiting (Sliding Window) ──────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket style rate limiter per client IP."""

    def __init__(self, app, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0  # 1 minute sliding window

        # Clean old entries outside the window
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if now - t < window
        ]

        # Check if over limit
        if len(self.requests[client_ip]) >= self.calls_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Max {self.calls_per_minute} requests per minute. Try again later.",
                },
            )

        # Record this request
        self.requests[client_ip].append(now)
        return await call_next(request)


# ─── App Init ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Signal.Engine...")
    
    # Capture the main FastAPI event loop so the scheduler thread can push tasks to it
    loop = asyncio.get_running_loop()
    
    # We wrap background_scan with asyncio.create_task inside a def so APScheduler can call it safely
    def run_scan_job():
        if not scan_state.is_scanning:
            scan_state.set_scanning(True)
            # Schedule the coroutine safely from the scheduler's background thread
            asyncio.run_coroutine_threadsafe(background_scan(), loop)
            
    start_scheduler(run_scan_job)
    
    yield  # Normal running application state
    
    # Teardown actions
    logger.info("Shutting down Signal.Engine...")
    stop_scheduler()

app = FastAPI(title="Signal.Engine API", version="2.0", lifespan=lifespan)

# ─── CORS Configuration ─────────────────────────────────────────────────────
# IMPORTANT: CORS must be the FIRST middleware added so it is the outermost
# ASGI layer. In Starlette, add_middleware() uses LIFO ordering — the first
# middleware added wraps everything else, meaning it sees the request first.
# If CORS is added after rate-limiting, OPTIONS preflights never get answered.

# Robust parsing: handle commas, spaces, and accidental quotes
_default_origins = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000"
headers_env = os.getenv("CORS_ORIGINS", _default_origins)

# Also accept FRONTEND_URL as a convenience env var (common in PaaS deployments)
frontend_url = os.getenv("FRONTEND_URL", "")
if frontend_url:
    headers_env = f"{headers_env},{frontend_url}"

CORS_ORIGINS = [
    origin.strip().strip('"').strip("'").rstrip("/")
    for origin in headers_env.split(",")
    if origin.strip()
]

logger.info(f"Parsed CORS_ORIGINS: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)

# Remaining middleware (applied after CORS in the LIFO stack)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(RequestIdMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, calls_per_minute=120)


# ─── Global Exception Handler ────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions and return a safe error response."""
    request_id = getattr(request.state, "request_id", None)
    logger.error(f"Unhandled exception [request_id={request_id}]: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": None,  # Never leak internal details
        },
    )


# ─── API Key Authentication (Constant-Time Comparison) ───────────────────────

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Security: Fail closed - if API_KEY is configured but env var is missing, deny all requests
if not API_KEY:
    logger.error(
        "FATAL: API_KEY environment variable is not set. "
        "Refusing to start in insecure mode. Set API_KEY to enable the API."
    )
    raise RuntimeError("API_KEY environment variable is required but not set. Exiting.")


async def verify_api_key(key: str = Security(api_key_header)):
    """Validate API key using constant-time comparison. Always required in production."""
    if not key or not hmac.compare_digest(key.encode("utf-8"), API_KEY.encode("utf-8")):
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


# ─── Thread-Safe State Store ─────────────────────────────────────────────────

SCAN_RESULTS_FILE = Path(__file__).parent.parent.parent / "scan_results.json"

@dataclass
class ScanState:
    """Thread-safe container for scan results. Persists to disk for server restarts."""
    decisions: List = field(default_factory=list)
    logs: List = field(default_factory=list)
    is_scanning: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        self._load_from_disk()

    def _load_from_disk(self):
        """Load last scan results from disk on startup."""
        if SCAN_RESULTS_FILE.exists():
            try:
                data = json.loads(SCAN_RESULTS_FILE.read_text(encoding="utf-8"))
                self.decisions = data.get("decisions", [])
                self.logs = data.get("logs", [])
                logger.info(f"Loaded {len(self.decisions)} previous scan results from disk.")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load scan results: {e}")

    def _save_to_disk(self):
        """Persist scan results to disk."""
        try:
            SCAN_RESULTS_FILE.write_text(
                json.dumps({"decisions": self.decisions, "logs": self.logs}, indent=2),
                encoding="utf-8"
            )
        except OSError as e:
            logger.warning(f"Failed to save scan results: {e}")

    def update(self, decisions: list, logs: list):
        with self._lock:
            self.decisions = decisions
            self.logs = logs
            self._save_to_disk()

    def read(self):
        with self._lock:
            return self.decisions.copy(), self.logs.copy(), self.is_scanning

    def set_scanning(self, value: bool):
        with self._lock:
            self.is_scanning = value


scan_state = ScanState()


# ─── Async Background Scan ───────────────────────────────────────────────────

async def background_scan():
    """Run brain.think() in a thread pool and broadcast results via WS."""
    try:
        logger.info("Brain started thinking...")
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
        logger.info("Brain finished thinking.")

        # Broadcast to WebSocket clients
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
    """Manual trigger to force the Hybrid Brain to execute a scan out-of-band."""
    if scan_state.is_scanning:
        return {"status": "busy", "message": "Brain is already thinking."}

    scan_state.set_scanning(True)
    asyncio.create_task(background_scan())
    return {"status": "started", "message": "Manual scan cycle triggered in background."}

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

# ─── Settings Sync ───────────────────────────────────────────────────────────

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
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
