"""
Pydantic v2 response schemas for Signal.Engine API.
These models validate response shapes and auto-generate OpenAPI docs at /docs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


# ─── Errors ──────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Structured error returned by global exception handler."""
    error: str
    request_id: Optional[str] = None
    detail: Optional[str] = None


# ─── Settings ────────────────────────────────────────────────────────────────

class SettingsPayload(BaseModel):
    """User-configurable settings. Extra fields are rejected to prevent injection."""
    universe: str = "nifty50"
    confidenceThreshold: int = 70
    maxPositions: int = 5

    class Config:
        extra = "forbid"


# ─── Health ──────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(description="Health status string, e.g. 'healthy'")
    version: str = Field(description="API version")
    uptime: float = Field(description="Server uptime in seconds")


# ─── Home ────────────────────────────────────────────────────────────────────

class HomeResponse(BaseModel):
    status: str
    message: str


# ─── Scan ────────────────────────────────────────────────────────────────────

class ScanTriggerResponse(BaseModel):
    status: str = Field(description="'started' or 'busy'")
    message: str


# ─── Decision / Quant ────────────────────────────────────────────────────────

class QuantRisk(BaseModel):
    WinRate: float = 0.0
    EV: float = 0.0
    VaR95: Optional[float] = None
    MaxDrawdown: Optional[float] = None

    class Config:
        extra = "allow"


class DecisionItem(BaseModel):
    Ticker: str
    Action: str
    Confidence: float = Field(ge=0, le=1)
    Rational: List[str] = Field(default_factory=list)
    History: Optional[List[Any]] = None
    QuantRisk: Optional[QuantRisk] = None

    class Config:
        extra = "allow"


# ─── Results ─────────────────────────────────────────────────────────────────

class ResultsResponse(BaseModel):
    status: str
    data: List[Any] = Field(default_factory=list)
    simulation: Optional[Dict[str, Any]] = None
    logs: List[Any] = Field(default_factory=list)
    is_thinking: bool = False


# ─── Simulation ──────────────────────────────────────────────────────────────

class PositionInfo(BaseModel):
    qty: float = 0
    avg_price: float = 0


class SimulationState(BaseModel):
    balance: float = 10000.0
    cash: float = 10000.0
    positions: Dict[str, PositionInfo] = Field(default_factory=dict)
    history: List[Any] = Field(default_factory=list)
    score: int = 0
    level: str = "Novice (Risk Taker)"
    status: str = "ALIVE"

    class Config:
        # Allow extra fields from get_portfolio() without breaking
        extra = "allow"


class ResetResponse(BaseModel):
    status: str
    state: SimulationState
