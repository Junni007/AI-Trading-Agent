"""
WebSocket endpoint for real-time updates.
Broadcasts scan results to all connected clients.
Supports optional API key authentication via query parameter.
"""
import os
import hmac
import asyncio
import logging
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

logger = logging.getLogger("WebSocket")

router = APIRouter()

# Max concurrent WebSocket connections to prevent resource exhaustion
MAX_CONNECTIONS = 50


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self, max_connections: int = MAX_CONNECTIONS):
        self.active_connections: Set[WebSocket] = set()
        self.max_connections = max_connections

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013, reason="Max connections reached")
            logger.warning(f"Rejected connection: limit of {self.max_connections} reached")
            return False
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
        return True

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)


manager = ConnectionManager()


def _verify_ws_token(token: str | None) -> bool:
    """Check WebSocket auth token against API_KEY using constant-time comparison."""
    api_key = os.getenv("API_KEY")
    if not api_key:
        return True  # Dev mode — no auth required
    if not token:
        return False
    return hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8"))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, token: str | None = Query(default=None)):
    """
    WebSocket endpoint for real-time signal updates.
    Authenticate by passing ?token=<API_KEY> in the connection URL.
    When API_KEY env var is not set, auth is disabled (dev mode).
    """
    # Validate authentication
    if not _verify_ws_token(token):
        await websocket.close(code=1008, reason="Unauthorized")
        logger.warning("WebSocket connection rejected: invalid or missing token")
        return

    # Validate origin (CORS does not apply to WebSocket upgrades)
    allowed_origins_env = os.getenv("CORS_ORIGINS", "")
    if allowed_origins_env:
        origin = websocket.headers.get("origin", "")
        allowed = [o.strip().strip('"').strip("'") for o in allowed_origins_env.split(",") if o.strip()]
        if origin and allowed and origin not in allowed:
            await websocket.close(code=1008, reason="Origin not allowed")
            logger.warning(f"WebSocket rejected: origin '{origin}' not in allowed list")
            return

    connected = await manager.connect(websocket)
    if not connected:
        return

    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()

            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
            # Ignore all other messages (no arbitrary input processing)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_update(data: dict):
    """Broadcast update to all connected clients."""
    await manager.broadcast(data)


def get_connection_count() -> int:
    """Get number of active connections."""
    return len(manager.active_connections)
