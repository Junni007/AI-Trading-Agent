"""
WebSocket endpoint for real-time updates.
Broadcasts scan results to all connected clients.
"""
import asyncio
import logging
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("WebSocket")

router = APIRouter()

# Store active connections
active_connections: Set[WebSocket] = set()


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")
    
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


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal updates.
    Clients receive updates when new scan results are available.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, handle incoming messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
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
