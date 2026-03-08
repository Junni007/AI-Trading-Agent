import os
import pytest
from fastapi.testclient import TestClient

# Must be set before importing app since dependencies evaluate API_KEY on import
os.environ["API_KEY"] = "test_super_secret_key"

from src.api.main import app, sim_engine, scan_state

# ─── Fixtures (AAA Pattern Setup) ───────────────────────────────────────────

@pytest.fixture
def client():
    """Provides a fresh FastAPI test client."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def auth_headers():
    """Provides valid authentication headers."""
    return {"X-API-Key": "test_super_secret_key"}

@pytest.fixture(autouse=True)
def reset_state():
    """Resets global state before each test to ensure Isolation (testing-patterns)."""
    sim_engine.reset()
    scan_state.set_scanning(False)
    yield

# ─── Integration Tests ───────────────────────────────────────────────────────

def test_home_endpoint(client):
    """Test public home route."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "Online", "message": "Sniper Agent is Ready."}

def test_health_check(client):
    """Test health check route."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "uptime" in data

# ─── Security / Auth Tests ───────────────────────────────────────────────────

def test_missing_api_key_returns_403(client):
    """Test endpoints protected by verify_api_key reject missing keys."""
    response = client.get("/api/results")
    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid or missing API key"}

def test_invalid_api_key_returns_403(client):
    """Test endpoints protected by verify_api_key reject invalid keys."""
    response = client.get("/api/results", headers={"X-API-Key": "wrong_key"})
    assert response.status_code == 403
    assert response.json() == {"detail": "Invalid or missing API key"}

# ─── Logic Tests ─────────────────────────────────────────────────────────────

def test_get_results_with_auth(client, auth_headers):
    """Test fetching scan results with auth."""
    response = client.get("/api/results", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data
    assert "simulation" in data
    assert "logs" in data

def test_simulation_state(client, auth_headers):
    """Test reading the current portfolio simulation state."""
    response = client.get("/api/simulation/state", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "cash" in data
    assert "positions" in data

def test_simulation_reset(client, auth_headers):
    """Test triggering a simulation reset via API."""
    response = client.post("/api/simulation/reset", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "reset"
    assert "state" in data

def test_get_settings(client, auth_headers):
    """Test fetching settings payload."""
    response = client.get("/api/settings", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    # verify model dump response structure
    assert "maxPositions" in data

def test_save_settings(client, auth_headers):
    """Test saving settings via API payload validation."""
    valid_payload = {
        "universe": "nifty50",
        "confidenceThreshold": 75,
        "maxPositions": 5
    }
    response = client.post("/api/settings", headers=auth_headers, json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "saved"
    assert data["settings"]["maxPositions"] == 5

def test_run_scan_busy_state(client, auth_headers):
    """Test the scan orchestration busy rejection logic."""
    scan_state.set_scanning(True)
    response = client.get("/api/scan", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == {"status": "busy", "message": "Brain is already thinking."}

