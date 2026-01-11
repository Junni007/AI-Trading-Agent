import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "Online"

def test_scan_trigger():
    # Test triggering the scan
    response = client.get("/api/scan")
    assert response.status_code == 200
    assert response.json()["status"] == "started"

def test_results_endpoint():
    response = client.get("/api/results")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "simulation" in data
    assert "is_thinking" in data
