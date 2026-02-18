"""
Phase 2 Security Tests — covers CORS, API key auth, security headers, and schemas.
brain.think() is mocked in conftest.py so tests finish in seconds.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


client = TestClient(app)


# ─── Health & Home (always public) ──────────────────────────────────────────

class TestPublicEndpoints:
    def test_home_returns_online(self):
        res = client.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "Online"
        assert "message" in data

    def test_health_returns_schema(self):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime" in data
        assert isinstance(data["uptime"], (int, float))


# ─── Security Headers (Step 2.3) ────────────────────────────────────────────

class TestSecurityHeaders:
    def test_x_content_type_options(self):
        res = client.get("/api/health")
        assert res.headers.get("X-Content-Type-Options") == "nosniff"

    def test_x_frame_options(self):
        res = client.get("/api/health")
        assert res.headers.get("X-Frame-Options") == "DENY"

    def test_x_xss_protection(self):
        res = client.get("/api/health")
        assert res.headers.get("X-XSS-Protection") == "1; mode=block"

    def test_referrer_policy(self):
        res = client.get("/api/health")
        assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


# ─── API Key Authentication (Step 2.2) ──────────────────────────────────────

class TestApiKeyAuth:
    def test_dev_mode_no_key_passes(self):
        """When API_KEY env is unset, all protected endpoints pass without auth."""
        res = client.get("/api/results")
        assert res.status_code == 200

    def test_dev_mode_scan_passes(self):
        """Scan endpoint also accessible in dev mode."""
        res = client.get("/api/scan")
        assert res.status_code == 200

    def test_dev_mode_sim_state_passes(self):
        """Simulation state endpoint accessible in dev mode."""
        res = client.get("/api/simulation/state")
        assert res.status_code == 200

    def test_auth_with_key_when_disabled(self):
        """Sending an API key when auth is disabled should still work."""
        res = client.get("/api/results", headers={"X-API-Key": "random_key"})
        assert res.status_code == 200


# ─── Pydantic Schema Validation (Step 2.4) ──────────────────────────────────

class TestSchemaValidation:
    def test_health_response_shape(self):
        res = client.get("/api/health")
        data = res.json()
        assert set(data.keys()) == {"status", "version", "uptime"}

    def test_results_response_shape(self):
        res = client.get("/api/results")
        data = res.json()
        assert "status" in data
        assert "data" in data
        assert "simulation" in data
        assert "is_thinking" in data
        assert isinstance(data["data"], list)

    def test_scan_response_shape(self):
        res = client.get("/api/scan")
        data = res.json()
        assert "status" in data
        assert "message" in data

    def test_home_response_shape(self):
        res = client.get("/")
        data = res.json()
        assert "status" in data
        assert "message" in data

    def test_openapi_has_schemas(self):
        """OpenAPI spec should include our Pydantic schema definitions."""
        res = client.get("/openapi.json")
        assert res.status_code == 200
        schemas = res.json().get("components", {}).get("schemas", {})
        assert "HealthResponse" in schemas
        assert "ScanTriggerResponse" in schemas
        assert "ResultsResponse" in schemas


# ─── Request ID Tracing (Step 4.2) ──────────────────────────────────────────

class TestRequestId:
    def test_request_id_auto_generated(self):
        """Responses should have X-Request-ID even without client sending one."""
        res = client.get("/api/health")
        assert "x-request-id" in res.headers
        # Should be a valid UUID-like string
        assert len(res.headers["x-request-id"]) >= 32

    def test_request_id_passthrough(self):
        """Client-sent X-Request-ID should be echoed back."""
        custom_id = "test-req-12345"
        res = client.get("/api/health", headers={"X-Request-ID": custom_id})
        assert res.headers.get("x-request-id") == custom_id

    def test_request_id_on_protected_endpoint(self):
        """Protected endpoints also get request IDs."""
        res = client.get("/api/results")
        assert "x-request-id" in res.headers
