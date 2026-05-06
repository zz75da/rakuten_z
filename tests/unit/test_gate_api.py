"""
Unit tests for gate-api endpoints
===================================
Purpose : Validate authentication and token-validation logic in gate-api
          without any external dependency (no MLflow, no Airflow, no DB).
          Uses FastAPI's TestClient exclusively.

Covered :
  TestHealthAndRoot
    - GET /health returns 200 with status="healthy" and service="gate-api"
    - GET /  returns a valid JSON body (endpoints list or status field)

  TestLogin
    - Admin and user logins succeed with correct credentials (200 + token)
    - Wrong password and unknown username both return 401
    - Empty credentials are rejected (400 or 401)
    - Returned token is a valid HS256 JWT with correct username/role claims
    - Token payload contains both "exp" and "iat" fields

  TestValidateToken
    - Valid admin and user tokens are accepted by POST /validate-token
    - Garbage strings and non-Bearer schemes are rejected (401 / 422)
    - Missing Authorization header returns 422

Dependencies : gate-api/app.py, PyJWT (no TF, no MLflow)
"""
import sys, os
import pytest
import jwt
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "gate-api"))
import importlib
_app_mod = importlib.import_module("app")
from app import app
SECRET = getattr(_app_mod, "SECRET_KEY", "default_secret")
ALGO   = "HS256"

@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------
class TestHealthAndRoot:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "gate-api"

    def test_root_returns_endpoints(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "endpoints" in body or "status" in body


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
class TestLogin:
    def test_admin_login_success(self, client):
        resp = client.post("/login", json={"username": "admin", "password": "admin_pass"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["role"] == "admin"
        assert data["token_type"] == "bearer"

    def test_user_login_success(self, client):
        resp = client.post("/login", json={"username": "user", "password": "user_pass"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "user"

    def test_wrong_password_is_401(self, client):
        resp = client.post("/login", json={"username": "admin", "password": "bad"})
        assert resp.status_code == 401

    def test_unknown_user_is_401(self, client):
        resp = client.post("/login", json={"username": "ghost", "password": "x"})
        assert resp.status_code == 401

    def test_empty_credentials_rejected(self, client):
        resp = client.post("/login", json={"username": "", "password": ""})
        assert resp.status_code in (400, 401)

    def test_returned_token_is_valid_jwt(self, client):
        resp = client.post("/login", json={"username": "admin", "password": "admin_pass"})
        token = resp.json()["token"]
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        assert payload["username"] == "admin"
        assert payload["role"] == "admin"

    def test_token_contains_expiry(self, client):
        resp = client.post("/login", json={"username": "admin", "password": "admin_pass"})
        token = resp.json()["token"]
        payload = jwt.decode(token, SECRET, algorithms=[ALGO])
        assert "exp" in payload
        assert "iat" in payload


# ---------------------------------------------------------------------------
# Token validation
# ---------------------------------------------------------------------------
class TestValidateToken:
    def _get_token(self, client, role="admin"):
        pwd = "admin_pass" if role == "admin" else "user_pass"
        resp = client.post("/login", json={"username": role, "password": pwd})
        return resp.json()["token"]

    def test_valid_admin_token_accepted(self, client):
        token = self._get_token(client, "admin")
        resp = client.post("/validate-token",
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "admin"
        assert data["role"] == "admin"
        assert data["valid"] is True

    def test_valid_user_token_accepted(self, client):
        token = self._get_token(client, "user")
        resp = client.post("/validate-token",
                           headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["role"] == "user"

    def test_garbage_token_is_401(self, client):
        resp = client.post("/validate-token",
                           headers={"Authorization": "Bearer not.a.token"})
        assert resp.status_code == 401

    def test_wrong_scheme_is_4xx(self, client):
        resp = client.post("/validate-token",
                           headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code in (401, 422)

    def test_missing_header_is_422(self, client):
        resp = client.post("/validate-token")
        assert resp.status_code == 422
