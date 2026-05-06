"""
Integration tests: cross-service authentication and token flow
==============================================================
Purpose : Verify that gate-api correctly issues and validates JWTs that can
          be consumed by downstream services.  Each service module is loaded
          by FILE PATH (not by module name) to prevent sys.modules["app"]
          collisions and Prometheus metric duplication when multiple service
          test files coexist in the same pytest session.

Covered :
  TestLoginAndValidate
    - POST /login with valid admin/user credentials returns 200 + token
    - The issued token is accepted by POST /validate-token (valid=True, role correct)
    - A tampered token is rejected with 401

  TestTrainAPIWithRealToken
    - The JWT issued by gate-api carries role="admin" and username="admin"
      in its payload (HS256 signature verified with the shared secret)
    - gate-api's /validate-token accepts its own freshly issued token

Implementation note :
  _find_loaded_module(file_path) scans sys.modules by __file__ attribute so
  gate-api is only executed once per session regardless of collection order,
  preventing duplicate Prometheus metric registration errors.

Dependencies : gate-api/app.py (no TF), PyJWT
"""
import sys, os
import importlib.util
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _find_loaded_module(file_path: str):
    """Return a module from sys.modules whose __file__ matches file_path, or None."""
    norm = os.path.normcase(os.path.normpath(file_path))
    for mod in sys.modules.values():
        try:
            if mod and os.path.normcase(os.path.normpath(mod.__file__ or "")) == norm:
                return mod
        except (AttributeError, TypeError):
            pass
    return None


def _load_once(module_name: str, file_path: str, extra_syspath: str = None):
    """Load a module from file_path using module_name, reusing if already loaded."""
    # Reuse if the same file is already in sys.modules under any name
    existing = _find_loaded_module(file_path)
    if existing is not None:
        return existing
    if extra_syspath and extra_syspath not in sys.path:
        sys.path.insert(0, extra_syspath)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load gate-api exactly once — reuses the module if test_gate_api.py already loaded it
_gate_mod = _load_once(
    "gate_api_app",
    os.path.join(PROJECT_ROOT, "gate-api", "app.py"),
    extra_syspath=os.path.join(PROJECT_ROOT, "gate-api"),
)


# ---------------------------------------------------------------------------
# Gate-API client fixture (module scope = one client for all gate tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gate_client():
    return TestClient(_gate_mod.app)


# ---------------------------------------------------------------------------
# Test: full login → token → validate flow (pure gate-api, no mocks needed)
# ---------------------------------------------------------------------------
class TestLoginAndValidate:
    def test_login_returns_valid_token(self, gate_client):
        resp = gate_client.post("/login",
                                json={"username": "admin", "password": "admin_pass"})
        assert resp.status_code == 200
        assert "token" in resp.json()

    def test_token_can_be_validated(self, gate_client):
        token = gate_client.post(
            "/login", json={"username": "admin", "password": "admin_pass"}
        ).json()["token"]
        resp = gate_client.post("/validate-token",
                                headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True
        assert resp.json()["role"] == "admin"

    def test_forged_token_rejected(self, gate_client):
        resp = gate_client.post("/validate-token",
                                headers={"Authorization": "Bearer tampered.token.here"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test: gate-api issues a token that carries the correct claims
# ---------------------------------------------------------------------------
class TestTrainAPIWithRealToken:
    @pytest.fixture(scope="class")
    def admin_token(self):
        c = TestClient(_gate_mod.app)
        return c.post(
            "/login", json={"username": "admin", "password": "admin_pass"}
        ).json()["token"]

    def test_gate_token_has_admin_role(self, admin_token):
        """Verify the JWT produced by gate-api carries role=admin."""
        import jwt as _jwt
        secret = getattr(_gate_mod, "SECRET_KEY", "default_secret")
        payload = _jwt.decode(admin_token, secret, algorithms=["HS256"])
        assert payload["role"] == "admin"
        assert payload["username"] == "admin"

    def test_gate_token_validates_against_gate(self, admin_token):
        """A token issued by gate-api is accepted by gate-api's validate endpoint."""
        c = TestClient(_gate_mod.app)
        resp = c.post("/validate-token",
                      headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        assert resp.json()["valid"] is True
