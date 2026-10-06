"""
Tests for production security middleware: API key auth and rate limiting.

These import a fresh FastAPI app instance per test (rather than reusing
tests/test_api.py's client) because auth/rate-limit behavior depends on
settings that must be set BEFORE the app module is imported.
"""
import importlib
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.api.middleware import RateLimiter


def _fresh_app(env_overrides: dict):
    """Re-imports app.config and app.api.main with the given env vars set,
    so each test gets an app built under its own settings."""
    old_environ = dict(os.environ)
    os.environ.update(env_overrides)

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(db_path)
    os.environ["HUMANLENS_DB_PATH"] = db_path

    for mod in ["app.config", "app.api.middleware", "app.api.main", "app.database.db"]:
        if mod in sys.modules:
            del sys.modules[mod]

    main = importlib.import_module("app.api.main")

    def cleanup():
        os.environ.clear()
        os.environ.update(old_environ)
        if os.path.exists(db_path):
            os.remove(db_path)

    return main.app, cleanup


def test_no_api_key_required_by_default():
    app, cleanup = _fresh_app({"HUMANLENS_API_KEY": ""})
    try:
        client = TestClient(app)
        resp = client.post("/analyze", json={"text": "Some plain text to analyze without a key set."})
        assert resp.status_code == 200
    finally:
        cleanup()


def test_api_key_required_when_configured():
    app, cleanup = _fresh_app({"HUMANLENS_API_KEY": "secret123"})
    try:
        client = TestClient(app)
        resp = client.post("/analyze", json={"text": "Some plain text to analyze."})
        assert resp.status_code == 401

        resp2 = client.post(
            "/analyze",
            json={"text": "Some plain text to analyze."},
            headers={"X-API-Key": "secret123"},
        )
        assert resp2.status_code == 200

        resp3 = client.post(
            "/analyze",
            json={"text": "Some plain text to analyze."},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp3.status_code == 401
    finally:
        cleanup()


def test_health_exempt_from_api_key():
    app, cleanup = _fresh_app({"HUMANLENS_API_KEY": "secret123"})
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
    finally:
        cleanup()


def test_docs_disabled_in_production():
    app, cleanup = _fresh_app({"HUMANLENS_ENV": "production", "HUMANLENS_API_KEY": ""})
    try:
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 404
    finally:
        cleanup()


def test_docs_enabled_in_development():
    app, cleanup = _fresh_app({"HUMANLENS_ENV": "development", "HUMANLENS_API_KEY": ""})
    try:
        client = TestClient(app)
        resp = client.get("/docs")
        assert resp.status_code == 200
    finally:
        cleanup()


def test_response_has_request_id_header():
    app, cleanup = _fresh_app({"HUMANLENS_API_KEY": ""})
    try:
        client = TestClient(app)
        resp = client.get("/health")
        assert "X-Request-ID" in resp.headers
    finally:
        cleanup()


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(limit_per_minute=3)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False  # 4th request in the window is blocked


def test_rate_limiter_tracks_clients_independently():
    limiter = RateLimiter(limit_per_minute=1)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True  # different client, own budget
    assert limiter.allow("client-a") is False


def test_rate_limiter_disabled_when_limit_zero():
    limiter = RateLimiter(limit_per_minute=0)
    for _ in range(50):
        assert limiter.allow("client-a") is True


def test_rate_limit_enforced_end_to_end():
    app, cleanup = _fresh_app({"HUMANLENS_API_KEY": "", "HUMANLENS_RATE_LIMIT_PER_MINUTE": "2"})
    try:
        client = TestClient(app)
        r1 = client.get("/stats")
        r2 = client.get("/stats")
        r3 = client.get("/stats")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 429
    finally:
        cleanup()
