import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Point the app at a throwaway database before importing it, so tests
# never touch the real humanlens.db used by the running app.
_tmp_db_fd, _tmp_db_path = tempfile.mkstemp(suffix=".db")
os.close(_tmp_db_fd)
os.remove(_tmp_db_path)
os.environ["HUMANLENS_DB_PATH"] = _tmp_db_path

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_normal_text():
    resp = client.post("/analyze", json={"text": "This is a perfectly normal sentence used for testing purposes, nothing more."})
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["ai_probability"] <= 1.0
    assert "analysis_id" in data


def test_analyze_empty_text_does_not_500():
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 200


def test_analyze_too_long_text_returns_422():
    huge_text = "word " * 60000  # exceeds default max_chars_for_analysis
    resp = client.post("/analyze", json={"text": huge_text})
    assert resp.status_code == 422


def test_analyze_paragraphs():
    resp = client.post(
        "/analyze/paragraphs",
        json={"text": "First paragraph here with some content.\n\nSecond paragraph with different content."},
    )
    assert resp.status_code == 200
    assert len(resp.json()["paragraphs"]) == 2


def test_humanize_basic():
    resp = client.post("/humanize", json={"text": "It is important to note that we utilize AI.", "use_ollama": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "revised_text" in data
    assert data["engine_used"] == "rule_based"


def test_ollama_status_endpoint():
    resp = client.get("/ollama/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "reachable" in data
    assert "models_pulled" in data
    # No Ollama server running in the test environment
    assert data["reachable"] is False
    assert data["models_pulled"] == []


def test_humanize_with_ollama_model_override_falls_back_cleanly():
    # No real Ollama server in the test env, so this must fall back and
    # report a specific reason, not silently pretend it used Ollama.
    resp = client.post(
        "/humanize",
        json={"text": "Utilize a robust approach.", "use_ollama": True, "ollama_model": "qwen2.5-coder:3b"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["engine_used"] == "rule_based"
    assert any("qwen2.5-coder:3b" in w or "reach" in w.lower() for w in data["warnings"])


def test_history_roundtrip():
    analyze_resp = client.post("/analyze", json={"text": "Some text to store in history for the roundtrip test."})
    aid = analyze_resp.json()["analysis_id"]

    history_resp = client.get("/history")
    assert history_resp.status_code == 200
    ids = [item["id"] for item in history_resp.json()["items"]]
    assert aid in ids

    detail_resp = client.get(f"/history/{aid}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["id"] == aid

    delete_resp = client.delete(f"/history/{aid}")
    assert delete_resp.status_code == 200

    missing_resp = client.get(f"/history/{aid}")
    assert missing_resp.status_code == 404


def test_history_detail_not_found():
    resp = client.get("/history/999999")
    assert resp.status_code == 404


def test_delete_nonexistent_returns_404():
    resp = client.delete("/history/999999")
    assert resp.status_code == 404


def test_stats_endpoint():
    resp = client.get("/stats")
    assert resp.status_code == 200
    assert "total_analyses" in resp.json()


def test_invalid_json_body_returns_422():
    resp = client.post("/analyze", json={"not_text": "oops"})
    assert resp.status_code == 422


def test_history_invalid_limit_returns_400():
    resp = client.get("/history?limit=9999")
    assert resp.status_code == 400
