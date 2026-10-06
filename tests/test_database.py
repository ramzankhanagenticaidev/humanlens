import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from app.database.db import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # let Database create it fresh
    database = Database(path)
    yield database
    if os.path.exists(path):
        os.remove(path)


def test_save_and_get_analysis(db):
    aid = db.save_analysis(
        word_count=100,
        ai_probability=0.65,
        confidence="Moderate",
        verdict="Possibly AI-influenced",
        model_used="ml_model",
        full_text="some sample text",
        result_dict={"foo": "bar"},
    )
    record = db.get_analysis(aid)
    assert record is not None
    assert record.word_count == 100
    assert record.full_text == "some sample text"


def test_get_nonexistent_analysis_returns_none(db):
    assert db.get_analysis(9999) is None


def test_list_history_ordering(db):
    id1 = db.save_analysis(10, 0.1, "Low", "Likely human-written", "ml_model", "first", {})
    id2 = db.save_analysis(10, 0.9, "High", "Likely AI-influenced", "ml_model", "second", {})
    history = db.list_history()
    assert history[0].id == id2  # most recent first
    assert history[1].id == id1


def test_delete_analysis(db):
    aid = db.save_analysis(10, 0.5, "Low", "Unclear / mixed signals", "ml_model", "text", {})
    assert db.delete_analysis(aid) is True
    assert db.get_analysis(aid) is None
    assert db.delete_analysis(aid) is False  # already deleted


def test_attach_humanize_result(db):
    aid = db.save_analysis(10, 0.5, "Low", "Unclear / mixed signals", "ml_model", "text", {})
    db.attach_humanize_result(aid, {"revised_text": "improved"})
    record = db.get_analysis(aid)
    assert "improved" in record.humanize_result_json


def test_stats_empty_db(db):
    stats = db.stats()
    assert stats["total_analyses"] == 0
    assert stats["avg_ai_probability"] is None


def test_stats_with_data(db):
    db.save_analysis(10, 0.4, "Low", "Possibly human-written", "ml_model", "a", {})
    db.save_analysis(20, 0.8, "High", "Likely AI-influenced", "ml_model", "b", {})
    stats = db.stats()
    assert stats["total_analyses"] == 2
    assert stats["avg_ai_probability"] == pytest.approx(0.6, abs=0.01)


def test_sql_injection_safe_text(db):
    malicious = "'; DROP TABLE analyses; --"
    aid = db.save_analysis(5, 0.5, "Low", "Unclear / mixed signals", "ml_model", malicious, {})
    record = db.get_analysis(aid)
    assert record.full_text == malicious
    # table should still exist and be queryable
    assert db.list_history() is not None
