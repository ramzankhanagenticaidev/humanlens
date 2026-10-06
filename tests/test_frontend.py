"""
Frontend UI tests using Streamlit's AppTest framework.

These simulate real user interactions (typing, clicking) against the
actual Streamlit script and assert no exception occurs and navigation
behaves correctly. This specifically guards against a real bug found
during development: programmatically switching pages from the home
page's pill buttons by directly setting `st.session_state["nav_page"]`
raises a StreamlitAPIException, because the sidebar nav radio (which
owns that key) has already been instantiated earlier in the same
script run — Streamlit only allows mutating an already-instantiated
widget's state from within an on_click callback. If a future edit
reintroduces a direct-call pattern instead of a callback, these tests
will catch it.

Requires a running API at HUMANLENS_API_HOST:HUMANLENS_API_PORT for the
full-flow tests (they're skipped automatically if it's unreachable).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.config import settings

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "frontend" / "streamlit_app.py")


def _api_reachable() -> bool:
    try:
        resp = requests.get(f"http://{settings.api_host}:{settings.api_port}/health", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


def _ss_get(at, key, default=None):
    try:
        return at.session_state[key]
    except Exception:
        return default


def test_home_page_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    assert list(at.exception) == []
    assert _ss_get(at, "nav_page") == "🏠 Home"


def test_home_pill_button_navigates_without_exception():
    """Regression test for the StreamlitAPIException bug: clicking a
    home-page pill button must not crash, and must actually switch
    pages via the nav_page session_state key the sidebar radio owns."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.text_area(key="hero_input").set_value("Some test text for the hero box.")
    at.run(timeout=30)

    detector_button = [b for b in at.button if "AI Detector" in b.label][0]
    detector_button.click()
    at.run(timeout=30)

    assert list(at.exception) == []
    assert _ss_get(at, "nav_page") == "🔍 Analyze"
    assert _ss_get(at, "analyze_input") == "Some test text for the hero box."
    assert any(t.value == "Analyze Writing" for t in at.title)


def test_home_humanizer_pill_carries_text_to_writing_improvement():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.text_area(key="hero_input").set_value("Utilize this text for the humanizer pill test.")
    at.run(timeout=30)

    humanizer_button = [b for b in at.button if "Humanizer" in b.label][0]
    humanizer_button.click()
    at.run(timeout=30)

    assert list(at.exception) == []
    assert _ss_get(at, "nav_page") == "✨ Writing Improvement"
    assert any(t.value == "Writing Improvement" for t in at.title)
    assert _ss_get(at, "humanize_input_box") == "Utilize this text for the humanizer pill test."


def test_manual_sidebar_navigation_still_works():
    """Confirms the callback-based fix for pill buttons didn't break
    ordinary manual clicking of the sidebar radio itself."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.sidebar.radio(key="nav_page").set_value("🕘 History")
    at.run(timeout=30)
    assert list(at.exception) == []
    assert any(t.value == "History" for t in at.title)


def test_settings_page_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.sidebar.radio(key="nav_page").set_value("⚙️ Settings")
    at.run(timeout=30)
    assert list(at.exception) == []
    assert any(t.value == "Settings" for t in at.title)


@pytest.mark.skipif(not _api_reachable(), reason="Requires a running API server")
def test_full_analyze_and_improve_flow():
    """End-to-end: analyze real text, then follow the 'Improve this
    writing' button through to the Writing Improvement page with the
    text correctly carried over. Requires a live API (started manually
    or via run.py) since this exercises real HTTP calls, not mocks."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    at.sidebar.radio(key="nav_page").set_value("🔍 Analyze")
    at.run(timeout=30)

    sample_text = (
        "In today's rapidly evolving digital landscape, it is important to note that businesses "
        "must adapt to remain competitive. Furthermore, leveraging cutting-edge technology plays "
        "a crucial role in driving sustainable growth."
    )
    at.text_area(key="analyze_input").set_value(sample_text)
    at.run(timeout=30)

    analyze_btn = [b for b in at.button if b.label == "Analyze Writing"][0]
    analyze_btn.click()
    at.run(timeout=60)

    assert list(at.exception) == []
    assert list(at.error) == []
    assert _ss_get(at, "last_analysis") is not None

    improve_btn = [b for b in at.button if "Improve this writing" in b.label]
    assert len(improve_btn) == 1
    improve_btn[0].click()
    at.run(timeout=30)

    assert list(at.exception) == []
    assert _ss_get(at, "nav_page") == "✨ Writing Improvement"
    assert _ss_get(at, "humanize_input_box") == sample_text
